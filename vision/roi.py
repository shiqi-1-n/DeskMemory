from __future__ import annotations

from common.types import TrackedFrame


ROI = tuple[float, float, float, float]

FULL_FRAME_ROI: ROI = (
    0.0,
    0.0,
    1.0,
    1.0,
)


def _validate_roi(
    roi_xyxy: ROI,
) -> ROI:
    """
    Validate and normalize a ROI.

    ROI uses normalized coordinates:
        x1, y1, x2, y2 in [0.0, 1.0]

    Reverse drag direction is allowed.
    """

    x1, y1, x2, y2 = (
        float(value)
        for value in roi_xyxy
    )

    values = (
        x1,
        y1,
        x2,
        y2,
    )

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError(
            f"ROI coordinates must be in [0, 1], got {roi_xyxy}"
        )

    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    if left == right or top == bottom:
        raise ValueError(
            f"ROI must have non-zero area, got {roi_xyxy}"
        )

    return (
        left,
        top,
        right,
        bottom,
    )


class DeskROIFilter:
    """
    Runtime ROI filter for DeskMemory.

    Two independent normalized rectangular ROIs are used:

    object_roi_xyxy:
        Used for all non-person objects.

    person_roi_xyxy:
        Used only for person.

    A bbox is considered inside its ROI when its center point
    lies inside the corresponding rectangle.
    """

    def __init__(
        self,
        object_roi_xyxy: ROI = FULL_FRAME_ROI,
        person_roi_xyxy: ROI = FULL_FRAME_ROI,
    ):
        self.object_roi_xyxy = _validate_roi(
            object_roi_xyxy
        )

        self.person_roi_xyxy = _validate_roi(
            person_roi_xyxy
        )

    def set_rois(
        self,
        object_roi_xyxy: ROI | None = None,
        person_roi_xyxy: ROI | None = None,
    ) -> None:
        """
        Update ROI configuration at runtime.

        None means keep the current ROI unchanged.
        """

        if object_roi_xyxy is not None:
            self.object_roi_xyxy = _validate_roi(
                object_roi_xyxy
            )

        if person_roi_xyxy is not None:
            self.person_roi_xyxy = _validate_roi(
                person_roi_xyxy
            )

    def contains(
        self,
        class_name: str,
        bbox_xyxy: tuple[
            float,
            float,
            float,
            float,
        ],
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        Return whether a bbox should be kept.

        The bbox uses original-image pixel coordinates.
        ROI uses normalized coordinates.
        """

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "image_width and image_height must be positive"
            )

        if class_name == "person":
            roi_xyxy = self.person_roi_xyxy
        else:
            roi_xyxy = self.object_roi_xyxy

        x1, y1, x2, y2 = bbox_xyxy

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        roi_x1, roi_y1, roi_x2, roi_y2 = (
            roi_xyxy
        )

        roi_x1 *= image_width
        roi_x2 *= image_width

        roi_y1 *= image_height
        roi_y2 *= image_height

        return (
            roi_x1 <= center_x <= roi_x2
            and
            roi_y1 <= center_y <= roi_y2
        )

    def split(
        self,
        frame: TrackedFrame,
    ) -> tuple[
        TrackedFrame,
        TrackedFrame,
    ]:
        """
        Split a TrackedFrame into:

        kept_frame:
            objects inside their ROI

        filtered_frame:
            objects outside their ROI

        This is useful for UI/debug visualization.
        """

        kept_objects = []
        filtered_objects = []

        for obj in frame.objects:

            if self.contains(
                class_name=obj.class_name,
                bbox_xyxy=obj.bbox_xyxy,
                image_width=frame.image_width,
                image_height=frame.image_height,
            ):
                kept_objects.append(
                    obj
                )
            else:
                filtered_objects.append(
                    obj
                )

        kept_frame = TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=kept_objects,
        )

        filtered_frame = TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=filtered_objects,
        )

        return (
            kept_frame,
            filtered_frame,
        )

    def update(
        self,
        frame: TrackedFrame,
    ) -> TrackedFrame:
        """
        Existing pipeline-compatible interface.

        Only objects kept by ROI are returned.
        """

        kept_frame, _ = self.split(
            frame
        )

        return kept_frame