from __future__ import annotations

from common.types import TrackedFrame


class DeskROIFilter:
    """
    桌面 ROI 过滤器。

    person 始终保留，因为 C 需要它判断用户是否在场。
    其他物品只有 bbox 中心点位于桌面 ROI 内时才保留。

    ROI 使用归一化坐标：
        0.0 ~ 1.0

    例如：
        (0.1, 0.3, 0.9, 0.95)

    表示：
        左边 10%
        上边 30%
        右边 90%
        下边 95%
    """

    def __init__(
        self,
        roi_xyxy: tuple[
            float,
            float,
            float,
            float,
        ] = (0.0, 0.0, 1.0, 1.0),
    ):
        self.roi_xyxy = roi_xyxy

    def _inside_roi(
        self,
        bbox_xyxy: tuple[
            float,
            float,
            float,
            float,
        ],
        image_width: int,
        image_height: int,
    ) -> bool:

        x1, y1, x2, y2 = bbox_xyxy

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        roi_x1, roi_y1, roi_x2, roi_y2 = (
            self.roi_xyxy
        )

        roi_x1 *= image_width
        roi_x2 *= image_width
        roi_y1 *= image_height
        roi_y2 *= image_height

        return (
            roi_x1 <= center_x <= roi_x2
            and roi_y1 <= center_y <= roi_y2
        )

    def update(
        self,
        frame: TrackedFrame,
    ) -> TrackedFrame:

        filtered_objects = []

        for obj in frame.objects:

            # person 不能被桌面 ROI 过滤掉
            if obj.class_name == "person":
                filtered_objects.append(obj)
                continue

            if self._inside_roi(
                bbox_xyxy=obj.bbox_xyxy,
                image_width=frame.image_width,
                image_height=frame.image_height,
            ):
                filtered_objects.append(obj)

        return TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=filtered_objects,
        )