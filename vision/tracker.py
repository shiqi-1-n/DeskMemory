from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from common.types import (
    Detection,
    DetectionFrame,
    TrackedFrame,
    TrackedObject,
)


BBox = tuple[float, float, float, float]


def bbox_iou(
    box_a: BBox,
    box_b: BBox,
) -> float:
    """
    Calculate IoU between two xyxy boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def bbox_center(
    bbox: BBox,
) -> tuple[float, float]:
    """
    Return bbox center point.
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def normalized_center_distance(
    box_a: BBox,
    box_b: BBox,
    image_width: int,
    image_height: int,
) -> float:
    """
    Center distance normalized by
    image diagonal.
    """

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    distance = hypot(
        ax - bx,
        ay - by,
    )

    image_diagonal = hypot(
        image_width,
        image_height,
    )

    if image_diagonal <= 0:
        return 1.0

    return (
        distance
        / image_diagonal
    )


@dataclass
class TrackState:
    track_id: int

    class_id: int
    class_name: str

    confidence: float
    bbox_xyxy: BBox

    missed_frames: int = 0


class SimpleTracker:
    """
    Simple tracker for DeskMemory v0.1.

    Matching rules:
    - same class
    - sufficient IoU OR
      sufficiently close center
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        center_distance_threshold: float = 0.10,
        max_missed_frames: int = 10,
    ):
        self.iou_threshold = (
            iou_threshold
        )

        self.center_distance_threshold = (
            center_distance_threshold
        )

        self.max_missed_frames = (
            max_missed_frames
        )

        self._tracks: dict[
            int,
            TrackState,
        ] = {}

        self._next_track_id = 1

    def _create_track(
        self,
        detection: Detection,
    ) -> None:
        track = TrackState(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            missed_frames=0,
        )

        self._tracks[
            self._next_track_id
        ] = track

        self._next_track_id += 1

    def update(
        self,
        frame: DetectionFrame,
    ) -> TrackedFrame:

        available_track_ids = set(
            self._tracks.keys()
        )

        matched_detection_indices: set[int] = (
            set()
        )

        # High-confidence detections
        # are matched first.
        indexed_detections = list(
            enumerate(frame.detections)
        )

        indexed_detections.sort(
            key=lambda item:
            item[1].confidence,
            reverse=True,
        )

        for (
            detection_index,
            detection,
        ) in indexed_detections:

            best_track_id = None
            best_score = float("-inf")

            for track_id in (
                available_track_ids
            ):
                track = self._tracks[
                    track_id
                ]

                # Different classes cannot match.
                if (
                    track.class_name
                    != detection.class_name
                ):
                    continue

                iou = bbox_iou(
                    track.bbox_xyxy,
                    detection.bbox_xyxy,
                )

                distance = (
                    normalized_center_distance(
                        track.bbox_xyxy,
                        detection.bbox_xyxy,
                        frame.image_width,
                        frame.image_height,
                    )
                )

                # Reject if both geometry
                # conditions are poor.
                if (
                    iou
                    < self.iou_threshold
                    and distance
                    > self.center_distance_threshold
                ):
                    continue

                score = (
                    2.0 * iou
                    - distance
                )

                if score > best_score:
                    best_score = score
                    best_track_id = (
                        track_id
                    )

            if best_track_id is None:
                continue

            track = self._tracks[
                best_track_id
            ]

            track.class_id = (
                detection.class_id
            )

            track.class_name = (
                detection.class_name
            )

            track.confidence = (
                detection.confidence
            )

            track.bbox_xyxy = (
                detection.bbox_xyxy
            )

            track.missed_frames = 0

            available_track_ids.remove(
                best_track_id
            )

            matched_detection_indices.add(
                detection_index
            )

        # Old tracks that were not matched
        # are temporarily kept.
        for track_id in (
            available_track_ids
        ):
            self._tracks[
                track_id
            ].missed_frames += 1

        # Unmatched detections start
        # new tracks.
        for (
            detection_index,
            detection,
        ) in enumerate(
            frame.detections
        ):
            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            self._create_track(
                detection
            )

        # Remove tracks that have been
        # missing for too long.
        expired_track_ids = [
            track_id
            for (
                track_id,
                track,
            ) in self._tracks.items()
            if (
                track.missed_frames
                > self.max_missed_frames
            )
        ]

        for track_id in expired_track_ids:
            del self._tracks[
                track_id
            ]

        objects = [
            TrackedObject(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                confidence=track.confidence,
                bbox_xyxy=track.bbox_xyxy,
                missed_frames=(
                    track.missed_frames
                ),
            )
            for track in sorted(
                self._tracks.values(),
                key=lambda item:
                item.track_id,
            )
        ]

        return TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=objects,
        )