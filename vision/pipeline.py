from __future__ import annotations

from common.types import DetectionFrame, TrackedFrame
from vision.stability import StabilityFilter
from vision.tracker import SimpleTracker


class VisionPipeline:
    """
    Public B -> C interface.

    DetectionFrame
        -> tracking
        -> stability filtering
        -> stable TrackedFrame
    """

    def __init__(self):
        self.tracker = SimpleTracker(
            iou_threshold=0.25,
            center_distance_threshold=0.10,
            max_missed_frames=10,
        )

        self.stability = StabilityFilter(
            min_observed_frames=3,
        )

    def update(
        self,
        frame: DetectionFrame,
    ) -> TrackedFrame:

        raw_tracked_frame = self.tracker.update(
            frame
        )

        stable_tracked_frame = (
            self.stability.update(
                raw_tracked_frame
            )
        )

        return stable_tracked_frame