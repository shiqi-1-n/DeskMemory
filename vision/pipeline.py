from __future__ import annotations

from common.types import DetectionFrame, TrackedFrame
from vision.roi import (
    DeskROIFilter,
    FULL_FRAME_ROI,
    ROI,
)
from vision.stability import StabilityFilter
from vision.tracker import SimpleTracker


class VisionPipeline:
    """
    Public B -> C interface.

    DetectionFrame
        -> tracking
        -> stability filtering
        -> ROI filtering
        -> TrackedFrame

    Runtime ROI configuration:
        - object ROI for non-person objects
        - person ROI for person

    Calling reset() clears tracker/stability history
    while keeping the current ROI configuration.
    """

    def __init__(
        self,
        object_roi_xyxy: ROI = FULL_FRAME_ROI,
        person_roi_xyxy: ROI = FULL_FRAME_ROI,
    ):
        self.roi = DeskROIFilter(
            object_roi_xyxy=object_roi_xyxy,
            person_roi_xyxy=person_roi_xyxy,
        )

        self.last_kept_frame: TrackedFrame | None = None
        self.last_filtered_frame: TrackedFrame | None = None

        self._reset_temporal_state()

    def _reset_temporal_state(
        self,
    ) -> None:
        """
        Recreate tracker and stability filter.

        ROI configuration is intentionally preserved.
        """

        self.tracker = SimpleTracker(
            iou_threshold=0.25,
            center_distance_threshold=0.10,
            max_missed_frames=10,
        )

        self.stability = StabilityFilter(
            min_observed_frames=3,
        )

        self.last_kept_frame = None
        self.last_filtered_frame = None

    def set_rois(
        self,
        object_roi_xyxy: ROI | None = None,
        person_roi_xyxy: ROI | None = None,
    ) -> None:
        """
        Update ROI configuration at runtime.

        Coordinates are normalized:
            0.0 ~ 1.0

        None means keep the current ROI unchanged.
        """

        self.roi.set_rois(
            object_roi_xyxy=object_roi_xyxy,
            person_roi_xyxy=person_roi_xyxy,
        )

    def reset(
        self,
    ) -> None:
        """
        Clear tracking and stability history.

        Current ROI settings are preserved.

        UI should call this after changing ROI,
        together with DeskMemoryEngine.reset().
        """

        self._reset_temporal_state()

    def update(
        self,
        frame: DetectionFrame,
    ) -> TrackedFrame:

        raw_tracked_frame = self.tracker.update(
            frame
        )

        stable_tracked_frame = self.stability.update(
            raw_tracked_frame
        )

        (
            kept_frame,
            filtered_frame,
        ) = self.roi.split(
            stable_tracked_frame
        )

        # For UI / debug visualization.
        self.last_kept_frame = kept_frame
        self.last_filtered_frame = filtered_frame

        # Only kept objects continue to C.
        return kept_frame