from __future__ import annotations

from dataclasses import dataclass

from common.types import TrackedFrame, TrackedObject


@dataclass
class StabilityState:
    track_id: int
    observed_frames: int = 0
    stable: bool = False


class StabilityFilter:
    """
    Filter short-lived noisy tracks.

    A track becomes stable after it has been
    actually observed for several frames.
    """

    def __init__(
        self,
        min_observed_frames: int = 3,
    ):
        self.min_observed_frames = (
            min_observed_frames
        )

        self._states: dict[
            int,
            StabilityState,
        ] = {}

    def update(
        self,
        frame: TrackedFrame,
    ) -> TrackedFrame:

        active_track_ids = {
            obj.track_id
            for obj in frame.objects
        }

        # Update stability counters.
        for obj in frame.objects:

            state = self._states.get(
                obj.track_id
            )

            if state is None:
                state = StabilityState(
                    track_id=obj.track_id
                )

                self._states[
                    obj.track_id
                ] = state

            # Only count frames where the
            # detector actually saw the object.
            if obj.missed_frames == 0:
                state.observed_frames += 1

            if (
                state.observed_frames
                >= self.min_observed_frames
            ):
                state.stable = True

        # Remove states whose tracker tracks
        # have already expired.
        expired_ids = [
            track_id
            for track_id in self._states
            if track_id
            not in active_track_ids
        ]

        for track_id in expired_ids:
            del self._states[track_id]

        stable_objects: list[
            TrackedObject
        ] = []

        for obj in frame.objects:
            state = self._states[
                obj.track_id
            ]

            if state.stable:
                stable_objects.append(
                    obj
                )

        return TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=stable_objects,
        )