from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionFrame:
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    missed_frames: int = 0


@dataclass(frozen=True)
class TrackedFrame:
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    objects: List[TrackedObject] = field(default_factory=list)