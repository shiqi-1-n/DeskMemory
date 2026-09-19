from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from common.types import Detection, DetectionFrame
from vision.tracker import SimpleTracker


# 兼容旧版 COCO 名称
CLASS_NAME_MAP = {
    "cell_phone": "phone",
    "cell phone": "phone",
}


# DeskMemory v0.1 当前关注的类别
ALLOWED_CLASSES = {
    "person",
    "phone",
    "book",
    "bottle",
}

CLASS_CONFIDENCE_THRESHOLDS = {
    "person": 0.30,
    "phone": 0.20,
    "book": 0.20,
    "bottle": 0.25,
}


def message_to_detection_frame(
    message: dict,
) -> DetectionFrame:
    """
    把 Atlas 的一条 JSON 消息
    转换成 DeskMemory 的 DetectionFrame。
    """

    detections: list[Detection] = []

    for item in message.get("detections", []):
        raw_name = str(item["class_name"])

        class_name = CLASS_NAME_MAP.get(
            raw_name,
            raw_name,
        )

        # 暂时忽略 cup、mouse、knife 等非核心类别
        if class_name not in ALLOWED_CLASSES:
            continue

        confidence = float(item["confidence"])

        threshold = CLASS_CONFIDENCE_THRESHOLDS.get(
            class_name,
            0.20,
        )

        if confidence < threshold:
            continue

        bbox = item["bbox_xyxy"]

        if len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox_xyxy: {bbox}"
            )

        detection = Detection(
            class_id=int(item["class_id"]),
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=(
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            ),
        )

        detections.append(detection)

    return DetectionFrame(
        frame_id=int(message["frame_id"]),
        timestamp=float(message["timestamp"]),
        image_width=int(message["image_width"]),
        image_height=int(message["image_height"]),
        detections=detections,
    )


def read_jsonl(
    path: str | Path,
) -> Iterator[DetectionFrame]:
    """
    一行一行读取 JSONL。
    每一行转换成一个 DetectionFrame。
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                message = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            yield message_to_detection_frame(
                message
            )


def main():
    path = (
        "vision/debug/"
        "atlas_detections.jsonl"
    )

    tracker = SimpleTracker(
        iou_threshold=0.25,
        center_distance_threshold=0.10,
        max_missed_frames=10,
    )

    for frame in read_jsonl(path):
        tracked_frame = tracker.update(
            frame
        )

        tracks = [
            (
                obj.track_id,
                obj.class_name,
                obj.missed_frames,
            )
            for obj in tracked_frame.objects
        ]

        print(
            f"frame={frame.frame_id:3d} "
            f"detections="
            f"{len(frame.detections):2d} "
            f"tracks={tracks}"
        )


if __name__ == "__main__":
    main()