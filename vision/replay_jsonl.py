from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from common.types import Detection, DetectionFrame


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

        bbox = item["bbox_xyxy"]

        if len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox_xyxy: {bbox}"
            )

        detection = Detection(
            class_id=int(item["class_id"]),
            class_name=class_name,
            confidence=float(item["confidence"]),
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

    frame_count = 0

    for frame in read_jsonl(path):
        frame_count += 1

        names = [
            detection.class_name
            for detection in frame.detections
        ]

        print(
            f"frame={frame.frame_id:3d} "
            f"size={frame.image_width}x"
            f"{frame.image_height} "
            f"detections={names}"
        )

    print()
    print(f"Total frames: {frame_count}")


if __name__ == "__main__":
    main()