import json

from common.types import (
    Detection,
    DetectionFrame,
)
from memory.engine import DeskMemoryEngine
from vision.pipeline import VisionPipeline


JSONL_PATH = "atlas_detections_sample.jsonl"


def json_to_detection_frame(
    data: dict,
) -> DetectionFrame:

    detections = []

    for item in data.get("detections", []):

        bbox = item["bbox_xyxy"]

        detections.append(
            Detection(
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                confidence=float(item["confidence"]),
                bbox_xyxy=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
            )
        )

    return DetectionFrame(
        frame_id=int(data["frame_id"]),
        timestamp=float(data["timestamp"]),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        detections=detections,
    )


def main():

    # ==========================================
    # B
    # ==========================================

    vision = VisionPipeline()

    # ==========================================
    # C
    # ==========================================

    engine = DeskMemoryEngine(
        baseline_min_hits=2,
        presence_grace_time=2.0,
        possible_leave_time=2.0,
        confirmed_leave_time=5.0,
        baseline_calibration_time=3.0,
    )

    frame_count = 0

    print()
    print("======================================")
    print(" Atlas JSONL -> B -> C Replay")
    print("======================================")
    print()

    with open(
        JSONL_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # ==================================
            # A JSON
            #       ↓
            # DetectionFrame
            # ==================================

            data = json.loads(line)

            detection_frame = (
                json_to_detection_frame(data)
            )

            # ==================================
            # B
            # DetectionFrame -> TrackedFrame
            # ==================================

            tracked_frame = vision.update(
                detection_frame
            )

            # ==================================
            # C
            # TrackedFrame -> ForgottenEvent
            # ==================================

            event = engine.update(
                tracked_frame
            )

            frame_count += 1

            detected = [
                obj.class_name
                for obj in detection_frame.detections
            ]

            tracked = [
                (
                    obj.track_id,
                    obj.class_name,
                    obj.missed_frames,
                )
                for obj in tracked_frame.objects
            ]

            print(
                f"Frame {detection_frame.frame_id:4d} | "
                f"Detected={detected} | "
                f"Tracked={tracked} | "
                f"State={engine.person_state.state.name}"
            )

            if event is not None:

                names = [
                    item.class_name
                    for item in event.items
                ]

                print()
                print(
                    ">>> FORGOTTEN EVENT:",
                    names,
                )
                print()

    print()
    print("======================================")
    print(f"Replay finished: {frame_count} frames")
    print("======================================")

    baseline = [
        obj.class_name
        for obj in engine.baseline.objects.values()
    ]

    print(
        "Baseline:",
        baseline,
    )

    if engine.session is None:

        session = []

    else:

        session = [
            (
                obj.class_name,
                obj.visible,
            )
            for obj in engine.session.objects.values()
        ]

    print(
        "Session:",
        session,
    )

    print(
        "Final state:",
        engine.person_state.state.name,
    )


if __name__ == "__main__":
    main()