from vision.pipeline import VisionPipeline
from vision.replay_jsonl import read_jsonl
from memory.engine import DeskMemoryEngine


JSONL_PATH = "atlas_detections_sample.jsonl"


def main():

    vision = VisionPipeline()

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

    for detection_frame in read_jsonl(
        JSONL_PATH
    ):

        # ==================================
        # B
        # ==================================

        tracked_frame = vision.update(
            detection_frame
        )

        # ==================================
        # C
        # ==================================

        event = engine.update(
            tracked_frame
        )

        frame_count += 1

        detected = [
            obj.class_name
            for obj
            in detection_frame.detections
        ]

        tracked = [
            (
                obj.track_id,
                obj.class_name,
                obj.missed_frames,
            )
            for obj
            in tracked_frame.objects
        ]

        print(
            f"Frame "
            f"{detection_frame.frame_id:4d} | "
            f"Detected={detected} | "
            f"Tracked={tracked} | "
            f"State="
            f"{engine.person_state.state.name}"
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
    print(
        f"Replay finished: "
        f"{frame_count} frames"
    )
    print("======================================")

    baseline = [
        obj.class_name
        for obj
        in engine.baseline.objects.values()
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
            for obj
            in engine.session.objects.values()
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