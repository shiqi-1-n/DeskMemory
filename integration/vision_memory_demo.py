from common.types import Detection, DetectionFrame
from vision.pipeline import VisionPipeline
from memory.engine import DeskMemoryEngine


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def det(
    class_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.95,
) -> Detection:

    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=bbox,
    )


def frame(
    frame_id: int,
    timestamp: float,
    detections: list[Detection],
) -> DetectionFrame:

    return DetectionFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        detections=detections,
    )


def print_result(
    tracked_frame,
    engine,
    event,
):

    objects = [
        (
            obj.track_id,
            obj.class_name,
            obj.missed_frames,
        )
        for obj in tracked_frame.objects
    ]

    print(
        f"\nFrame {tracked_frame.frame_id}"
        f"  t={tracked_frame.timestamp:.1f}"
    )

    print(
        "Tracked:",
        objects,
    )

    print(
        "Person state:",
        engine.person_state.state.name,
    )

    if engine.session is not None:

        session_objects = [
            (
                obj.track_id,
                obj.class_name,
                obj.visible,
            )
            for obj in engine.session.objects.values()
        ]

        print(
            "Session:",
            session_objects,
        )

    if event is not None:

        print(
            ">>> FORGOTTEN EVENT:",
            [
                item.class_name
                for item in event.items
            ],
        )


def main():

    # ==================================================
    # B：Vision
    # ==================================================

    vision = VisionPipeline()

    # ==================================================
    # C：Memory
    #
    # confirmed_leave_time 使用 5 秒方便演示
    # 正式配置可以继续使用公共配置中的 10 秒
    # ==================================================

    engine = DeskMemoryEngine(
        baseline_min_hits=2,
        presence_grace_time=2.0,
        possible_leave_time=2.0,
        confirmed_leave_time=5.0,
        baseline_calibration_time=3.0,
    )

    # ==================================================
    # 测试序列
    #
    # 1. 空桌状态：book 是原有物品
    # 2. 用户进入
    # 3. 用户带来 bottle
    # 4. 用户离开
    # 5. bottle 留在桌面
    # ==================================================

    book = det(
        73,
        "book",
        (100, 150, 300, 420),
    )

    person = det(
        0,
        "person",
        (450, 80, 850, 700),
    )

    bottle = det(
        39,
        "bottle",
        (900, 300, 1000, 600),
    )

    frames = [

        # --------------------------------------------------
        # Baseline
        #
        # StabilityFilter 需要连续观察多帧，
        # 因此连续提供 book
        # --------------------------------------------------

        frame(
            0,
            0.0,
            [book],
        ),

        frame(
            1,
            1.0,
            [book],
        ),

        frame(
            2,
            2.0,
            [book],
        ),

        frame(
            3,
            3.2,
            [book],
        ),

        # --------------------------------------------------
        # User enters
        #
        # person 连续出现三帧，
        # 让 StabilityFilter 确认 person
        # --------------------------------------------------

        frame(
            4,
            4.0,
            [book, person],
        ),

        frame(
            5,
            4.5,
            [book, person],
        ),

        frame(
            6,
            5.0,
            [book, person],
        ),

        # --------------------------------------------------
        # User brings bottle
        #
        # bottle 连续出现三帧，
        # 使其成为稳定 session object
        # --------------------------------------------------

        frame(
            7,
            6.0,
            [book, person, bottle],
        ),

        frame(
            8,
            6.5,
            [book, person, bottle],
        ),

        frame(
            9,
            7.0,
            [book, person, bottle],
        ),

        # --------------------------------------------------
        # User leaves
        #
        # bottle 留在桌面
        # --------------------------------------------------

        frame(
            10,
            8.0,
            [book, bottle],
        ),

        # 离开超过 possible_leave_time
        frame(
            11,
            10.5,
            [book, bottle],
        ),

        # 离开超过 confirmed_leave_time
        frame(
            12,
            13.5,
            [book, bottle],
        ),
    ]

    # ==================================================
    # B -> C Integration
    # ==================================================

    for detection_frame in frames:

        # B:
        # DetectionFrame -> TrackedFrame
        tracked_frame = vision.update(
            detection_frame
        )

        # C:
        # TrackedFrame -> ForgottenEvent
        event = engine.update(
            tracked_frame
        )

        print_result(
            tracked_frame,
            engine,
            event,
        )


if __name__ == "__main__":
    main()