from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.engine import DeskMemoryEngine
from ui.main_window import DeskMemoryUI


def obj(
    track_id,
    class_name,
    bbox=(100, 100, 200, 200),
):

    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.95,
        bbox_xyxy=bbox,
    )


def frame(
    frame_id,
    timestamp,
    objects,
):

    return TrackedFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_width=1280,
        image_height=720,
        objects=objects,
    )


engine = DeskMemoryEngine(
    baseline_min_hits=2,
    presence_grace_time=2.0,
    confirmed_leave_time=5.0,
)


# ------------------------------
# Baseline
# ------------------------------

engine.update(
    frame(
        1,
        0.0,
        [
            obj(1, "book"),
            obj(2, "bottle"),
        ],
    )
)

engine.update(
    frame(
        2,
        1.0,
        [
            obj(1, "book"),
            obj(2, "bottle"),
        ],
    )
)


# ------------------------------
# 用户进入
# ------------------------------

engine.update(
    frame(
        3,
        2.0,
        [
            obj(100, "person"),

            obj(1, "book"),
            obj(2, "bottle"),

            obj(
                3,
                "phone",
                (400, 300, 550, 420),
            ),

            obj(
                4,
                "student_card",
                (700, 300, 800, 370),
            ),
        ],
    )
)


# ------------------------------
# phone 被拿走
# ------------------------------

engine.update(
    frame(
        4,
        5.0,
        [
            obj(100, "person"),

            obj(1, "book"),
            obj(2, "bottle"),

            obj(
                4,
                "student_card",
                (700, 300, 800, 370),
            ),
        ],
    )
)


# ------------------------------
# 用户离开
# ------------------------------

engine.update(
    frame(
        5,
        6.0,
        [
            obj(1, "book"),
            obj(2, "bottle"),

            obj(
                4,
                "student_card",
                (700, 300, 800, 370),
            ),
        ],
    )
)


# ------------------------------
# 确认离场
# ------------------------------

event = engine.update(
    frame(
        6,
        11.5,
        [
            obj(1, "book"),
            obj(2, "bottle"),

            obj(
                4,
                "student_card",
                (700, 300, 800, 370),
            ),
        ],
    )
)


# ------------------------------
# UI
# ------------------------------

ui = DeskMemoryUI()

ui.update_view(
    engine,
    event,
)

ui.run()