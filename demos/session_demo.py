from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.baseline import BaselineMemory
from memory.session import SessionMemory


def obj(
    track_id,
    class_name,
    bbox,
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


# ==================================
# 1. 建立 Baseline
# ==================================

baseline = BaselineMemory(
    min_hits=2
)

baseline.observe(
    frame(
        1,
        0.0,
        [
            obj(
                1,
                "book",
                (100, 100, 250, 250),
            ),
            obj(
                2,
                "bottle",
                (400, 100, 500, 300),
            ),
        ],
    )
)

baseline.observe(
    frame(
        2,
        1.0,
        [
            obj(
                1,
                "book",
                (102, 101, 252, 251),
            ),
            obj(
                2,
                "bottle",
                (401, 100, 501, 300),
            ),
        ],
    )
)

baseline.freeze()


# ==================================
# 2. 创建 Session
# ==================================

session = SessionMemory(
    baseline=baseline,
    presence_grace_time=2.0,
)


# ==================================
# 3. 用户带来了 phone + card
# ==================================

f3 = frame(
    3,
    2.0,
    [
        obj(
            1,
            "book",
            (102, 101, 252, 251),
        ),
        obj(
            2,
            "bottle",
            (401, 100, 501, 300),
        ),
        obj(
            3,
            "phone",
            (600, 300, 720, 420),
        ),
        obj(
            4,
            "student_card",
            (800, 300, 900, 370),
        ),
    ],
)

session.update(f3)


print("Session objects:")

for item in session.objects.values():
    print(
        "-",
        item.track_id,
        item.class_name,
        item.visible,
    )


# ==================================
# 4. phone 被拿走
# card 仍然在桌上
# ==================================

f4 = frame(
    4,
    5.0,
    [
        obj(
            1,
            "book",
            (102, 101, 252, 251),
        ),
        obj(
            2,
            "bottle",
            (401, 100, 501, 300),
        ),
        obj(
            4,
            "student_card",
            (800, 300, 900, 370),
        ),
    ],
)

session.update(f4)


print("\nAfter phone removed:")

for item in session.objects.values():
    print(
        "-",
        item.track_id,
        item.class_name,
        item.visible,
    )


print("\nVisible session objects:")

for item in session.get_visible_objects():
    print(
        "-",
        item.class_name,
    )