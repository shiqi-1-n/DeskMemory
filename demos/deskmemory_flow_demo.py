from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.baseline import BaselineMemory
from memory.session import SessionMemory
from memory.person_state import (
    PersonStateMachine,
)
from memory.forgotten import ForgottenDetector


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


# ==========================================
# Step 1：建立 Baseline
# 桌上原本有 book + bottle
# ==========================================

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


print("=== Baseline ===")

for item in baseline.objects.values():
    print("-", item.class_name)


# ==========================================
# Step 2：初始化 Session / 状态机
# ==========================================

session = SessionMemory(
    baseline=baseline,
    presence_grace_time=2.0,
)

person_state = PersonStateMachine(
    confirmed_leave_time=5.0,
)

forgotten_detector = ForgottenDetector(
    session=session,
    person_state_machine=person_state,
)


# ==========================================
# Step 3：用户进入
# 带来了 phone + student_card
# ==========================================

f3 = frame(
    3,
    2.0,
    [
        obj(
            100,
            "person",
            (700, 50, 1100, 700),
        ),
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

person_state.update(f3)
session.update(f3)

print(
    "\nUser state:",
    person_state.state.name,
)

print("Session objects:")

for item in session.objects.values():
    print(
        "-",
        item.class_name,
        item.visible,
    )


# ==========================================
# Step 4：用户拿走 phone
# student_card 仍然留着
# ==========================================

f4 = frame(
    4,
    5.0,
    [
        obj(
            100,
            "person",
            (700, 50, 1100, 700),
        ),
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

person_state.update(f4)
session.update(f4)


print("\nAfter phone removed:")

for item in session.objects.values():
    print(
        "-",
        item.class_name,
        item.visible,
    )


# ==========================================
# Step 5：用户刚刚离开
# ==========================================

f5 = frame(
    5,
    6.0,
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

person_state.update(f5)
session.update(f5)

event = forgotten_detector.update(
    timestamp=f5.timestamp
)

print(
    "\nAt t=6:",
    person_state.state.name,
)

print(
    "Forgotten event:",
    event,
)


# ==========================================
# Step 6：离开超过 5 秒
# ==========================================

f6 = frame(
    6,
    11.5,
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

person_state.update(f6)
session.update(f6)

event = forgotten_detector.update(
    timestamp=f6.timestamp
)


print(
    "\nAt t=11.5:",
    person_state.state.name,
)


if event is None:

    print(
        "Forgotten event: None"
    )

else:

    print(
        "Forgotten event generated!"
    )

    print(
        "Event ID:",
        event.event_id,
    )

    print(
        "Items:"
    )

    for item in event.items:
        print(
            "-",
            item.class_name,
        )


# ==========================================
# Step 7：再来一帧
# 验证不会重复报警
# ==========================================

f7 = frame(
    7,
    12.0,
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

person_state.update(f7)
session.update(f7)

event2 = forgotten_detector.update(
    timestamp=f7.timestamp
)

print(
    "\nRepeat event:",
    event2,
)