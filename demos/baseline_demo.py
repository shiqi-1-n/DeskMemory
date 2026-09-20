from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.baseline import BaselineMemory


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


baseline = BaselineMemory(
    min_hits=2
)


# 用户进入前，桌上已有 book 和 bottle
f1 = frame(
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

f2 = frame(
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

baseline.observe(f1)
baseline.observe(f2)

baseline.freeze()


print("Baseline objects:")

for item in baseline.objects.values():
    print(
        "-",
        item.track_id,
        item.class_name,
        item.hits,
    )


# 用户进入后放入 phone
phone = obj(
    3,
    "phone",
    (600, 300, 720, 420),
)

print(
    "\nIs phone baseline?",
    baseline.is_baseline_object(phone),
)


book = obj(
    1,
    "book",
    (103, 102, 253, 252),
)

print(
    "Is book baseline?",
    baseline.is_baseline_object(book),
)