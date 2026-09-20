from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.engine import DeskMemoryEngine


class DeskMemoryReceiver:
    """
    B -> C 的统一接口。

    B 只需要不断把 TrackedFrame 传进来。
    """

    def __init__(self):

        self.engine = DeskMemoryEngine(
            baseline_min_hits=2,
            presence_grace_time=2.0,
            confirmed_leave_time=5.0,
        )

    def process(
        self,
        tracked_frame: TrackedFrame,
    ):
        """
        输入:
            B 输出的 TrackedFrame

        返回:
            ForgottenEvent 或 None
        """

        return self.engine.update(
            tracked_frame
        )


# ==================================================
# 以下部分只是模拟 B 的输出
# 真正联调时由 B 的 Tracker 替代
# ==================================================

def make_object(
    track_id: int,
    class_name: str,
    bbox_xyxy: tuple[
        float,
        float,
        float,
        float,
    ],
    confidence: float = 0.95,
):

    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
    )


def make_frame(
    frame_id: int,
    timestamp: float,
    objects: list[TrackedObject],
):

    return TrackedFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_width=1280,
        image_height=720,
        objects=objects,
    )


if __name__ == "__main__":

    receiver = DeskMemoryReceiver()

    # -----------------------------------------
    # 模拟 B 输出的连续 TrackedFrame
    # -----------------------------------------

    frames = [

        # 空桌面 Baseline
        make_frame(
            1,
            0.0,
            [
                make_object(
                    1,
                    "book",
                    (100, 100, 250, 250),
                ),
                make_object(
                    2,
                    "bottle",
                    (350, 100, 450, 300),
                ),
            ],
        ),

        make_frame(
            2,
            1.0,
            [
                make_object(
                    1,
                    "book",
                    (100, 100, 250, 250),
                ),
                make_object(
                    2,
                    "bottle",
                    (350, 100, 450, 300),
                ),
            ],
        ),

        # 用户进入并带来 phone
        make_frame(
            3,
            2.0,
            [
                make_object(
                    100,
                    "person",
                    (700, 50, 1100, 700),
                ),
                make_object(
                    1,
                    "book",
                    (100, 100, 250, 250),
                ),
                make_object(
                    2,
                    "bottle",
                    (350, 100, 450, 300),
                ),
                make_object(
                    4,
                    "phone",
                    (700, 300, 800, 370),
                ),
            ],
        ),

        # 用户离开
        make_frame(
            4,
            3.0,
            [
                make_object(
                    1,
                    "book",
                    (100, 100, 250, 250),
                ),
                make_object(
                    2,
                    "bottle",
                    (350, 100, 450, 300),
                ),
                make_object(
                    4,
                    "phone",
                    (700, 300, 800, 370),
                ),
            ],
        ),

        # 确认离场
        make_frame(
            5,
            8.5,
            [
                make_object(
                    1,
                    "book",
                    (100, 100, 250, 250),
                ),
                make_object(
                    2,
                    "bottle",
                    (350, 100, 450, 300),
                ),
                make_object(
                    4,
                    "student_card",
                    (700, 300, 800, 370),
                ),
            ],
        ),
    ]

    for tracked_frame in frames:

        event = receiver.process(
            tracked_frame
        )

        print(
            f"frame={tracked_frame.frame_id}",
            f"state={receiver.engine.person_state.state.name}",
        )

        if event is not None:

            print(
                "Forgotten:",
                [
                    item.class_name
                    for item in event.items
                ],
            )