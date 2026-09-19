from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.engine import DeskMemoryEngine
from memory.event_log import EventLogger
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


# ==================================================
# DeskMemory Engine
# ==================================================

engine = DeskMemoryEngine(
    baseline_min_hits=2,
    presence_grace_time=2.0,
    possible_leave_time=2.0,
    confirmed_leave_time=5.0,

    # Demo 中 t=0、t=1 为无人状态，
    # 因此 1 秒即可完成 Baseline 校准
    baseline_calibration_time=1.0,
)


# ==================================================
# Event Logger
# ==================================================

logger = EventLogger(
    log_path="logs/events.jsonl"
)


# ==================================================
# UI
# ==================================================

ui = DeskMemoryUI()


# ==================================================
# 模拟真实摄像头连续产生的 TrackedFrame
#
# 第一版真实检测类别统一为：
# person / phone / book / bottle
# ==================================================

frames = [

    # --------------------------------------------------
    # Frame 1
    # 空桌面：开始建立 Baseline
    #
    # 桌面原本只有 book
    # --------------------------------------------------

    frame(
        1,
        0.0,
        [
            obj(
                1,
                "book",
                (100, 100, 300, 260),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 2
    # 连续无人 1 秒
    #
    # Baseline 校准完成
    # --------------------------------------------------

    frame(
        2,
        1.0,
        [
            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 3
    # 用户进入
    #
    # 用户带来了：
    # phone
    # bottle
    #
    # 此时 Baseline 冻结
    # --------------------------------------------------

    frame(
        3,
        2.0,
        [
            obj(
                100,
                "person",
                (700, 50, 1100, 700),
            ),

            # 原有 Baseline 物品
            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            # 用户带来的 phone
            obj(
                3,
                "phone",
                (400, 300, 550, 420),
            ),

            # 用户带来的 bottle
            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 4
    # 用户仍然在位
    #
    # phone + bottle 都还在
    # --------------------------------------------------

    frame(
        4,
        3.0,
        [
            obj(
                100,
                "person",
                (700, 50, 1100, 700),
            ),

            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            obj(
                3,
                "phone",
                (400, 300, 550, 420),
            ),

            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 5
    # 用户拿走 phone
    #
    # bottle 仍然留在桌面
    # --------------------------------------------------

    frame(
        5,
        6.0,
        [
            obj(
                100,
                "person",
                (700, 50, 1100, 700),
            ),

            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 6
    # 用户刚刚离开
    #
    # missing time = 0 秒
    #
    # 根据新状态机：
    # 此时仍然保持 PRESENT
    # --------------------------------------------------

    frame(
        6,
        7.0,
        [
            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 7
    # 用户持续离开
    #
    # missing time = 2 秒
    #
    # 进入 POSSIBLE_LEAVE
    # --------------------------------------------------

    frame(
        7,
        9.0,
        [
            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),

    # --------------------------------------------------
    # Frame 8
    # 用户持续离开超过 5 秒
    #
    # t = 12.5
    # 第一次消失 t = 7.0
    #
    # 12.5 - 7.0 = 5.5 秒
    #
    # → CONFIRMED_LEAVE
    # → bottle 仍在
    # → ForgottenEvent
    # --------------------------------------------------

    frame(
        8,
        12.5,
        [
            obj(
                1,
                "book",
                (102, 101, 302, 261),
            ),

            obj(
                4,
                "bottle",
                (700, 250, 800, 500),
            ),
        ],
    ),
]


# ==================================================
# 实时播放模拟数据
# ==================================================

def run_step(index=0):

    # 所有模拟帧播放结束
    if index >= len(frames):
        return

    current_frame = frames[index]

    # 把 B 模块未来会提供的 TrackedFrame
    # 送给 C 的 DeskMemoryEngine
    event = engine.update(
        current_frame
    )

    # 更新 UI
    ui.update_view(
        engine,
        event,
    )

    # 如果产生遗落事件
    if event is not None:

        # 保存日志
        logger.write(event)

        print(
            "Forgotten event:",
            [
                item.class_name
                for item in event.items
            ],
        )

    # 1.5 秒后播放下一帧
    ui.root.after(
        1500,
        lambda: run_step(index + 1),
    )


# ==================================================
# 启动 Demo
# ==================================================

# 窗口出现 0.5 秒后开始播放
ui.root.after(
    500,
    run_step,
)

# 启动 Tkinter
ui.run()