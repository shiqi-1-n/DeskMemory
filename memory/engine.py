from enum import Enum, auto

from common.types import (
    TrackedFrame,
    ForgottenEvent,
)

from memory.baseline import BaselineMemory
from memory.session import SessionMemory
from memory.person_state import PersonStateMachine
from memory.forgotten import ForgottenDetector


class EngineState(Enum):
    BUILDING_BASELINE = auto()
    ACTIVE = auto()


class DeskMemoryEngine:

    def __init__(
        self,
        baseline_min_hits: int = 2,
        presence_grace_time: float = 2.0,
        possible_leave_time: float = 2.0,
        confirmed_leave_time: float = 10.0,
        baseline_calibration_time: float = 3.0,
    ):
        self.state = EngineState.BUILDING_BASELINE

        # ========================================
        # Baseline 参数
        # ========================================

        self.baseline_min_hits = baseline_min_hits
        self.baseline_calibration_time = baseline_calibration_time

        # 连续无人状态开始时间
        self._baseline_empty_since: float | None = None

        # Baseline 是否已经完成校准
        self._baseline_ready = False

        self.baseline = BaselineMemory(
            min_hits=baseline_min_hits
        )

        # ========================================
        # Session
        # ========================================

        self.presence_grace_time = presence_grace_time

        self.session: SessionMemory | None = None

        # ========================================
        # Person State
        # ========================================

        self.person_state = PersonStateMachine(
            possible_leave_time=possible_leave_time,
            confirmed_leave_time=confirmed_leave_time,
        )

        # ========================================
        # Forgotten Detector
        # ========================================

        self.forgotten_detector: ForgottenDetector | None = None

    # ==================================================
    # 当前真实可见目标过滤
    # ==================================================

    def _visible_frame(
        self,
        frame: TrackedFrame,
    ) -> TrackedFrame:
        """
        只保留当前帧真正被检测到的目标。

        missed_frames == 0:
            当前帧真实可见。

        missed_frames > 0:
            Tracker 暂时保留的历史轨迹，
            不视为当前真实可见。
        """

        visible_objects = [
            obj
            for obj in frame.objects
            if obj.missed_frames == 0
        ]

        return TrackedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_width=frame.image_width,
            image_height=frame.image_height,
            objects=visible_objects,
        )

    # ==================================================
    # 主更新函数
    # ==================================================

    def update(
        self,
        frame: TrackedFrame,
    ) -> ForgottenEvent | None:

        # ========================================
        # 1. 过滤 Tracker 暂时保留的漏检轨迹
        # ========================================

        visible_frame = self._visible_frame(
            frame
        )

        person_present = any(
            obj.class_name == "person"
            for obj in visible_frame.objects
        )

        # ========================================
        # 2. 阶段一：建立 Baseline
        # ========================================

        if self.state == EngineState.BUILDING_BASELINE:

            # ------------------------------------
            # 当前没有真正检测到人
            # → 继续 Baseline 校准
            # ------------------------------------

            if not person_present:

                # 第一次进入连续无人状态
                if self._baseline_empty_since is None:
                    self._baseline_empty_since = (
                        frame.timestamp
                    )

                # 观察当前桌面物品
                self.baseline.observe(
                    visible_frame
                )

                empty_time = (
                    frame.timestamp
                    - self._baseline_empty_since
                )

                # 连续无人时间达到要求
                if (
                    empty_time
                    >= self.baseline_calibration_time
                ):
                    self._baseline_ready = True

                return None

            # ------------------------------------
            # 检测到人，但 Baseline
            # 还没有完成校准
            # ------------------------------------

            if not self._baseline_ready:

                # 本轮 Baseline 数据作废。
                # 防止用户已经在桌前时，
                # 把用户带来的物品误加入 Baseline。
                self.baseline = BaselineMemory(
                    min_hits=self.baseline_min_hits
                )

                self._baseline_empty_since = None

                return None

            # ------------------------------------
            # Baseline 已经校准完成
            # 用户第一次真正进入
            # ------------------------------------

            self.baseline.freeze()

            self.session = SessionMemory(
                baseline=self.baseline,
                presence_grace_time=self.presence_grace_time,
            )

            self.forgotten_detector = (
                ForgottenDetector(
                    session=self.session,
                    person_state_machine=self.person_state,
                )
            )

            self.state = EngineState.ACTIVE

        # ========================================
        # 3. 阶段二：Session 正式运行
        # ========================================

        self.person_state.update(
            visible_frame
        )

        assert self.session is not None
        assert self.forgotten_detector is not None

        self.session.update(
            visible_frame
        )

        # ========================================
        # 4. 判断是否产生遗落事件
        # ========================================

        event = self.forgotten_detector.update(
            timestamp=frame.timestamp
        )

        return event

    def reset(self) -> None:
        """
        重置 DeskMemory，
        重新进入 Baseline 校准阶段。
        """

        self.state = EngineState.BUILDING_BASELINE

        # 重新创建 Baseline
        self.baseline = BaselineMemory(
            min_hits=self.baseline_min_hits
        )

        # 清除 Baseline 校准状态
        self._baseline_empty_since = None
        self._baseline_ready = False

        # 清除 Session
        self.session = None

        # 重置人员状态机
        self.person_state = PersonStateMachine(
            possible_leave_time=self.person_state.possible_leave_time,
            confirmed_leave_time=self.person_state.confirmed_leave_time,
        )

        # 清除遗落检测器
        self.forgotten_detector = None