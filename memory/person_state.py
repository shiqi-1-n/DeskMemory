from enum import Enum, auto

from common.types import TrackedFrame


class PersonState(Enum):
    EMPTY = auto()
    PRESENT = auto()
    POSSIBLE_LEAVE = auto()
    CONFIRMED_LEAVE = auto()


class PersonStateMachine:

    def __init__(
        self,
        possible_leave_time: float = 2.0,
        confirmed_leave_time: float = 10.0,
    ):
        self.state = PersonState.EMPTY

        self.possible_leave_time = possible_leave_time
        self.confirmed_leave_time = confirmed_leave_time

        # 第一次检测不到用户的时间
        self.person_missing_since: float | None = None


    def update(
        self,
        frame: TrackedFrame,
    ) -> PersonState:

        person_present = any(
            obj.class_name == "person"
            for obj in frame.objects
        )

        now = frame.timestamp

        # ========================================
        # 当前检测到用户
        # ========================================

        if person_present:

            self.person_missing_since = None
            self.state = PersonState.PRESENT

            return self.state

        # ========================================
        # 系统启动后从未检测到过用户
        # ========================================

        if self.state == PersonState.EMPTY:
            return self.state

        # ========================================
        # 第一次检测不到用户
        # ========================================

        if self.person_missing_since is None:
            self.person_missing_since = now

        missing_time = (
            now - self.person_missing_since
        )

        # ========================================
        # 已达到确认离场时间
        # ========================================

        if missing_time >= self.confirmed_leave_time:

            self.state = PersonState.CONFIRMED_LEAVE

            return self.state

        # ========================================
        # 已达到可能离场时间
        # ========================================

        if missing_time >= self.possible_leave_time:

            self.state = PersonState.POSSIBLE_LEAVE

            return self.state

        # ========================================
        # 只是短暂消失
        # ========================================

        self.state = PersonState.PRESENT

        return self.state