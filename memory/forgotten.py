import uuid

from common.types import (
    ForgottenEvent,
    ForgottenItem,
)

from memory.session import SessionMemory
from memory.person_state import (
    PersonState,
    PersonStateMachine,
)


class ForgottenDetector:

    def __init__(
        self,
        session: SessionMemory,
        person_state_machine: PersonStateMachine,
    ):
        self.session = session
        self.person_state_machine = person_state_machine

        # 防止确认离场后每一帧都重复报警
        self.event_sent = False


    def update(
        self,
        timestamp: float,
    ) -> ForgottenEvent | None:

        state = self.person_state_machine.state

        # 用户回来了，可以重新允许下一次报警
        if state == PersonState.PRESENT:
            self.event_sent = False
            return None

        # 还没有真正离场
        if state != PersonState.CONFIRMED_LEAVE:
            return None

        # 这一轮已经发送过事件
        if self.event_sent:
            return None

        visible_objects = (
            self.session.get_visible_objects()
        )

        # 没有 Session 物品残留
        if not visible_objects:
            return None

        items = [
            ForgottenItem(
                track_id=obj.track_id,
                class_name=obj.class_name,
                bbox_xyxy=obj.bbox_xyxy,
                last_seen_time=obj.last_seen_time,
            )
            for obj in visible_objects
        ]

        event = ForgottenEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp,
            items=items,
        )

        self.event_sent = True

        return event