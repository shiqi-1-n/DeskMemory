import unittest

from common.types import TrackedFrame, TrackedObject
from memory.person_state import (
    PersonState,
    PersonStateMachine,
)


def person():
    return TrackedObject(
        track_id=100,
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox_xyxy=(100, 100, 300, 600),
    )


def frame(
    frame_id: int,
    timestamp: float,
    objects,
):
    return TrackedFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_width=1280,
        image_height=720,
        objects=objects,
    )


class TestPersonStateMachine(unittest.TestCase):

    def test_leave_state_transition(self):

        state = PersonStateMachine(
            possible_leave_time=2.0,
            confirmed_leave_time=5.0,
        )

        # 用户出现
        result = state.update(
            frame(
                1,
                0.0,
                [person()],
            )
        )

        self.assertEqual(
            result,
            PersonState.PRESENT,
        )

        # 第一次看不到用户
        result = state.update(
            frame(
                2,
                1.0,
                [],
            )
        )

        self.assertEqual(
            result,
            PersonState.PRESENT,
        )

        # 消失还不到 2 秒
        result = state.update(
            frame(
                3,
                2.5,
                [],
            )
        )

        self.assertEqual(
            result,
            PersonState.PRESENT,
        )

        # 消失超过 2 秒
        result = state.update(
            frame(
                4,
                3.1,
                [],
            )
        )

        self.assertEqual(
            result,
            PersonState.POSSIBLE_LEAVE,
        )

        # 从第一次消失开始累计超过 5 秒
        result = state.update(
            frame(
                5,
                6.1,
                [],
            )
        )

        self.assertEqual(
            result,
            PersonState.CONFIRMED_LEAVE,
        )


    def test_temporary_leave_and_return(self):

        state = PersonStateMachine(
            possible_leave_time=2.0,
            confirmed_leave_time=5.0,
        )

        state.update(
            frame(
                1,
                0.0,
                [person()],
            )
        )

        # 暂时消失
        state.update(
            frame(
                2,
                1.0,
                [],
            )
        )

        self.assertEqual(
            state.state,
            PersonState.PRESENT,
        )

        # 2 秒内回来
        state.update(
            frame(
                3,
                2.0,
                [person()],
            )
        )

        self.assertEqual(
            state.state,
            PersonState.PRESENT,
        )

        self.assertIsNone(
            state.person_missing_since
        )


    def test_empty_before_first_user(self):

        state = PersonStateMachine(
            possible_leave_time=2.0,
            confirmed_leave_time=5.0,
        )

        result = state.update(
            frame(
                1,
                0.0,
                [],
            )
        )

        self.assertEqual(
            result,
            PersonState.EMPTY,
        )


if __name__ == "__main__":
    unittest.main()