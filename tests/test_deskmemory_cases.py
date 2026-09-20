import unittest

from common.types import TrackedFrame, TrackedObject

from memory.baseline import BaselineMemory
from memory.session import SessionMemory
from memory.person_state import (
    PersonState,
    PersonStateMachine,
)
from memory.forgotten import ForgottenDetector
from memory.engine import (
    DeskMemoryEngine,
    EngineState,
)

def obj(
    track_id,
    class_name,
    bbox=(100, 100, 200, 200),
    missed_frames=0,
):
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.95,
        bbox_xyxy=bbox,
        missed_frames=missed_frames,
    )


def frame(frame_id, timestamp, objects):
    return TrackedFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_width=1280,
        image_height=720,
        objects=objects,
    )


def make_system():
    baseline = BaselineMemory(min_hits=2)

    baseline.observe(
        frame(
            1,
            0.0,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )
    )

    baseline.observe(
        frame(
            2,
            1.0,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )
    )

    baseline.freeze()

    session = SessionMemory(
        baseline=baseline,
        presence_grace_time=2.0,
    )

    person_state = PersonStateMachine(
        confirmed_leave_time=5.0,
    )

    detector = ForgottenDetector(
        session=session,
        person_state_machine=person_state,
    )

    return baseline, session, person_state, detector

class TestDeskMemory(unittest.TestCase):

    def test_forgotten_student_card(self):
        _, session, state, detector = make_system()

        # 用户在位，带来学生卡
        f1 = frame(
            3,
            2.0,
            [
                obj(100, "person"),
                obj(1, "book"),
                obj(2, "bottle"),
                obj(3, "student_card"),
            ],
        )

        state.update(f1)
        session.update(f1)

        # 用户离开，卡还在
        f2 = frame(
            4,
            3.0,
            [
                obj(1, "book"),
                obj(2, "bottle"),
                obj(3, "student_card"),
            ],
        )

        state.update(f2)
        session.update(f2)

        # 超过离场时间
        f3 = frame(
            5,
            8.5,
            [
                obj(1, "book"),
                obj(2, "bottle"),
                obj(3, "student_card"),
            ],
        )

        state.update(f3)
        session.update(f3)

        event = detector.update(f3.timestamp)

        self.assertIsNotNone(event)

        names = [
            item.class_name
            for item in event.items
        ]

        self.assertEqual(
            names,
            ["student_card"],
        )

    def test_nothing_forgotten(self):
        _, session, state, detector = make_system()

        f1 = frame(
            3,
            2.0,
            [
                obj(100, "person"),
                obj(1, "book"),
                obj(2, "bottle"),
                obj(3, "phone"),
            ],
        )

        state.update(f1)
        session.update(f1)

        # phone 已经拿走
        f2 = frame(
            4,
            5.0,
            [
                obj(100, "person"),
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f2)
        session.update(f2)

        # 用户离开
        f3 = frame(
            5,
            6.0,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f3)
        session.update(f3)

        f4 = frame(
            6,
            11.5,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f4)
        session.update(f4)

        event = detector.update(f4.timestamp)

        self.assertIsNone(event)


    def test_temporary_leave_no_warning(self):
        _, session, state, detector = make_system()

        f1 = frame(
            3,
            2.0,
            [
                obj(100, "person"),
                obj(3, "student_card"),
            ],
        )

        state.update(f1)
        session.update(f1)

        # 暂时消失
        f2 = frame(
            4,
            3.0,
            [
                obj(3, "student_card"),
            ],
        )

        state.update(f2)
        session.update(f2)

        self.assertEqual(
            state.state,
            PersonState.PRESENT,
        )

        # 很快回来
        f3 = frame(
            5,
            5.0,
            [
                obj(100, "person"),
                obj(3, "student_card"),
            ],
        )

        state.update(f3)
        session.update(f3)

        event = detector.update(f3.timestamp)

        self.assertEqual(
            state.state,
            PersonState.PRESENT,
        )

        self.assertIsNone(event)

    def test_baseline_objects_not_forgotten(self):
        _, session, state, detector = make_system()

        # 用户出现，但没有带来任何新东西
        f1 = frame(
            3,
            2.0,
            [
                obj(100, "person"),
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f1)
        session.update(f1)

        # 用户离开
        f2 = frame(
            4,
            3.0,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f2)
        session.update(f2)

        f3 = frame(
            5,
            8.5,
            [
                obj(1, "book"),
                obj(2, "bottle"),
            ],
        )

        state.update(f3)
        session.update(f3)

        event = detector.update(f3.timestamp)

        self.assertIsNone(event)

    def test_baseline_track_id_change(self):

        baseline = BaselineMemory(
            min_hits=2
        )

        # 原来的 book，ID = 1
        baseline.observe(
            frame(
                1,
                0.0,
                [
                    obj(
                        1,
                        "book",
                        (100, 100, 250, 250),
                    )
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
                    )
                ],
            )
        )

        baseline.freeze()

        # Tracker 重新分配 ID
        # 但是位置基本没变
        same_book_new_id = obj(
            17,
            "book",
            (105, 103, 255, 253),
        )

        self.assertTrue(
            baseline.is_baseline_object(
                same_book_new_id
            )
        )

    def test_session_track_id_change(self):
        _, session, state, _ = make_system()

        # 用户带来 phone，初始 Track ID = 3
        f1 = frame(
            3,
            2.0,
            [
                obj(
                    100,
                    "person",
                    (700, 50, 1100, 700),
                ),
                obj(
                    3,
                    "phone",
                    (500, 300, 650, 420),
                ),
            ],
        )

        state.update(f1)
        session.update(f1)

        self.assertEqual(
            len(session.objects),
            1,
        )

        # 同一个 phone 被 Tracker 重新赋予 ID = 18
        f2 = frame(
            4,
            3.0,
            [
                obj(
                    100,
                    "person",
                    (700, 50, 1100, 700),
                ),
                obj(
                    18,
                    "phone",
                    (505, 302, 655, 422),
                ),
            ],
        )

        state.update(f2)
        session.update(f2)

        # 应该仍然只有一个 phone
        self.assertEqual(
            len(session.objects),
            1,
        )

        self.assertIn(
            18,
            session.objects,
        )

        self.assertEqual(
            session.objects[18].class_name,
            "phone",
        )

    def test_missed_person_not_visible(self):

        engine = DeskMemoryEngine(
            baseline_min_hits=2,
            presence_grace_time=2.0,
            confirmed_leave_time=5.0,
            baseline_calibration_time=1.0,
        )

        # =====================================
        # 1. 空桌面建立 Baseline
        # =====================================

        engine.update(
            frame(
                1,
                0.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        engine.update(
            frame(
                2,
                1.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        # =====================================
        # 2. 用户真正出现
        # missed_frames = 0
        # =====================================

        engine.update(
            frame(
                3,
                2.0,
                [
                    obj(
                        100,
                        "person",
                        missed_frames=0,
                    ),
                    obj(1, "book"),
                    obj(3, "phone"),
                ],
            )
        )

        self.assertEqual(
            engine.person_state.state,
            PersonState.PRESENT,
        )

        # =====================================
        # 3. Tracker 仍保留 person
        # 但当前帧实际已经没看到人
        # =====================================

        engine.update(
            frame(
                4,
                3.0,
                [
                    obj(
                        100,
                        "person",
                        missed_frames=1,
                    ),
                    obj(1, "book"),
                    obj(3, "phone"),
                ],
            )
        )

        # 刚开始漏检，还没有达到 2 秒
        self.assertEqual(
            engine.person_state.state,
            PersonState.PRESENT,
        )

        engine.update(
            frame(
                5,
                5.1,
                [
                    obj(
                        100,
                        "person",
                        missed_frames=3,
                    ),
                    obj(1, "book"),
                    obj(3, "phone"),
                ],
            )
        )

        # 已经持续不可见超过 2 秒
        self.assertEqual(
            engine.person_state.state,
            PersonState.POSSIBLE_LEAVE,
        )

        # =====================================
        # 4. person 轨迹仍然被 Tracker 保留
        # 但已经持续不可见超过 5 秒
        # =====================================

        event = engine.update(
            frame(
                6,
                8.5,
                [
                    obj(
                        100,
                        "person",
                        missed_frames=6,
                    ),
                    obj(1, "book"),
                    obj(3, "phone"),
                ],
            )
        )

        self.assertEqual(
            engine.person_state.state,
            PersonState.CONFIRMED_LEAVE,
        )

        # phone 仍真实留在桌面，所以应产生遗落事件
        self.assertIsNotNone(event)

        names = [
            item.class_name
            for item in event.items
        ]

        self.assertIn(
            "phone",
            names,
        )

    def test_baseline_waits_for_empty_desk(self):
        engine = DeskMemoryEngine(
            baseline_min_hits=2,
            presence_grace_time=2.0,
            confirmed_leave_time=5.0,
            baseline_calibration_time=2.0,
        )

        # =====================================
        # 1. 程序启动时，人已经坐在桌前
        # Baseline 不能直接完成
        # =====================================

        engine.update(
            frame(
                1,
                0.0,
                [
                    obj(100, "person"),
                    obj(3, "phone"),
                ],
            )
        )

        self.assertEqual(
            engine.state,
            EngineState.BUILDING_BASELINE,
        )

        self.assertFalse(
            engine._baseline_ready
        )

        self.assertEqual(
            len(engine.baseline.objects),
            0,
        )

        # =====================================
        # 2. 用户离开，桌面恢复为空桌状态
        # 只剩原本就在桌上的 book
        # =====================================

        engine.update(
            frame(
                2,
                1.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        self.assertFalse(
            engine._baseline_ready
        )

        # =====================================
        # 3. 连续空桌 1 秒
        # 还没有达到 2 秒
        # =====================================

        engine.update(
            frame(
                3,
                2.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        self.assertFalse(
            engine._baseline_ready
        )

        # =====================================
        # 4. 连续空桌达到 2 秒
        # Baseline 校准完成
        # =====================================

        engine.update(
            frame(
                4,
                3.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        self.assertTrue(
            engine._baseline_ready
        )

        # =====================================
        # 5. 用户重新进入
        # 此时才正式 freeze Baseline
        # =====================================

        engine.update(
            frame(
                5,
                4.0,
                [
                    obj(100, "person"),
                    obj(1, "book"),
                    obj(4, "phone"),
                ],
            )
        )

        self.assertEqual(
            engine.state,
            EngineState.ACTIVE,
        )

        # Baseline 中只能有原本的 book
        baseline_names = [
            item.class_name
            for item in engine.baseline.objects.values()
        ]

        self.assertEqual(
            baseline_names,
            ["book"],
        )

        # 用户后来带来的 phone 应该进入 Session
        session_names = [
            item.class_name
            for item in engine.session.objects.values()
        ]

        self.assertIn(
            "phone",
            session_names,
        )

    def test_engine_reset_for_recalibration(self):
        engine = DeskMemoryEngine(
            baseline_min_hits=2,
            presence_grace_time=2.0,
            possible_leave_time=2.0,
            confirmed_leave_time=5.0,
            baseline_calibration_time=1.0,
        )

        # 建立 Baseline
        engine.update(
            frame(
                1,
                0.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        engine.update(
            frame(
                2,
                1.0,
                [
                    obj(1, "book"),
                ],
            )
        )

        # 用户进入
        engine.update(
            frame(
                3,
                2.0,
                [
                    obj(100, "person"),
                    obj(1, "book"),
                    obj(3, "phone"),
                ],
            )
        )

        self.assertEqual(
            engine.state,
            EngineState.ACTIVE,
        )

        self.assertIsNotNone(
            engine.session
        )

        # =====================================
        # 重新校准
        # =====================================

        engine.reset()

        self.assertEqual(
            engine.state,
            EngineState.BUILDING_BASELINE,
        )

        self.assertFalse(
            engine._baseline_ready
        )

        self.assertIsNone(
            engine.session
        )

        self.assertIsNone(
            engine.forgotten_detector
        )

        self.assertEqual(
            engine.person_state.state,
            PersonState.EMPTY,
        )

        self.assertEqual(
            len(engine.baseline.objects),
            0,
        )

if __name__ == "__main__":
    unittest.main()