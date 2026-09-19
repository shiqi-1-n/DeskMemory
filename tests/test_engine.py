import unittest

from common.types import (
    TrackedFrame,
    TrackedObject,
)

from memory.engine import DeskMemoryEngine
from memory.person_state import PersonState


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


class TestDeskMemoryEngine(unittest.TestCase):

    def test_full_forgotten_flow(self):

        engine = DeskMemoryEngine(
            baseline_min_hits=2,
            presence_grace_time=2.0,
            possible_leave_time=2.0,
            confirmed_leave_time=5.0,
            baseline_calibration_time=1.0,
        )

        # ========================================
        # 1. Baseline
        # ========================================

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

        self.assertTrue(
            engine._baseline_ready
        )

        # ========================================
        # 2. 用户进入
        # 带来 phone + bottle
        # ========================================

        engine.update(
            frame(
                3,
                2.0,
                [
                    obj(
                        100,
                        "person",
                        (700, 50, 1100, 700),
                    ),
                    obj(1, "book"),
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
            )
        )

        self.assertEqual(
            engine.person_state.state,
            PersonState.PRESENT,
        )

        self.assertIsNotNone(
            engine.session
        )

        # ========================================
        # 3. 用户拿走 phone
        # ========================================

        engine.update(
            frame(
                4,
                5.0,
                [
                    obj(
                        100,
                        "person",
                        (700, 50, 1100, 700),
                    ),
                    obj(1, "book"),
                    obj(
                        4,
                        "bottle",
                        (700, 250, 800, 500),
                    ),
                ],
            )
        )

        # ========================================
        # 4. 用户刚离开
        # ========================================

        event = engine.update(
            frame(
                5,
                6.0,
                [
                    obj(1, "book"),
                    obj(
                        4,
                        "bottle",
                        (700, 250, 800, 500),
                    ),
                ],
            )
        )

        self.assertIsNone(event)

        self.assertEqual(
            engine.person_state.state,
            PersonState.PRESENT,
        )

        # ========================================
        # 5. 消失超过 2 秒
        # ========================================

        event = engine.update(
            frame(
                6,
                8.1,
                [
                    obj(1, "book"),
                    obj(
                        4,
                        "bottle",
                        (700, 250, 800, 500),
                    ),
                ],
            )
        )

        self.assertIsNone(event)

        self.assertEqual(
            engine.person_state.state,
            PersonState.POSSIBLE_LEAVE,
        )

        # ========================================
        # 6. 消失超过 5 秒
        # ========================================

        event = engine.update(
            frame(
                7,
                11.5,
                [
                    obj(1, "book"),
                    obj(
                        4,
                        "bottle",
                        (700, 250, 800, 500),
                    ),
                ],
            )
        )

        self.assertEqual(
            engine.person_state.state,
            PersonState.CONFIRMED_LEAVE,
        )

        self.assertIsNotNone(event)

        names = [
            item.class_name
            for item in event.items
        ]

        # bottle 被遗落
        self.assertIn(
            "bottle",
            names,
        )

        # phone 已经被拿走
        self.assertNotIn(
            "phone",
            names,
        )

        # Baseline book 不能被误报
        self.assertNotIn(
            "book",
            names,
        )


if __name__ == "__main__":
    unittest.main()