import unittest

from common.types import (
    Detection,
    DetectionFrame,
)

from vision.pipeline import VisionPipeline


class RuntimeROITest(unittest.TestCase):

    def make_frame(
        self,
        frame_id: int,
    ) -> DetectionFrame:

        return DetectionFrame(
            frame_id=frame_id,
            timestamp=float(frame_id),
            image_width=640,
            image_height=480,
            detections=[
                Detection(
                    class_id=67,
                    class_name="phone",
                    confidence=0.9,
                    bbox_xyxy=(50, 100, 100, 180),
                ),
                Detection(
                    class_id=39,
                    class_name="bottle",
                    confidence=0.9,
                    bbox_xyxy=(500, 100, 550, 200),
                ),
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.9,
                    bbox_xyxy=(400, 50, 520, 450),
                ),
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.8,
                    bbox_xyxy=(50, 50, 170, 450),
                ),
            ],
        )

    def test_object_and_person_use_independent_rois(
        self,
    ):
        vision = VisionPipeline()

        vision.set_rois(
            object_roi_xyxy=(
                0.0,
                0.0,
                0.5,
                1.0,
            ),
            person_roi_xyxy=(
                0.5,
                0.0,
                1.0,
                1.0,
            ),
        )

        for frame_id in range(1, 4):
            vision.update(
                self.make_frame(frame_id)
            )

        kept = [
            (
                obj.class_name,
                obj.bbox_xyxy,
            )
            for obj in vision.last_kept_frame.objects
        ]

        filtered = [
            (
                obj.class_name,
                obj.bbox_xyxy,
            )
            for obj in vision.last_filtered_frame.objects
        ]

        self.assertIn(
            (
                "phone",
                (50, 100, 100, 180),
            ),
            kept,
        )

        self.assertIn(
            (
                "person",
                (400, 50, 520, 450),
            ),
            kept,
        )

        self.assertIn(
            (
                "bottle",
                (500, 100, 550, 200),
            ),
            filtered,
        )

        self.assertIn(
            (
                "person",
                (50, 50, 170, 450),
            ),
            filtered,
        )

    def test_reset_clears_tracker_and_stability(
        self,
    ):
        vision = VisionPipeline()

        vision.set_rois(
            object_roi_xyxy=(
                0.0,
                0.0,
                0.5,
                1.0,
            ),
            person_roi_xyxy=(
                0.5,
                0.0,
                1.0,
                1.0,
            ),
        )

        for frame_id in range(1, 4):
            vision.update(
                self.make_frame(frame_id)
            )

        first_ids = [
            obj.track_id
            for obj in vision.last_kept_frame.objects
        ]

        self.assertEqual(
            first_ids,
            [1, 3],
        )

        vision.reset()

        # Stability should also restart,
        # so first two observations are not stable yet.
        result = vision.update(
            self.make_frame(4)
        )

        self.assertEqual(
            result.objects,
            [],
        )

        result = vision.update(
            self.make_frame(5)
        )

        self.assertEqual(
            result.objects,
            [],
        )

        result = vision.update(
            self.make_frame(6)
        )

        second_ids = [
            obj.track_id
            for obj in result.objects
        ]

        self.assertEqual(
            second_ids,
            [1, 3],
        )

    def test_reverse_drag_roi_is_normalized(
        self,
    ):
        vision = VisionPipeline()

        vision.set_rois(
            object_roi_xyxy=(
                0.5,
                1.0,
                0.0,
                0.0,
            ),
        )

        self.assertEqual(
            vision.roi.object_roi_xyxy,
            (
                0.0,
                0.0,
                0.5,
                1.0,
            ),
        )

    def test_invalid_roi_is_rejected(
        self,
    ):
        vision = VisionPipeline()

        with self.assertRaises(
            ValueError
        ):
            vision.set_rois(
                object_roi_xyxy=(
                    -0.1,
                    0.0,
                    0.5,
                    1.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()