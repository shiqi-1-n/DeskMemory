from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from common.types import Detection, DetectionFrame
from vision.my_utils import (
    get_labels_from_txt,
    letterbox,
    nms,
    scale_coords,
)


# DeskMemory v0.1 暂时只关注这些类别
ALLOWED_CLASSES = {
    "person",
    "phone",
    "book",
    "bottle",
}


# COCO 类别名 -> DeskMemory 内部统一名称
CLASS_NAME_MAP = {
    "cell_phone": "phone",
    "cell phone": "phone",
}


class PCDetector:
    """
    PC 端 ONNX 目标检测器。

    输入:
        OpenCV BGR 图像

    输出:
        DetectionFrame
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )

        # PC 开发阶段使用 CPU ONNX Runtime
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.labels = get_labels_from_txt(
            str(self.labels_path)
        )

    @staticmethod
    def _preprocess(img_bgr: np.ndarray):
        """
        与 Atlas OM 侧保持统一：

        BGR image
            ↓
        letterbox 640x640
            ↓
        BGR -> RGB
            ↓
        HWC -> CHW
            ↓
        float32 / 255.0
        """

        img, scale_ratio, pad_size = letterbox(
            img_bgr,
            new_shape=[640, 640],
        )

        # BGR -> RGB，同时 HWC -> CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)

        # CHW -> NCHW
        img = np.expand_dims(
            img,
            axis=0,
        ).astype(np.float32)

        img = np.ascontiguousarray(img) / 255.0

        return img, scale_ratio, pad_size

    def detect(
        self,
        img_bgr: np.ndarray,
        frame_id: int = 0,
        timestamp: float | None = None,
    ) -> DetectionFrame:

        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")

        if timestamp is None:
            timestamp = time.time()

        image_height, image_width = img_bgr.shape[:2]

        # 1. 预处理
        input_tensor, scale_ratio, pad_size = self._preprocess(
            img_bgr
        )

        # 2. ONNX 推理
        output = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor,
            },
        )[0]

        # 3. NMS
        boxout = nms(
            torch.tensor(output),
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
        )

        pred_all = boxout[0].cpu().numpy()

        detections: list[Detection] = []

        if len(pred_all) > 0:
            # 4. 将 640x640 坐标还原到原图
            scale_coords(
                [640, 640],
                pred_all[:, :4],
                img_bgr.shape,
                ratio_pad=(scale_ratio, pad_size),
            )

            # 5. 转成 DeskMemory Detection
            for pred in pred_all:
                x1, y1, x2, y2, confidence, class_id = pred[:6]

                class_id = int(class_id)

                raw_name = self.labels.get(
                    class_id,
                    str(class_id),
                )

                class_name = CLASS_NAME_MAP.get(
                    raw_name,
                    raw_name,
                )

                # v0.1 暂时忽略不相关类别
                if class_name not in ALLOWED_CLASSES:
                    continue

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox_xyxy=(
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ),
                    )
                )

        # 6. 输出统一 DetectionFrame
        return DetectionFrame(
            frame_id=frame_id,
            timestamp=float(timestamp),
            image_width=image_width,
            image_height=image_height,
            detections=detections,
        )

def draw_detection_frame(
    img_bgr: np.ndarray,
    frame: DetectionFrame,
) -> np.ndarray:
    result = img_bgr.copy()

    for detection in frame.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        cv2.rectangle(
            result,
            p1,
            p2,
            (0, 255, 0),
            3,
        )

        text = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            result,
            text,
            (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    return result

def main():
    model_path = "vision/models/yolov5s.onnx"
    labels_path = "vision/assets/coco_names.txt"
    image_path = "vision/debug/test.jpg"

    detector = PCDetector(
    model_path=model_path,
    labels_path=labels_path,
    conf_threshold=0.25,
)

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    frame = detector.detect(
        img_bgr,
        frame_id=0,
    )

    print()
    print("========== DetectionFrame ==========")
    print(f"frame_id: {frame.frame_id}")
    print(f"timestamp: {frame.timestamp}")
    print(
        f"image_size: "
        f"{frame.image_width} x {frame.image_height}"
    )
    print(f"detection_count: {len(frame.detections)}")
    print("------------------------------------")

    for detection in frame.detections:
        print(
            f"{detection.class_name:10s} "
            f"conf={detection.confidence:.3f} "
            f"bbox={detection.bbox_xyxy}"
        )
    result_img = draw_detection_frame(
        img_bgr,
        frame,
    )

    output_path = "vision/debug/result.jpg"

    cv2.imwrite(
        output_path,
        result_img,
    )

    print()
    print(f"Detection image saved to: {output_path}")


if __name__ == "__main__":
    main()