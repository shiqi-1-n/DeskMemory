import os
import time

import cv2
import numpy as np
import torch

from mindx.sdk import Tensor
from mindx.sdk import base

from my_utils import draw_bbox, get_labels_from_txt, letterbox, nms, scale_coords


DEVICE_ID = 0
CAMERA_ID = 0
MODEL_PATH = "model/yolov5s.om"
LABELS_PATH = "coco_names.txt"
OUTPUT_DIR = "out/camera_test"
INPUT_SIZE = (640, 640)
WARMUP_FRAMES = 10
MEASURED_FRAMES = 100
SAVE_FRAME_IDS = {1, 50, 100}
CONFIDENCE_THRESHOLD = 0.20


def milliseconds(start, end):
    return (end - start) * 1000.0


def summarize(name, values):
    array = np.asarray(values, dtype=np.float64)
    print(
        f"{name:<12} avg={array.mean():8.2f} ms  "
        f"min={array.min():8.2f} ms  max={array.max():8.2f} ms"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing MindX SDK...")
    base.mx_init()
    print(f"Loading model: {MODEL_PATH}")
    model = base.model(modelPath=MODEL_PATH, deviceId=DEVICE_ID)
    labels = get_labels_from_txt(LABELS_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {CAMERA_ID}")

    camera_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []
    failed_reads = 0
    processed_frames = 0
    measured_frames = 0

    print(
        f"Starting camera test: {WARMUP_FRAMES} warmup frames + "
        f"{MEASURED_FRAMES} measured frames"
    )

    try:
        while measured_frames < MEASURED_FRAMES:
            total_start = time.perf_counter()

            camera_start = time.perf_counter()
            ok, frame = cap.read()
            camera_end = time.perf_counter()
            if not ok or frame is None:
                failed_reads += 1
                print(f"Warning: camera read failed ({failed_reads})")
                if failed_reads >= 20:
                    raise RuntimeError("Camera read failed 20 times")
                continue

            processed_frames += 1

            preprocess_start = time.perf_counter()
            image, scale_ratio, pad_size = letterbox(frame, new_shape=list(INPUT_SIZE))
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.expand_dims(image, 0).astype(np.float32)
            image = np.ascontiguousarray(image) / 255.0
            input_tensor = Tensor(image)
            preprocess_end = time.perf_counter()

            inference_start = time.perf_counter()
            output_tensor = model.infer([input_tensor])[0]
            output_tensor.to_host()
            output = np.array(output_tensor)
            inference_end = time.perf_counter()

            postprocess_start = time.perf_counter()
            detections = nms(
                torch.tensor(output),
                conf_thres=CONFIDENCE_THRESHOLD,
                iou_thres=0.5,
            )[0].numpy()
            if len(detections) > 0:
                scale_coords(
                    list(INPUT_SIZE),
                    detections[:, :4],
                    frame.shape,
                    ratio_pad=(scale_ratio, pad_size),
                )
            postprocess_end = time.perf_counter()
            total_end = time.perf_counter()

            if processed_frames <= WARMUP_FRAMES:
                if processed_frames == WARMUP_FRAMES:
                    print("Warmup complete; starting measurements.")
                continue

            measured_frames += 1
            camera_times.append(milliseconds(camera_start, camera_end))
            preprocess_times.append(milliseconds(preprocess_start, preprocess_end))
            inference_times.append(milliseconds(inference_start, inference_end))
            postprocess_times.append(milliseconds(postprocess_start, postprocess_end))
            total_times.append(milliseconds(total_start, total_end))

            if measured_frames in SAVE_FRAME_IDS:
                rendered = draw_bbox(
                    detections, frame.copy(), (0, 255, 0), 2, labels
                )
                output_path = os.path.join(
                    OUTPUT_DIR, f"frame_{measured_frames:03d}.jpg"
                )
                cv2.imwrite(output_path, rendered)
                print(
                    f"Frame {measured_frames:3d}/{MEASURED_FRAMES}: "
                    f"detections={len(detections)}, saved={output_path}"
                )
            elif measured_frames % 10 == 0:
                print(
                    f"Frame {measured_frames:3d}/{MEASURED_FRAMES}: "
                    f"detections={len(detections)}, "
                    f"total={total_times[-1]:.2f} ms"
                )

    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        cap.release()
        print("Camera released.")

    if not total_times:
        print("No measured frames were completed.")
        return

    print("\n=== Performance summary ===")
    summarize("Camera", camera_times)
    summarize("Preprocess", preprocess_times)
    summarize("Inference", inference_times)
    summarize("Postprocess", postprocess_times)
    summarize("Total", total_times)
    average_total_ms = float(np.mean(total_times))
    print(f"Pipeline FPS avg={1000.0 / average_total_ms:.2f}")
    print(f"Measured frames={len(total_times)}, failed reads={failed_reads}")


if __name__ == "__main__":
    main()
