import json
import socket
import time

import cv2
import numpy as np
import torch

from mindx.sdk import Tensor
from mindx.sdk import base

from my_utils import get_labels_from_txt, letterbox, nms, scale_coords


PC_HOST = "192.168.0.1"
PC_PORT = 9999
DEVICE_ID = 0
CAMERA_ID = 0
MODEL_PATH = "model/yolov5s.om"
LABELS_PATH = "coco_names.txt"
INPUT_SIZE = (640, 640)
WARMUP_FRAMES = 10
SEND_FRAMES = 100
CONFIDENCE_THRESHOLD = 0.20
CLASS_NAME_ALIASES = {"cell_phone": "phone"}


def detection_to_dict(row, labels):
    class_id = int(row[5])
    return {
        "class_id": class_id,
        "class_name": CLASS_NAME_ALIASES.get(labels[class_id], labels[class_id]),
        "confidence": round(float(row[4]), 6),
        "bbox_xyxy": [round(float(value), 2) for value in row[:4]],
    }


def main():
    print("Initializing MindX SDK...")
    base.mx_init()
    print(f"Loading model: {MODEL_PATH}")
    model = base.model(modelPath=MODEL_PATH, deviceId=DEVICE_ID)
    labels = get_labels_from_txt(LABELS_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {CAMERA_ID}")

    sent_frames = 0
    processed_frames = 0
    failed_reads = 0
    frames_with_detections = 0
    total_detections = 0
    send_times_ms = []
    pipeline_times_ms = []

    print(f"Connecting to PC at {PC_HOST}:{PC_PORT}...")

    try:
        with socket.create_connection((PC_HOST, PC_PORT), timeout=10) as connection:
            connection.settimeout(10)
            print("Connected; starting live inference.")

            while sent_frames < SEND_FRAMES:
                pipeline_start = time.perf_counter()

                ok, frame = cap.read()
                captured_at = time.time()
                if not ok or frame is None:
                    failed_reads += 1
                    print(f"Warning: camera read failed ({failed_reads})")
                    if failed_reads >= 20:
                        raise RuntimeError("Camera read failed 20 times")
                    continue

                processed_frames += 1

                image, scale_ratio, pad_size = letterbox(
                    frame, new_shape=list(INPUT_SIZE)
                )
                image = image[:, :, ::-1].transpose(2, 0, 1)
                image = np.expand_dims(image, 0).astype(np.float32)
                image = np.ascontiguousarray(image) / 255.0

                output_tensor = model.infer([Tensor(image)])[0]
                output_tensor.to_host()
                raw_output = np.array(output_tensor)

                detections = nms(
                    torch.tensor(raw_output),
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

                if processed_frames <= WARMUP_FRAMES:
                    if processed_frames == WARMUP_FRAMES:
                        print("Warmup complete; sending live DetectionFrame messages.")
                    continue

                sent_frames += 1
                detection_list = [
                    detection_to_dict(row, labels) for row in detections
                ]
                message = {
                    "frame_id": sent_frames,
                    "timestamp": captured_at,
                    "image_width": int(frame.shape[1]),
                    "image_height": int(frame.shape[0]),
                    "detections": detection_list,
                }
                payload = (
                    json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")

                send_start = time.perf_counter()
                connection.sendall(payload)
                send_end = time.perf_counter()
                pipeline_end = time.perf_counter()

                send_times_ms.append((send_end - send_start) * 1000.0)
                pipeline_times_ms.append((pipeline_end - pipeline_start) * 1000.0)

                if detection_list:
                    frames_with_detections += 1
                    total_detections += len(detection_list)

                if sent_frames == 1 or sent_frames % 10 == 0:
                    print(
                        f"Sent {sent_frames:3d}/{SEND_FRAMES}: "
                        f"detections={len(detection_list)}, "
                        f"pipeline={pipeline_times_ms[-1]:.2f} ms, "
                        f"send={send_times_ms[-1]:.2f} ms"
                    )

    except KeyboardInterrupt:
        print("Live test interrupted by user.")
    finally:
        cap.release()
        print("Camera released.")

    print("\n=== Live send summary ===")
    print(f"Messages sent: {sent_frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Failed camera reads: {failed_reads}")
    if pipeline_times_ms:
        print(f"Average pipeline time: {np.mean(pipeline_times_ms):.2f} ms")
        print(f"Average pipeline FPS: {1000.0 / np.mean(pipeline_times_ms):.2f}")
        print(f"Average TCP send time: {np.mean(send_times_ms):.2f} ms")


if __name__ == "__main__":
    main()
