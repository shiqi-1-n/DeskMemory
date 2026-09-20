from __future__ import annotations

import argparse
import json
import socket

from common.types import Detection, DetectionFrame
from memory.engine import DeskMemoryEngine
from vision.pipeline import VisionPipeline


# DeskMemory v0.1 暂时关注的类别
ALLOWED_CLASSES = {
    "person",
    "phone",
    "book",
    "bottle",
}


# 防止不同检测端命名不一致
CLASS_NAME_MAP = {
    "cell_phone": "phone",
    "cell phone": "phone",
}


class StreamClock:
    """
    使用 Atlas timestamp 驱动 C 的时序逻辑，
    但忽略明显的数据中断时间。

    这样：
    - 正常连续图像：时间正常推进
    - TCP/Atlas 中断几秒：这几秒不计入“用户离场时间”
    """

    def __init__(
        self,
        max_valid_gap: float = 0.5,
    ):
        self.max_valid_gap = max_valid_gap

        self.last_raw_timestamp: float | None = None
        self.logical_timestamp: float | None = None

    def convert(
        self,
        raw_timestamp: float,
    ) -> float:

        if self.last_raw_timestamp is None:

            self.last_raw_timestamp = raw_timestamp
            self.logical_timestamp = raw_timestamp

            return raw_timestamp

        dt = raw_timestamp - self.last_raw_timestamp

        self.last_raw_timestamp = raw_timestamp

        if dt < 0:
            dt = 0.0

        # 正常约 14 FPS，帧间隔约 0.07s。
        # 如果突然跳了很长时间，认为是数据中断，
        # 不把中断时间算进 FSM。
        if dt > self.max_valid_gap:

            print(
                f"[DATA GAP] {dt:.2f}s gap ignored "
                f"for DeskMemory timing"
            )

            dt = 0.0

        assert self.logical_timestamp is not None

        self.logical_timestamp += dt

        return self.logical_timestamp


def json_to_detection_frame(
    data: dict,
    timestamp: float,
) -> DetectionFrame:

    detections: list[Detection] = []

    for item in data.get(
        "detections",
        [],
    ):

        raw_name = str(
            item["class_name"]
        )

        class_name = CLASS_NAME_MAP.get(
            raw_name,
            raw_name,
        )

        # 屏蔽 cup / mouse / knife / keyboard 等
        # 当前 DeskMemory 不关心的类别
        if class_name not in ALLOWED_CLASSES:
            continue

        bbox = item["bbox_xyxy"]

        if len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox_xyxy: {bbox}"
            )

        detections.append(
            Detection(
                class_id=int(
                    item["class_id"]
                ),
                class_name=class_name,
                confidence=float(
                    item["confidence"]
                ),
                bbox_xyxy=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
            )
        )

    return DetectionFrame(
        frame_id=int(
            data["frame_id"]
        ),
        timestamp=timestamp,
        image_width=int(
            data["image_width"]
        ),
        image_height=int(
            data["image_height"]
        ),
        detections=detections,
    )


def create_engine() -> DeskMemoryEngine:

    # 第一轮联调为了节约等待时间，
    # confirmed_leave_time 暂时使用 5 秒。
    return DeskMemoryEngine(
        baseline_min_hits=2,
        presence_grace_time=2.0,
        possible_leave_time=2.0,
        confirmed_leave_time=5.0,
        baseline_calibration_time=3.0,
    )


def get_session_status(
    engine: DeskMemoryEngine,
):

    if engine.session is None:
        return []

    return [
        (
            obj.track_id,
            obj.class_name,
            obj.visible,
        )
        for obj
        in engine.session.objects.values()
    ]


def process_json_line(
    line: bytes,
    vision: VisionPipeline,
    engine: DeskMemoryEngine,
    clock: StreamClock,
):

    text = line.decode(
        "utf-8"
    ).strip()

    if not text:
        return None

    data = json.loads(
        text
    )

    raw_timestamp = float(
        data["timestamp"]
    )

    logical_timestamp = (
        clock.convert(
            raw_timestamp
        )
    )

    # ==========================================
    # A JSON
    #    ↓
    # DetectionFrame
    # ==========================================

    detection_frame = (
        json_to_detection_frame(
            data,
            logical_timestamp,
        )
    )

    # ==========================================
    # B
    # DetectionFrame
    #    ↓
    # VisionPipeline
    #    ↓
    # TrackedFrame
    # ==========================================

    tracked_frame = vision.update(
        detection_frame
    )

    # ==========================================
    # C
    # TrackedFrame
    #    ↓
    # DeskMemoryEngine
    #    ↓
    # ForgottenEvent
    # ==========================================

    event = engine.update(
        tracked_frame
    )

    return (
        detection_frame,
        tracked_frame,
        event,
    )


def run_server(
    host: str,
    port: int,
):

    vision = VisionPipeline()

    engine = create_engine()

    clock = StreamClock()

    frame_count = 0

    last_state = None
    last_session = None

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as server:

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.bind(
            (
                host,
                port,
            )
        )

        server.listen(1)

        print()
        print(
            "=============================================="
        )
        print(
            " DeskMemory Atlas -> B -> C TCP Receiver"
        )
        print(
            "=============================================="
        )

        print(
            f"Listening on {host}:{port}"
        )

        print()
        print(
            "Protocol: UTF-8 JSON Lines"
        )

        print(
            "Pipeline:"
        )

        print(
            "Atlas JSON -> DetectionFrame "
            "-> VisionPipeline "
            "-> DeskMemoryEngine"
        )

        print()
        print(
            "Waiting for Atlas..."
        )
        print()

        while True:

            conn, addr = (
                server.accept()
            )

            print()
            print(
                f"[CONNECTED] {addr[0]}:{addr[1]}"
            )
            print()

            # 不因为新 TCP connection
            # 自动重置 engine。
            # 短暂断线不能等同于新会话。

            with conn:

                conn.settimeout(
                    2.0
                )

                buffer = b""

                data_interrupted = False

                while True:

                    try:

                        chunk = conn.recv(
                            65536
                        )

                    except socket.timeout:

                        if not data_interrupted:

                            print()
                            print(
                                "[DATA INTERRUPTED] "
                                "No DetectionFrame received."
                            )

                            print(
                                "DeskMemory FSM is paused; "
                                "this is NOT treated as user leave."
                            )

                            print()

                            data_interrupted = True

                        # 关键：
                        # timeout 时什么都不送给 Vision/C
                        continue

                    except (
                        ConnectionResetError,
                        ConnectionAbortedError,
                    ):

                        print()
                        print(
                            "[DISCONNECTED] "
                            "Atlas connection reset."
                        )
                        print()

                        break

                    if not chunk:

                        print()
                        print(
                            "[DISCONNECTED] "
                            "Atlas sender closed connection."
                        )

                        print(
                            "Waiting for reconnection; "
                            "no empty frames are generated."
                        )
                        print()

                        break

                    if data_interrupted:

                        print()
                        print(
                            "[DATA RESUMED]"
                        )
                        print()

                        data_interrupted = False

                    # ==================================
                    # JSON Lines Buffer
                    #
                    # 不能假设：
                    # 一次 recv == 一帧
                    # ==================================

                    buffer += chunk

                    while b"\n" in buffer:

                        line, buffer = (
                            buffer.split(
                                b"\n",
                                1,
                            )
                        )

                        if not line.strip():
                            continue

                        try:

                            result = (
                                process_json_line(
                                    line,
                                    vision,
                                    engine,
                                    clock,
                                )
                            )

                        except (
                            json.JSONDecodeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:

                            print(
                                "[BAD FRAME]",
                                repr(exc),
                            )

                            continue

                        if result is None:
                            continue

                        (
                            detection_frame,
                            tracked_frame,
                            event,
                        ) = result

                        frame_count += 1

                        state_name = (
                            engine.person_state
                            .state.name
                        )

                        session_status = (
                            get_session_status(
                                engine
                            )
                        )

                        # ==================================
                        # 状态变化时输出
                        # ==================================

                        if (
                            state_name
                            != last_state
                            or
                            session_status
                            != last_session
                            or
                            frame_count % 30 == 0
                        ):

                            detected = [
                                d.class_name
                                for d
                                in detection_frame.detections
                            ]

                            tracked = [
                                (
                                    obj.track_id,
                                    obj.class_name,
                                    obj.missed_frames,
                                )
                                for obj
                                in tracked_frame.objects
                            ]

                            print(
                                f"Frame "
                                f"{detection_frame.frame_id}"
                            )

                            print(
                                "  Detected:",
                                detected,
                            )

                            print(
                                "  Tracked :",
                                tracked,
                            )

                            print(
                                "  State   :",
                                state_name,
                            )

                            print(
                                "  Session :",
                                session_status,
                            )

                            print()

                            last_state = (
                                state_name
                            )

                            last_session = (
                                session_status
                            )

                        # ==================================
                        # Forgotten Event
                        # ==================================

                        if event is not None:

                            forgotten_names = [
                                item.class_name
                                for item
                                in event.items
                            ]

                            print()
                            print(
                                "########################################"
                            )

                            print(
                                ">>> FORGOTTEN EVENT:",
                                forgotten_names,
                            )

                            print(
                                "########################################"
                            )
                            print()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "DeskMemory Atlas TCP "
            "integration receiver"
        )
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9999,
    )

    args = parser.parse_args()

    try:

        run_server(
            host=args.host,
            port=args.port,
        )

    except KeyboardInterrupt:

        print()
        print(
            "DeskMemory receiver stopped."
        )


if __name__ == "__main__":
    main()