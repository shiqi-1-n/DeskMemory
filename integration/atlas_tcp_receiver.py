from __future__ import annotations

import argparse
import json
import socket

from memory.engine import DeskMemoryEngine
from vision.pipeline import VisionPipeline
from vision.replay_jsonl import message_to_detection_frame


class StreamClock:
    """
    使用 Atlas timestamp 驱动 C 的时序逻辑，
    但忽略明显的数据中断时间。

    正常连续数据：
        时间正常推进

    TCP / Atlas 中断：
        中断时间不计入用户离场时间
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

        # Atlas 当前约 14 FPS，
        # 正常帧间隔约 0.07 秒。
        #
        # 如果出现明显时间跳跃，
        # 认为中间发生了数据中断。
        if dt > self.max_valid_gap:

            print(
                f"[DATA GAP] {dt:.2f}s gap ignored "
                f"for DeskMemory timing"
            )

            dt = 0.0

        assert self.logical_timestamp is not None

        self.logical_timestamp += dt

        return self.logical_timestamp


def create_engine() -> DeskMemoryEngine:

    # 第一轮联调为了方便观察，
    # confirmed_leave_time 暂时设为 5 秒。
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

    # ==========================================
    # TCP JSON
    # ==========================================

    message = json.loads(
        text
    )

    # ==========================================
    # 数据中断时间保护
    #
    # 不修改 B 的转换函数，
    # 只在送入它之前修正 timestamp。
    # ==========================================

    raw_timestamp = float(
        message["timestamp"]
    )

    logical_timestamp = clock.convert(
        raw_timestamp
    )

    message = dict(
        message
    )

    message["timestamp"] = (
        logical_timestamp
    )

    # ==========================================
    # B 公共转换入口
    #
    # Atlas JSON
    #     ↓
    # message_to_detection_frame()
    #     ↓
    # DetectionFrame
    #
    # 类别映射、MVP 类别过滤、
    # confidence threshold 均由 B 统一负责。
    # ==========================================

    detection_frame = (
        message_to_detection_frame(
            message
        )
    )

    # ==========================================
    # B
    #
    # DetectionFrame
    #     ↓
    # Tracking + Stability
    #     ↓
    # TrackedFrame
    # ==========================================

    tracked_frame = vision.update(
        detection_frame
    )

    # ==========================================
    # C
    #
    # TrackedFrame
    #     ↓
    # DeskMemory
    #     ↓
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

        print()
        print(
            "Pipeline:"
        )

        print(
            "Atlas JSON"
            " -> message_to_detection_frame()"
            " -> VisionPipeline"
            " -> DeskMemoryEngine"
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
                f"[CONNECTED] "
                f"{addr[0]}:{addr[1]}"
            )
            print()

            # 注意：
            # 新 TCP connection 不自动 reset。
            #
            # 因为一次短暂断线
            # 不应该等价于新的 DeskMemory 会话。

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
                                "DeskMemory FSM paused. "
                                "This is NOT treated "
                                "as user leave."
                            )
                            print()

                            data_interrupted = True

                        # timeout：
                        # 不生成空 DetectionFrame，
                        # 不更新 Vision/C。
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
                            "Atlas sender closed "
                            "connection."
                        )

                        print(
                            "Waiting for reconnection. "
                            "No empty frames generated."
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
                    # JSON Lines buffer
                    #
                    # 一次 recv 可能是：
                    #
                    # 半条 JSON
                    # 1 条 JSON
                    # 多条 JSON
                    #
                    # 所以必须按照 \n 拆包。
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
                        # 关键状态变化时打印
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
                                obj.class_name
                                for obj
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