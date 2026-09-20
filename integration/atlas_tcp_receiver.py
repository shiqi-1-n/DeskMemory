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

        # 第一帧
        if self.last_raw_timestamp is None:

            self.last_raw_timestamp = raw_timestamp
            self.logical_timestamp = raw_timestamp

            return raw_timestamp

        # 当前帧与上一帧的真实时间差
        dt = (
            raw_timestamp
            - self.last_raw_timestamp
        )

        self.last_raw_timestamp = raw_timestamp

        # 防止异常时间戳倒退
        if dt < 0:
            dt = 0.0

        # Atlas 当前约 14 FPS，
        # 正常帧间隔大约 0.07 秒。
        #
        # 如果出现明显的大时间间隔，
        # 则认为中间发生过数据中断，
        # 不让这个中断时间参与离场计时。
        if dt > self.max_valid_gap:

            print(
                f"[DATA GAP] "
                f"{dt:.2f}s gap ignored "
                f"for DeskMemory timing"
            )

            dt = 0.0

        assert (
            self.logical_timestamp
            is not None
        )

        self.logical_timestamp += dt

        return self.logical_timestamp


def create_engine() -> DeskMemoryEngine:
    """
    创建 C 端 DeskMemory Engine。

    第一轮真机联调时将确认离场时间设为 5 秒，
    方便现场快速观察完整流程。
    """

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
    """
    获取当前 Session Memory 中的物品状态，
    主要用于联调日志打印。
    """

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
    """
    处理一整行 Atlas JSON 消息。

    Atlas JSON
        ↓
    message_to_detection_frame()
        ↓
    DetectionFrame
        ↓
    VisionPipeline.update()
        ↓
    TrackedFrame
        ↓
    DeskMemoryEngine.update()
        ↓
    ForgottenEvent | None
    """

    # ==================================================
    # Decode JSON line
    # ==================================================

    text = line.decode(
        "utf-8"
    ).strip()

    if not text:
        return None

    message = json.loads(
        text
    )

    # ==================================================
    # Stream Clock
    #
    # 数据中断不能被算进“用户离场时间”
    # ==================================================

    raw_timestamp = float(
        message["timestamp"]
    )

    logical_timestamp = (
        clock.convert(
            raw_timestamp
        )
    )

    # 不修改原始 message
    message = dict(
        message
    )

    message["timestamp"] = (
        logical_timestamp
    )

    # ==================================================
    # B：Atlas JSON -> DetectionFrame
    #
    # 类别名称统一：
    # cell_phone -> phone
    #
    # MVP 类别过滤：
    # person / phone / book / bottle
    #
    # confidence threshold：
    # 全部由 B 的统一函数负责
    # ==================================================

    detection_frame = (
        message_to_detection_frame(
            message
        )
    )

    # ==================================================
    # B：DetectionFrame -> TrackedFrame
    #
    # VisionPipeline 内部：
    #
    # SimpleTracker
    #       ↓
    # StabilityFilter
    # ==================================================

    tracked_frame = vision.update(
        detection_frame
    )

    # ==================================================
    # C：TrackedFrame -> ForgottenEvent
    # ==================================================

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

    # ==================================================
    # B
    # ==================================================

    vision = VisionPipeline()

    # ==================================================
    # C
    # ==================================================

    engine = create_engine()

    # ==================================================
    # 时间保护
    # ==================================================

    clock = StreamClock()

    # ==================================================
    # Debug / Log State
    # ==================================================

    frame_count = 0

    last_state = None
    last_session = None

    # 用于确保 BASELINE READY 只打印一次
    last_baseline_ready = False

    # ==================================================
    # TCP Server
    # ==================================================

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

        # ==================================================
        # Startup Info
        # ==================================================

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
        print()

        print(
            f"Listening on {host}:{port}"
        )

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
            " -> ForgottenEvent"
        )

        print()
        print(
            "Waiting for Atlas..."
        )
        print()

        # ==================================================
        # Accept Connections
        # ==================================================

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

            # --------------------------------------------------
            # 注意：
            #
            # TCP 重新连接并不代表新会话。
            #
            # 因此断线重连时不自动：
            #
            # engine.reset()
            # vision = VisionPipeline()
            #
            # 避免短暂网络抖动直接清空 DeskMemory 状态。
            # --------------------------------------------------

            with conn:

                conn.settimeout(
                    2.0
                )

                # JSONL 接收缓冲区
                buffer = b""

                # 数据中断状态
                data_interrupted = False

                while True:

                    # ==========================================
                    # Receive TCP Data
                    # ==========================================

                    try:

                        chunk = conn.recv(
                            65536
                        )

                    except socket.timeout:

                        # --------------------------------------
                        # TCP timeout
                        #
                        # 数据中断 != 用户离开
                        #
                        # 因此：
                        # 不创建空 DetectionFrame
                        # 不更新 VisionPipeline
                        # 不更新 DeskMemoryEngine
                        # --------------------------------------

                        if not data_interrupted:

                            print()
                            print(
                                "[DATA INTERRUPTED]"
                            )

                            print(
                                "No DetectionFrame received."
                            )

                            print(
                                "DeskMemory FSM is paused."
                            )

                            print(
                                "This is NOT treated "
                                "as user leave."
                            )

                            print()

                            data_interrupted = True

                        continue

                    except (
                        ConnectionResetError,
                        ConnectionAbortedError,
                    ):

                        print()
                        print(
                            "[DISCONNECTED]"
                        )

                        print(
                            "Atlas connection reset."
                        )
                        print()

                        break

                    # ==========================================
                    # Client closed connection
                    # ==========================================

                    if not chunk:

                        print()
                        print(
                            "[DISCONNECTED]"
                        )

                        print(
                            "Atlas sender closed "
                            "the connection."
                        )

                        print(
                            "No empty frames are generated."
                        )

                        print(
                            "Waiting for reconnection..."
                        )
                        print()

                        break

                    # ==========================================
                    # Data resumed
                    # ==========================================

                    if data_interrupted:

                        print()
                        print(
                            "[DATA RESUMED]"
                        )
                        print()

                        data_interrupted = False

                    # ==========================================
                    # JSON Lines Buffer
                    #
                    # TCP 是字节流：
                    #
                    # 一次 recv 可能得到：
                    #
                    # 1. 半条 JSON
                    # 2. 一条 JSON
                    # 3. 多条 JSON
                    #
                    # 所以必须缓存，并按照 \n 拆分。
                    # ==========================================

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

                        # ======================================
                        # A -> B -> C
                        # ======================================

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

                        # ======================================
                        # BASELINE READY
                        # ======================================

                        if (
                            engine._baseline_ready
                            and
                            not last_baseline_ready
                        ):

                            print()
                            print(
                                "========================================"
                            )

                            print(
                                "[BASELINE READY]"
                            )

                            print(
                                "Desk baseline calibration "
                                "completed."
                            )

                            baseline_names = [
                                obj.class_name
                                for obj
                                in engine.baseline.objects.values()
                            ]

                            print(
                                "Baseline objects:",
                                baseline_names,
                            )

                            print(
                                "========================================"
                            )
                            print()

                            last_baseline_ready = True

                        # ======================================
                        # Current State
                        # ======================================

                        state_name = (
                            engine.person_state
                            .state.name
                        )

                        session_status = (
                            get_session_status(
                                engine
                            )
                        )

                        # ======================================
                        # Debug Log
                        #
                        # 以下情况输出：
                        #
                        # 1. Person State 改变
                        # 2. Session 改变
                        # 3. 每 30 帧打印一次
                        # ======================================

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

                        # ======================================
                        # Forgotten Event
                        # ======================================

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
        help=(
            "TCP server bind address "
            "(default: 0.0.0.0)"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help=(
            "TCP server port "
            "(default: 9999)"
        ),
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