import json
import socket


HOST = "0.0.0.0"
PORT = 9999
EXPECTED_MESSAGES = 100


def validate_message(message):
    required = {
        "frame_id",
        "timestamp",
        "image_width",
        "image_height",
        "detections",
    }
    missing = required - message.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(message["detections"], list):
        raise ValueError("detections must be a list")


def main():
    received = 0
    frames_with_detections = 0
    total_detections = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Listening on {HOST}:{PORT}")

        connection, address = server.accept()
        print(f"Atlas connected from {address[0]}:{address[1]}")

        with connection, connection.makefile("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                message = json.loads(line)
                validate_message(message)
                received += 1
                count = len(message["detections"])
                if count:
                    frames_with_detections += 1
                    total_detections += count

                if received == 1 or received % 10 == 0:
                    print(
                        f"Received {received:3d}: "
                        f"frame_id={message['frame_id']}, detections={count}"
                    )

                if received >= EXPECTED_MESSAGES:
                    break

    print("\n=== Receive summary ===")
    print(f"Messages received: {received}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections: {total_detections}")


if __name__ == "__main__":
    main()
