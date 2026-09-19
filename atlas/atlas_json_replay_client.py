import socket
import time


PC_HOST = "192.168.0.1"
PC_PORT = 9999
INPUT_PATH = "out/detections.jsonl"
SEND_INTERVAL_SECONDS = 0.05


def main():
    with open(INPUT_PATH, "rb") as input_file:
        messages = [line for line in input_file if line.strip()]

    print(f"Loaded {len(messages)} messages from {INPUT_PATH}")
    print(f"Connecting to PC at {PC_HOST}:{PC_PORT}...")

    with socket.create_connection((PC_HOST, PC_PORT), timeout=10) as connection:
        print("Connected; starting replay.")
        for index, message in enumerate(messages, start=1):
            connection.sendall(message.rstrip(b"\r\n") + b"\n")
            if index == 1 or index % 10 == 0:
                print(f"Sent {index:3d}/{len(messages)}")
            time.sleep(SEND_INTERVAL_SECONDS)

    print("Replay complete; connection closed.")


if __name__ == "__main__":
    main()
