import json
from dataclasses import asdict
from pathlib import Path

from common.types import ForgottenEvent


class EventLogger:

    def __init__(
        self,
        log_path: str = "logs/events.jsonl",
    ):
        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(
        self,
        event: ForgottenEvent,
    ) -> None:

        data = asdict(event)

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
                + "\n"
            )