import json
import tempfile
import unittest
from pathlib import Path

from common.types import (
    ForgottenEvent,
    ForgottenItem,
)

from memory.event_log import EventLogger


class TestEventLogger(unittest.TestCase):

    def test_write_event(self):

        # 使用临时目录，测试结束后自动删除
        with tempfile.TemporaryDirectory() as temp_dir:

            log_path = Path(temp_dir) / "events.jsonl"

            logger = EventLogger(
                log_path=str(log_path)
            )

            event = ForgottenEvent(
                event_id="test-event-001",
                timestamp=100.0,
                items=[
                    ForgottenItem(
                        track_id=4,
                        class_name="phone",
                        bbox_xyxy=(
                            700,
                            300,
                            800,
                            370,
                        ),
                        last_seen_time=99.0,
                    )
                ],
            )

            # 写入事件
            logger.write(event)

            # 日志文件必须存在
            self.assertTrue(
                log_path.exists()
            )

            # 读取日志内容
            with log_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                lines = f.readlines()

            # 应该只有一条事件
            self.assertEqual(
                len(lines),
                1,
            )

            data = json.loads(
                lines[0]
            )

            # 检查事件基本信息
            self.assertEqual(
                data["event_id"],
                "test-event-001",
            )

            self.assertEqual(
                data["timestamp"],
                100.0,
            )

            # 检查遗落物
            self.assertEqual(
                len(data["items"]),
                1,
            )

            self.assertEqual(
                data["items"][0]["class_name"],
                "phone",
            )

            self.assertEqual(
                data["items"][0]["track_id"],
                4,
            )


if __name__ == "__main__":
    unittest.main()