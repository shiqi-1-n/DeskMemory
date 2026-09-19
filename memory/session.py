from dataclasses import dataclass

from common.types import TrackedFrame, TrackedObject
from memory.baseline import BaselineMemory
from memory.geometry import bbox_iou


@dataclass
class SessionObject:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]

    first_seen_time: float
    last_seen_time: float

    visible: bool = True


class SessionMemory:

    def __init__(
        self,
        baseline: BaselineMemory,
        presence_grace_time: float = 2.0,
        iou_threshold: float = 0.3,
    ):
        self.baseline = baseline
        self.presence_grace_time = presence_grace_time
        self.iou_threshold = iou_threshold

        self.objects: dict[int, SessionObject] = {}

    def update(self, frame: TrackedFrame):

        now = frame.timestamp

        seen_ids: set[int] = set()

        for obj in frame.objects:

            # 人不属于 Session 物品
            if obj.class_name == "person":
                continue

            # Baseline 原有物品不属于本次 Session
            if self.baseline.is_baseline_object(obj):
                continue

            # 查找是不是已经记录过的 Session 物体
            match = self._find_match(obj)

            # -----------------------------
            # 新物体
            # -----------------------------
            if match is None:

                self.objects[obj.track_id] = SessionObject(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    bbox_xyxy=obj.bbox_xyxy,
                    first_seen_time=now,
                    last_seen_time=now,
                    visible=True,
                )

                seen_ids.add(obj.track_id)

            # -----------------------------
            # 已有物体
            # -----------------------------
            else:

                old_track_id, session_obj = match

                # Tracker 给同一物体重新分配 ID
                if old_track_id != obj.track_id:

                    del self.objects[old_track_id]

                    session_obj.track_id = obj.track_id

                    self.objects[obj.track_id] = session_obj

                session_obj.bbox_xyxy = obj.bbox_xyxy
                session_obj.last_seen_time = now
                session_obj.visible = True

                seen_ids.add(obj.track_id)

        # --------------------------------
        # 检查本帧没有看到的 Session 物体
        # --------------------------------

        for track_id, session_obj in self.objects.items():

            if track_id in seen_ids:
                continue

            lost_time = now - session_obj.last_seen_time

            if lost_time > self.presence_grace_time:
                session_obj.visible = False

    def _find_match(
        self,
        obj: TrackedObject,
    ) -> tuple[int, SessionObject] | None:

        # 第一优先级：Track ID 相同
        if obj.track_id in self.objects:
            return (
                obj.track_id,
                self.objects[obj.track_id],
            )

        # 第二优先级：类别相同 + 空间高度重合
        for track_id, session_obj in self.objects.items():

            if session_obj.class_name != obj.class_name:
                continue

            iou = bbox_iou(
                session_obj.bbox_xyxy,
                obj.bbox_xyxy,
            )

            if iou >= self.iou_threshold:
                return (
                    track_id,
                    session_obj,
                )

        return None

    def get_visible_objects(
        self,
    ) -> list[SessionObject]:

        return [
            obj
            for obj in self.objects.values()
            if obj.visible
        ]