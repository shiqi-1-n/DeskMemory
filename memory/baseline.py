from dataclasses import dataclass

from memory.geometry import bbox_iou
from common.types import TrackedFrame, TrackedObject


@dataclass
class BaselineObject:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    hits: int = 1


class BaselineMemory:

    def __init__(
        self,
        min_hits: int = 2,
    ):
        self.min_hits = min_hits

        self._candidates: dict[int, BaselineObject] = {}
        self.objects: dict[int, BaselineObject] = {}

        self.frozen = False


    def observe(self, frame: TrackedFrame):

        if self.frozen:
            return

        for obj in frame.objects:

            # person 不属于桌面物品
            if obj.class_name == "person":
                continue

            if obj.track_id not in self._candidates:

                self._candidates[obj.track_id] = (
                    BaselineObject(
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        bbox_xyxy=obj.bbox_xyxy,
                        hits=1,
                    )
                )

            else:

                candidate = self._candidates[
                    obj.track_id
                ]

                candidate.hits += 1
                candidate.bbox_xyxy = (
                    obj.bbox_xyxy
                )


    def freeze(self):

        self.objects = {
            track_id: obj
            for track_id, obj
            in self._candidates.items()
            if obj.hits >= self.min_hits
        }

        self.frozen = True

    def is_baseline_object(
            self,
            obj: TrackedObject,
    ) -> bool:

        for baseline_obj in self.objects.values():

            # 类别不同，不可能是同一个物体
            if baseline_obj.class_name != obj.class_name:
                continue

            # Track ID 没变化
            if baseline_obj.track_id == obj.track_id:
                return True

            # Track ID 变了，但空间位置仍高度重合
            iou = bbox_iou(
                baseline_obj.bbox_xyxy,
                obj.bbox_xyxy,
            )

            if iou >= 0.3:
                return True

        return False