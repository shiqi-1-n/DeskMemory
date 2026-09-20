# DeskMemory C 模块接口说明

## 1. 数据流

B 模块负责目标跟踪，并输出：

TrackedFrame

C 模块接收 TrackedFrame，完成桌面记忆、用户离场判断和遗落物检测：

TrackedFrame
→ DeskMemoryEngine
→ ForgottenEvent / None

---

## 2. 输入接口

公共数据结构位于：

common/types.py

B 模块需要构造 TrackedFrame：

```python
TrackedFrame(
    frame_id=frame_id,
    timestamp=timestamp,
    image_width=image_width,
    image_height=image_height,
    objects=tracked_objects,
)