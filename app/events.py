# -*- coding: utf-8 -*-
"""事件与总线：SSE 推送的最小实现。"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class Event:
    task_id: str
    stage: str
    pct: float = -1.0
    msg: str = ""
    level: str = "info"
    kind: str = "status"     # status|log|progress：progress 只更新进度条，不进日志
    ts: float = field(default_factory=time.time)
    seq: int = 0          # 同一任务内单调递增，供前端去重（ts 精度不足，可能重复）

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "pct": round(self.pct, 4),
            "msg": self.msg,
            "level": self.level,
            "kind": self.kind,
            "ts": round(self.ts, 3),
            "seq": self.seq,
        }


class EventBus:
    """按 task_id 分发事件，并保留最近若干条供断线回放。"""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._history: Dict[str, Deque[Event]] = {}
        self._subs: Dict[str, List[queue.Queue]] = {}
        self._seq: Dict[str, int] = {}

    def publish(self, event: Event) -> None:
        with self._lock:
            seq = self._seq.get(event.task_id, 0) + 1
            self._seq[event.task_id] = seq
            event.seq = seq
            history = self._history.setdefault(event.task_id, deque(maxlen=self.capacity))
            history.append(event)
            subscribers = list(self._subs.get(event.task_id, []))
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self, task_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(task_id)
            if subs and q in subs:
                subs.remove(q)

    def history(self, task_id: str, since: float = 0.0) -> List[dict]:
        with self._lock:
            events = list(self._history.get(task_id, []))
        return [e.to_dict() for e in events if e.ts >= since]
