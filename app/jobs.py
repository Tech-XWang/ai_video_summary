# -*- coding: utf-8 -*-
"""单任务队列：串行执行（GPU 独占），负责状态机、事件推送与取消。"""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bilinote import utils
from bilinote.utils import Cancelled

from .events import Event, EventBus
from .service import JobParams, MissingComponentError, NoContentError, run_job


@dataclass
class Job:
    id: str
    params: JobParams
    status: str = "queued"          # queued|running|done|failed|cancelled
    stage: str = "queued"
    pct: float = -1.0
    msg: str = ""
    files: List[str] = field(default_factory=list)
    out_dir: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "task_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "pct": round(self.pct, 4),
            "msg": self.msg,
            "files": self.files,
            "out_dir": self.out_dir,
            "error": self.error,
            "title": self.params.url,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
        }


class JobManager:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._jobs: Dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._worker.start()

    # --- 对外接口 ---
    def submit(self, params: JobParams) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._publish(job, "queued", -1.0, "已加入队列")
        self._queue.put(job)
        return job

    def get(self, task_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(task_id)

    def list(self) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def cancel(self, task_id: str) -> bool:
        job = self.get(task_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            self._publish(job, "cancelled", job.pct, "已取消", "warn")
        else:
            self._publish(job, job.stage, job.pct, "正在取消……", "warn")
        return True

    # --- 内部 ---
    def _publish(self, job: Job, stage: str, pct: float, msg: str, level: str = "info",
                 kind: str = "status") -> None:
        job.stage = stage
        job.pct = pct
        job.msg = msg
        self.bus.publish(Event(task_id=job.id, stage=stage, pct=pct, msg=msg, level=level,
                               kind=kind))

    def _report(self, job: Job, stage: str, pct: float, msg: str) -> None:
        # 进度类的 msg 只用于进度条展示，不写入运行日志
        self._publish(job, stage, pct, msg, kind="progress")

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        self._publish(job, "queued", -1.0, "开始执行")
        utils.set_log_hook(lambda msg: self._publish(job, job.stage, job.pct, msg, "info", "log"))
        utils.set_reporter(lambda stage, pct, msg: self._report(job, stage, pct, msg))
        utils.set_cancel_check(lambda: job.cancel_event.is_set())
        try:
            result = run_job(job.params)
            job.files = result.get("files", [])
            job.out_dir = result.get("out_dir", "")
            job.status = "done"
            self._publish(job, "done", 1.0, "完成")
        except Cancelled:
            job.status = "cancelled"
            self._publish(job, "cancelled", job.pct, "已取消", "warn")
        except NoContentError as e:
            job.status = "failed"
            job.error = str(e)
            self._publish(job, "error", job.pct, str(e), "error")
        except MissingComponentError as e:
            job.status = "failed"
            job.error = str(e)
            self._publish(job, "error", job.pct, str(e), "error")
        except Exception as e:  # noqa: BLE001  GUI 需要把异常完整带回界面
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            self._publish(job, "error", job.pct, job.error, "error")
        finally:
            job.finished = time.time()
            utils.set_log_hook(None)
            utils.set_reporter(None)
            utils.set_cancel_check(None)
