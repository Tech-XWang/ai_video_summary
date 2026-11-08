# -*- coding: utf-8 -*-
"""本地组件状态检测与安装/下载调度（GUI 高级配置里的“本地状态”）。"""

import importlib.util
import subprocess
import sys
import threading
from typing import Callable, Dict, Tuple

from bilinote import asr, local_llm, media, ocr

Progress = Callable[[float, str], None]


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cookie_status() -> dict:
    try:
        from bilinote import bili_api

        cookie = bili_api.load_cookie()
    except Exception:
        cookie = ""
    ok = bool(cookie)
    return {"ok": ok, "state": "ready" if ok else "missing", "label": "已配置" if ok else "未配置"}


class EnvManager:
    """同一时间只跑一个安装任务；状态供 /api/env 轮询。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = ""
        self._states: Dict[str, dict] = {}

    # ---------- 查询 ----------
    def status(self) -> dict:
        with self._lock:
            current = self._current
            states = {k: dict(v) for k, v in self._states.items()}
        data = {
            "cookie": cookie_status(),
            "llm": local_llm.status(),
            "asr": asr.status(),
            "ocr": ocr.status(),
            "ffmpeg": media.ffmpeg_status(),
            "installing": current,
        }
        for name in ("llm", "asr", "ocr", "ffmpeg"):
            st = states.get(name)
            if not st:
                continue
            data[name].update({
                "running": bool(st.get("running")),
                "pct": st.get("pct", -1.0),
                "error": st.get("error", ""),
            })
            if st.get("running"):
                data[name]["ok"] = False
                data[name]["label"] = st.get("text") or "安装中…"
            elif st.get("error"):
                data[name]["ok"] = False
                data[name]["label"] = "安装失败：" + str(st["error"])
        return data

    # ---------- 安装 ----------
    def start(self, name: str) -> Tuple[bool, str]:
        if name not in ("llm", "asr", "ocr"):
            return False, "未知组件"
        with self._lock:
            if self._current:
                return False, f"已有安装任务进行中：{self._current}"
            self._current = name
            self._states[name] = {"running": True, "pct": -1.0, "text": "准备中…", "error": ""}
        thread = threading.Thread(target=self._run, args=(name,), name=f"install-{name}", daemon=True)
        thread.start()
        return True, ""

    def _progress(self, name: str) -> Progress:
        def fn(pct: float, text: str) -> None:
            with self._lock:
                st = self._states.setdefault(name, {})
                st.update(running=True, pct=pct, text=text, error="")
        return fn

    def _run(self, name: str) -> None:
        progress = self._progress(name)
        try:
            getattr(self, f"_install_{name}")(progress)
        except Exception as e:  # noqa: BLE001  安装失败需要回传界面
            with self._lock:
                self._states[name] = {
                    "running": False, "pct": -1.0, "text": "", "error": str(e)[:500],
                }
        else:
            with self._lock:
                self._states[name] = {"running": False, "pct": 1.0, "text": "完成", "error": ""}
        finally:
            with self._lock:
                self._current = ""

    # ---------- 各组件动作 ----------
    def _install_llm(self, progress: Progress) -> None:
        vram, _ = local_llm.gpu_info()
        if local_llm.recommend(vram) is None:
            raise RuntimeError(f"显存 {vram:.1f}GB ≤ 6GB，不支持本地 LLM 总结")
        local_llm.ensure_runtime(progress)
        local_llm.download_recommended(progress)

    def _install_asr(self, progress: Progress) -> None:
        deps = ["funasr>=1.3.27", "modelscope>=1.11", "sentencepiece",
                "tiktoken", "safetensors"]
        if not _find_spec("torch"):
            deps.append("torch")
        if not _find_spec("torchaudio"):
            deps.append("torchaudio")
        self._pip(deps, progress)
        asr.download_models(progress)

    def _install_ocr(self, progress: Progress) -> None:
        deps = ["rapidocr-onnxruntime>=1.3", "opencv-python>=4.8"]
        if not (_find_spec("onnxruntime") or _find_spec("onnxruntime_gpu")):
            deps.append("onnxruntime>=1.16")
        self._pip(deps, progress)

    def _pip(self, packages: list, progress: Progress) -> None:
        cmd = [sys.executable, "-m", "pip", "install",
               "--upgrade-strategy", "only-if-needed"] + packages
        progress(-1.0, "安装依赖：" + " ".join(packages))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = []
        if proc.stdout is None:
            raise RuntimeError("pip 启动失败：无法读取输出")
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            lines = lines[-8:]
            progress(-1.0, line[:160])
        code = proc.wait()
        if code != 0:
            raise RuntimeError("pip 安装失败：" + " | ".join(lines[-3:])[:400])


__all__ = ["EnvManager", "cookie_status"]
