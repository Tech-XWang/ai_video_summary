# -*- coding: utf-8 -*-
"""音频语音识别：Fun-ASR-Nano（Windows 走 FunASR AutoModel，不使用 vLLM）。"""

import importlib.util
import json
import logging
import os
import re
import sys
import time
from typing import List

from .config import ASR_MODEL_DIR, ASR_MODEL_REPO, ASR_MODELS_DIR, ASR_VAD_DIR, ASR_VAD_REPO, PROJECT_DIR
from .cues import Cue
from .utils import Cancelled, Ticker, check_cancel, log, report

PATHS_FILE = os.path.join(ASR_MODELS_DIR, "paths.json")


def _fix_import_path() -> None:
    """项目根目录下的 funasr/ 源码目录会遮蔽已安装包，导入前先移出 sys.path。"""
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != PROJECT_DIR]


def _load_paths() -> dict:
    try:
        with open(PATHS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_path(repo: str, path: str) -> None:
    os.makedirs(ASR_MODELS_DIR, exist_ok=True)
    data = _load_paths()
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, os.path.abspath(ASR_MODELS_DIR))
    except ValueError:
        rel = abs_path
    data[repo] = abs_path if rel.startswith("..") else rel
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_saved(rel: str) -> str:
    if not rel:
        return ""
    return rel if os.path.isabs(rel) else os.path.join(ASR_MODELS_DIR, rel)


def _dir_ready(path: str, min_model_bytes: int = 0) -> bool:
    """快照目录里必须同时有配置文件和有足够大小的模型权重，避免半成品。"""
    if not os.path.isdir(path):
        return False
    has_cfg = any(os.path.isfile(os.path.join(path, n)) for n in
                  ("config.yaml", "config.json", "configuration.json"))
    has_model = False
    for name in ("model.pt", "model.safetensors"):
        p = os.path.join(path, name)
        try:
            if os.path.isfile(p) and os.path.getsize(p) >= min_model_bytes:
                has_model = True
                break
        except OSError:
            continue
    return has_cfg and has_model


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_dependencies() -> List[str]:
    """只做轻量检测，不 import torch/funasr，避免 GUI 轮询时初始化 CUDA。"""
    _fix_import_path()
    return [
        name for name in ("funasr", "torch", "torchaudio", "modelscope", "transformers",
                          "sentencepiece", "tiktoken", "safetensors")
        if not _find_spec(name)
    ]


def model_path() -> str:
    path = _resolve_saved(_load_paths().get(ASR_MODEL_REPO, ""))
    if path and _dir_ready(path, 100 * 1024 * 1024):
        return path
    return ASR_MODEL_DIR if _dir_ready(ASR_MODEL_DIR, 100 * 1024 * 1024) else ""


def vad_path() -> str:
    path = _resolve_saved(_load_paths().get(ASR_VAD_REPO, ""))
    if path and _dir_ready(path, 1 * 1024 * 1024):
        return path
    return ASR_VAD_DIR if _dir_ready(ASR_VAD_DIR, 1 * 1024 * 1024) else ""


def status() -> dict:
    deps = missing_dependencies()
    model_dir = model_path()
    vad_dir = vad_path()
    if deps:
        label = "缺少依赖：" + "、".join(deps)
        state = "missing_deps"
    elif not model_dir:
        label = "依赖已安装，ASR 模型未下载"
        state = "missing_model"
    elif not vad_dir:
        label = "依赖已安装，VAD 模型未下载"
        state = "missing_vad"
    else:
        label = "已安装（Fun-ASR-Nano）"
        state = "ready"
    return {
        "ok": not deps and bool(model_dir) and bool(vad_dir),
        "state": state,
        "label": label,
        "model": model_dir,
        "vad": vad_dir,
    }


def check_installed() -> bool:
    return bool(status()["ok"])


def _snapshot(repo: str, target: str) -> str:
    """下载 ModelScope 快照；local_dir 不可用时退回缓存目录并记录实际路径。"""
    from modelscope.hub.snapshot_download import snapshot_download

    os.makedirs(ASR_MODELS_DIR, exist_ok=True)
    try:
        path = snapshot_download(repo, local_dir=target)
    except TypeError:
        path = snapshot_download(repo, cache_dir=ASR_MODELS_DIR)
    return os.path.abspath(path)


def download_models(progress=None) -> None:
    """下载 Fun-ASR-Nano 与 VAD；供安装按钮调用。"""
    if progress:
        progress(-1.0, "下载 Fun-ASR-Nano 模型…")
    path = _snapshot(ASR_MODEL_REPO, ASR_MODEL_DIR)
    if not _dir_ready(path, 100 * 1024 * 1024):
        raise RuntimeError("Fun-ASR-Nano 模型文件不完整，请重试安装")
    _save_path(ASR_MODEL_REPO, path)
    if progress:
        progress(-1.0, "下载 VAD 模型…")
    path = _snapshot(ASR_VAD_REPO, ASR_VAD_DIR)
    if not _dir_ready(path, 1 * 1024 * 1024):
        raise RuntimeError("VAD 模型文件不完整，请重试安装")
    _save_path(ASR_VAD_REPO, path)
    if progress:
        progress(1.0, "ASR 模型下载完成")


def pick_asr_device(requested: str) -> str:
    """auto：有 CUDA 就尝试 GPU（失败由 run_asr 重试并回退 CPU）；也可显式 cpu。"""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_model_dtype(device: str) -> str:
    """LLM 推理精度：CPU 用 fp32；GPU 优先 bf16，老卡不支持 bf16 时用 fp16。"""
    if device.split(":", 1)[0] != "cuda":
        return "fp32"
    try:
        import torch

        if not torch.cuda.is_available():
            return "fp32"
        if torch.cuda.is_bf16_supported():
            return "bf16"
    except Exception:
        return "fp32"
    return "fp16"


def _split_hotwords(hotwords: str) -> List[str]:
    return [w for w in re.split(r"[\s,，;；、]+", hotwords or "") if w]


def _to_sec(value, duration: float) -> float:
    """兼容毫秒/秒两种时间戳。"""
    try:
        sec = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if duration > 0 and sec > duration:
        sec /= 1000.0
    return max(0.0, sec)


def _text_cues(text: str, duration: float) -> List[Cue]:
    """没有句子时间戳时，按标点拆分并把总时长按字数分配。"""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;\n])", text) if p.strip()]
    if len(parts) <= 1:
        return [Cue(0.0, max(duration, 0.1), text, "asr")]
    total_chars = sum(len(p) for p in parts) or 1
    cues: List[Cue] = []
    t = 0.0
    for p in parts:
        span = duration * len(p) / total_chars if duration > 0 else 0.0
        cues.append(Cue(t, max(t + span, t + 0.1), p, "asr"))
        t += span
    return cues


def _cues_from_tokens(r: dict, duration: float) -> List[Cue]:
    """用 Fun-ASR-Nano 输出的 token 级时间戳，按标点切成句子。"""
    tokens = r.get("timestamps") or []
    cues: List[Cue] = []
    buf = ""
    start = None
    end = 0.0
    for t in tokens:
        tok = str(t.get("token") or "")
        if not tok:
            continue
        try:
            s = float(t.get("start_time") or 0)
            e = float(t.get("end_time") or s)
        except (TypeError, ValueError):
            continue
        if start is None:
            start = s
        buf += tok
        end = e
        if tok.strip().endswith(("。", "！", "？", "!", "?", "；", ";", "，", ",")):
            text = buf.strip()
            if text:
                cues.append(Cue(start, max(end, start + 0.1), text, "asr"))
            buf = ""
            start = None
    if buf.strip():
        cues.append(Cue(start or 0.0, max(end, (start or 0.0) + 0.1), buf.strip(), "asr"))
    if not cues:
        return _text_cues(str(r.get("text") or "").strip(), duration)
    return cues


def _clear_cuda_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_asr_once(wav: str, device: str, hotwords: str, duration: float) -> List[Cue]:
    """在指定设备上加载模型并识别一次。"""
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("funasr").setLevel(logging.WARNING)
    logging.getLogger("modelscope").setLevel(logging.WARNING)
    logging.getLogger("modelscope_hub").setLevel(logging.ERROR)
    try:
        from funasr import AutoModel
    except ImportError as e:
        raise RuntimeError(f"FunASR 不可用：{e}")

    model_dir = model_path()
    vad_dir = vad_path()
    if not model_dir:
        raise RuntimeError("Fun-ASR-Nano 模型未安装，请到高级配置点击“安装”")
    if not vad_dir:
        raise RuntimeError("VAD 模型未安装，请到高级配置点击“安装”")

    llm_dtype = pick_model_dtype(device)
    log(f"  加载 Fun-ASR-Nano（device={device}, llm_dtype={llm_dtype}）……")
    report("asr", -1.0, "加载 Fun-ASR-Nano 模型")
    check_cancel()
    t0 = time.time()
    kwargs = {
        "model": model_dir,
        "vad_model": vad_dir,
        "vad_kwargs": {"max_single_segment_time": 30000},
        "device": device,
        "disable_update": True,
        "disable_log": True,
        "log_level": "WARNING",
    }
    model = None
    try:
        with Ticker("  加载模型"):
            model = AutoModel(**kwargs)
        log(f"  模型就绪（{time.time() - t0:.1f}s），开始识别……")
        report("asr", -1.0, "ASR 识别中")
        t1 = time.time()

        gen = {
            "input": [wav],
            "cache": {},
            "batch_size": 1,
            "language": "中文",
            "itn": True,
            "llm_dtype": llm_dtype,
            "disable_pbar": True,
        }
        hot = _split_hotwords(hotwords)
        if hot:
            gen["hotwords"] = hot
        with Ticker("  ASR 识别"):
            res = model.generate(**gen)
        check_cancel()
        log(f"  识别完成（{time.time() - t1:.1f}s）")

        r = res[0] if res else {}
        cues: List[Cue] = []
        for s in r.get("sentence_info") or []:
            text = str(s.get("text") or s.get("sentence") or "").strip()
            if not text:
                continue
            start = _to_sec(s.get("start", 0), duration)
            end = _to_sec(s.get("end", 0), duration)
            cues.append(Cue(start, max(end, start + 0.1), text, "asr"))
        if not cues:
            cues = _cues_from_tokens(r, duration)
        report("asr", 1.0, f"{len(cues)} 句")
        return cues
    finally:
        del model
        _clear_cuda_cache()


def run_asr(wav: str, device: str, hotwords: str = "", duration: float = 0.0) -> List[Cue]:
    _fix_import_path()
    if device != "auto":
        return _run_asr_once(wav, device, hotwords, duration)

    target = pick_asr_device(device)
    if target != "cuda":
        log("  未检测到可用 GPU，使用 CPU 运行 ASR")
        return _run_asr_once(wav, "cpu", hotwords, duration)

    err = ""
    for attempt in (1, 2):
        if attempt > 1:
            log("  等待 4 秒后重试自动GPU……")
            time.sleep(4)
        try:
            return _run_asr_once(wav, "cuda", hotwords, duration)
        except Cancelled:
            raise
        except Exception as e:
            err = str(e)
        log(f"  自动GPU 第 {attempt}/2 次启动失败：{err}")
        _clear_cuda_cache()
    log("  自动GPU 连续 2 次失败，改用 CPU 运行")
    return _run_asr_once(wav, "cpu", hotwords, duration)
