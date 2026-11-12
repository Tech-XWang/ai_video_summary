# -*- coding: utf-8 -*-
"""媒体处理：ffmpeg/ffprobe 封装、音频转码、抽帧。"""

import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from tqdm import tqdm

from .config import FFMPEG_DIR
from .utils import log, report


def _local_tool(name: str) -> Optional[str]:
    if not os.path.isdir(FFMPEG_DIR):
        return None
    for path in (os.path.join(FFMPEG_DIR, name), os.path.join(FFMPEG_DIR, "bin", name)):
        if os.path.isfile(path):
            return path
    for root, _, files in os.walk(FFMPEG_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


def find_ffmpeg() -> Optional[str]:
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return _local_tool(name) or shutil.which("ffmpeg")


def find_ffprobe() -> Optional[str]:
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    return _local_tool(name) or shutil.which("ffprobe")


def ffmpeg_status() -> dict:
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()
    if not ffmpeg or not ffprobe:
        return {"ok": False, "state": "missing", "label": "缺失（需手动安装）"}
    local = _local_tool("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return {
        "ok": True,
        "state": "ready",
        "label": "正常（本地）" if local else "正常（系统 PATH）",
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
    }


def media_duration(src: str) -> float:
    """媒体总时长（秒）；取不到时返回 0。"""
    probe = find_ffprobe()
    if not probe:
        return 0.0
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", src],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def run_ffmpeg(cmd: List[str], src: str, desc: str, stage: str = "media") -> None:
    """执行 ffmpeg，用 -progress 管道按媒体时长驱动 tqdm，并上报进度。"""
    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError("未找到 ffmpeg，请到高级配置点击“安装”")
    cmd = list(cmd)
    cmd[0] = exe
    total = media_duration(src)
    bar = tqdm(
        total=round(total) if total > 0 else None, unit="s", desc=desc, leave=False,
        file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}s/{total:.0f}s" if total > 0
        else "{desc}: {n:.0f}s",
    )
    proc = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    errs: List[str] = []
    last_pct = -1.0
    try:
        for line in proc.stdout:
            key, _, val = line.partition("=")
            val = val.strip()
            sec = None
            if key == "out_time_us" and val.isdigit():
                sec = int(val) / 1_000_000
            elif key == "out_time_ms" and val.isdigit():
                # 旧版 ffmpeg 输出毫秒（新版为 out_time_us 微秒），除数不同
                sec = int(val) / 1000
            elif line.strip() and not line.startswith("progress="):
                errs.append(line.strip())
            if sec is None:
                continue
            bar.n = min(sec, total) if total > 0 else sec
            bar.refresh()
            pct = min(sec / total, 1.0) if total > 0 else -1.0
            if pct - last_pct >= 0.01 or pct >= 1.0:
                report(stage, pct, desc)
                last_pct = pct
        code = proc.wait()
    finally:
        bar.close()
    if code != 0:
        raise RuntimeError(f"ffmpeg 失败：{' | '.join(errs[-3:])[:400]}")


def to_wav(src: str, work: str) -> Optional[str]:
    """把任意音频转成 16k 单声道 wav，供 ASR 使用。"""
    wav = os.path.join(work, "audio16k.wav")
    if os.path.isfile(wav) and os.path.getsize(wav) > 1024:
        return wav
    if not find_ffmpeg():
        raise RuntimeError("未找到 ffmpeg，请到高级配置点击“安装”")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", wav]
    log("  音频转 16k 单声道 wav……")
    run_ffmpeg(cmd, src, "转码 wav", stage="asr")
    return wav if os.path.isfile(wav) else None


def extract_frames(video: str, frames_dir: str, fps: float, region: str) -> List[str]:
    """按 fps 抽帧并裁到目标区域，返回排好序的帧文件名。已有帧则直接复用。"""
    if os.path.isdir(frames_dir) and os.listdir(frames_dir):
        log("  OCR：复用已抽帧")
    else:
        os.makedirs(frames_dir, exist_ok=True)
        vf = f"fps={fps}"
        if region == "bottom":
            vf += ",crop=iw:ih*0.3:0:ih*0.7"
        elif region == "center":
            vf += ",crop=iw:ih*0.5:0:ih*0.25"
        # 宽度超过 1280 时先缩小，减少 jpg 体积、解码和 OCR 开销
        vf += ",scale='min(1280,iw)':-2"
        out_pat = os.path.join(frames_dir, "%06d.jpg")
        log(f"  抽帧（fps={fps}, region={region}）……")
        t0 = time.time()
        run_ffmpeg(
            ["ffmpeg", "-y", "-v", "error", "-i", video, "-vf", vf, "-q:v", "3", out_pat],
            video, "抽帧", stage="ocr",
        )
        log(f"  抽帧完成：{len(os.listdir(frames_dir))} 帧（{time.time() - t0:.1f}s）")

    return sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
