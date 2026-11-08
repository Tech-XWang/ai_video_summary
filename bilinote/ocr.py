# -*- coding: utf-8 -*-
"""画面硬字幕识别（RapidOCR）+ 持续性水印/台标过滤。

水印判定基于"行内容连续出现"的时长：
  * 同一行 norm_text 连续出现记一条链，中断不超过 1 个观测帧仍算同一条链；
  * 链跨度 ≥8s 视为水印/台标，整条链删除；
  * 链跨度 4~8s 视为重复出现，只保留首次那一次。
判定必须放在合并文本之前，否则水印会和字幕粘连成一句。
"""

import importlib.util
import os
import re
import sys
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from .media import extract_frames
from .models import Cue
from .utils import check_cancel, is_meaningful, join_ocr, log, norm_text, report

# 一帧出现多个识别区域时，字数超过阈值视为非字幕，整块舍弃。
# 字数按"中文算 2 个字符、其它算 1 个"计权。
MAX_REGION_CHAR = 100

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_dependencies() -> List[str]:
    return [
        name for name in ("cv2", "rapidocr_onnxruntime", "onnxruntime")
        if not _find_spec(name)
    ]


def status() -> dict:
    deps = missing_dependencies()
    if deps:
        return {
            "ok": False,
            "state": "missing_deps",
            "label": "缺少依赖：" + "、".join(deps),
        }
    return {"ok": True, "state": "ready", "label": "已安装（RapidOCR）"}


def check_installed() -> bool:
    return bool(status()["ok"])


def text_weight(text: str) -> int:
    """字数计权：中文按 2 个字符，其它按 1 个。"""
    return len(text) + len(_CJK_RE.findall(text))


def drop_long_regions(entries: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], int]:
    """一帧内有多个区域时，丢弃字数 ≥100 的区域，返回剩余条目与丢弃数量。

    调用点必须在连续链计数（counter）更新之后：超长区域仍要参与去重与链统计，
    只是不进入最终文本。只有一个区域时不适用该规则（无法区分字幕与说明文字）。
    """
    if len(entries) <= 1:
        return entries, 0
    kept = [e for e in entries if text_weight(e[0]) < MAX_REGION_CHAR]
    return kept, len(entries) - len(kept)


def run_ocr(video: str, work: str, fps: float, region: str, diff_thr: float = 2.5) -> List[Cue]:
    try:
        import cv2
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        log(f"  OCR 依赖缺失（{e}），跳过画面识别")
        return []

    # 目录名带上 fps/region，换参数重跑时不会误复用旧帧
    frames_dir = os.path.join(work, f"frames_{fps:g}_{region}")
    files = extract_frames(video, frames_dir, fps, region)
    if not files:
        return []

    # onnxruntime-gpu 需要 CUDA/cuDNN 动态库；PyTorch 已自带，用官方 preload_dlls 预加载，
    # 否则 CUDA provider 会因找不到依赖而回退 CPU。没有 GPU 环境时保持 CPU。
    cuda_ok = False
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as e:
        log(f"  [Warn] CUDA 预加载失败，OCR 使用 CPU：{e}")

    # 限制检测输入为最长边 960（默认 min 模式下小图会被放大到 4096 宽）；
    # 字幕不会倒置，关掉方向分类。det_model_path 传空串走默认模型路径。
    engine = RapidOCR(
        det_model_path="",
        det_limit_type="max",
        det_limit_side_len=960,
        use_angle_cls=False,
        det_use_cuda=cuda_ok,
    )
    # 第一遍：逐帧识别，按行收集。水印判定依赖"连续出现"的时长，所以先不合并文本。
    # frames 元素为 [起始时间, [(行文本, 连续链编号), ...], 该内容可见到的最后时间]
    frames: List[List] = []
    runs: Dict[int, List[float]] = {}   # 连续链编号 -> [首次出现时间, 最后出现时间]
    live: Dict[str, List[int]] = {}     # norm_text -> [连续链编号, 消失计数]
    next_run = 0
    prev_gray = None
    skipped = 0
    dropped_long = 0
    t0 = time.time()

    for i, name in enumerate(tqdm(files, desc="  OCR", unit="帧", leave=False, file=sys.stdout)):
        if i % 10 == 0:
            report("ocr", i / len(files), f"识别 {i}/{len(files)}")
        check_cancel()
        img = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if float(img.std()) < 3.0:  # 纯黑/纯色帧跳过
            continue
        t = i / fps
        diff = 255.0 if prev_gray is None else float(cv2.absdiff(img, prev_gray).mean())
        prev_gray = img
        if diff < diff_thr:
            # 字幕带几乎无变化，沿用上一帧内容，省掉一次 OCR
            skipped += 1
            if frames:
                frames[-1][2] = t  # 上一帧内容持续到当前帧
            continue
        try:
            res, _ = engine(img)
        except Exception:
            continue
        entries = []
        seen = set()
        for item in res or []:
            txt, score = str(item[1]).strip(), float(item[2])
            if score < 0.7 or not is_meaningful(txt):
                continue
            n = norm_text(txt)
            if not n:
                continue
            state = live.get(n)
            if state is None or state[1] <= 0:
                # 首次出现，或上一段连续链已结束：新开一条链，重新计入字幕
                rid = next_run
                next_run += 1
                runs[rid] = [t, t]
                live[n] = [rid, 2]
            else:
                # 连续出现：counter 补满，并把链的结束时间延展到当前帧
                rid = state[0]
                state[1] = 2
                runs[rid][1] = t
            entries.append((txt, rid))
            seen.add(n)
        # 本帧未出现的行：counter--；归零说明已消失，下次出现重新计入字幕
        for n, state in live.items():
            if n not in seen and state[1] > 0:
                state[1] -= 1
        # counter 之后才丢弃超长区域：它们已经参与了上面的连续链计数与去重，
        # 只是多区域帧里字数 ≥100 的整块说明文字不进入最终文本。
        entries, dropped = drop_long_regions(entries)
        dropped_long += dropped
        frames.append([t, entries, t])

    # 第二遍（合并前）：按连续链判定水印。
    drop_runs: set = set()
    keep_first_runs: set = set()
    for rid, (start, last) in runs.items():
        span = last - start
        if span >= 8.0:
            drop_runs.add(rid)
        elif span >= 4.0:
            keep_first_runs.add(rid)

    cues: List[Cue] = []
    kept_first: set = set()
    last_norm, last_start = "", 0.0
    for t, entries, end_t in frames:
        picked: List[str] = []
        for txt, rid in entries:
            if rid in drop_runs:
                continue
            if rid in keep_first_runs:
                if rid in kept_first:
                    continue
                kept_first.add(rid)
            picked.append(txt)
        # 同一帧内不同识别区域之间用空格分隔
        text = " ".join(picked)
        n = norm_text(text)
        if not n:
            last_norm = ""
            continue
        if n == last_norm:
            if cues:
                cues[-1].end = max(cues[-1].end, end_t)
            continue
        if cues and t - last_start < 0.6:  # 同一句的抖动，合并
            cues[-1].text = join_ocr([cues[-1].text, text])
            cues[-1].end = max(cues[-1].end, end_t)
            last_norm = n
            continue
        cues.append(Cue(t, max(end_t, t + 1.0 / fps), text, "ocr"))
        last_norm, last_start = n, t

    # 兜底：纯水印画面（无字幕）会形成超长 cue，行级规则管不到时在此丢弃
    cues = [c for c in cues if not (c.end - c.start > 10 and len(c.text) < 15)]
    for c in cues:
        c.text = re.sub(r"\s+", " ", c.text).strip()
    log(
        f"  OCR 完成：{len(cues)} 条（跳帧 {skipped}，超长区域丢弃 {dropped_long} 个，"
        f"水印行 删除 {len(drop_runs)} 段 / 仅留首次 {len(keep_first_runs)} 段，"
        f"用时 {time.time() - t0:.1f}s）"
    )
    report("ocr", 1.0, f"{len(cues)} 条")
    return cues
