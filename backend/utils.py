# -*- coding: utf-8 -*-
"""通用工具：日志、进度、取消、计时条、文本归一化、时间格式化、cue 聚合。"""

import re
import sys
import threading
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm

from .cues import Cue


class Cancelled(Exception):
    """任务被用户取消。"""


# --- 三个可注入的 hook：CLI 用默认实现，GUI 换成事件推送 ---
_log_hook: Optional[Callable[[str], None]] = None
_reporter: Optional[Callable[[str, float, str], None]] = None
_cancel_check: Optional[Callable[[], bool]] = None
# 控制台是否输出 debug 级日志（GUI 后台 cmd 打开，前端始终只收 info）
_console_debug: bool = False


def set_log_hook(fn: Optional[Callable[[str], None]]) -> None:
    global _log_hook
    _log_hook = fn


def set_console_debug(on: bool) -> None:
    global _console_debug
    _console_debug = on


def set_reporter(fn: Optional[Callable[[str, float, str], None]]) -> None:
    """fn(stage, pct, msg)，pct 为 0~1，-1 表示不确定进度。"""
    global _reporter
    _reporter = fn


def set_cancel_check(fn: Optional[Callable[[], bool]]) -> None:
    global _cancel_check
    _cancel_check = fn


def log(msg: str, level: str = "info") -> None:
    """写日志。debug 只进后台控制台；hook（前端）只收 info 及以上的内容。"""
    if level == "debug":
        if _console_debug:
            print(msg, flush=True)
        return
    if _log_hook is not None:
        _log_hook(msg)
    print(msg, flush=True)


def report(stage: str, pct: float, msg: str = "") -> None:
    if _reporter is not None:
        _reporter(stage, pct, msg)


def check_cancel() -> None:
    if _cancel_check is not None and _cancel_check():
        raise Cancelled("任务已取消")


class Ticker:
    """阻塞式长任务（模型加载、ASR 识别）用的 tqdm 计时条。"""

    def __init__(self, desc: str) -> None:
        self.bar = tqdm(total=None, unit="s", desc=desc, leave=False, file=sys.stdout,
                        bar_format="{desc}: {n:.0f}s")
        self._stop = threading.Event()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.bar.n += 0.5
            self.bar.refresh()

    def __enter__(self) -> "Ticker":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=2)
        self.bar.close()


def sanitize(name: str, limit: int = 60) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n]+', "_", name).strip(" .")
    # 截断后可能正好落在空格/句点上：Windows 会建立时去掉结尾空格，
    # 后续再用带空格的路径写文件就会 FileNotFoundError，所以截断后要再清理一次
    return name[:limit].strip(" .") or "untitled"


def fmt_ts(sec: float, sep: str = ":") -> str:
    sec = max(0, float(sec))
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), int(sec % 60)
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}"


def srt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def norm_text(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()《》\-—_]+", "", s).lower()


def coverage(cues: List[Cue], duration: float) -> float:
    """时间轴上被 cue 覆盖的总长度（重叠区间只算一次）/ 视频时长。"""
    if duration <= 0 or not cues:
        return 0.0
    spans = sorted((max(0.0, c.start), max(max(0.0, c.start), c.end)) for c in cues)
    total = 0.0
    start, end = spans[0]
    for s, e in spans[1:]:
        if s > end:
            total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    total += end - start
    return min(1.0, total / duration)


def is_meaningful(text: str) -> bool:
    """至少有一个有效字符，用于过滤 OCR 噪声。"""
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)) >= 1


def join_ocr(parts: List[str]) -> str:
    """拼接短语：两侧都是英文/数字时补一个空格，中文直接相连。"""
    text = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if text and re.search(r"[A-Za-z0-9]$", text) and re.match(r"[A-Za-z0-9]", p):
            text += " " + p
        else:
            text += p
    return text


def group_cues(cues: List[Cue], max_chars: int = 220, max_gap: float = 2.5,
               merge: bool = True) -> List[Tuple[float, str]]:
    """把碎片化的字幕/ASR 句合并成便于阅读的段落块。

    merge=False 时不做任何合并，每条 cue 独立返回（人工字幕 / AI 字幕用）。
    OCR 相邻 cue 之间统一补一个空格分隔，避免识别结果粘连成一句。
    """
    if not merge:
        return [(c.start, c.text.strip()) for c in cues if c.text.strip()]
    blocks: List[List] = []  # [start, text, end]
    for c in cues:
        text = c.text.strip()
        if not text:
            continue
        if blocks and blocks[-1][2] >= c.start - max_gap and len(blocks[-1][1]) < max_chars:
            prev = blocks[-1]
            if c.src == "ocr":
                prev[1] = f"{prev[1]} {text}"
            else:
                prev[1] = join_ocr([prev[1], text])
            prev[2] = max(prev[2], c.end)
        else:
            blocks.append([c.start, text, c.end])
    return [(b[0], b[1]) for b in blocks]
