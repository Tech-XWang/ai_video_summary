# -*- coding: utf-8 -*-
"""多路文本汇合：保留所有启用的来源，各路内部按自己的规则合并。

人工字幕 / AI 字幕每条独立；OCR / ASR 继续合并相邻同文。跨路不做去重，
导出时仍按来源分文件，LLM 侧也把字幕/ASR 与 OCR 作为独立文本块使用。
"""

from typing import Dict, List, Tuple

from .config import INDEPENDENT_SRC, SRC_DESC
from .models import Cue
from .utils import log, norm_text

_ORDER = ("cc", "ai-sub", "ocr", "asr")


def pick_subtitle_track(groups: Dict[str, List[Cue]]) -> Tuple[List[Cue], str]:
    """只在 B 站字幕内部按 人工 CC > AI 字幕 选一条（不含 OCR）。"""
    keys = [k for k in ("cc", "ai-sub") if groups.get(k)]
    if not keys:
        return [], ""
    key = keys[0]
    return groups[key], key


def merge_sources(groups: Dict[str, List[Cue]]) -> Tuple[List[Cue], str]:
    """把所有启用的来源各自合并后按时间拼接。"""
    groups = {k: sorted(v, key=lambda c: c.start) for k, v in groups.items() if v}
    if not groups:
        return [], ""
    keys = [k for k in _ORDER if groups.get(k)]
    keys += [k for k in groups if k not in _ORDER]

    timeline: List[Cue] = []
    for key in keys:
        collapse = key not in INDEPENDENT_SRC
        merged: List[Cue] = []
        for c in groups[key]:
            if (collapse and merged and norm_text(merged[-1].text) == norm_text(c.text)
                    and c.start - merged[-1].end < 1.0):
                merged[-1].end = max(merged[-1].end, c.end)
            else:
                merged.append(Cue(c.start, c.end, c.text, c.src))
        timeline.extend(merged)

    timeline.sort(key=lambda c: c.start)
    desc = "+".join(SRC_DESC.get(k, k) for k in keys)
    log(f"  保留轨道：{desc}（{len(timeline)} 条）")
    return timeline, desc
