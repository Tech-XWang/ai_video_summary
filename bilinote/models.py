# -*- coding: utf-8 -*-
"""数据模型。"""

from dataclasses import dataclass


@dataclass
class Cue:
    """一条带时间轴的文本（字幕 / OCR / ASR 通用）。"""

    start: float
    end: float
    text: str
    src: str = "asr"
