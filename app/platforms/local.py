# -*- coding: utf-8 -*-
"""本地媒体文件适配器：支持视频/纯音频/纯视频，可优先提取内嵌文本字幕。"""

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse

from backend import ffmpeg
from backend.cues import Cue
from backend.utils import log


def _to_path(raw: str) -> str:
    """把用户输入整理成绝对路径：支持裸路径、file:// URI、引号包裹的相对路径。"""
    text = (raw or "").strip().strip('"').strip("'")
    if text.lower().startswith("file://"):
        path = unquote(urlparse(text).path)
        # Windows 的 file:///D:/a.mp4 解析出来是 /D:/a.mp4
        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        text = path
    return os.path.abspath(os.path.expanduser(text)) if text else ""


def _srt_seconds(value: str) -> float:
    m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)", value or "")
    if not m:
        return 0.0
    h, minute, sec, ms = (int(x) for x in m.groups())
    return h * 3600 + minute * 60 + sec + ms / 1000.0


def _parse_srt(text: str) -> List[Cue]:
    """把 SRT 文本解析成 cue；去掉 <i> / {\\an8} 之类的样式标记。"""
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        left, _, right = lines[idx].partition("-->")
        body = " ".join(lines[idx + 1:])
        body = re.sub(r"<[^>]+>|\{[^}]*\}", "", body).strip()
        if not body:
            continue
        start = _srt_seconds(left)
        end = max(_srt_seconds(right), start + 0.1)
        cues.append(Cue(start, end, body, "cc"))
    return cues


class LocalPlatform:
    name = "local"
    label = "本地"
    needs_cookie = False
    supports_subtitles = True

    def __init__(self) -> None:
        # download_audio / download_video 只拿得到 bvid，这里用它反查源文件
        self._sources = {}
        # 同一个任务里 resolve / video_info / probe_media 会重复读取，缓存一次探测结果
        self._probes = {}

    # --- Cookie（本地文件不需要登录）---
    def load_cookie(self, explicit: Optional[str] = None,
                    cookie_file: Optional[str] = None) -> str:
        return ""

    def print_cookie_help(self) -> None:
        return

    # --- 媒体探测 ---
    def _probe(self, raw: str) -> dict:
        path = _to_path(raw)
        if not path:
            raise RuntimeError("请填写本地媒体文件路径")
        if not os.path.exists(path):
            raise RuntimeError(f"本地文件不存在：{path}")
        if not os.path.isfile(path):
            raise RuntimeError(f"请选择媒体文件，而不是文件夹：{path}")

        cached = self._probes.get(path)
        if cached:
            return dict(cached)

        media = ffmpeg.probe_media_info(path)
        if not media["has_video"] and not media["has_audio"]:
            raise RuntimeError("该文件既没有音频也没有视频，无法提炼")
        if media["duration"] <= 0:
            raise RuntimeError("无法读取媒体时长，文件可能已损坏")
        info = {"path": path, **media}
        self._probes[path] = info
        return dict(info)

    def probe_media(self, url: str) -> dict:
        """供 service 在跑依赖检查前判断文件包含哪些轨道。"""
        return self._probe(url)

    # --- 媒体信息 ---
    def resolve(self, url: str) -> Tuple[str, int]:
        return self._probe(url)["path"], 0

    def video_info(self, vid: str) -> dict:
        media = self._probe(vid)
        title = os.path.splitext(os.path.basename(vid))[0] or "本地媒体"
        bvid = "local_" + hashlib.md5(vid.encode("utf-8")).hexdigest()[:8]
        self._sources[bvid] = vid
        return {
            "bvid": bvid,
            "title": title,
            "owner": {"name": "本地文件"},
            "duration": media["duration"],
            "desc": "",
            "videos": 1,
            "has_video": media["has_video"],
            "has_audio": media["has_audio"],
        }

    def video_tags(self, vid: str, cookie: str = "") -> List[str]:
        return []

    def pick_page(self, info: dict, part: int) -> dict:
        # 本地文件没有分 P，part 参数忽略
        return {"cid": 0, "page": 1, "part": "本地文件",
                "duration": float(info.get("duration") or 0),
                "has_video": bool(info.get("has_video")),
                "has_audio": bool(info.get("has_audio"))}

    def build_url(self, info: dict, bvid: str, page: dict) -> str:
        src = self._sources.get(bvid, "")
        try:
            return Path(src).as_uri() if src else ""
        except ValueError:
            return src

    # --- 媒体 ---
    def fetch_subtitles(self, bvid: str, cid: int, cookie: str,
                        duration: float) -> Tuple[List[Cue], str]:
        """提取内嵌文本字幕轨；没有文本字幕时返回空，让上层回退到 ASR。"""
        src = self._source(bvid)
        index = self._probe(src).get("subtitle_index")
        if index is None:
            return [], ""
        try:
            text = ffmpeg.extract_subtitle_text(src, int(index))
        except RuntimeError as e:
            log(f"  内嵌字幕提取失败：{e}")
            return [], ""
        return _parse_srt(text), "cc"

    def download_audio(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return self._source(bvid)

    def download_video(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return self._source(bvid)

    def _source(self, bvid: str) -> str:
        path = self._sources.get(bvid, "")
        if not path or not os.path.isfile(path):
            raise RuntimeError("源文件已不可用，请重新提交任务")
        return path
