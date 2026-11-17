# -*- coding: utf-8 -*-
"""本地视频文件适配器：不抓字幕、不下载，直接把文件交给 ASR / OCR。"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse

from backend import ffmpeg
from backend.cues import Cue


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


class LocalPlatform:
    name = "local"
    label = "本地"
    needs_cookie = False
    supports_subtitles = False

    def __init__(self) -> None:
        # download_audio / download_video 只拿得到 bvid，这里用它反查源文件
        self._sources = {}

    # --- Cookie（本地文件不需要登录）---
    def load_cookie(self, explicit: Optional[str] = None,
                    cookie_file: Optional[str] = None) -> str:
        return ""

    def print_cookie_help(self) -> None:
        return

    # --- 视频信息 ---
    def resolve(self, url: str) -> Tuple[str, int]:
        path = _to_path(url)
        if not path:
            raise RuntimeError("请填写本地视频文件路径")
        if not os.path.exists(path):
            raise RuntimeError(f"本地文件不存在：{path}")
        if not os.path.isfile(path):
            raise RuntimeError(f"请选择视频文件，而不是文件夹：{path}")
        return path, 0

    def video_info(self, vid: str) -> dict:
        streams = ffmpeg.probe_streams(vid)
        if "video" not in streams:
            raise RuntimeError("该文件没有视频画面，请选择正常的视频文件")
        duration = ffmpeg.media_duration(vid)
        if duration <= 0:
            raise RuntimeError("无法读取视频时长，文件可能已损坏")
        title = os.path.splitext(os.path.basename(vid))[0] or "本地视频"
        bvid = "local_" + hashlib.md5(vid.encode("utf-8")).hexdigest()[:8]
        self._sources[bvid] = vid
        return {
            "bvid": bvid,
            "title": title,
            "owner": {"name": "本地文件"},
            "duration": duration,
            "desc": "",
            "videos": 1,
        }

    def video_tags(self, vid: str, cookie: str = "") -> List[str]:
        return []

    def pick_page(self, info: dict, part: int) -> dict:
        # 本地文件没有分 P，part 参数忽略
        return {"cid": 0, "page": 1, "part": "本地文件",
                "duration": float(info.get("duration") or 0)}

    def build_url(self, info: dict, bvid: str, page: dict) -> str:
        src = self._sources.get(bvid, "")
        try:
            return Path(src).as_uri() if src else ""
        except ValueError:
            return src

    # --- 媒体 ---
    def fetch_subtitles(self, bvid: str, cid: int, cookie: str,
                        duration: float) -> Tuple[List[Cue], str]:
        return [], ""

    def download_audio(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return self._source(bvid)

    def download_video(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return self._source(bvid)

    def _source(self, bvid: str) -> str:
        path = self._sources.get(bvid, "")
        if not path or not os.path.isfile(path):
            raise RuntimeError("源文件已不可用，请重新提交任务")
        return path
