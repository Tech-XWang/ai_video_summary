# -*- coding: utf-8 -*-
"""B 站平台适配器：全部转调 bilinote.bili_api，不新增逻辑。"""

from typing import List, Optional, Tuple

from bilinote import bili_api
from bilinote.models import Cue


class BilibiliPlatform:
    name = "bilibili"
    label = "B站"

    # --- Cookie ---
    def load_cookie(self, explicit: Optional[str] = None,
                    cookie_file: Optional[str] = None) -> str:
        return bili_api.load_cookie(explicit, cookie_file)

    def print_cookie_help(self) -> None:
        bili_api.print_manual_cookie_help()

    def login_qr(self, sess=None) -> Tuple[Optional[str], Optional[bytes]]:
        return bili_api.login_qr(sess)

    def login_poll(self, key: str, sess=None) -> Tuple[Optional[int], Optional[str]]:
        return bili_api.login_poll(key, sess)

    def save_cookie(self, cookie: str) -> None:
        bili_api.save_cookie(cookie)

    @property
    def login_state_msg(self) -> dict:
        return bili_api.LOGIN_STATE_MSG

    # --- 视频信息 ---
    def resolve(self, url: str) -> Tuple[str, int]:
        return bili_api.resolve_url(url)

    def video_info(self, vid: str) -> dict:
        return bili_api.get_video_info(vid)

    def video_tags(self, vid: str, cookie: str = "") -> List[str]:
        return bili_api.get_video_tags(vid, cookie)

    def pick_page(self, info: dict, part: int) -> dict:
        return bili_api.pick_page(info, part)

    def build_url(self, info: dict, bvid: str, page: dict) -> str:
        suffix = f"?p={page['page']}" if (info.get("videos") or 1) > 1 else ""
        return f"https://www.bilibili.com/video/{bvid}{suffix}"

    # --- 媒体 ---
    def fetch_subtitles(self, bvid: str, cid: int, cookie: str,
                        duration: float) -> Tuple[List[Cue], str]:
        return bili_api.fetch_subtitles(bvid, cid, cookie, duration)

    def download_audio(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return bili_api.download_audio(bvid, cid, cookie, work)

    def download_video(self, bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
        return bili_api.download_video(bvid, cid, cookie, work)
