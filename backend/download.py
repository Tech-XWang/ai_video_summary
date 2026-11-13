# -*- coding: utf-8 -*-
"""下载工具：断点续传下载与 GitHub Release 资产回退。"""

import os
from typing import Callable, Optional

import requests

from .config import UA

Progress = Callable[[float, str], None]


def _http_headers() -> dict:
    return {"User-Agent": UA}



def _format_bytes(n: float) -> str:
    gb = n / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f}GB"
    return f"{n / (1024 ** 2):.0f}MB"


def download_file(url: str, dest: str, progress: Optional[Progress] = None,
                  extra_headers: Optional[dict] = None) -> None:
    """带断点续传的下载；中断后保留 .part，下次从已有字节继续。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    pos = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = _http_headers()
    if extra_headers:
        headers.update(extra_headers)
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with requests.get(url, headers=headers, stream=True, timeout=(15, 60),
                      allow_redirects=True) as r:
        if r.status_code == 416 and os.path.isfile(part):
            os.replace(part, dest)
            return
        r.raise_for_status()
        append = pos > 0 and r.status_code == 206
        if not append:
            pos = 0
        total = int(r.headers.get("Content-Length") or 0)
        if append:
            total += pos
        mode = "ab" if append else "wb"
        last = -1.0
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                if progress is not None:
                    pct = pos / total if total else -1.0
                    if pct < 0 or pct - last >= 0.01 or pct >= 1.0:
                        last = pct
                        if pct < 0:
                            text = f"下载中（{_format_bytes(pos)}）"
                        else:
                            text = f"下载中 {pct:.0%}（{_format_bytes(pos)} / {_format_bytes(total)}）"
                        progress(pct, text)
    if total and pos < total:
        raise RuntimeError(f"下载不完整：{_format_bytes(pos)} / {_format_bytes(total)}")
    os.replace(part, dest)


def _download_asset(asset: dict, dest: str, progress: Optional[Progress] = None) -> None:
    """下载 GitHub Release 资产。

    部分网络环境下 github.com 的连接会被拦截/超时，而 api.github.com 正常。
    API 资产地址会直接 302 到 CDN，因此优先走 API 通道，失败再回退到网页下载地址。
    """
    candidates = []
    if asset.get("url"):
        candidates.append((str(asset["url"]), {"Accept": "application/octet-stream"}))
    if asset.get("browser_download_url"):
        candidates.append((str(asset["browser_download_url"]), None))
    last_err = None
    for url, headers in candidates:
        try:
            if progress is not None:
                # 连接阶段可能长时间无输出，先给界面一个“仍在尝试”的提示
                progress(-1.0, "连接下载源…")
            download_file(url, dest, progress, headers)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"下载失败（可检查网络或代理）：{last_err}")
