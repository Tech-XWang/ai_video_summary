# -*- coding: utf-8 -*-
"""B 站网络交互：Cookie/登录、URL 解析、视频信息、字幕、播放地址与下载。

所有 HTTP 请求复用同一个 Session；需要登录态的请求通过 ``cookie_with_headers``
在单次请求上覆盖 Cookie 头，不影响全局默认头。
"""

import hashlib
import io
import locale
import os
import re
import sys
import threading
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from .config import API, CACHE_DIR, COOKIE_FILE, HEADERS
from .cues import Cue
from .utils import coverage, log

# 复用 TCP 连接；每次请求仍可通过 headers= 覆盖 Cookie
HTTP = requests.Session()
HTTP.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# Cookie / 登录
# ---------------------------------------------------------------------------
def _read_cookie_lines(path: str) -> List[str]:
    """读取 Cookie 文件；兼容被编辑器另存为 ANSI/GBK 的情况。"""
    encodings = ["utf-8", locale.getpreferredencoding(False), "gbk"]
    for enc in dict.fromkeys(encodings):
        try:
            with open(path, "r", encoding=enc) as f:
                return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        except UnicodeDecodeError:
            continue
        except OSError:
            return []
    return []


def load_cookie(explicit: Optional[str] = None, cookie_file: Optional[str] = None) -> str:
    """返回可直接放进请求头的 Cookie 字符串，取不到则返回空串。"""
    if explicit:
        return explicit.strip()
    path = cookie_file or COOKIE_FILE
    if os.path.isfile(path):
        lines = _read_cookie_lines(path)
        if lines:
            parts: List[str] = []
            for ln in lines:
                for kv in ln.split(";"):
                    kv = kv.strip()
                    if not kv:
                        continue
                    # 兼容只粘贴 SESSDATA 值、漏写键名的情况
                    if "=" not in kv and len(kv) > 20:
                        kv = "SESSDATA=" + kv
                    parts.append(kv)
            if parts:
                return "; ".join(parts)
    sess = os.environ.get("BILI_SESSDATA", "").strip()
    return f"SESSDATA={sess}" if sess else ""


def cookie_with_headers(cookie: str) -> Dict[str, str]:
    h = dict(HEADERS)
    if cookie:
        h["Cookie"] = cookie
    return h


def login_qr(sess: Optional[requests.Session] = None) -> Tuple[Optional[str], Optional[bytes]]:
    """生成扫码登录二维码，返回 (qrcode_key, PNG 字节)；失败返回 (None, None)。"""
    try:
        import qrcode
    except ImportError:
        log("缺少 qrcode 依赖，可执行：python -m pip install qrcode")
        return None, None
    sess = sess or requests.Session()
    try:
        r = sess.get(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/generate",
            headers=HEADERS, timeout=15,
        ).json()
    except (requests.RequestException, ValueError) as e:
        log(f"请求登录二维码失败：{e}")
        return None, None
    if r.get("code") != 0:
        log(f"生成登录二维码失败：{r.get('code')} {r.get('message')}")
        return None, None
    data = r["data"]
    buf = io.BytesIO()
    qrcode.make(data["url"]).save(buf, format="PNG")
    return data["qrcode_key"], buf.getvalue()


def login_poll(key: str, sess: Optional[requests.Session] = None) -> Tuple[Optional[int], Optional[str]]:
    """轮询扫码状态，返回 (state, cookie)。state==0 且 cookie 非空表示登录成功。"""
    sess = sess or requests.Session()
    try:
        p = sess.get(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
            params={"qrcode_key": key, "source": "main-fe-header"},
            headers=HEADERS, timeout=15,
        ).json()
    except (requests.RequestException, ValueError) as e:
        log(f"轮询登录状态失败：{e}")
        return None, None
    d = p.get("data") or {}
    state = d.get("code")
    if state != 0:
        return state, None
    # 优先从回调 URL 解析，Set-Cookie 有时取不全
    q = urllib.parse.parse_qs(urllib.parse.urlparse(d.get("url", "")).query)
    pairs = {}
    for k in ("SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5", "sid"):
        if q.get(k):
            pairs[k] = q[k][0]
    if not pairs:
        pairs = sess.cookies.get_dict()
    if "SESSDATA" not in pairs:
        log("登录成功但未拿到 SESSDATA，请改用浏览器手动复制。")
        return None, None
    return 0, "; ".join(f"{k}={v}" for k, v in pairs.items() if v)


def save_cookie(cookie: str) -> None:
    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
        f.write(cookie + "\n")


LOGIN_STATE_MSG = {86101: "等待扫码", 86090: "已扫码，请在手机上确认", 86100: "确认中", 86038: "二维码已过期"}


def login_flow() -> str:
    """扫码登录并把 cookie 写入本地文件（CLI 用）。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sess = requests.Session()
    key, png = login_qr(sess)
    if not key or png is None:
        return ""
    qr_path = os.path.join(CACHE_DIR, "login_qr.png")
    with open(qr_path, "wb") as f:
        f.write(png)
    try:
        os.startfile(qr_path)  # noqa: S606  Windows 下用默认看图工具打开
    except Exception:
        pass
    log(f"已打开登录二维码：{qr_path}")
    log("请用 B 站手机客户端扫码（有效期约 3 分钟）……")

    deadline = time.time() + 180
    last = None
    while time.time() < deadline:
        time.sleep(2)
        state, cookie = login_poll(key, sess)
        if state is None:
            continue
        if state != last:
            msg = LOGIN_STATE_MSG.get(state)
            if msg:
                log(f"  {msg}")
            last = state
        if state == 0:
            if not cookie:
                return ""
            save_cookie(cookie)
            log(f"登录成功，Cookie 已保存到 {COOKIE_FILE}")
            return cookie
        if state == 86038:
            break
    log("登录超时或二维码已失效，请重试。")
    return ""


def print_manual_cookie_help() -> None:
    log(
        "\n手动获取 Cookie 的方法：\n"
        "  1. 浏览器登录 https://www.bilibili.com\n"
        "  2. F12 → Application(应用) → Cookies → https://www.bilibili.com\n"
        "  3. 复制 SESSDATA 的值（形如 1a2b3c...%2C...）\n"
        f"  4. 新建 {COOKIE_FILE}，写入一行：SESSDATA=粘贴的值\n"
        "  5. 重新运行本脚本即可读取\n"
    )


# ---------------------------------------------------------------------------
# 地址解析与通用接口
# ---------------------------------------------------------------------------
def resolve_url(text: str) -> Tuple[str, int]:
    """把各种输入（BV/av/短链）解析成 bvid 与分P序号。"""
    text = text.strip()
    if re.fullmatch(r"BV[0-9A-Za-z]{10}", text):
        return text, 1
    if re.fullmatch(r"av\d+", text, re.I):
        return text.lower(), 1

    if "b23.tv" in text:
        try:
            r = HTTP.get(text, timeout=15, allow_redirects=True)
            text = r.url
        except requests.RequestException as e:
            raise RuntimeError(f"短链解析失败：{e}")

    m = re.search(r"(BV[0-9A-Za-z]{10})", text)
    if m:
        p = re.search(r"[?&]p=(\d+)", text)
        return m.group(1), int(p.group(1)) if p else 1
    m = re.search(r"av(\d+)", text, re.I)
    if m:
        p = re.search(r"[?&]p=(\d+)", text)
        return "av" + m.group(1), int(p.group(1)) if p else 1
    raise RuntimeError(f"无法从输入中识别视频号：{text[:80]}")


def api_get(url: str, cookie: str, params: Optional[dict] = None, tries: int = 3) -> dict:
    last_err: Optional[Exception] = None
    for i in range(tries):
        try:
            r = HTTP.get(url, params=params, headers=cookie_with_headers(cookie), timeout=20)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if i < tries - 1:
                log(f"  接口请求失败，第 {i + 1}/{tries} 次重试：{e}")
                time.sleep(1.0 + i)
    raise RuntimeError(f"接口请求失败 {url}：{last_err}")


def get_video_info(vid: str) -> dict:
    if vid.startswith("av"):
        r = api_get(f"{API}/x/web-interface/view", "", {"aid": vid[2:]})
    else:
        r = api_get(f"{API}/x/web-interface/view", "", {"bvid": vid})
    if r.get("code") != 0:
        raise RuntimeError(f"获取视频信息失败：{r.get('code')} {r.get('message')}")
    return r["data"]


def get_video_tags(vid: str, cookie: str = "") -> List[str]:
    """取视频标签；接口异常或没有标签时返回空列表，不影响主流程。"""
    params = {"aid": vid[2:]} if vid.startswith("av") else {"bvid": vid}
    try:
        r = api_get(f"{API}/x/web-interface/view/detail", cookie, params, tries=2)
    except RuntimeError as e:
        log(f"  获取视频标签失败：{e}")
        return []
    if r.get("code") != 0:
        return []
    tags = ((r.get("data") or {}).get("Tags")) or []
    return [str(t.get("tag_name") or "").strip() for t in tags
            if str(t.get("tag_name") or "").strip()]


def pick_page(info: dict, part: int) -> dict:
    pages = info.get("pages") or []
    if not pages:
        return {"cid": info["cid"], "page": 1, "part": info.get("title", ""), "duration": info.get("duration", 0)}
    part = min(max(1, part), len(pages))
    pg = pages[part - 1]
    return {
        "cid": pg["cid"],
        "page": pg["page"],
        "part": pg.get("part", ""),
        "duration": pg.get("duration", info.get("duration", 0)),
    }


# ---- WBI 签名（player/wbi/v2 兜底需要）----
_WBI_MIXIN = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]

_WBI_CACHE = {"expire": 0.0, "key": "", "raw": ""}


def _wbi_keys() -> Tuple[str, str]:
    now = time.time()
    if _WBI_CACHE["key"] and _WBI_CACHE["expire"] > now:
        return _WBI_CACHE["key"], _WBI_CACHE["raw"]
    nav = api_get(f"{API}/x/web-interface/nav", "")
    wbi = (nav.get("data") or {}).get("wbi_img") or {}
    img = wbi.get("img_url", "").rsplit("/", 1)[-1].split(".")[0]
    sub = wbi.get("sub_url", "").rsplit("/", 1)[-1].split(".")[0]
    raw = img + sub
    if len(raw) < 64:
        return "", ""
    key = "".join(raw[i] for i in _WBI_MIXIN if i < len(raw))[:32]
    _WBI_CACHE.update(expire=now + 600, key=key, raw=raw)
    return key, raw


def wbi_sign(params: dict) -> dict:
    key, _ = _wbi_keys()
    if not key:
        return params
    signed = dict(params)
    signed["wts"] = int(time.time())
    q = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted(signed.items()))
    signed["w_rid"] = hashlib.md5((q + key).encode()).hexdigest()
    return signed


# ---------------------------------------------------------------------------
# 字幕
# ---------------------------------------------------------------------------
def rank_subtitle(item: dict) -> Tuple[int, str]:
    lan = (item.get("lan") or "").lower()
    doc = item.get("lan_doc") or ""
    is_ai = ("ai" in lan) or ("自动生成" in doc) or (item.get("ai_type") or 0) > 0
    return (1 if is_ai else 0, lan)


def _player_subtitles(bvid: str, cid: int, cookie: str) -> List[dict]:
    """优先走 WBI 签名接口：未签名的 player/v2 会偶发返回其他视频的字幕。"""
    try:
        r = api_get(f"{API}/x/player/wbi/v2", cookie, wbi_sign({"bvid": bvid, "cid": cid}))
        subs = ((r.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
    except RuntimeError:
        subs = []
    if not subs:
        r = api_get(f"{API}/x/player/v2", cookie, {"bvid": bvid, "cid": cid})
        subs = ((r.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
    return subs


def _valid_track(item: dict, cid: int) -> bool:
    """校验字幕条目确实属于当前视频。

    B 站偶发返回其他视频的字幕 URL；AI 字幕的 URL 形如
    .../ai_subtitle/prod/<aid><cid><hash>，可用 cid 校验。人工 CC 的 URL 是内容哈希，
    无法直接校验，只要与一条正确 AI 轨同时出现即可信。
    """
    url = item.get("subtitle_url") or ""
    if not url:
        return False
    if rank_subtitle(item)[0] == 0:
        return True
    return str(cid) in url


def get_subtitle_meta(bvid: str, cid: int, cookie: str) -> List[dict]:
    """取字幕列表。B 站偶发只返回 AI 轨或别的视频的字幕，需要重试并校验。"""
    best: List[dict] = []
    for attempt in range(6):
        subs = _player_subtitles(bvid, cid, cookie)
        valid = [s for s in subs if _valid_track(s, cid)]
        if any(rank_subtitle(s)[0] == 0 for s in valid):
            return valid
        if valid and not best:
            best = valid
        if not subs:
            return best
        if attempt == 0 and valid:
            log("  字幕列表仅返回 AI 字幕，重试查找人工字幕……")
        elif not valid:
            log("  字幕列表异常（可能返回了其他视频的字幕），重试……")
        if attempt < 5:
            time.sleep(0.8)
    return best


def fetch_subtitles(bvid: str, cid: int, cookie: str, duration: float) -> Tuple[List[Cue], str]:
    """取一条 B 站字幕，返回 (cues, src)；src 为 cc / ai-sub / 空串。"""
    subs = get_subtitle_meta(bvid, cid, cookie)
    if not subs:
        if cookie:
            log("  未获取到字幕：Cookie 可能已失效（可重新 --login），或该视频确实没有字幕")
        else:
            log("  未获取到字幕：B 站字幕接口需要登录态，请先 --login")
        return [], ""

    zh = [s for s in subs if (s.get("lan") or "").lower().startswith(("zh", "ai-zh"))] or subs
    zh.sort(key=rank_subtitle)
    chosen = zh[0]
    url = chosen.get("subtitle_url") or ""
    if not url:
        return [], ""
    if url.startswith("//"):
        url = "https:" + url
    try:
        raw = HTTP.get(url, headers=cookie_with_headers(cookie), timeout=20).json()
    except (requests.RequestException, ValueError) as e:
        log(f"  字幕下载失败：{e}")
        return [], ""

    src = "ai-sub" if rank_subtitle(chosen)[0] == 1 else "cc"
    cues = []
    for b in raw.get("body") or []:
        text = re.sub(r"\s+", " ", str(b.get("content", ""))).strip()
        if not text:
            continue
        cues.append(Cue(float(b.get("from", 0)), float(b.get("to", 0)), text, src))
    cues.sort(key=lambda c: c.start)
    log(f"  字幕：{chosen.get('lan_doc') or chosen.get('lan')}，{len(cues)} 条，覆盖 {coverage(cues, duration):.0%}")
    return cues, src


# ---------------------------------------------------------------------------
# 播放地址与媒体下载
# ---------------------------------------------------------------------------
_PLAYURL_CACHE: Dict[Tuple[str, int], dict] = {}
_PLAYURL_LOCK = threading.Lock()


def get_play_url(bvid: str, cid: int, cookie: str) -> dict:
    """同一次运行内音频/视频两路复用同一份 playurl 响应。

    并发下载时会同时调用这里，锁住缓存查询和请求，避免重复请求接口。
    """
    key = (bvid, cid)
    with _PLAYURL_LOCK:
        if key in _PLAYURL_CACHE:
            return _PLAYURL_CACHE[key]
        r = api_get(f"{API}/x/player/playurl", cookie,
                    {"bvid": bvid, "cid": cid, "fnval": 16, "fourk": 1})
        if r.get("code") != 0:
            raise RuntimeError(f"获取播放地址失败：{r.get('code')} {r.get('message')}")
        data = r.get("data") or {}
        _PLAYURL_CACHE[key] = data
        return data


def download(url: str, path: str, cookie: str, label: str) -> bool:
    if os.path.isfile(path):
        cached_size = os.path.getsize(path)
        if cached_size > 1024:
            log(f"  {label}：使用缓存 {os.path.basename(path)}（{cached_size / 1048576:.1f}MB）")
            return True
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".part"
    h = cookie_with_headers(cookie)
    for attempt in range(3):
        try:
            with HTTP.get(url, headers=h, timeout=60, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                with tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
                          desc=f"  {label}下载", leave=False, file=sys.stdout) as bar, \
                        open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            if os.path.getsize(tmp) < 1024:
                raise RuntimeError("文件过小")
            os.replace(tmp, path)
            log(f"  {label}：完成 {os.path.getsize(path) / 1048576:.1f}MB")
            return True
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            log(f"  {label}下载失败（第{attempt + 1}次）：{e}")
            time.sleep(1 + attempt)
    return False


def download_audio(bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
    data = get_play_url(bvid, cid, cookie)
    audios = ((data.get("dash") or {}).get("audio")) or []
    if not audios:
        log("  未拿到 DASH 音频流，尝试 durl……")
        durl = data.get("durl") or []
        if not durl:
            return None
        url = durl[0].get("url")
        out = os.path.join(work, "audio.m4s")
        return out if download(url, out, cookie, "音频") else None

    best = max(audios, key=lambda a: a.get("bandwidth", 0))
    out = os.path.join(work, f"audio_{best.get('bandwidth', 0)}.m4s")
    if not download(best.get("baseUrl"), out, cookie, "音频"):
        backup = (best.get("backupUrl") or [None])[0]
        if not backup or not download(backup, out, cookie, "音频"):
            return None
    return out


def download_video(bvid: str, cid: int, cookie: str, work: str) -> Optional[str]:
    data = get_play_url(bvid, cid, cookie)
    videos = ((data.get("dash") or {}).get("video")) or []
    if not videos:
        return None
    # 优先 AVC（兼容性最好），再按清晰度取最低
    avc = [v for v in videos if v.get("codecid") == 7]
    pool = avc or videos
    best = min(pool, key=lambda v: (v.get("id", 999), v.get("bandwidth", 1 << 60)))
    out = os.path.join(work, f"video_{best.get('id', 0)}.mp4")
    log(f"  OCR 视频流：{best.get('id')}P / {best.get('bandwidth', 0) // 1000}kbps")
    if not download(best.get("baseUrl"), out, cookie, "视频"):
        backup = (best.get("backupUrl") or [None])[0]
        if not backup or not download(backup, out, cookie, "视频"):
            return None
    return out
