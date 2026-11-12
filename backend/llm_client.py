# -*- coding: utf-8 -*-
"""LLM 服务管理（本地 llama-server / 兼容 OpenAI 接口）与两阶段总结。"""

import datetime
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import requests

from . import llm_runtime, prompts
from .config import (
    CACHE_DIR,
    LLAMA_BASE,
    LLAMA_HOST,
    LLAMA_PORT,
    LLAMA_SLEEP_IDLE,
    LLM_REASONING_EFFORT,
    INDEPENDENT_SRC,
    SRC_DESC,
)
from .cues import Cue
from .utils import check_cancel, fmt_ts, group_cues, log, report

# 探测模型列表/健康检查用的会话（与 B 站 HTTP 会话相互独立）
_HTTP = requests.Session()

# 本次进程启动的 llama-server 句柄；GUI 关窗时据此释放显存
_MANAGED_PROC: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# 服务探测
# ---------------------------------------------------------------------------
def resolve_one(base_url: str, model: str) -> Tuple[Optional[str], Optional[str]]:
    """只探测指定地址，不做其它候选回退。"""
    try:
        r = _HTTP.get(base_url.rstrip("/") + "/models", timeout=3)
        if r.status_code != 200:
            return None, None
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        if not ids:
            return None, None
        chosen = model or os.environ.get("LLAMA_MODEL") or ids[0]
        if chosen not in ids:
            chosen = ids[0]
        return base_url.rstrip("/"), chosen
    except (requests.RequestException, ValueError):
        return None, None


def resolve_llm(base_url: str, model: str) -> Tuple[Optional[str], Optional[str]]:
    for candidate in ([base_url] if base_url else []) + [
        os.environ.get("LLAMA_BASE_URL", ""),
        LLAMA_BASE,
        "http://localhost:11434/v1",
    ]:
        if not candidate:
            continue
        base, mdl = resolve_one(candidate, model)
        if base:
            return base, mdl
    return None, None


def _llama_health() -> Optional[int]:
    try:
        return _HTTP.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health", timeout=2).status_code
    except requests.RequestException:
        return None


def _llama_sleeping() -> bool:
    """本地 llama-server 是否处于休眠（权重与 KV cache 已从显存卸载）。"""
    try:
        r = _HTTP.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/props", timeout=3)
        return r.status_code == 200 and bool((r.json() or {}).get("is_sleeping"))
    except (requests.RequestException, ValueError):
        return False


def _wait_llama(proc: Optional[subprocess.Popen], timeout: float = 600.0) -> bool:
    t0 = time.time()
    last_log = t0
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        if _llama_health() == 200:
            return True
        if time.time() - last_log >= 15:
            log(f"    等待模型加载…… {int(time.time() - t0)}s")
            last_log = time.time()
        time.sleep(1.0)
    return False


def start_local_llm() -> Optional[subprocess.Popen]:
    """按本地模型清单启动 llama-server；返回进程句柄，已在运行则返回 None。"""
    global _MANAGED_PROC
    if _llama_health() is not None:
        log("  检测到 llama-server 已存在，等待加载完成……")
        _wait_llama(None)
        return None

    exe = llm_runtime.llama_server_exe()
    model = llm_runtime.resolve_model_path()
    if not exe or not model:
        log("  本地 LLM 未就绪：缺少 llama.cpp 运行时或模型文件")
        return None

    log("  未检测到 LLM 服务，启动 llama-server（需加载约1分钟）……")
    os.makedirs(CACHE_DIR, exist_ok=True)
    log_path = os.path.join(CACHE_DIR, "llama-server.log")
    cmd = [
        exe, "-m", model,
        "-ngl", "999", "--parallel", "1", "-c", str(llm_runtime.context_size()),
        "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
        "--reasoning-format", "deepseek", "--alias", llm_runtime.model_alias(),
        "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
        "--sleep-idle-seconds", str(LLAMA_SLEEP_IDLE),
    ]
    # 每次启动覆盖旧日志，避免长期运行后日志文件无限增长
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(
            cmd, cwd=os.path.dirname(exe) or None, stdout=lf, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    try:
        if _wait_llama(proc):
            log("  llama-server 就绪")
            _MANAGED_PROC = proc
            return proc
    except BaseException:
        stop_local_llm(proc)
        raise
    log(f"  启动超时或异常退出，日志见 {log_path}")
    stop_local_llm(proc)
    return None


def stop_local_llm(proc: Optional[subprocess.Popen]) -> None:
    """只关闭本次启动的实例；已存在的服务不动。"""
    global _MANAGED_PROC
    if proc is _MANAGED_PROC:
        _MANAGED_PROC = None
    if proc is None or proc.poll() is not None:
        return
    log("  关闭 llama-server，释放资源……")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def stop_managed_llm() -> bool:
    """关闭本进程启动、仍存活的 llama-server（GUI 退出时调用）。

    只处理自己启动的实例；用户事先用 start-chatbox.bat 开的服务不受影响。
    """
    proc = _MANAGED_PROC
    if proc is None:
        return False
    stop_local_llm(proc)
    return True


# ---------------------------------------------------------------------------
# 分块与调用
# ---------------------------------------------------------------------------
def chunk_ranges(cues: List[Cue], max_chars: int) -> List[Tuple[float, float]]:
    """按时间顺序切分参考时间轴，返回每个分块的 [起, 止]。"""
    ranges: List[Tuple[float, float]] = []
    cur: List[Cue] = []
    size = 0
    for c in cues:
        line_len = len(c.text) + 1  # 只算正文，不含时间前缀
        if size + line_len > max_chars and cur:
            ranges.append((cur[0].start, cur[-1].end))
            cur, size = [], 0
        cur.append(c)
        size += line_len
    if cur:
        ranges.append((cur[0].start, cur[-1].end))
    return ranges


def track_text(cues: List[Cue], t0: float, t1: float, merge: bool = True) -> str:
    """取出与 [t0, t1] 有交集的文本，按段落合并成带时间戳的多行。

    merge=False 时每条 cue 独立成行（人工字幕 / AI 字幕用）。
    """
    sel = [c for c in cues if c.end >= t0 - 5 and c.start <= t1 + 5]  # 留5秒重叠窗口
    if not sel:
        return ""
    return "\n".join(f"[{fmt_ts(s)}] {t}" for s, t in group_cues(sel, merge=merge))


def _public_base_url(url: str) -> str:
    """规整 base_url：去掉末尾斜杠，兼容用户直接填 .../chat/completions。"""
    base = (url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def _public_error(e: Exception) -> RuntimeError:
    """把 OpenAI SDK 异常转成面向用户的可读错误，便于界面直接展示。"""
    status = getattr(e, "status_code", None)
    detail = str(getattr(e, "message", None) or e)
    hints = {
        401: "API Key 无效或未授权（401）",
        403: "无权访问该接口或模型（403）",
        404: "接口地址或模型不存在（404），请检查 base_url 是否含 /v1 及 model 名称",
        429: "请求受限或额度不足（429）",
    }
    if status in hints:
        message = f"{hints[status]}：{detail}"
    elif status:
        message = f"外部 LLM 接口返回 HTTP {status}：{detail}"
    elif "Timeout" in type(e).__name__:
        message = f"外部 LLM 请求超时：{detail}"
    elif "Connection" in type(e).__name__:
        message = f"无法连接外部 LLM 接口：{detail}"
    else:
        message = f"调用外部 LLM 失败（{type(e).__name__}）：{detail}"
    return RuntimeError(message)


def _public_llm_call(client, model: str, system: str, user: str, max_tokens: int) -> str:
    """public 模式的生成函数：标准 OpenAI 兼容请求，失败直接上抛，不回退本地。"""
    log(f"[LLM 请求] messages.system:\n{system}", level="debug")
    log(f"[LLM 请求] messages.user:\n{user}", level="debug")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise _public_error(e) from e
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        raise RuntimeError(
            f"外部 LLM 返回空内容（finish_reason={resp.choices[0].finish_reason}）"
        )
    log(f"[LLM 响应] {text}", level="debug")
    return text


def _llm_call(client, model: str, system: str, user: str, max_tokens: int) -> str:
    """调用本地模型。默认开启思考（中等强度）；若思考吃满 token 导致正文为空，再关闭思考兜底。"""
    log(f"[LLM 请求] messages.system:\n{system}\n", level="debug")
    log(f"[LLM 请求] messages.user:\n{user}\n", level="debug")
    plans = [
        ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": LLM_REASONING_EFFORT}},
         max_tokens),
        ({"chat_template_kwargs": {"enable_thinking": False}}, max_tokens),
        ({}, int(max_tokens * 0.5)),
    ]
    last_err: Optional[Exception] = None
    for extra, budget in plans:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.6,
                max_tokens=budget,
                extra_body=extra,
            )
        except Exception as e:
            # 不再静默：保留最后一次异常，全部失败后输出，便于排查
            last_err = e
            log(f"[Warning] LLM 调用异常：{e}")
            continue
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if text:
            log(f"[LLM 响应] {text}", level="debug")
            return text
        reason = getattr(msg, "reasoning_content", "") or ""
        log(f"[Warning] LLM返回空正文（finish={resp.choices[0].finish_reason}，思考 {len(reason)} 字）…")
    if last_err is not None:
        log(f"[Warning] 本次 LLM 调用全部重试失败，最后一次异常：{last_err}")
    return ""


# ---------------------------------------------------------------------------
# 两阶段总结
# ---------------------------------------------------------------------------
def llm_summary(groups: Dict[str, List[Cue]], timeline: List[Cue], title: str, url: str,
                duration: float, base_url: str, model: str, max_chars: int,
                keep_llm: bool = False, desc: str = "",
                tags: Optional[List[str]] = None,
                llm_source: str = "local", api_key: str = "") -> Optional[str]:
    """keep_llm=True 时不关闭本次启动的 llama-server（GUI 常驻复用，省去重载）。

    desc 为视频简介（最多注入前 1000 字），tags 为视频标签，二者都会写入两阶段 prompt。
    llm_source=public 时改用 base_url/api_key/model 调用外部 OpenAI 兼容接口，
    分块、prompt 与结果解析仍与本地模式共用。
    """
    desc = (desc or "").strip()[:1000]
    tags = [t.strip() for t in (tags or []) if t.strip()]
    desc_text = desc or "无"
    tags_text = "、".join(tags) or "无"
    try:
        from openai import OpenAI
    except ImportError:
        log("缺少 openai 依赖，跳过 LLM 总结")
        return None

    explicit = (base_url or "").strip().rstrip("/")
    today = datetime.date.today().isoformat()
    local_default = {LLAMA_BASE, f"http://127.0.0.1:{LLAMA_PORT}/v1"}
    public = llm_source == "public"
    proc = None
    try:
        if public:
            # public 失败一律上抛，不静默回退本地
            if not explicit:
                raise RuntimeError("public 模式缺少 base_url")
            if not api_key.strip():
                raise RuntimeError("public 模式缺少 api_key")
            if not model.strip():
                raise RuntimeError("public 模式缺少 model")
            base = _public_base_url(explicit)
            mdl = model.strip()
            log(f"  LLM：{base} / {mdl}（public，OpenAI 兼容接口）")
            client = OpenAI(base_url=base, api_key=api_key.strip(), timeout=600.0)
            call = _public_llm_call
        else:
            call = _llm_call
            if explicit and explicit not in local_default:
                base, mdl = resolve_llm(explicit, model)
                if not base:
                    log(f"未检测到 LLM 服务：{explicit}（自定义地址不自动启动）")
                    return None
            else:
                base, mdl = resolve_one(LLAMA_BASE, model)
                if not base:
                    proc = start_local_llm()
                    base, mdl = resolve_one(LLAMA_BASE, model)
                if not base:
                    log("未检测到可用的本地 LLM 服务，本次仅导出转写稿。")
                    return None

            log(f"  LLM：{base} / {mdl}")
            if base in local_default and _llama_sleeping():
                log("  LLM 处于休眠（显存已释放），首次请求会自动唤醒并重载模型……")
            client = OpenAI(base_url=base, api_key="sk-local", timeout=1800.0)
        sub_cues = groups.get("cc") or groups.get("ai-sub") or []
        sub_key = "cc" if groups.get("cc") else ("ai-sub" if groups.get("ai-sub") else "")
        asr_cues = groups.get("asr") or []
        ocr_cues = groups.get("ocr") or []
        sub_merge = sub_key not in INDEPENDENT_SRC   # 人工/AI 字幕保持每条独立
        sub_label = SRC_DESC.get(sub_key, "字幕") if sub_key else "无"
        ranges = chunk_ranges(timeline, max_chars)
        total = len(ranges)
        log(f"    字幕源：{sub_label}（{len(sub_cues)} 条）｜ASR：{len(asr_cues)} 条"
            f"｜OCR：{len(ocr_cues)} 条｜分块 {total} 个")
        points: List[str] = []

        for i, (t0, t1) in enumerate(ranges, 1):
            check_cancel()
            report("llm", (i - 1) / total, f"分段提炼 {i}/{total}")
            log(f"    分段提炼 {i}/{total} …")
            sub_text = track_text(sub_cues, t0, t1, merge=sub_merge) or "（本段无字幕）"
            asr_text = track_text(asr_cues, t0, t1) or "（本段无ASR）"
            ocr_text = track_text(ocr_cues, t0, t1) or "（本段无OCR）"
            user = prompts.USER_CHUNK.format(
                date=today, title=title, desc=desc_text, tags=tags_text,
                i=i, n=total, start=fmt_ts(t0), end=fmt_ts(t1),
                sub_label=sub_label, sub_text=sub_text, asr_text=asr_text,
                ocr_text=ocr_text,
            )
            txt = call(client, mdl, prompts.SYSTEM_CHUNK, user, 10000)
            if txt:
                points.append(txt)
            else:
                log(f"    分段 {i}/{total} 提炼失败，用原始文本代替")
                fallback = sub_text if sub_cues else (ocr_text if ocr_text else asr_cues)
                points.append(f"[第{i}段原始转写，LLM 提炼失败]\n{fallback[:4000]}")

        log(f"    整合全文（{total} 段要点）…")
        report("llm", 1.0, "整合全文")
        user = prompts.USER_FINAL.format(
            date=today, title=title, duration=fmt_ts(duration),
            desc=desc_text, tags=tags_text,
            points=chr(10).join(points),
        )
        body = call(client, mdl, prompts.SYSTEM_FINAL, user, 10000)
        if not body:
            log("  整合失败：模型未返回有效内容")
            return None
        return body
    finally:
        if not keep_llm:
            stop_local_llm(proc)
