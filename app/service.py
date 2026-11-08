# -*- coding: utf-8 -*-
"""流水线：解析 → 并发下载资源（字幕/音频/视频）→ 音频 ASR → 画面 OCR → 汇合 → LLM → 导出。

CLI 与 GUI 都调用 ``run_job``，进度与日志通过 bilinote.utils 的 hook 外发。
"""

import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

from bilinote import asr as asr_mod, export, llm, media, merge, ocr
from bilinote import local_llm
from bilinote.config import CACHE_DIR, OCR_DIFF, OCR_FPS, OUT_DIR
from bilinote.models import Cue
from bilinote.utils import coverage, fmt_ts, log, report, sanitize

from .platforms import get_platform


class NoContentError(RuntimeError):
    """没有任何可用文本来源。"""


class MissingComponentError(RuntimeError):
    """缺少本地组件，必须终止执行。"""


@dataclass
class JobParams:
    """一次转写任务的完整参数；GUI 与 CLI 共用。"""

    url: str
    platform: str = "bilibili"          # 预留：后续接入抖音/快手
    mode: str = "auto"                  # auto | manual
    part: int = 0
    use_subtitle: bool = True           # 手动模式下是否获取 B 站字幕
    use_asr: bool = True                # 手动模式下是否跑本地 ASR
    use_ocr: bool = True                # 手动模式下是否跑本地 OCR
    ocr_fps: float = OCR_FPS
    ocr_diff: float = OCR_DIFF
    ocr_region: str = "bottom"
    asr_device: str = "auto"
    hotword: str = ""
    no_llm: bool = False
    no_ocr: bool = False
    no_asr: bool = False
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    llm_source: str = "local"           # local=本地 llama-server；public=OpenAI 兼容接口
    max_chars: int = 8000
    out_dir: str = OUT_DIR
    keep_temp: bool = False
    keep_llm: bool = False
    cookie: Optional[str] = None
    cookie_file: Optional[str] = None

    @classmethod
    def from_namespace(cls, args) -> "JobParams":
        """由 argparse 命名空间构造（CLI 用）；旧的 --mode 值映射成手动勾选。"""
        old = getattr(args, "mode", "auto")
        if old in ("subtitle", "asr", "ocr", "all"):
            mode = "manual"
            use_subtitle = old in ("subtitle", "all")
            use_asr = old in ("asr", "all") and not args.no_asr
            use_ocr = old in ("ocr", "all") and not args.no_ocr
        else:
            mode = "auto"
            use_subtitle, use_asr, use_ocr = True, True, True
        return cls(
            url=args.url,
            mode=mode,
            part=args.part,
            use_subtitle=use_subtitle,
            use_asr=use_asr,
            use_ocr=use_ocr,
            ocr_fps=args.ocr_fps,
            ocr_diff=args.ocr_diff,
            ocr_region=args.ocr_region,
            asr_device=args.asr_device,
            hotword=args.hotword,
            no_llm=args.no_llm,
            no_ocr=args.no_ocr,
            no_asr=args.no_asr,
            model=args.model,
            base_url=args.base_url,
            api_key=getattr(args, "api_key", ""),
            llm_source="public" if getattr(args, "api_key", "") else "local",
            max_chars=args.max_chars,
            out_dir=args.out,
            keep_temp=args.keep_temp,
            cookie=args.cookie,
            cookie_file=args.cookie_file,
        )


def _wants(p: JobParams) -> tuple:
    """各路是否启用。auto 由优先级与 --no-* 决定；manual 完全按勾选。"""
    if p.mode == "manual":
        return p.use_subtitle, p.use_asr, p.use_ocr
    return True, not p.no_asr, not p.no_ocr


def _merge_hotwords(user: str, tags: List[str]) -> str:
    """用户热词 + 不超过 5 字的视频标签，去重后以空格拼接。"""
    words = re.split(r"[\s,，;；、]+", user or "")
    words += [t for t in tags if len(t) <= 5]
    seen = set()
    out = []
    for w in words:
        w = w.strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out)


def run_job(p: JobParams) -> dict:
    """执行一次完整任务，返回 {files, out_dir, info, timeline, coverage}。"""
    t_all = time.time()
    platform = get_platform(p.platform)

    # LLM 缺失一律终止；手动模式勾选的本地依赖缺失也终止；自动模式非 LLM 缺失只跳过
    if not p.no_llm:
        if p.llm_source == "public":
            missing = [name for name, value in (
                ("base_url", p.base_url), ("api_key", p.api_key), ("model", p.model),
            ) if not str(value).strip()]
            if missing:
                raise MissingComponentError(
                    "public 模式 LLM 配置不完整，缺少：" + "、".join(missing)
                    + "。请在左侧「LLM 配置」中填写后重试。"
                )
        elif not p.base_url:
            llm_state = local_llm.status()
            if not llm_state.get("ok"):
                raise MissingComponentError(
                    "本地 LLM 不可用：" + llm_state.get("label", "未就绪")
                    + "。请到“高级配置 → 本地状态”下载模型/运行时后再执行。"
                )

    cookie = platform.load_cookie(p.cookie, p.cookie_file)
    want_sub, want_asr, want_ocr = _wants(p)
    ffmpeg_ok = media.ffmpeg_status().get("ok", False)
    if p.mode == "manual":
        missing = []
        if want_sub and not cookie:
            missing.append("B站字幕（未配置 Cookie）")
        if want_asr and not asr_mod.check_installed():
            missing.append("本地 ASR（未安装）")
        if want_ocr and not ocr.check_installed():
            missing.append("本地 OCR（未安装）")
        if (want_asr or want_ocr) and not ffmpeg_ok:
            missing.append("ffmpeg（未安装）")
        if missing:
            raise MissingComponentError(
                "手动模式缺少依赖，已终止执行：" + "；".join(missing)
                + "。请先安装，或取消对应勾选。"
            )
    else:
        if not cookie:
            log("提示：未配置 Cookie，自动模式将跳过 B 站字幕，改用 ASR/OCR。")
            platform.print_cookie_help()
        if want_asr and not asr_mod.check_installed():
            log("  本地 ASR 未安装，自动模式跳过 ASR（可在高级配置安装）")
            want_asr = False
        if want_ocr and not ocr.check_installed():
            log("  本地 OCR 未安装，自动模式跳过 OCR（可在高级配置安装）")
            want_ocr = False
        if (want_asr or want_ocr) and not ffmpeg_ok:
            log("  本地 ffmpeg 缺失，自动模式跳过 ASR/OCR（可在高级配置安装）")
            want_asr = False
            want_ocr = False

    # 1. 解析视频
    report("resolve", 0.0, "解析视频")
    log("\n[1/6] 解析视频……")
    vid, part_in_url = platform.resolve(p.url)
    info = platform.video_info(vid)
    bvid = info.get("bvid") or vid
    part = p.part or part_in_url
    page = platform.pick_page(info, part)
    duration = float(page.get("duration") or info.get("duration") or 0)
    video_url = platform.build_url(info, bvid, page)
    log(f"  {info.get('title')}")
    log(f"  UP：{(info.get('owner') or {}).get('name', '')} ｜ 时长 {fmt_ts(duration)} ｜ cid {page['cid']}")

    # 简介与标签：简介注入 prompt（截前 1000 字），短标签并入 ASR 热词
    desc = str(info.get("desc") or "").strip()
    if desc == "-":   # B 站空简介返回 "-"
        desc = ""
    tags = platform.video_tags(bvid, cookie)
    hotwords = _merge_hotwords(p.hotword, tags)
    if desc or tags:
        log(f"  简介 {len(desc)} 字 ｜ 标签 {len(tags)} 个")
    report("resolve", 1.0, info.get("title", ""))

    work = os.path.join(CACHE_DIR, f"{bvid}_p{page['page']}")
    os.makedirs(work, exist_ok=True)

    # 2. 下载资源：字幕 / 音频 / 视频并发
    task_names = []
    if want_sub:
        task_names.append("字幕")
    if want_asr:
        task_names.append("音频")
    if want_ocr:
        task_names.append("视频")
    log("\n[2/6] 下载视频资源……")
    if task_names:
        log("  并发获取：" + " / ".join(task_names))
        report("download", -1.0, "并发获取：" + " / ".join(task_names))

    subs: List[Cue] = []
    sub_src = ""
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    if task_names:
        with ThreadPoolExecutor(max_workers=len(task_names),
                                thread_name_prefix="resources") as pool:
            f_sub = (pool.submit(platform.fetch_subtitles, bvid, page["cid"], cookie, duration)
                     if want_sub else None)
            f_audio = (pool.submit(platform.download_audio, bvid, page["cid"], cookie, work)
                       if want_asr else None)
            f_video = (pool.submit(platform.download_video, bvid, page["cid"], cookie, work)
                       if want_ocr else None)

            if f_sub:
                try:
                    subs, sub_src = f_sub.result()
                except Exception as e:  # noqa: BLE001  单路失败不影响其它资源
                    log(f"  字幕获取失败：{e}")
            if f_audio:
                try:
                    audio_path = f_audio.result()
                except Exception as e:  # noqa: BLE001
                    log(f"  音频下载失败：{e}")
            if f_video:
                try:
                    video_path = f_video.result()
                except Exception as e:  # noqa: BLE001
                    log(f"  视频流下载失败：{e}")

    if want_sub:
        if not subs:
            if p.mode == "auto":
                log("  未获取到 B 站字幕，自动模式改用本地 ASR（若可用）")
            else:
                log("  未获取到 B 站字幕")
        elif p.mode == "auto" and sub_src in ("cc", "ai-sub"):
            if sub_src == "cc":
                log("  已获取人工字幕，自动跳过 B站 AI 字幕与本地 ASR")
            else:
                log("  已获取 B站 AI 字幕，自动跳过本地 ASR")
            want_asr = False
    else:
        log("  跳过 B 站字幕（手动模式未勾选）")

    parts = []
    if want_sub:
        parts.append(f"字幕 {len(subs)} 条")
    if want_asr:
        parts.append("音频已就绪" if audio_path else "音频未就绪")
    if want_ocr:
        parts.append("视频流已就绪" if video_path else "视频流未就绪")
    if parts:
        log("  " + " ｜ ".join(parts))

    sub_cov = coverage(subs, duration)
    if want_sub:
        report("subtitle", 1.0, f"{len(subs)} 条")

    # 3. 音频 ASR
    log("\n[3/6] 音频ASR……")
    asr_cues: List[Cue] = []
    if want_asr:
        if audio_path:
            report("asr", 0.0, "音频转写")
            wav = media.to_wav(audio_path, work)
            if wav:
                asr_cues = asr_mod.run_asr(wav, p.asr_device, hotwords, duration)
                log(f"  ASR：{len(asr_cues)} 句，覆盖 {coverage(asr_cues, duration):.0%}")
            else:
                log("  音频转码失败，跳过 ASR")
                report("asr", 1.0, "已跳过")
        else:
            log("  音频未就绪，跳过 ASR")
            report("asr", 1.0, "已跳过")
    else:
        if p.mode == "auto" and subs:
            log("  跳过 ASR（已使用 B 站字幕；音频已下载缓存）"
                if audio_path else "  跳过 ASR（已使用 B 站字幕）")
        elif p.mode == "auto":
            log("  跳过 ASR（未启用或组件缺失）")
        else:
            log("  跳过 ASR（手动模式未勾选）")
        report("asr", 1.0, "已跳过")

    # 4. 画面 OCR
    log("\n[4/6] 画面硬字幕 OCR……")
    ocr_cues: List[Cue] = []
    if want_ocr:
        if video_path:
            report("ocr", 0.0, "画面 OCR")
            ocr_cues = ocr.run_ocr(video_path, work, p.ocr_fps, p.ocr_region, p.ocr_diff)
            log(f"  OCR：{len(ocr_cues)} 条，覆盖 {coverage(ocr_cues, duration):.0%}")
        else:
            log("  视频流未就绪，跳过 OCR")
            report("ocr", 1.0, "已跳过")
    else:
        if p.mode == "manual":
            log("  跳过 OCR（手动模式未勾选）")
        else:
            log("  跳过 OCR（--no-ocr 或组件缺失）")
        report("ocr", 1.0, "已跳过")

    # 5. 融合
    log("\n[5/6] 汇总数据源")
    report("merge", 0.0, "汇合时间线")
    groups = {"cc": [], "ai-sub": [], "ocr": ocr_cues, "asr": asr_cues}
    for c in subs:
        groups.setdefault(c.src, []).append(c)
    timeline, base = merge.merge_sources(groups)
    if not timeline:
        log("没有任何可用文本来源，退出。")
        if not cookie:
            log("提示：配置 Cookie 后可获取 B 站字幕，通常是最快最准的来源。")
        raise NoContentError("没有任何可用文本来源")

    total_chars = sum(len(c.text) for c in timeline)
    counts = {k: len(v) for k, v in groups.items()}
    meta = {
        "counts": counts,
        "base": base,
        "coverage": coverage(timeline, duration),
        "chars": total_chars,
        "subtitle_coverage": sub_cov,
    }
    log(f"  时间线 {len(timeline)} 条，{total_chars} 字，覆盖 {meta['coverage']:.0%}")
    report("merge", 1.0, f"{len(timeline)} 条")

    # 6. LLM
    summary = None
    if not p.no_llm:
        log("\n[6/6] LLM 提炼……")
        summary = llm.llm_summary(
            groups, timeline, info.get("title", ""), video_url, duration,
            p.base_url, p.model, p.max_chars, keep_llm=p.keep_llm,
            desc=desc, tags=tags, llm_source=p.llm_source, api_key=p.api_key,
        )
    else:
        log("\n[6/6] 跳过 LLM")
        report("llm", 1.0, "已跳过")

    # 导出
    report("export", 0.0, "生成报告")
    title = info.get("title", "")
    pno = page["page"]
    # 多分P视频共用同一个标题，文件名/目录名带上 P 号避免互相覆盖
    stem = f"{bvid}_p{pno}_{sanitize(title)}"
    out_dir = os.path.join(p.out_dir, f"{bvid}_p{pno}_{sanitize(title, limit=30)}")
    files = export.write_outputs(
        out_dir, stem,
        {
            "title": info.get("title", ""),
            "url": video_url,
            "owner": (info.get("owner") or {}).get("name", ""),
            "duration": duration,
            "part": page.get("part") or page["page"],
            "bvid": bvid,
            "cid": page["cid"],
        },
        timeline, meta, summary,
    )

    if not p.keep_temp:
        shutil.rmtree(work, ignore_errors=True)

    log("\n任务完成，用时 {:.1f}s".format(time.time() - t_all))
    for f in files:
        if f.lower().endswith(".html"):
            log(f"  {f}")
    report("export", 1.0, "导出完成")
    report("done", 1.0, "完成")

    return {
        "files": files,
        "out_dir": out_dir,
        "info": {"title": info.get("title", ""), "url": video_url, "bvid": bvid, "duration": duration},
        "timeline": len(timeline),
        "coverage": meta["coverage"],
        "elapsed": time.time() - t_all,
    }
