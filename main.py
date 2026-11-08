#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B 站视频文字提炼 —— 命令行入口。

流水线本身在 app/service.py，GUI 与 CLI 共用：
  解析视频 → 并发下载资源（字幕/音频/视频）→ 音频 ASR → 画面 OCR → 汇合 → LLM 总结 → 导出

示例：
  python main.py --login                       # 扫码登录，保存 cookie
  python main.py BV1xx411c7mD                  # 自动选源并总结
  python main.py https://b23.tv/xxxx --no-llm  # 只导出转写稿
  python main.py BV1xx411c7mD --mode all       # 三路全跑后融合
"""

import argparse
import os
import sys

from app.service import JobParams, NoContentError, run_job
from bilinote import bili_api
from bilinote.config import OUT_DIR
from bilinote.utils import log

# 调试开关：由 --debug 参数打开，异常时输出完整堆栈
DEBUG = False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B 站视频字幕/OCR/ASR 融合转写 + LLM 结构化总结",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--debug", action="store_true", help="异常时输出完整堆栈，便于排查")
    ap.add_argument("url", nargs="?", help="B 站视频链接、BV 号或 av 号")
    ap.add_argument("--login", action="store_true", help="扫码登录并保存 Cookie")
    ap.add_argument("--cookie", help="直接指定 Cookie 字符串（如 SESSDATA=xxx）")
    ap.add_argument("--cookie-file", help="Cookie 文件路径（默认 bili_cookie.txt）")
    ap.add_argument("--mode", default="auto", choices=["auto", "subtitle", "asr", "ocr", "all"],
                    help="auto=人工字幕 > AI字幕 > 本地ASR，OCR 始终执行；"
                         "subtitle/asr/ocr/all=手动模式，按指定来源获取（默认 auto）")
    ap.add_argument("--part", type=int, default=0, help="分P 序号，默认取链接里的 p 或第 1P")
    ap.add_argument("--ocr-fps", type=float, default=1.0, help="OCR 抽帧帧率（默认 1）")
    ap.add_argument("--ocr-diff", type=float, default=2.0,
                    help="字幕带帧间差异阈值，小于该值跳过 OCR（默认 2.0）")
    ap.add_argument("--ocr-region", default="bottom", choices=["bottom", "center", "full"],
                    help="OCR 区域，默认底部字幕带")
    ap.add_argument("--asr-device", default="auto", choices=["auto", "cpu"],
                    help="ASR 设备：auto=自动GPU（失败重试后回退 CPU），cpu=只用 CPU")
    ap.add_argument("--hotword", default="", help="ASR 热词，用空格分隔，可提升人名/术语准确率")
    ap.add_argument("--no-llm", action="store_true", help="只导出转写，不调用 LLM")
    ap.add_argument("--no-ocr", action="store_true", help="关闭画面 OCR")
    ap.add_argument("--no-asr", action="store_true", help="关闭音频 ASR")
    ap.add_argument("--model", default="", help="LLM 模型名，默认自动探测")
    ap.add_argument("--base-url", default="",
                    help="OpenAI 兼容接口地址，默认 http://localhost:8080/v1")
    ap.add_argument("--api-key", default="",
                    help="外部 OpenAI 兼容接口的 API Key；填写后走 public 模式，不回退本地")
    ap.add_argument("--max-chars", type=int, default=8000, help="LLM 分段最大字符数")
    ap.add_argument("--out", default=OUT_DIR, help="输出目录")
    ap.add_argument("--keep-temp", action="store_true", help="保留缓存（音频/视频/抽帧）")
    return ap


def run(args: argparse.Namespace) -> int:
    try:
        run_job(JobParams.from_namespace(args))
    except NoContentError:
        return 2
    return 0


def main() -> int:
    global DEBUG
    ap = build_parser()
    args = ap.parse_args()
    DEBUG = args.debug

    if args.login:
        bili_api.login_flow()
        bili_api.print_manual_cookie_help()
        return 0
    if not args.url:
        ap.print_help()
        bili_api.print_manual_cookie_help()
        return 1
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n已中断")
        sys.exit(130)
    except Exception as e:
        # 默认只给简短信息；加 --debug 可输出完整堆栈便于排查
        if DEBUG:
            import traceback

            traceback.print_exc()
        log(f"\n出错：{e}")
        sys.exit(1)
