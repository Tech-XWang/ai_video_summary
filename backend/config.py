# -*- coding: utf-8 -*-
"""全局配置：目录、B 站接口参数、本地 LLM 服务参数、来源优先级。"""

import os

# 包目录与项目根目录
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)

CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
OUT_DIR = os.path.join(PROJECT_DIR, "out")
COOKIE_FILE = os.path.join(PROJECT_DIR, "bili_cookie.txt")

# 本地组件安装目录（模型与工具都在项目内，便于整体迁移）
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LLM_MODELS_DIR = os.path.join(MODELS_DIR, "llm")
ASR_MODELS_DIR = os.path.join(MODELS_DIR, "asr")
TOOLS_DIR = os.path.join(PROJECT_DIR, "tools")
FFMPEG_DIR = os.path.join(TOOLS_DIR, "ffmpeg")
LLAMA_RUNTIME_DIR = os.path.join(TOOLS_DIR, "llama.cpp")

# LLM 模型下载：优先国内镜像，失败回退官方站
HF_ENDPOINT = "https://hf-mirror.com"
HF_FALLBACK_ENDPOINT = "https://huggingface.co"

# ASR：Fun-ASR-Nano（ModelScope 官方仓库）
ASR_MODEL_REPO = "FunAudioLLM/Fun-ASR-Nano-2512"
ASR_VAD_REPO = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
ASR_MODEL_DIR = os.path.join(ASR_MODELS_DIR, "Fun-ASR-Nano-2512")
ASR_VAD_DIR = os.path.join(ASR_MODELS_DIR, "fsmn-vad")

# OCR 抽帧固定参数（GUI 不再暴露，保留给 CLI）
OCR_FPS = 1.0
OCR_DIFF = 2.0

# 本地 llama-server 运行时：项目内 tools/llama.cpp（目录可通过 junction 指向外部运行时）
LLAMA_EXE = os.path.join(LLAMA_RUNTIME_DIR, "llama-server.exe")
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_ALIAS = "qwen38-27b"
LLAMA_BASE = f"http://localhost:{LLAMA_PORT}/v1"
# LLM 空闲多少秒后休眠：卸载权重与 KV cache 释放显存，下次请求自动重载
LLAMA_SLEEP_IDLE = 5
# 思考强度：low / medium / high。medium 可显著缩短推理长度与耗时
LLM_REASONING_EFFORT = "medium"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://www.bilibili.com/"}
API = "https://api.bilibili.com"

# 字幕源优先级：数字越小越可信
SRC_RANK = {"cc": 0, "ai-sub": 2, "ocr": 1, "asr": 3}
SRC_DESC = {"cc": "人工字幕", "ai-sub": "AI字幕", "ocr": "OCR", "asr": "语音识别"}
SUB_KEYS = ("cc", "ocr", "ai-sub")
# 这些来源的 cue 不做相邻合并，每条独立进入时间线与导出
INDEPENDENT_SRC = ("cc", "ai-sub")
