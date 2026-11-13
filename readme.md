# AI\_video\_summary

[中文](说明文档.md) | **English**

Turn website videos into readable structured notes: automatically collect subtitles, run speech recognition and on-screen subtitle OCR, aggregate the video's text content, then use an LLM to generate "Key Information / Content Summary / Key Data or Cases", and finally export an HTML report.

- The project provides both a local Web GUI and a CLI, sharing the same pipeline (`app/service.py`).
- Speech recognition: Fun-ASR-Nano
- Subtitle recognition: RapidOCR
- Local LLM: Qwen3 series models (GGUF quantized, served by llama.cpp)

## Code Structure

```text
ai_video_summary/
├─ main.py                # CLI entry point
├─ app/                   # GUI / FastAPI service
│  ├─ desktop.py          # Desktop shell (pywebview/browser)
│  ├─ server.py           # Local HTTP API + SSE events
│  ├─ service.py          # Main pipeline
│  ├─ jobs.py             # Serial single-task queue
│  ├─ settings.py         # LLM config persistence
│  ├─ installer.py        # Model component installation
│  ├─ platforms/          # Platform adapters (Bilibili only for now)
│  └─ web/                # Frontend (index.html / app.js / style.css)
├─ backend/               # Core library: site APIs, ASR, OCR, merging, LLM, export
├─ models/                # Local AI model assets
├─ tools/                 # ffmpeg, llama.cpp runtime
├─ out/                   # Default output directory
├─ .cache/                # Runtime cache (config, logs, temp files)
├─ requirements.txt       # Python dependencies
├─ start_gui_windows.bat  # Windows launcher
└─ start_gui_linux.sh     # Linux / macOS launcher
```

## Quick Start

### 1. Prerequisites

| Item        | Description                                                                                                                                                                                                             |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python      | Python >= 3.10 recommended; install the [PyTorch](https://pytorch.org/get-started/locally/) build for your platform beforehand                                                                                            |
| Python deps | `python -m pip install -r requirements.txt`                                                                                                                                                                             |
| ffmpeg      | `ffmpeg` and `ffprobe` are required; place them in `tools/ffmpeg` or on the system `PATH`. Quick install: Windows `winget install Gyan.FFmpeg`, macOS `brew install ffmpeg`, Linux `apt install ffmpeg`                  |
| GPU         | Local models require **at least 8 GB of VRAM**; a 16 GB+ NVIDIA GPU is recommended (newer AMD/Intel/Apple environments are also supported in theory)                                                                    |

### 2. Launch

On Windows, click `start_gui_windows.bat`; on Linux / macOS, run `start_gui_linux.sh`.

This starts a background cmd process and a Python frontend. Do not operate the cmd backend manually — it exits automatically when the frontend is closed.

### 3. Get a Bilibili Cookie (Optional)

Without a Cookie, Bilibili subtitles are unavailable and the pipeline can only rely on local ASR / OCR, which is much slower.

Click the `扫码刷新` button in the `本地状态` card at the bottom left and scan it on your phone to save the Cookie. Refresh the Cookie weekly to avoid expiry.

You can also manually copy the `SESSDATA` field of your Bilibili Cookie into `bili\_cookie.txt`.

### 4. Prepare the LLM

**Option A: Local model**

Complete two independent steps in the `本地状态` card at the bottom left (model and runtime are separate):

- Click `下载模型` next to `LLM`: the GGUF quantized file of the model recommended for your VRAM (see the table below) is downloaded to `models/llm` (tries `hf-mirror` first, falls back to `HuggingFace`).
  - You can also manually place a single GGUF model file under `models/llm`; the program will pick it up automatically.
- Click `安装` next to `llama.cpp`: downloads and installs the llama.cpp runtime for your platform into `tools/llama.cpp` (NVIDIA → CUDA, AMD/Intel → Vulkan, Apple → Metal)

**Option B: External API**

In `LLM 配置`, set `模型来源` to `public` and fill in your `base_url`, `api_key`, and `model`.

The endpoint must be compatible with OpenAI chat/completions. Platforms such as DeepSeek, Volcano Ark, Alibaba Cloud Bailian, and SiliconFlow are supported.

### 5. Run

Once the local components are ready, enter a video URL or BV ID in the top-left and click `开始`.

A single summary takes several minutes. Progress is shown in `运行日志` at the bottom right, and results appear in `查看结果` at the top right — click `在浏览器打开` to view them.

## LLM Technical Details

### Local mode (local)

- Uses llama.cpp's `llama-server`, always listening on `127.0.0.1:8080`
- Prefers reusing a service already running on port 8080; if none is detected, starts `llama-server` from `tools/llama.cpp` with a GGUF model in `models/llm`
- llama-server logs are saved to `.cache/llama-server.log`
- `LLM 常驻`: (enabled by default) llama-server stays alive after a task so subsequent tasks can reuse it; if unchecked, the local LLM exits when the task ends

VRAM and default model tiers (determined by rounding total VRAM up):

| Total VRAM | Model                   | File size |
| ---------- | ----------------------- | --------- |
| ≥ 20 GB    | Qwen3.8-27B Q4\_K\_M    | ~16.5 GB  |
| ≥ 16 GB    | Qwen3.8-27B Q3\_K\_XL   | ~13.1 GB  |
| ≥ 11 GB    | Qwen3-14B Q4\_K\_S      | ~8.6 GB   |
| ≥ 7 GB     | Qwen3.5-9B Q4\_K\_S     | ~5.4 GB   |
| ≤ 6 GB     | Local LLM not supported; use the public API instead | — |

Other local-mode parameters:

- Context length: 32768 when VRAM >= 20 GB, otherwise 25600
- KV cache set to `q8_0`
- `--sleep-idle-seconds 5`: frees VRAM when idle to prevent a later ASR run from running out of VRAM (requires sufficient system memory; otherwise loading will be slower)
- Local LLM calls enable `enable\_thinking` by default (reasoning\_effort=medium); if that fails, they retry twice with `enable\_thinking` disabled as a fallback.

### External API mode (public)

- Standard OpenAI-compatible protocol via the `openai` SDK

### Two-stage summarization

1. Split the timeline into chunks by `max\_chars` (default 8000, adjustable in `高级配置`);
2. Extract key points from each chunk;
3. Merge the chunk summaries into a final report containing Key Information, Content Summary, and Key Data or Cases.

Both prompts include the current date, video title, video description (first 1000 characters), and video tags; tags of 5 characters or fewer are automatically added to the ASR hotwords.

<br />

## Output

Output directory: `out/<bvid>_p<part>_<first 30 chars of title>/`

| File                           | Content                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `<stem>.transcript.cc.txt` etc. | One plain-text transcript per source, formatted as `[mm:ss] text`                                                                          |
| `<stem>.json`                  | `video` (video info), `sources` (source statistics), `cues` (text entries grouped by source)                                               |
| `<stem>.html`                  | HTML summary report: video info, key information, content summary, key data or cases, and source text (collapsed by default)               |

`<stem>` looks like `BV1xx432c1mD_p1_video title`.

Cache directory: `.cache/<bvid>_p<part>` is deleted after a successful task by default.

## FAQ

0. **For any execution issue, using an AI coding platform for automatic debugging is recommended**
1. ASR fails as soon as it starts: check whether `numpy >= 2.x` is causing dependency conflicts; try downgrading numpy to 1.x.
2. ASR/OCR cannot use the GPU: check whether torch is the CPU build; if so, reinstall the GPU build.
3. Local ASR/OCR is too slow: set `模式` to `手动（按勾选获取）` and check only `B站字幕` (this only works for videos that have subtitles).
4. System stutters while running: check whether other programs are using VRAM, such as a large 3D game.
5. Out of memory / OOM while running: restart the app and uncheck `LLM 常驻`.
