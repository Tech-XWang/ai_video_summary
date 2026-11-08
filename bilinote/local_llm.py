# -*- coding: utf-8 -*-
"""本地 LLM 管理：显存档位、GGUF 模型下载、llama.cpp 运行时与状态检测。"""

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from typing import Callable, Optional, Tuple

import requests

from .config import (
    HF_ENDPOINT,
    HF_FALLBACK_ENDPOINT,
    LLAMA_ALIAS,
    LLAMA_EXE,
    LLAMA_RUNTIME_DIR,
    LLM_MODELS_DIR,
    TOOLS_DIR,
    UA,
)

Progress = Callable[[float, str], None]

# 按总显存选型：>=20G Q4、>=16G Q3、>10G 14B Q4、>6G 9B Q4、<=6G 不支持
MODEL_TIERS = [
    {
        "min_vram": 20.0,
        "label": "Qwen3.8-27B Q4",
        "repo": "nsloth/Qwen3.8-27B-GGUF",
        "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "size_gb": 16.5,
    },
    {
        "min_vram": 16.0,
        "label": "Qwen3.8-27B Q3",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "file": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "size_gb": 13.1,
    },
    {
        "min_vram": 10.0,
        "label": "Qwen3-14B Q4",
        "repo": "unsloth/Qwen3-14B-GGUF",
        "file": "Qwen3-14B-Q4_K_S.gguf",
        "size_gb": 8.6,
    },
    {
        "min_vram": 6.0,
        "label": "Qwen3.5-9B Q4",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q4_K_S.gguf",
        "size_gb": 5.4,
    },
]

MANIFEST = os.path.join(LLM_MODELS_DIR, "manifest.json")
_GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"
_GPU_CACHE = {"ts": 0.0, "total": 0.0, "name": ""}


def _http_headers() -> dict:
    return {"User-Agent": UA}


def gpu_info() -> Tuple[float, str]:
    """返回 (多卡总显存 GB, 第一张卡名称)；取不到时为 (0, "")。"""
    now = time.time()
    if _GPU_CACHE["ts"] and now - _GPU_CACHE["ts"] < 30:
        return float(_GPU_CACHE["total"]), str(_GPU_CACHE["name"])
    total, name = 0.0, ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            for row in out.splitlines():
                parts = [p.strip() for p in row.split(",")]
                if len(parts) < 2:
                    continue
                name = name or parts[0]
                total += float(parts[-1]) / 1024.0
        except Exception:
            pass
    _GPU_CACHE.update(ts=now, total=total, name=name)
    return total, name


def recommend(vram_gb: float) -> Optional[dict]:
    """按显存返回推荐模型；显存 ≤6G 返回 None。

    nvidia-smi 的 MiB 换算成 GiB 后 16G 卡通常显示 15.9G，这里向上取整，
    避免把 16G 卡误判到 14B 档。
    """
    vram = float(math.ceil(vram_gb))
    if vram >= 20.0:
        return MODEL_TIERS[0]
    if vram >= 16.0:
        return MODEL_TIERS[1]
    if vram > 10.0:
        return MODEL_TIERS[2]
    if vram > 6.0:
        return MODEL_TIERS[3]
    return None


def context_size() -> int:
    vram, _ = gpu_info()
    return 32768 if vram >= 20 else 25600


def _is_gguf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _read_manifest() -> dict:
    try:
        with open(MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_manifest(data: dict) -> None:
    os.makedirs(LLM_MODELS_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def installed_model() -> Optional[dict]:
    """已下载的模型；优先清单，其次扫描 models/llm 下的 .gguf（含链接文件）。"""
    man = _read_manifest()
    path = os.path.join(LLM_MODELS_DIR, man["file"]) if man.get("file") else ""
    if path:
        try:
            if (os.path.isfile(path) and os.path.getsize(path) > (1 << 20)
                    and _is_gguf(path)):
                return {**man, "path": path, "size": os.path.getsize(path)}
        except OSError:
            pass

    if os.path.isdir(LLM_MODELS_DIR):
        files = [
            os.path.join(LLM_MODELS_DIR, name)
            for name in os.listdir(LLM_MODELS_DIR)
            if name.lower().endswith(".gguf")
        ]
        valid = []
        for p in files:
            try:
                if os.path.getsize(p) > (1 << 20) and _is_gguf(p):
                    valid.append(p)
            except OSError:
                continue   # 链接目标被删除时跳过，避免状态接口报错
        if valid:
            path = max(valid, key=os.path.getsize)
            return {
                "label": os.path.splitext(os.path.basename(path))[0],
                "path": path,
                "size": os.path.getsize(path),
            }
    return None


def _runtime_exe() -> Optional[str]:
    if os.path.isdir(LLAMA_RUNTIME_DIR):
        for root, _, files in os.walk(LLAMA_RUNTIME_DIR):
            if "llama-server.exe" in files:
                return os.path.join(root, "llama-server.exe")
            if "llama-server" in files:
                return os.path.join(root, "llama-server")
    return None


def llama_server_exe() -> Optional[str]:
    if os.path.isfile(LLAMA_EXE):
        return LLAMA_EXE
    exe = _runtime_exe()
    if exe:
        return exe
    return shutil.which("llama-server")


def status() -> dict:
    """本地 LLM 的完整状态，供 GUI 展示。"""
    vram, gpu = gpu_info()
    rec = recommend(vram)
    model = installed_model()
    runtime = llama_server_exe()

    if rec is None:
        return {
            "ok": False,
            "state": "unsupported",
            "label": f"不支持本地 LLM（显存 {vram:.1f}G ≤ 6G）",
            "vram_gb": round(vram, 1),
            "gpu": gpu,
            "recommended": None,
            "model": model,
            "runtime": runtime,
        }
    if not runtime:
        if model:
            label = f"缺 llama.cpp 运行时（模型：{model.get('label', '')}）"
        else:
            label = f"缺 llama.cpp 运行时，模型未下载（推荐 {rec['label']}，约 {rec['size_gb']}GB）"
        return {
            "ok": False,
            "state": "no_runtime",
            "label": label,
            "vram_gb": round(vram, 1),
            "gpu": gpu,
            "recommended": rec,
            "model": model,
            "runtime": None,
        }
    if not model:
        return {
            "ok": False,
            "state": "missing",
            "label": f"未下载（推荐 {rec['label']}，约 {rec['size_gb']}GB）",
            "vram_gb": round(vram, 1),
            "gpu": gpu,
            "recommended": rec,
            "model": None,
            "runtime": runtime,
        }
    return {
        "ok": True,
        "state": "ready",
        "label": f"已就绪（{model.get('label', '')}）",
        "vram_gb": round(vram, 1),
        "gpu": gpu,
        "recommended": rec,
        "model": model,
        "runtime": runtime,
    }


def resolve_model_path() -> Optional[str]:
    model = installed_model()
    return model["path"] if model else None


def model_alias() -> str:
    """llama-server 的模型别名；旧模型沿用现有别名。"""
    model = installed_model()
    if not model or model.get("legacy"):
        return LLAMA_ALIAS
    label = str(model.get("label") or "")
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or LLAMA_ALIAS


def _format_bytes(n: float) -> str:
    gb = n / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f}GB"
    return f"{n / (1024 ** 2):.0f}MB"


def download_file(url: str, dest: str, progress: Optional[Progress] = None) -> None:
    """带断点续传的下载；中断后保留 .part，下次从已有字节继续。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    pos = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = _http_headers()
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
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


def _model_urls(spec: dict) -> list:
    rel = f"{spec['repo']}/resolve/main/{spec['file']}?download=true"
    return [f"{base}/{rel}" for base in (HF_ENDPOINT, HF_FALLBACK_ENDPOINT)]


def download_model(spec: dict, progress: Optional[Progress] = None) -> str:
    """下载并校验一个 GGUF；已存在时直接返回。"""
    dest = os.path.join(LLM_MODELS_DIR, spec["file"])
    if os.path.isfile(dest) and os.path.getsize(dest) > (1 << 20) and _is_gguf(dest):
        if progress is not None:
            progress(1.0, "模型已存在")
        return dest

    os.makedirs(LLM_MODELS_DIR, exist_ok=True)
    free_gb = shutil.disk_usage(LLM_MODELS_DIR).free / (1024 ** 3)
    if spec.get("size_gb") and free_gb < float(spec["size_gb"]) * 1.1:
        raise RuntimeError(
            f"磁盘空间不足：需要约 {spec['size_gb'] * 1.1:.0f}GB，当前剩余 {free_gb:.0f}GB"
        )

    last_err: Optional[Exception] = None
    for url in _model_urls(spec):
        try:
            download_file(url, dest, progress)
            break
        except Exception as e:  # noqa: BLE001  镜像失败要回退官方站
            last_err = e
    else:
        raise RuntimeError(f"模型下载失败：{last_err}")
    if not _is_gguf(dest):
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError("下载的文件不是有效的 GGUF 模型")

    _write_manifest({
        "label": spec["label"],
        "repo": spec["repo"],
        "file": spec["file"],
        "size_gb": spec["size_gb"],
        "vram_min_gb": spec["min_vram"],
        "downloaded_at": time.time(),
    })
    if progress is not None:
        progress(1.0, f"已下载 {spec['label']}")
    return dest


def _has_nvidia() -> bool:
    return bool(shutil.which("nvidia-smi"))


def _runtime_target() -> Tuple[str, str]:
    """返回 llama.cpp 发布资产的平台/架构标签。"""
    if os.name == "nt":
        plat = "win"
    elif sys.platform == "darwin":
        plat = "macos"
    else:
        plat = "ubuntu"   # Linux 发布包使用 ubuntu-* 命名
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return plat, arch


def _archive_ok(name: str) -> bool:
    return name.endswith(".zip") or name.endswith(".tar.gz")


def _arch_match(name: str, arch: str) -> bool:
    if arch == "arm64":
        return "arm64" in name
    return "x64" in name or "amd64" in name


def _asset_score(name: str, plat: str, arch: str, nvidia: bool) -> int:
    n = name.lower()
    if "cudart" in n or "llama-b" not in n:
        return -1
    if plat == "win":
        if not n.endswith(".zip") or "win" not in n:
            return -1
    elif plat == "macos":
        if "macos" not in n or not _archive_ok(n):
            return -1
    else:
        if not _archive_ok(n) or ("ubuntu" not in n and "linux" not in n):
            return -1
        if any(k in n for k in ("android", "snapdragon", "s390x")):
            return -1
    if not _arch_match(n, arch):
        return -1
    if plat == "macos":
        return 5
    if "cuda" in n:
        if not nvidia:
            return -1
        ver = re.search(r"cuda-(\d+\.\d+)", n)
        # CUDA 12.x 兼容面更广，优先于 13.x
        return 6 if ver and ver.group(1).startswith("12") else 5
    if "vulkan" in n:
        return 4
    if any(k in n for k in ("rocm", "openvino", "sycl", "opencl", "zdnn")):
        return -1
    return 3


def _pick_cudart(assets: list, main_name: str, plat: str, arch: str) -> tuple:
    if "cuda" not in main_name.lower():
        return None
    ver = re.search(r"cuda-(\d+\.\d+)", main_name.lower())
    for a in assets:
        n = str(a.get("name", "")).lower()
        if "cudart" not in n or not _archive_ok(n):
            continue
        if plat == "win" and "win" not in n:
            continue
        if plat == "ubuntu" and "ubuntu" not in n:
            continue
        if plat == "macos":
            continue
        if not _arch_match(n, arch):
            continue
        if ver and f"cuda-{ver.group(1)}" not in n:
            continue
        return a
    return None


def _pick_runtime_assets(assets: list, plat: str, arch: str, nvidia: bool) -> tuple:
    main = max(assets, key=lambda a: _asset_score(str(a.get("name", "")), plat, arch, nvidia),
               default=None)
    if main is None or _asset_score(str(main.get("name", "")), plat, arch, nvidia) < 3:
        return None, None
    return main, _pick_cudart(assets, str(main.get("name", "")), plat, arch)


def _extract_archive(archive: str, dest: str) -> None:
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)


def _merge_tree(src_dir: str, dst_dir: str) -> None:
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src) and os.path.isdir(dst):
            _merge_tree(src, dst)
        else:
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.move(src, dst)


def _is_shared_lib(name: str) -> bool:
    n = name.lower()
    if os.name == "nt":
        return n.endswith(".dll")
    return n.endswith(".dylib") or ".so" in n


def download_runtime(progress: Optional[Progress] = None) -> str:
    """下载当前平台可用的 llama.cpp 运行时（CUDA/Vulkan/CPU，macOS 为 Metal 通用包）。"""
    plat, arch = _runtime_target()
    nvidia = _has_nvidia()

    try:
        r = requests.get(
            _GITHUB_API,
            headers={**_http_headers(), "Accept": "application/vnd.github+json"},
            timeout=20,
        )
        r.raise_for_status()
        releases = r.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"获取 llama.cpp 发布信息失败：{e}")

    main = cudart = None
    for rel in releases if isinstance(releases, list) else []:
        main, cudart = _pick_runtime_assets(rel.get("assets") or [], plat, arch, nvidia)
        if main is not None:
            break
    if main is None:
        raise RuntimeError(f"没有找到适用于 {plat}/{arch} 的 llama.cpp 二进制，请手动安装 llama-server")

    os.makedirs(TOOLS_DIR, exist_ok=True)
    tmp_pkg = os.path.join(TOOLS_DIR, "llama-runtime.pkg")
    cudart_pkg = os.path.join(TOOLS_DIR, "llama-cudart.pkg")
    tmp_dir = os.path.join(TOOLS_DIR, "llama-runtime-extract")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    def main_progress(pct: float, text: str) -> None:
        if progress is not None:
            progress(pct * 0.7 if pct >= 0 else -1.0, text)

    try:
        if progress is not None:
            progress(0.0, "下载 llama.cpp 运行时…")
        download_file(str(main["browser_download_url"]), tmp_pkg, main_progress)
        _extract_archive(tmp_pkg, tmp_dir)

        if cudart is not None:
            if progress is not None:
                progress(0.75, "下载 CUDA 运行库…")
            download_file(
                str(cudart["browser_download_url"]),
                cudart_pkg,
                lambda pct, text: progress(0.7 + pct * 0.2, text) if progress and pct >= 0 else None,
            )
            _extract_archive(cudart_pkg, tmp_dir)

        if progress is not None:
            progress(0.92, "解压运行时…")
        os.makedirs(LLAMA_RUNTIME_DIR, exist_ok=True)
        _merge_tree(tmp_dir, LLAMA_RUNTIME_DIR)

        exe = _runtime_exe()
        if not exe:
            raise RuntimeError("llama.cpp 压缩包中未找到 llama-server")
        if os.name != "nt":
            os.chmod(exe, 0o755)
        # CUDA 运行库可能被解压到另一个顶层目录，统一放到 exe 同目录保证可加载
        exe_dir = os.path.dirname(exe)
        for root, _, files in os.walk(LLAMA_RUNTIME_DIR):
            for name in files:
                if not _is_shared_lib(name):
                    continue
                src = os.path.join(root, name)
                dst = os.path.join(exe_dir, name)
                if os.path.abspath(src) != os.path.abspath(dst) and not os.path.isfile(dst):
                    shutil.copy2(src, dst)
        if progress is not None:
            progress(1.0, "llama.cpp 运行时就绪")
        return exe
    finally:
        for path in (tmp_pkg, cudart_pkg, tmp_dir):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def ensure_runtime(progress: Optional[Progress] = None) -> str:
    exe = llama_server_exe()
    if exe:
        if progress is not None:
            progress(1.0, "llama.cpp 运行时已存在")
        return exe
    return download_runtime(progress)


def download_recommended(progress: Optional[Progress] = None) -> str:
    vram, _ = gpu_info()
    spec = recommend(vram)
    if spec is None:
        raise RuntimeError(f"显存 {vram:.1f}GB ≤ 6GB，不支持本地 LLM 总结")
    return download_model(spec, progress)


def model_storage() -> str:
    """模型目录，供界面提示。"""
    return LLM_MODELS_DIR


__all__ = [
    "MODEL_TIERS",
    "context_size",
    "download_model",
    "download_recommended",
    "download_runtime",
    "ensure_runtime",
    "gpu_info",
    "installed_model",
    "llama_server_exe",
    "model_alias",
    "model_storage",
    "recommend",
    "resolve_model_path",
    "status",
]
