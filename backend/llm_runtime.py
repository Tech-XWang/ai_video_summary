# -*- coding: utf-8 -*-
"""本地 LLM 运行时：llama.cpp 下载/解压、状态汇总与公共入口。"""

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from typing import Callable, Optional, Tuple

import requests

from .config import LLAMA_EXE, LLAMA_RUNTIME_DIR, TOOLS_DIR
from .download import _download_asset, _http_headers, download_file
from .hardware import context_size, gpu_backend, gpu_info
from .model_store import (
    MODEL_TIERS,
    download_model,
    download_recommended,
    installed_model,
    model_alias,
    model_storage,
    recommend,
    resolve_model_path,
)

Progress = Callable[[float, str], None]

_GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"

_BACKEND_LABELS = {
    "cuda": "CUDA",
    "vulkan": "Vulkan",
    "metal": "Metal",
    "cpu": "CPU",
}


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


def runtime_status() -> dict:
    """llama.cpp 运行时的独立状态，与模型下载解耦。"""
    exe = llama_server_exe()
    backend = str(gpu_backend().get("backend") or "cpu")
    return {
        "ok": bool(exe),
        "state": "ready" if exe else "missing",
        "label": f"已安装（{_BACKEND_LABELS.get(backend, backend.upper())}）" if exe else "未安装",
        "path": exe,
        "backend": backend,
    }


def status() -> dict:
    """本地 LLM 的完整状态（模型 + 运行时），供 GUI 展示。"""
    info = gpu_backend()
    vram, gpu = float(info["vram_gb"]), str(info["name"])
    mem = "统一内存" if info["shared_mem"] else "显存"
    rec = recommend(vram)
    model = installed_model()
    runtime = llama_server_exe()
    model_ok, runtime_ok = bool(model), bool(runtime)
    base = {
        "vram_gb": round(vram, 1),
        "gpu": gpu,
        "backend": info["backend"],
        "vendor": info["vendor"],
        "shared_mem": info["shared_mem"],
        "model_ok": model_ok,
        "runtime_ok": runtime_ok,
    }

    if rec is None:
        if info["backend"] == "cpu":
            label = "未检测到可用 GPU（需 NVIDIA / AMD / Intel 显卡或 Apple Silicon）"
        else:
            label = f"不支持本地 LLM（{mem} {vram:.1f}G ≤ 6G）"
        return {
            **base,
            "ok": False,
            "state": "unsupported",
            "label": label,
            "recommended": None,
            "model": model,
            "runtime": runtime,
        }
    if not model_ok:
        return {
            **base,
            "ok": False,
            "state": "missing",
            "label": f"未下载（推荐 {rec['label']}，约 {rec['size_gb']}GB）",
            "recommended": rec,
            "model": None,
            "runtime": runtime,
        }
    if not runtime_ok:
        return {
            **base,
            "ok": False,
            "state": "no_runtime",
            "label": f"已下载（{model.get('label', '')}）",
            "recommended": rec,
            "model": model,
            "runtime": None,
        }
    return {
        **base,
        "ok": True,
        "state": "ready",
        "label": f"已就绪（{model.get('label', '')}）",
        "recommended": rec,
        "model": model,
        "runtime": runtime,
    }


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


def _asset_score(name: str, plat: str, arch: str, backend: str) -> int:
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
        # macOS 发布包自带 Metal，无需区分后端
        return 6 if backend == "metal" else 5
    if "cuda" in n:
        if backend != "cuda":
            return -1
        ver = re.search(r"cuda-(\d+\.\d+)", n)
        # CUDA 12.x 兼容面更广，优先于 13.x
        return 7 if ver and ver.group(1).startswith("12") else 6
    if "vulkan" in n:
        # AMD / Intel 走 Vulkan；NVIDIA 环境也允许作为 CUDA 不可用时的兜底
        return 7 if backend == "vulkan" else (5 if backend == "cuda" else 4)
    if any(k in n for k in ("rocm", "openvino", "sycl", "opencl", "zdnn")):
        return -1
    # 纯 CPU 包只在没有 GPU 时优先，其余情况作为兜底
    return 6 if backend == "cpu" else 2


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


def _pick_runtime_assets(assets: list, plat: str, arch: str, backend: str) -> tuple:
    main = max(assets, key=lambda a: _asset_score(str(a.get("name", "")), plat, arch, backend),
               default=None)
    if main is None or _asset_score(str(main.get("name", "")), plat, arch, backend) < 2:
        return None, None
    return main, _pick_cudart(assets, str(main.get("name", "")), plat, arch)


def _extract_archive(archive: str, dest: str) -> None:
    # 临时文件统一命名为 *.pkg，不能靠后缀判断格式，按文件内容嗅探
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"无法识别的压缩包：{os.path.basename(archive)}")


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
    """下载当前平台可用的 llama.cpp 运行时（NVIDIA→CUDA，AMD/Intel→Vulkan，Apple→Metal）。"""
    plat, arch = _runtime_target()
    gpu = gpu_backend()
    backend = str(gpu.get("backend") or "cpu")
    backend_label = _BACKEND_LABELS.get(backend, backend.upper())

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
        main, cudart = _pick_runtime_assets(rel.get("assets") or [], plat, arch, backend)
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
            progress(0.0, f"下载 llama.cpp 运行时（{backend_label}，{os.path.basename(str(main.get('name', '')))}）…")
        _download_asset(main, tmp_pkg, main_progress)
        _extract_archive(tmp_pkg, tmp_dir)

        if cudart is not None:
            if progress is not None:
                progress(0.75, "下载 CUDA 运行库…")
            _download_asset(
                cudart,
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


__all__ = [
    "MODEL_TIERS",
    "context_size",
    "download_model",
    "download_recommended",
    "download_runtime",
    "ensure_runtime",
    "gpu_backend",
    "gpu_info",
    "installed_model",
    "llama_server_exe",
    "model_alias",
    "model_storage",
    "recommend",
    "resolve_model_path",
    "runtime_status",
    "status",
]
