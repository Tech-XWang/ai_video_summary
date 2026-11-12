# -*- coding: utf-8 -*-
"""模型仓库：显存档位、GGUF 扫描、下载与清单管理。"""

import json
import math
import os
import re
import shutil
import time
from typing import Optional

from .config import HF_ENDPOINT, HF_FALLBACK_ENDPOINT, LLAMA_ALIAS, LLM_MODELS_DIR
from .download import Progress, download_file
from .hardware import gpu_info

# 按总显存选型：>=20G Q4、>=16G Q3、>10G 14B Q4、>6G 9B Q4、<=6G 不支持
MODEL_TIERS = [
    {
        "min_vram": 19.9,
        "label": "Qwen3.8-27B Q4",
        "repo": "nsloth/Qwen3.8-27B-GGUF",
        "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "size_gb": 16.5,
    },
    {
        "min_vram": 15.9,
        "label": "Qwen3.8-27B Q3",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "file": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "size_gb": 13.1,
    },
    {
        "min_vram": 10.1,
        "label": "Qwen3-14B Q4",
        "repo": "unsloth/Qwen3-14B-GGUF",
        "file": "Qwen3-14B-Q4_K_S.gguf",
        "size_gb": 8.6,
    },
    {
        "min_vram": 6.1,
        "label": "Qwen3.5-9B Q4",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q4_K_S.gguf",
        "size_gb": 5.4,
    },
]

MANIFEST = os.path.join(LLM_MODELS_DIR, "manifest.json")


def recommend(vram_gb: float) -> Optional[dict]:
    """按显存返回推荐模型；显存 ≤6G 返回 None。
    """
    vram_gb = math.ceil(vram_gb * 10) / 10
    if vram_gb > 19.89:
        return MODEL_TIERS[0]
    if vram_gb > 15.89:
        return MODEL_TIERS[1]
    if vram_gb > 10.11:
        return MODEL_TIERS[2]
    if vram_gb > 6.11:
        return MODEL_TIERS[3]
    return None


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


def download_recommended(progress: Optional[Progress] = None) -> str:
    vram, _ = gpu_info()
    spec = recommend(vram)
    if spec is None:
        raise RuntimeError(f"显存 {vram:.1f}GB ≤ 6GB，不支持本地 LLM 总结")
    return download_model(spec, progress)


def model_storage() -> str:
    """模型目录，供界面提示。"""
    return LLM_MODELS_DIR
