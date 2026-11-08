# -*- coding: utf-8 -*-
"""GUI 配置：默认值 + LLM 相关字段的本地持久化。

只有「模型来源 / LLM 常驻 / base_url / api_key / model」会写入
``.bili_cache/config.json``；其余参数每次启动仍从默认值开始。
"""

import json
import os
from typing import Any, Dict

from bilinote.config import CACHE_DIR, OUT_DIR

CONFIG_PATH = os.path.join(CACHE_DIR, "config.json")
PERSIST_KEYS = ("llm_source", "keep_llm", "base_url", "api_key", "model")

DEFAULTS: Dict[str, Any] = {
    "mode": "auto",
    "part": 0,
    "use_subtitle": True,
    "use_asr": True,
    "use_ocr": True,
    "ocr_region": "bottom",
    "asr_device": "auto",
    "hotword": "",
    "max_chars": 8000,
    "keep_llm": True,
    "llm_source": "local",
    "base_url": "",
    "api_key": "",
    "model": "",
    "out_dir": OUT_DIR,
}


def default_config() -> Dict[str, Any]:
    """返回一份默认配置的副本。"""
    return dict(DEFAULTS)


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """只保留可持久化字段，并做基本类型归一。"""
    out = {}
    for key in PERSIST_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key == "keep_llm":
            out[key] = bool(value)
        elif key == "llm_source":
            out[key] = "public" if value == "public" else "local"
        else:
            out[key] = str(value or "").strip()
    return out


def load_config() -> Dict[str, Any]:
    """默认值 + 上次保存的 LLM 配置。"""
    cfg = default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return cfg
    if isinstance(data, dict):
        saved = _normalize(data)
        cfg.update(saved)
        if not saved.get("llm_source"):
            cfg["llm_source"] = "local"
    return cfg


def save_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    """合并并写入配置文件，返回完整配置。"""
    cfg = load_config()
    cfg.update(_normalize(patch))
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({key: cfg[key] for key in PERSIST_KEYS}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)
    if os.name != "nt":
        try:
            os.chmod(CONFIG_PATH, 0o600)   # 配置中含 api_key
        except OSError:
            pass
    return cfg
