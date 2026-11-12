# -*- coding: utf-8 -*-
"""硬件探测：系统内存、GPU 后端（CUDA / Vulkan / Metal）与可用显存估算。"""

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Tuple

_GPU_CACHE = {"ts": 0.0, "info": None}


def _system_ram_gb() -> float:
    """系统物理内存 GB；用于统一内存（Apple / 核显）估算可用显存。"""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            return float(out) / (1024 ** 3)
        except Exception:
            return 0.0
    if os.name == "nt":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024 ** 2)
    except (OSError, ValueError):
        pass
    return 0.0


def _vendor_of(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "rtx ", "quadro", "tesla")):
        return "nvidia"
    if any(k in n for k in ("amd", "radeon", "ati ")):
        return "amd"
    if "intel" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return ""


def _is_virtual_gpu(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("virtual", "remote", "basic display", "parsec",
                                "vmware", "citrix", "mirror", "indirect"))


def _is_integrated(name: str) -> bool:
    """核显/APU 判断：这类显卡没有独立显存，只能共享系统内存。"""
    n = name.lower()
    if any(k in n for k in ("geforce", "rtx", "quadro", "tesla", "radeon pro")):
        return False
    if re.search(r"\b(rx|pro\s+w)\s?\d", n):
        return False
    if re.search(r"arc(\(tm\))?\s?[ab]\d", n):
        return False
    return any(k in n for k in ("graphics", "vega", "iris", "uhd", "intel", "radeon"))


def _nvidia_info() -> Tuple[float, str]:
    """nvidia-smi 汇总的多卡总显存 GB 与第一张卡名称。"""
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
    return total, name


def _windows_registry_gpus() -> list:
    """读注册表枚举显卡；WMI 的 AdapterRAM 是 uint32，超过 4GB 会截断，不可用。"""
    try:
        import winreg
    except ImportError:
        return []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    rows, seen = [], set()
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not re.fullmatch(r"\d{4}", sub):
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    try:
                        vram = float(winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0])
                    except OSError:
                        vram = 0.0
            except OSError:
                continue
            if not name or name in seen or _is_virtual_gpu(name):
                continue
            seen.add(name)
            rows.append({"name": name, "vendor": _vendor_of(name), "vram_gb": vram / (1024 ** 3)})
    return rows


def _windows_wmi_gpus() -> list:
    """注册表拿不到时的兜底：PowerShell CIM 查询。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        data = json.loads(out) if out else []
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data:
        name = str(item.get("Name") or "")
        if not name or _is_virtual_gpu(name):
            continue
        ram = float(item.get("AdapterRAM") or 0)
        rows.append({
            "name": name,
            "vendor": _vendor_of(name),
            "vram_gb": ram / (1024 ** 3) if 0 < ram < 4.2e9 else 0.0,
        })
    return rows


def _linux_gpus() -> list:
    rows = []
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            with open(os.path.join(dev, "vendor"), "r", encoding="utf-8") as f:
                vendor_id = f.read().strip().lower()
        except OSError:
            continue
        vendor = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}.get(vendor_id, "")
        if not vendor:
            continue
        vram = 0.0
        try:
            with open(os.path.join(dev, "mem_info_vram_total"), "r", encoding="utf-8") as f:
                vram = float(f.read().strip()) / (1024 ** 3)
        except (OSError, ValueError):
            pass
        rows.append({
            "name": {"amd": "AMD GPU", "intel": "Intel GPU", "nvidia": "NVIDIA GPU"}[vendor],
            "vendor": vendor,
            "vram_gb": vram,
        })
    return rows


def _detect_gpu() -> dict:
    """识别 GPU 后端与可用于模型的显存。

    NVIDIA 走 CUDA，AMD/Intel 走 Vulkan，Apple Silicon 走 Metal（统一内存）。
    核显/统一内存没有独立显存，按系统内存的比例估算可用容量。
    """
    ram = _system_ram_gb()
    if sys.platform == "darwin":
        name = "Apple GPU"
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if cpu:
                name = cpu
        except Exception:
            pass
        # Metal 与系统共用内存，按半内存估算，避免推荐推理过慢的档位
        return {"backend": "metal", "vendor": "apple", "name": name,
                "vram_gb": ram * 0.5, "shared_mem": True}

    total, name = _nvidia_info()
    if total > 0:
        return {"backend": "cuda", "vendor": "nvidia", "name": name,
                "vram_gb": total, "shared_mem": False}

    gpus = _windows_registry_gpus() if os.name == "nt" else _linux_gpus()
    if os.name == "nt" and not any(g["vendor"] in ("amd", "intel") for g in gpus):
        gpus = _windows_wmi_gpus() or gpus
    gpus = [g for g in gpus if g["vendor"] in ("amd", "intel")]
    if gpus:
        discrete = [g for g in gpus if not _is_integrated(g["name"])]
        if discrete:
            vram = sum(max(g["vram_gb"], 0.0) for g in discrete)
            # 驱动未上报容量时按半内存估算，避免误判为不支持
            if vram < 1.0:
                vram = ram * 0.5
            return {"backend": "vulkan", "vendor": discrete[0]["vendor"],
                    "name": " / ".join(g["name"] for g in discrete),
                    "vram_gb": vram, "shared_mem": False}
        return {"backend": "vulkan", "vendor": gpus[0]["vendor"],
                "name": " / ".join(g["name"] for g in gpus),
                # 核显带宽有限，按 40% 内存估算并限制 8GB 上限，避免推荐推理过慢的档位
                "vram_gb": min(ram * 0.4, 8.0), "shared_mem": True}

    return {"backend": "cpu", "vendor": "", "name": "", "vram_gb": 0.0, "shared_mem": False}


def gpu_backend() -> dict:
    """当前 GPU 后端：{backend, vendor, name, vram_gb, shared_mem}（30s 缓存）。"""
    now = time.time()
    cached = _GPU_CACHE.get("info")
    if cached and now - float(_GPU_CACHE["ts"]) < 30:
        return dict(cached)
    info = _detect_gpu()
    _GPU_CACHE["ts"] = now
    _GPU_CACHE["info"] = info
    return dict(info)


def gpu_info() -> Tuple[float, str]:
    """返回 (可用于模型的总显存 GB, 显卡名称)；取不到时为 (0, "")。"""
    info = gpu_backend()
    return float(info["vram_gb"]), str(info["name"])


def context_size() -> int:
    vram, _ = gpu_info()
    return 32768 if vram >= 20 else 25600
