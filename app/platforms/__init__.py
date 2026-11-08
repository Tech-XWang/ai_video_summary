# -*- coding: utf-8 -*-
"""平台适配层：只做转发，不复制业务逻辑。

当前只实现 B 站；``PLATFORMS`` 里预留了注册位，接入抖音/快手时新增一个
模块并注册即可，service 层无需改动。
"""

from typing import List

from .bilibili import BilibiliPlatform

PLATFORMS = {
    "bilibili": BilibiliPlatform(),
}


def get_platform(name: str):
    platform = PLATFORMS.get(name)
    if platform is None:
        raise RuntimeError(f"暂不支持平台 {name}（当前仅实现 B站）")
    return platform


def available_platforms() -> List[dict]:
    return [{"id": key, "label": p.label} for key, p in PLATFORMS.items()]
