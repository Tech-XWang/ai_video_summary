# -*- coding: utf-8 -*-
"""平台适配层：只做转发，不复制业务逻辑。

``PLATFORMS`` 是注册表：接入新平台时新增一个模块并注册即可，
service 层通过统一接口调用。
"""

from typing import List

from .bilibili import BilibiliPlatform
from .local import LocalPlatform

PLATFORMS = {
    "bilibili": BilibiliPlatform(),
    "local": LocalPlatform(),
}


def get_platform(name: str):
    platform = PLATFORMS.get(name)
    if platform is None:
        raise RuntimeError(f"暂不支持平台 {name}")
    return platform


def available_platforms() -> List[dict]:
    return [{"id": key, "label": p.label} for key, p in PLATFORMS.items()]
