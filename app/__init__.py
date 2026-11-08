# -*- coding: utf-8 -*-
"""GUI 应用层：FastAPI 服务、任务队列、pywebview 外壳。

与 bilinote 的关系：bilinote 只放算法与平台能力，本包只负责编排、事件和界面，
因此 CLI（main.py）与 GUI 共用 app/service.py，不存在两套流水线。
"""

__version__ = "0.1.0"
