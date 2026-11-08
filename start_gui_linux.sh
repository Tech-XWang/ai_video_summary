#!/usr/bin/env bash
# Linux / macOS 启动脚本。
#
# 系统依赖（pywebview 桌面窗口）：
#   Debian/Ubuntu: sudo apt install python3-gi gir1.2-webkit2-4.1
#   Fedora:        sudo dnf install python3-gobject webkit2gtk4.1
#   macOS:         pip install pyobjc
# 缺少上述依赖时会退化为浏览器模式，或直接手动运行：python3 -m app.desktop

set -euo pipefail
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

exec "$PY" -m app.desktop
