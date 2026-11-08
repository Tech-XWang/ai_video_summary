# -*- coding: utf-8 -*-
"""pywebview 桌面外壳：先起本地服务，再开窗口；缺 webview 时退化为浏览器模式。

Windows 上 pywebview + pythonnet + WebView2 偶现渲染层问题（窗口打开但空白），
因此默认使用 webbrowser 打开，用 ``BILI_USE_PYWEBVIEW=1`` 环境变量显式启用
pywebview 桌面壳。非 Windows 平台仍默认 pywebview。
"""

import os
import secrets
import socket
import threading
import time
import urllib.request

import uvicorn

from bilinote import llm, utils

from .server import create_app


def _use_pywebview() -> bool:
    """决定是否用 pywebview 桌面壳；否则用 webbrowser 退化为浏览器模式。"""
    if os.environ.get("BILI_USE_PYWEBVIEW") == "1":
        return True
    if os.name == "nt":                       # Windows 默认走浏览器，稳定可靠
        return False
    try:
        import webview                        # 其它平台（macOS/Linux）默认 pywebview
        return True
    except ImportError:
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    print("** 此为AI视频总结后台程序，请勿手动关闭，关闭前端后会自动退出 **", flush=True)
    # GUI 的后台 cmd 输出完整日志（含 debug 级 LLM 请求内容），前端仍只显示 info
    utils.set_console_debug(True)
    token = secrets.token_urlsafe(16)
    port = _free_port()
    app = create_app(token)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    if not _wait_ready(port):
        print("本地服务启动失败")
        return 1

    url = f"http://127.0.0.1:{port}/?token={token}"
    try:
        if not _use_pywebview():
            import webbrowser

            webbrowser.open(url)
            print(f"已在浏览器打开：{url}\n按 Ctrl+C 退出。", flush=True)
            try:
                while thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return 0

        import webview
        # pywebview 加载 URL 是异步的，_wait_ready 只确认后端已就绪；
        # 给渲染层留一点时间，避免 create_window 过早触发 URL 加载。
        time.sleep(0.2)
        webview.create_window("B站视频总结", url, width=1180, height=860, min_size=(960, 640),
                              text_select=True)
        webview.start()
        return 0
    finally:
        # 关窗/退出时释放本次启动的 llama-server 显存（外部已开的服务不动）
        llm.stop_managed_llm()
        server.should_exit = True


if __name__ == "__main__":
    raise SystemExit(main())
