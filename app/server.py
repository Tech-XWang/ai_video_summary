# -*- coding: utf-8 -*-
"""FastAPI 服务：任务、SSE 事件、登录、配置、报告与历史。

只监听 127.0.0.1，并用启动时生成的 token 校验 /api/* 请求。
"""

import asyncio
import base64
import json
import os
import queue as queue_lib
import webbrowser
from pathlib import Path
from typing import Optional

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import bilibili_api, ffmpeg
from backend.config import OCR_DIFF, OCR_FPS, OUT_DIR

from .events import EventBus
from .installer import EnvManager
from .jobs import JobManager
from .platforms import available_platforms
from .service import JobParams
from .settings import load_config, save_config

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
FFMPEG_DOWNLOAD_PAGE = "https://ffmpeg.org/download.html"


class TaskRequest(BaseModel):
    url: str
    platform: str = "bilibili"
    mode: str = "auto"
    part: int = 0
    use_subtitle: bool = True
    use_asr: bool = True
    use_ocr: bool = True
    ocr_region: str = "bottom"
    ocr_fps: float = OCR_FPS
    asr_device: str = "auto"
    hotword: str = ""
    max_chars: int = 8000
    keep_temp: bool = False
    keep_llm: bool = True
    llm_source: str = "local"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class ConfigPatch(BaseModel):
    """仅允许更新可持久化的 LLM 配置字段。"""

    llm_source: Optional[str] = None
    keep_llm: Optional[bool] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


def create_app(token: str) -> FastAPI:
    app = FastAPI(title="bili_summary GUI", docs_url=None, redoc_url=None)
    bus = EventBus()
    manager = JobManager(bus)
    env = EnvManager()
    app.state.bus = bus
    app.state.manager = manager
    app.state.env = env

    def guard(x_app_token: str = Header(default=""), token_q: str = Query(default="", alias="token")):
        if x_app_token != token and token_q != token:
            raise HTTPException(status_code=401, detail="invalid token")

    # ---------- 页面与静态资源 ----------
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    # ---------- 任务 ----------
    @app.post("/api/tasks")
    def create_task(req: TaskRequest, _=Depends(guard)):
        mode = req.mode if req.mode in ("auto", "manual") else "auto"
        if mode == "manual" and not (req.use_subtitle or req.use_asr or req.use_ocr):
            raise HTTPException(status_code=400, detail="手动模式至少勾选一个数据源")
        # local 模式忽略 public 的地址/密钥/模型名，避免误连外部接口
        llm_source = req.llm_source if req.llm_source in ("local", "public") else "local"
        public = llm_source == "public"
        ocr_fps = req.ocr_fps if req.ocr_fps and 4 >= req.ocr_fps > 0 else OCR_FPS
        params = JobParams(
            url=req.url.strip(),
            platform=req.platform,
            mode=mode,
            part=req.part,
            use_subtitle=req.use_subtitle,
            use_asr=req.use_asr,
            use_ocr=req.use_ocr,
            ocr_fps=ocr_fps,
            ocr_diff=OCR_DIFF,
            ocr_region=req.ocr_region,
            asr_device=req.asr_device,
            hotword=req.hotword,
            model=req.model if public else "",
            base_url=req.base_url if public else "",
            api_key=req.api_key if public else "",
            llm_source=llm_source,
            max_chars=req.max_chars,
            out_dir=OUT_DIR,
            keep_temp=req.keep_temp,
            keep_llm=req.keep_llm,
        )
        return manager.submit(params).to_dict()

    @app.get("/api/tasks")
    def list_tasks(_=Depends(guard)):
        return [j.to_dict() for j in manager.list()[:30]]

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, _=Depends(guard)):
        job = manager.get(task_id)
        if job is None:
            raise HTTPException(status_code=404, detail="task not found")
        return job.to_dict()

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, _=Depends(guard)):
        return {"ok": manager.cancel(task_id)}

    # ---------- SSE ----------
    @app.get("/api/events")
    async def events(task_id: str, since: float = 0.0, _=Depends(guard)):
        q = bus.subscribe(task_id)

        async def gen():
            try:
                for ev in bus.history(task_id, since):
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                while True:
                    try:
                        ev = await asyncio.to_thread(q.get, True, 15)
                    except queue_lib.Empty:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(ev.to_dict(), ensure_ascii=False)}\n\n"
            finally:
                bus.unsubscribe(task_id, q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------- 报告与历史 ----------
    @app.get("/api/outputs")
    def outputs(_=Depends(guard)):
        """扫描整个 out 目录，按文件创建时间倒序返回历史结果。"""
        items = []
        try:
            for name in os.listdir(OUT_DIR):
                folder = os.path.join(OUT_DIR, name)
                if not os.path.isdir(folder):
                    continue
                html_files = [f for f in os.listdir(folder) if f.endswith(".html")]
                if not html_files:
                    continue
                target = os.path.join(folder, sorted(html_files)[0])
                stat = os.stat(target)
                items.append({
                    "name": name,
                    "path": target,
                    # Windows 上 st_ctime 即创建时间；其它平台退化为修改时间
                    "created": getattr(stat, "st_ctime", stat.st_mtime),
                    "mtime": stat.st_mtime,
                })
        except OSError:
            pass
        items.sort(key=lambda x: x["created"], reverse=True)
        return items[:100]

    @app.post("/api/open-report")
    def open_report(path: str, _=Depends(guard)):
        """用系统默认浏览器打开 out 目录下的报告，不做页面内预览。"""
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(OUT_DIR)):
            raise HTTPException(status_code=403, detail="path not allowed")
        if not os.path.isfile(real):
            raise HTTPException(status_code=404, detail="report not found")
        try:
            webbrowser.open(Path(real).as_uri())
        except Exception as e:  # noqa: BLE001  打开失败需要回传界面
            raise HTTPException(status_code=500, detail=f"打开失败：{e}")
        return {"ok": True}

    # ---------- 配置与状态 ----------
    @app.get("/api/config")
    def get_config(_=Depends(guard)):
        cfg = load_config()
        cfg["platforms"] = available_platforms()
        return cfg

    @app.put("/api/config")
    def put_config(patch: ConfigPatch, _=Depends(guard)):
        """只保存 LLM 相关字段；base_url / api_key / model 改动即落盘。"""
        data = {k: v for k, v in patch.model_dump().items() if v is not None}
        return save_config(data)

    @app.get("/api/env")
    def env_status(_=Depends(guard)):
        return app.state.env.status()

    @app.post("/api/env/{component}/install")
    def env_install(component: str, _=Depends(guard)):
        if component == "ffmpeg":
            if ffmpeg.ffmpeg_status().get("ok"):
                return {"ok": True, "component": component,
                        "message": "ffmpeg 已可用，无需安装"}
            webbrowser.open(FFMPEG_DOWNLOAD_PAGE)
            return {"ok": True, "component": component, "manual": True,
                    "message": "已打开 ffmpeg 下载页，请安装后将其加入系统 PATH"}
        ok, message = app.state.env.start(component)
        if not ok:
            raise HTTPException(status_code=409, detail=message)
        return {"ok": True, "component": component}

    # ---------- 扫码登录 ----------
    login_sess = requests.Session()

    @app.post("/api/login/start")
    def login_start(_=Depends(guard)):
        key, png = bilibili_api.login_qr(login_sess)
        if not key or png is None:
            return JSONResponse({"ok": False, "message": "二维码生成失败"}, status_code=400)
        image = "data:image/png;base64," + base64.b64encode(png).decode()
        return {"ok": True, "key": key, "image": image}

    @app.get("/api/login/status")
    def login_status(key: str, _=Depends(guard)):
        state, cookie = bilibili_api.login_poll(key, login_sess)
        message = bilibili_api.LOGIN_STATE_MSG.get(state, "") if state is not None else ""
        if state == 0 and cookie:
            bilibili_api.save_cookie(cookie)
            return {"state": 0, "message": "登录成功，Cookie 已保存", "saved": True}
        return {"state": state, "message": message, "saved": False}

    return app
