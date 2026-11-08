# -*- coding: utf-8 -*-
"""B 站视频转写与总结工具。

模块划分：
  config    常量与路径
  models    Cue 数据模型
  utils     日志、时间格式化、文本归一化与聚合
  bili_api  网络抓取（登录、视频信息、字幕、播放地址、下载）
  media     ffmpeg 封装（转码、抽帧）
  asr       音频语音识别
  ocr       画面硬字幕识别与水印过滤
  merge     多路文本汇合
  llm       LLM 服务管理与两阶段总结
  export    结果导出
"""

__version__ = "2.0.0"
