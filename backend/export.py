# -*- coding: utf-8 -*-
"""导出：按来源拆分转写稿，另出结构化 JSON 与自包含 HTML 报告。

HTML 报告直接由数据构造，不经过 Markdown 转换：
  * 标题、视频信息、核心信息、内容提炼、关键数据或案例、完整转写稿
  * 每个模块用 <details> 实现展开/收起，完整转写稿默认收起
"""

import html
import json
import os
import re
from dataclasses import asdict
from typing import Dict, List, Optional

from .config import INDEPENDENT_SRC, SRC_DESC
from .cues import Cue
from .utils import fmt_ts, group_cues

# 总结的三个固定章节（与 prompts 里的输出要求一致）
SECTION_TITLES = ("核心信息", "内容提炼", "关键数据或案例")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.、)]\s+(.*)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")

_HTML_STYLE = """
:root {
  --fg: #1f2328; --muted: #64748b; --line: #e2e8f0;
  --accent: #2563eb; --bg: #f1f5f9; --card: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.75 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.page { max-width: 900px; margin: 0 auto; padding: 40px 20px 64px; }
.page-header { margin-bottom: 24px; }
.page-header h1 { font-size: 1.75rem; line-height: 1.35; margin: 0 0 10px; }
.page-header .link {
  color: var(--accent); text-decoration: none; font-size: .9rem;
  word-break: break-all;
}
.page-header .link:hover { text-decoration: underline; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  margin-bottom: 14px; overflow: hidden;
}
.card summary {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 16px 20px; font-weight: 600; cursor: pointer; user-select: none;
  list-style: none;
}
.card summary::-webkit-details-marker { display: none; }
.card summary::after {
  content: "▾"; color: var(--muted); font-size: .85rem;
  transition: transform .18s ease;
}
.card[open] summary::after { transform: rotate(180deg); }
.card[open] summary { border-bottom: 1px solid var(--line); }
.card summary:hover { background: #f8fafc; }
.card-body { padding: 18px 22px 22px; }
.card-body p { margin: .7em 0; }
.card-body ul, .card-body ol { margin: .4em 0; padding-left: 1.35em; }
.card-body li { margin: .38em 0; }
.src-block {
  border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px;
  overflow: hidden;
}
.src-block > summary {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 9px 14px; font-size: .88rem; font-weight: 600; color: var(--muted);
  cursor: pointer; list-style: none; user-select: none;
}
.src-block > summary::-webkit-details-marker { display: none; }
.src-block > summary::after { content: "▾"; font-size: .75rem; transition: transform .18s ease; }
.src-block[open] > summary { border-bottom: 1px solid var(--line); }
.src-block[open] > summary::after { transform: rotate(180deg); }
.src-block > summary:hover { background: #f8fafc; }
.src-block .lines { padding: 4px 14px 8px; }
.src-block .lines li { font-size: .72rem; line-height: 1.65; }
.src-block .ts { font-size: .72rem; }
.empty { color: var(--muted); }
.notice {
  margin-bottom: 14px; padding: 10px 14px; font-size: .9rem; color: #b45309;
  background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
}
.info-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 24px; margin: 0;
}
.info-grid .wide { grid-column: 1 / -1; }
.info-grid dt { font-size: .8rem; color: var(--muted); }
.info-grid dd { margin: 2px 0 0; font-size: .95rem; word-break: break-word; }
.lines { list-style: none; margin: 0; padding: 0; }
.lines li { padding: 6px 0; border-bottom: 1px dashed #eef2f7; }
.lines li:last-child { border-bottom: 0; }
.ts {
  display: inline-block; margin-right: 8px; padding: 1px 6px; color: var(--accent);
  background: #eff6ff; border-radius: 4px; white-space: nowrap;
  font: .82rem/1.6 ui-monospace, SFMono-Regular, Consolas, monospace;
}
@media (max-width: 620px) {
  .page { padding: 24px 14px; }
  .page-header h1 { font-size: 1.4rem; }
  .card-body { padding: 16px 16px 18px; }
  .info-grid { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# 文本解析：把总结文本切成三个章节，并渲染为原生 HTML
# ---------------------------------------------------------------------------
def _inline(text: str) -> str:
    """转义后处理行内 **加粗** 与 `代码`，避免任何未转义的标签。"""
    s = html.escape(text.strip())
    s = _BOLD_RE.sub(r"<strong>\1</strong>", s)
    s = _CODE_RE.sub(r"<code>\1</code>", s)
    return s


def _section_of(title: str) -> Optional[str]:
    if "核心" in title or "观点" in title:
        return "核心信息"
    if "提炼" in title:
        return "内容提炼"
    if "数据" in title or "案例" in title:
        return "关键数据或案例"
    return None


def _split_sections(summary: str) -> Dict[str, List[str]]:
    """按 ## 标题切分总结；无法识别的标题内容归入「内容提炼」。"""
    buckets: Dict[str, List[str]] = {name: [] for name in SECTION_TITLES}
    current = "内容提炼"  # 没有标题时按正文处理
    for line in summary.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            current = _section_of(m.group(1)) or "内容提炼"
            continue
        stripped = line.strip()
        if not stripped or stripped in ("---", "***", "___"):
            continue
        buckets[current].append(line)
    return buckets


def _render_body(lines: List[str]) -> str:
    """把章节文本渲染成段落 / 有序无序列表。"""
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _BULLET_RE.match(line):
            items = []
            while i < n and _BULLET_RE.match(lines[i]):
                items.append(_BULLET_RE.match(lines[i]).group(1))
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ul>")
            continue
        if _NUMBERED_RE.match(line):
            items = []
            while i < n and _NUMBERED_RE.match(lines[i]):
                items.append(_NUMBERED_RE.match(lines[i]).group(1))
                i += 1
            out.append("<ol>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ol>")
            continue
        para: List[str] = []
        while i < n and not _BULLET_RE.match(lines[i]) and not _NUMBERED_RE.match(lines[i]):
            para.append(lines[i])
            i += 1
        out.append("<p>" + "<br>".join(_inline(t) for t in para) + "</p>")
    return "\n".join(out)


def _card(title: str, body: str, expanded: bool = True, extra_class: str = "") -> str:
    cls = f"card {extra_class}".strip()
    opened = " open" if expanded else ""
    return (
        f'<details class="{cls}"{opened}>\n'
        f"<summary>{html.escape(title)}</summary>\n"
        f'<div class="card-body">\n{body}\n</div>\n'
        f"</details>"
    )


def _render_summary(summary: Optional[str]) -> List[str]:
    """生成「核心信息 / 内容提炼 / 关键数据或案例」三块折叠卡片。"""
    blocks: List[str] = []
    if not summary or not summary.strip():
        blocks.append('<p class="notice">（未生成总结：LLM 服务不可用）</p>')
        for name in SECTION_TITLES:
            blocks.append(_card(name, '<p class="empty">（无）</p>'))
        return blocks
    buckets = _split_sections(summary)
    for name in SECTION_TITLES:
        body = _render_body(buckets[name]) or '<p class="empty">（无）</p>'
        blocks.append(_card(name, body))
    return blocks


# ---------------------------------------------------------------------------
# HTML 页面
# ---------------------------------------------------------------------------
def _render_info(info: dict, meta: dict) -> str:
    esc = html.escape
    src_line = "、".join(
        f"{SRC_DESC.get(k, k)}（{v} 条）" for k, v in meta["counts"].items() if v
    ) or "无"
    rows = [
        ("UP 主", esc(info.get("owner", "") or "—"), False),
        ("时长", fmt_ts(info["duration"]), False),
        ("分P", esc(str(info.get("part") or 1)), True),
        ("文本来源", esc(src_line), False),
        ("保留轨道", esc(meta["base"]), False),
    ]
    items: List[str] = []
    for label, value, wide in rows:
        cls = ' class="wide"' if wide else ""
        items.append(f"<div{cls}><dt>{esc(label)}</dt><dd>{value}</dd></div>")
    return '<dl class="info-grid">\n' + "\n".join(items) + "\n</dl>"


def _render_transcript(srcs: List[str], by_src: Dict[str, List[Cue]]) -> str:
    parts: List[str] = []
    for src in srcs:
        label = html.escape(SRC_DESC.get(src, src))
        # 每个来源一个可收缩的子块
        parts.append('<details class="src-block" open>')
        parts.append(f"<summary>{label}（{len(by_src[src])} 条）</summary>")
        parts.append('<ul class="lines">')
        # 人工字幕 / AI 字幕每条 cue 独立展示，不做相邻合并
        for start, text in group_cues(by_src[src], merge=src not in INDEPENDENT_SRC):
            parts.append(
                f'<li><span class="ts">[{fmt_ts(start)}]</span>{html.escape(text)}</li>'
            )
        parts.append("</ul>")
        parts.append("</details>")
    return "\n".join(parts)


def _render_html(info: dict, srcs: List[str], by_src: Dict[str, List[Cue]],
                 meta: dict, summary: Optional[str]) -> str:
    title = html.escape(info.get("title", ""))
    url = html.escape(info.get("url", ""))
    blocks: List[str] = []
    blocks.append(_card("视频信息", _render_info(info, meta)))
    blocks.extend(_render_summary(summary))
    blocks.append(
        _card("源文本", _render_transcript(srcs, by_src),
              expanded=False, extra_class="transcript")
    )

    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>《{title}》</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<div class="page">
<header class="page-header">
<h1>{title}</h1>
<a class="link" href="{url}" target="_blank" rel="noopener noreferrer" style="font-size:20px;">{url}</a>
</header>

{body}
</div>
</body>
</html>
"""


def write_outputs(out_dir: str, stem: str, info: dict, cues: List[Cue],
                  meta: dict, summary: Optional[str]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    files = []

    # 按来源分组：不同 src 分开保存，避免多路混排
    by_src: Dict[str, List[Cue]] = {}
    for c in sorted(cues, key=lambda x: x.start):
        by_src.setdefault(c.src, []).append(c)
    srcs = [k for k in SRC_DESC if k in by_src]
    srcs += [k for k in by_src if k not in SRC_DESC]

    # 转写稿：每个来源一个文件
    for src in srcs:
        txt_path = os.path.join(out_dir, f"{stem}.transcript.{src}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for c in by_src[src]:
                f.write(f"[{fmt_ts(c.start)}] {c.text}\n")
        files.append(txt_path)

    # # SRT
    # srt_path = os.path.join(out_dir, stem + ".srt")
    # with open(srt_path, "w", encoding="utf-8") as f:
    #     for i, c in enumerate(cues, 1):
    #         f.write(f"{i}\n{srt_ts(c.start)} --> {srt_ts(c.end)}\n{c.text}\n\n")
    # files.append(srt_path)

    # 结构化 JSON
    js_path = os.path.join(out_dir, stem + ".json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": info,
                "sources": meta,
                "cues": {src: [asdict(c) for c in by_src[src]] for src in srcs},
            },
            f, ensure_ascii=False, indent=2,
        )
    files.append(js_path)

    # HTML 报告
    html_path = os.path.join(out_dir, stem + ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(info, srcs, by_src, meta, summary))
    files.append(html_path)
    return files
