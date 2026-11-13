/* B站视频总结 GUI 前端 */
(function () {
  "use strict";

  var params = new URLSearchParams(location.search);
  var TOKEN = params.get("token") || "";
  var STAGE_LABEL = {
    queued: "排队中", resolve: "解析视频", download: "下载资源", subtitle: "获取字幕",
    asr: "语音识别", ocr: "画面 OCR", merge: "汇合时间线", llm: "LLM 提炼",
    export: "导出报告", done: "完成", error: "出错", cancelled: "已取消"
  };
  var ENV_ROWS = [
    { key: "cookie", state: "env-cookie", btn: "btn-cookie" },
    { key: "llm", state: "env-llm", btn: "btn-llm" },
    { key: "llama", state: "env-llama", btn: "btn-llama" },
    { key: "asr", state: "env-asr", btn: "btn-asr" },
    { key: "ocr", state: "env-ocr", btn: "btn-ocr" },
    { key: "ffmpeg", state: "env-ffmpeg", btn: "btn-ffmpeg" }
  ];

  var el = function (id) { return document.getElementById(id); };
  var state = { taskId: null, es: null, lastSeq: 0, loginTimer: null, loginKey: null, env: null, pendingInstall: null };
  var ENV_LABELS = { llm: "本地 LLM", llama: "llama.cpp", asr: "ASR", ocr: "OCR", ffmpeg: "FFmpeg", cookie: "Cookie" };
  var BACKEND_LABELS = { cuda: "CUDA", vulkan: "Vulkan", metal: "Metal", cpu: "CPU" };

  function api(path, options) {
    options = options || {};
    options.headers = Object.assign({}, options.headers || {}, { "X-App-Token": TOKEN });
    if (options.body && typeof options.body !== "string") {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.body);
    }
    return fetch(path, options).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) { throw new Error(t || ("HTTP " + r.status)); });
      }
      return r.json();
    });
  }

  function withToken(path) {
    return path + (path.indexOf("?") >= 0 ? "&" : "?") + "token=" + encodeURIComponent(TOKEN);
  }

  function appendLog(text, level) {
    var box = el("log");
    // 只有停在底部时才自动滚动，避免用户正在框选时被强制滚走
    var stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    var line = document.createElement("div");
    if (level === "error") line.className = "err";
    if (level === "warn") line.className = "warn";
    line.textContent = text;
    box.appendChild(line);
    if (stick) box.scrollTop = box.scrollHeight;
  }

  function setProgress(stage, pct, msg) {
    el("stage").textContent = STAGE_LABEL[stage] || stage || "空闲";
    if (pct >= 0) {
      el("pct").textContent = Math.round(pct * 100) + "%";
      el("bar").style.width = (pct * 100) + "%";
    } else {
      el("pct").textContent = "";
      el("bar").style.width = "0%";
    }
    el("msg").textContent = msg || "";
  }

  function fmtDate(sec) {
    var d = new Date(sec * 1000);
    var p = function (n) { return n < 10 ? "0" + n : "" + n; };
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
           " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  function showWarn(text) {
    el("warn-text").textContent = text;
    el("warnModal").classList.remove("hidden");
  }

  function closeWarn() {
    el("warnModal").classList.add("hidden");
  }

  // ---------- 配置 ----------
  var CONFIG_FIELDS = ["mode", "part", "use_subtitle", "use_asr", "use_ocr",
                       "ocr_region", "asr_device", "hotword", "max_chars", "keep_llm",
                       "base_url", "api_key", "model"];

  function getLlmSource() {
    var radio = document.querySelector('input[name="llm_source"]:checked');
    return radio ? radio.value : "local";
  }

  function syncLlmSource() {
    var source = getLlmSource();
    el("llm-local").classList.toggle("hidden", source !== "local");
    el("llm-public").classList.toggle("hidden", source !== "public");
  }

  var llmSaveTimer = null;
  function saveLlmConfig() {
    clearTimeout(llmSaveTimer);
    llmSaveTimer = setTimeout(function () {
      api("/api/config", { method: "PUT", body: {
        llm_source: getLlmSource(),
        keep_llm: el("keep_llm").checked,
        base_url: el("base_url").value.trim(),
        api_key: el("api_key").value.trim(),
        model: el("model").value.trim()
      }}).catch(function (e) {
        appendLog("保存 LLM 配置失败：" + e.message, "error");
      });
    }, 400);
  }

  function applyConfig(cfg) {
    CONFIG_FIELDS.forEach(function (k) {
      var node = el(k);
      if (!node) return;
      if (node.type === "checkbox") node.checked = !!cfg[k];
      else node.value = cfg[k];
    });
    var sel = el("platform");
    sel.innerHTML = "";
    (cfg.platforms || []).forEach(function (p) {
      var o = document.createElement("option");
      o.value = p.id; o.textContent = p.label;
      sel.appendChild(o);
    });
    var source = cfg.llm_source === "public" ? "public" : "local";
    document.querySelectorAll('input[name="llm_source"]').forEach(function (r) {
      r.checked = (r.value === source);
    });
    syncLlmSource();
    syncMode();
  }

  function currentConfig() {
    var out = {};
    CONFIG_FIELDS.forEach(function (k) {
      var node = el(k);
      if (!node) return;
      if (node.type === "checkbox") out[k] = node.checked;
      else if (node.type === "number") out[k] = Number(node.value);
      else out[k] = node.value;
    });
    out.llm_source = getLlmSource();
    // local 模式只用本地 llama-server：public 的地址/密钥/模型名是隐藏字段，
    // 不能把历史保存的值带进任务，否则会误连到外部接口
    if (out.llm_source !== "public") {
      out.base_url = "";
      out.api_key = "";
      out.model = "";
    }
    return out;
  }

  function syncMode() {
    var manual = el("mode").value === "manual";
    ["use_subtitle", "use_asr", "use_ocr"].forEach(function (id) {
      var node = el(id);
      if (node) node.disabled = !manual;
    });
  }

  function loadConfig() {
    api("/api/config").then(applyConfig).catch(function (e) {
      appendLog("读取配置失败：" + e.message, "error");
    });
  }

  // ---------- 本地组件状态 ----------
  var envPollTimer = null;

  function renderEnv(env) {
    var busy = env.installing || "";
    ENV_ROWS.forEach(function (row) {
      var data = env[row.key] || {};
      var node = el(row.state);
      node.textContent = data.label || "未知";
      node.title = data.label || "";
      // 本地 LLM 额外展示识别到的 GPU 与推理后端，便于确认平台适配是否生效
      if (row.key === "llm" && data.backend && data.backend !== "cpu") {
        node.title += "\n" + (data.gpu || "") + " · " + (BACKEND_LABELS[data.backend] || data.backend) +
          " · " + data.vram_gb + "GB" + (data.shared_mem ? "（共享内存）" : "（显存）");
      }
      var unsupported = row.key === "llm" && data.state === "unsupported";
      // LLM 行只反映模型是否下载；运行时状态由 llama.cpp 行单独展示
      var ok = row.key === "llm" ? (!unsupported && !!data.model_ok) : !!data.ok;
      node.className = "env-state " + (ok ? "ok" : (data.running ? "run" : "bad"));

      var btn = el(row.btn);
      if (row.key === "cookie") {
        btn.disabled = false;
      } else {
        // 模型已存在且与当前显存推荐档位一致时无需重复下载；旧模型/档位不一致仍可手动下载
        var sameAsRecommended = row.key === "llm" && data.model && data.recommended &&
          data.model.size_gb &&
          Math.abs(data.model.size_gb - data.recommended.size_gb) < 0.2;
        // asr / ocr / ffmpeg / llama.cpp 已就绪时禁用安装按钮，避免重复执行安装
        var ready = row.key !== "llm" && !!data.ok;
        // 显存不足时本地 LLM 无法运行，装 llama.cpp 没有意义，也避免误触长时间下载
        var unusable = row.key === "llama" && env.llm && env.llm.state === "unsupported";
        btn.disabled = !!busy || unsupported || sameAsRecommended || ready || unusable;
        if (unusable) btn.title = "当前显存不支持本地 LLM，无需安装 llama.cpp";
      }
    });
  }

  function refreshEnv() {
    if (envPollTimer) {
      clearTimeout(envPollTimer);
      envPollTimer = null;
    }
    return api("/api/env").then(function (env) {
      var wasInstalling = (state.env && state.env.installing) || "";
      state.env = env;
      renderEnv(env);
      // 安装结束后补一条日志：wasInstalling 覆盖轮询路径，pendingInstall 覆盖
      // 安装太快、POST 返回前就已结束的路径。
      var doneName = !env.installing && (wasInstalling || state.pendingInstall);
      if (doneName) {
        var info = env[doneName] || {};
        var name = ENV_LABELS[doneName] || doneName;
        if (info.ok) appendLog(name + " 安装完成：" + (info.label || "已就绪"));
        else appendLog(name + " 安装未完成：" + (info.label || "请查看状态"), "error");
        state.pendingInstall = null;
      }
      // 安装/下载在服务端后台线程执行，POST 返回时通常尚未结束；
      // 安装期间保持轮询，结束后再拉一次即可看到最终状态。
      if (env.installing) {
        envPollTimer = setTimeout(refreshEnv, 1500);
      }
      return env;
    }).catch(function (e) {
      appendLog("读取本地状态失败：" + e.message, "error");
      // 仅在安装期间重试，避免无安装任务时反复刷错误日志
      if (state.env && state.env.installing) {
        envPollTimer = setTimeout(refreshEnv, 3000);
      }
      return null;
    });
  }

  function install(name) {
    appendLog("开始处理本地组件：" + name);
    api("/api/env/" + name + "/install", { method: "POST" }).then(function (res) {
      state.pendingInstall = name;
      appendLog((res && res.message) || ("已提交安装/下载任务：" + name));
      refreshEnv();
    }).catch(function (e) {
      appendLog("安装/下载失败：" + e.message, "error");
      refreshEnv();
    });
  }

  // ---------- 任务 ----------
  function openReport(path) {
    api("/api/open-report?path=" + encodeURIComponent(path), { method: "POST" })
      .then(function () { appendLog("已在浏览器打开：" + path); })
      .catch(function (e) { appendLog("打开失败：" + e.message, "error"); });
  }

  function loadHistory() {
    api("/api/outputs").then(function (items) {
      var list = el("history");
      list.innerHTML = "";
      if (!items.length) {
        list.innerHTML = '<li class="muted">暂无</li>';
        return;
      }
      items.forEach(function (it) {
        var li = document.createElement("li");
        li.className = "history-item";
        li.title = it.path;

        var name = document.createElement("div");
        name.className = "h-name";
        name.textContent = it.name;

        var date = document.createElement("span");
        date.className = "h-date";
        date.textContent = fmtDate(it.created || it.mtime);

        var open = document.createElement("button");
        open.className = "mini";
        open.textContent = "在浏览器打开";
        open.addEventListener("click", function (e) {
          e.stopPropagation();
          openReport(it.path);
        });

        li.appendChild(name);
        li.appendChild(date);
        li.appendChild(open);
        li.addEventListener("click", function () { openReport(it.path); });
        list.appendChild(li);
      });
    }).catch(function (e) {
      appendLog("读取历史结果失败：" + e.message, "error");
    });
  }

  function subscribe(taskId) {
    if (state.es) state.es.close();
    state.lastSeq = 0;
    var es = new EventSource(withToken("/api/events?task_id=" + taskId));
    state.es = es;
    es.onmessage = function (e) {
      var ev = JSON.parse(e.data);
      if (ev.seq && ev.seq <= state.lastSeq) return;   // 断线重连时跳过已处理事件
      state.lastSeq = ev.seq || state.lastSeq;
      setProgress(ev.stage, ev.pct, ev.msg);
      // 进度类事件只更新进度条，避免每次刷新都在运行日志里新增一行
      if (ev.msg && ev.kind !== "progress") appendLog(ev.msg, ev.level);
      if (ev.stage === "done") {
        es.close();
        state.es = null;
        setRunning(false);
        appendLog("报告已生成，可在右上方「查看结果」中打开");
        loadHistory();
        refreshEnv();
      } else if (ev.stage === "error" || ev.stage === "cancelled") {
        es.close();
        state.es = null;
        setRunning(false);
        if (ev.stage === "error" && ev.msg &&
            (ev.msg.indexOf("本地 LLM 不可用") === 0 || ev.msg.indexOf("手动模式缺少依赖") === 0)) {
          showWarn(ev.msg);
        }
        refreshEnv();
      }
    };
    es.onerror = function () { appendLog("事件连接中断，正在重连…", "warn"); };
  }

  function setRunning(running) {
    el("start").disabled = running;
    el("cancel").disabled = !running;
  }

  function submitTask(url, cfg) {
    el("log").innerHTML = "";
    setProgress("queued", 0, "提交任务…");
    setRunning(true);
    var body = Object.assign({ url: url, platform: el("platform").value, keep_temp: false }, cfg);
    api("/api/tasks", { method: "POST", body: body }).then(function (job) {
      state.taskId = job.task_id;
      appendLog("任务已创建：" + job.task_id);
      subscribe(job.task_id);
    }).catch(function (e) {
      appendLog("创建任务失败：" + e.message, "error");
      setRunning(false);
    });
  }

  function startTask() {
    var url = el("url").value.trim();
    if (!url) { appendLog("请先填写视频链接", "error"); return; }
    // 以最新的本地状态做前置校验：LLM 缺失一律终止；手动模式勾选的依赖缺失也终止
    api("/api/env").then(function (env) {
      state.env = env;
      renderEnv(env);
      var cfg = currentConfig();
      if (env.installing) {
        showWarn("正在安装/下载本地组件（" + env.installing + "），请等待完成后再开始。");
        return;
      }
      if (cfg.llm_source === "public") {
        if (!cfg.base_url || !cfg.api_key || !cfg.model) {
          showWarn("public 模式需要完整填写 base_url、api_key、model。");
          return;
        }
      } else if (!env.llm || !env.llm.ok) {
        if (env.llm && env.llm.state === "unsupported") {
          showWarn("当前显存不支持本地 LLM 总结：\n" + env.llm.label +
                   "\n\nGUI 任务需要本地 LLM。若只想导出转写稿，可运行：\n" +
                   "python main.py <视频链接> --no-llm");
        } else {
          var detail = (env.llm && env.llm.label) || "状态未知";
          if (env.llm && env.llm.state === "no_runtime") detail += "，缺少 llama.cpp 运行时";
          showWarn("本地 LLM 不可用：\n" + detail +
                   "\n\n请到「高级配置 → 本地状态」下载模型 / 安装 llama.cpp 后再执行。");
        }
        return;
      }
      if (cfg.mode === "manual") {
        var missing = [];
        if (cfg.use_subtitle && !(env.cookie && env.cookie.ok)) {
          missing.push("B站字幕依赖 Cookie，当前未配置");
        }
        if (cfg.use_asr && !(env.asr && env.asr.ok)) {
          missing.push("本地 ASR 未安装");
        }
        if (cfg.use_ocr && !(env.ocr && env.ocr.ok)) {
          missing.push("本地 OCR 未安装");
        }
        if ((cfg.use_asr || cfg.use_ocr) && !(env.ffmpeg && env.ffmpeg.ok)) {
          missing.push("ffmpeg 未安装");
        }
        if (!cfg.use_subtitle && !cfg.use_asr && !cfg.use_ocr) {
          missing.push("手动模式至少勾选一个数据源");
        }
        if (missing.length) {
          showWarn("手动模式缺少依赖，已终止执行：\n\n" + missing.join("\n") +
                   "\n\n请先安装，或取消对应勾选。");
          return;
        }
      }
      submitTask(url, cfg);
    }).catch(function (e) {
      appendLog("读取本地状态失败：" + e.message, "error");
    });
  }

  function cancelTask() {
    if (!state.taskId) return;
    api("/api/tasks/" + state.taskId + "/cancel", { method: "POST" })
      .then(function () { appendLog("已请求取消", "warn"); })
      .catch(function (e) { appendLog("取消失败：" + e.message, "error"); });
  }

  // ---------- 扫码登录 ----------
  function openLogin() {
    el("loginModal").classList.remove("hidden");
    el("qr-box").textContent = "生成中…";
    el("qr-state").textContent = "";
    api("/api/login/start", { method: "POST" }).then(function (res) {
      state.loginKey = res.key;
      el("qr-box").innerHTML = '<img alt="登录二维码" src="' + res.image + '">';
      el("qr-state").textContent = "请用 B 站手机客户端扫码";
      clearInterval(state.loginTimer);
      state.loginTimer = setInterval(pollLogin, 2000);
    }).catch(function (e) {
      el("qr-box").textContent = "生成失败";
      el("qr-state").textContent = e.message;
    });
  }

  function pollLogin() {
    if (!state.loginKey) return;
    api("/api/login/status?key=" + encodeURIComponent(state.loginKey)).then(function (res) {
      if (res.message) el("qr-state").textContent = res.message;
      if (res.saved || res.state === 0 || res.state === 86038) {
        clearInterval(state.loginTimer);
        state.loginTimer = null;
        refreshEnv();
        if (res.saved) setTimeout(closeLogin, 800);
      }
    }).catch(function () {});
  }

  function closeLogin() {
    clearInterval(state.loginTimer);
    state.loginTimer = null;
    el("loginModal").classList.add("hidden");
  }

  // ---------- 事件绑定 ----------
  document.addEventListener("DOMContentLoaded", function () {
    el("start").addEventListener("click", startTask);
    el("cancel").addEventListener("click", cancelTask);
    el("refresh").addEventListener("click", loadHistory);
    el("mode").addEventListener("change", syncMode);
    document.querySelectorAll('input[name="llm_source"]').forEach(function (r) {
      r.addEventListener("change", function () {
        syncLlmSource();
        saveLlmConfig();
      });
    });
    ["base_url", "api_key", "model"].forEach(function (id) {
      el(id).addEventListener("input", saveLlmConfig);
    });
    el("keep_llm").addEventListener("change", saveLlmConfig);
    el("btn-cookie").addEventListener("click", openLogin);
    el("btn-llm").addEventListener("click", function () { install("llm"); });
    el("btn-llama").addEventListener("click", function () { install("llama"); });
    el("btn-asr").addEventListener("click", function () { install("asr"); });
    el("btn-ocr").addEventListener("click", function () { install("ocr"); });
    el("btn-ffmpeg").addEventListener("click", function () { install("ffmpeg"); });
    el("login-close").addEventListener("click", closeLogin);
    el("warn-close").addEventListener("click", closeWarn);
    el("url").addEventListener("keydown", function (e) {
      if (e.key === "Enter") startTask();
    });
    loadConfig();
    refreshEnv();
    loadHistory();
    setInterval(refreshEnv, 2000);
  });
})();
