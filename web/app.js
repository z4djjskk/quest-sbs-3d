const BUILD_ID = "2025-12-20-4";

const defaults = {
  baseline: 0.064,
  baselineMin: 0,
  maxDisp: 60,
  fov: 60,
  cutThreshold: 0.9,
  minShot: 0.01,
  maxShot: 0.05,
  minInliers: 70,
  maxReproj: 2,
  crf: 18,
  preset: "slow",
  inpaint: 2,
  copyAudio: true,
  audioCodec: "copy",
  debugInterval: 30,
  device: "auto",
  ioBackend: "ffmpeg",
  decode: "nvdec",
  encode: "hevc_nvenc",
  trackBackend: "cpu",
  keyframeMode: "normal",
  perFrameBatch: 1,
  perFramePipeline: 2,
  perFrameCache: false,
  clearCacheOnExit: true,
  amp: false,
};

const ids = [
  "videoPath",
  "outPath",
  "debugDir",
  "baseline",
  "baselineMin",
  "maxDisp",
  "fov",
  "cutThreshold",
  "minShot",
  "maxShot",
  "minInliers",
  "maxReproj",
  "crf",
  "preset",
  "inpaint",
  "copyAudio",
  "audioCodec",
  "debugInterval",
  "maxFrames",
  "device",
  "ioBackend",
  "decode",
  "encode",
  "trackBackend",
  "keyframeMode",
  "perFrameBatch",
  "perFramePipeline",
  "perFrameCache",
  "clearCacheOnExit",
  "amp",
];

const els = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const statusPill = document.getElementById("statusPill");
const progressBar = document.getElementById("progressBar");
const progressInfo = document.getElementById("progressInfo");
const logBox = document.getElementById("logBox");
const dispValue = document.getElementById("dispValue");
const baselineValue = document.getElementById("baselineValue");
const comfortLabel = document.getElementById("comfortLabel");
const videoPicker = document.getElementById("videoPicker");
const videoPickBtn = document.getElementById("videoPickBtn");
const downloadBtn = document.getElementById("downloadBtn");
const videoHint = document.getElementById("videoHint");
const outHint = document.getElementById("outHint");

const presets = {
  comfort: {
    baseline: 0.028,
    maxDisp: 24,
    fov: 60,
    minInliers: 80,
    maxReproj: 2.2,
    inpaint: 3,
  },
  depth: {
    baseline: 0.04,
    maxDisp: 40,
    fov: 58,
    minInliers: 70,
    maxReproj: 2.6,
    inpaint: 2,
  },
  stable: {
    baseline: 0.025,
    maxDisp: 20,
    fov: 62,
    minInliers: 110,
    maxReproj: 2.0,
    inpaint: 4,
  },
};

let stream = null;
let backendOk = false;
let isRunning = false;
let lastOutputName = "";

function updateComfortLabel(maxDisp) {
  if (maxDisp <= 28) return "舒适";
  if (maxDisp <= 35) return "偏强";
  return "强烈";
}

function buildPayload() {
  return {
    video: els.videoPath.value.trim(),
    out: els.outPath.value.trim(),
    debug_dir: els.debugDir.value.trim(),
    baseline_m: parseFloat(els.baseline.value),
    baseline_min_m: parseFloat(els.baselineMin.value),
    max_disp_px: parseFloat(els.maxDisp.value),
    fov_deg: parseFloat(els.fov.value),
    cut_threshold: parseFloat(els.cutThreshold.value),
    min_shot_len: parseFloat(els.minShot.value),
    max_shot_len: parseFloat(els.maxShot.value),
    min_inliers: parseFloat(els.minInliers.value),
    max_reproj: parseFloat(els.maxReproj.value),
    ffmpeg_crf: parseInt(els.crf.value, 10),
    ffmpeg_preset: els.preset.value,
    inpaint_radius: parseInt(els.inpaint.value, 10),
    copy_audio: Boolean(els.copyAudio.checked),
    audio_codec: els.audioCodec.value,
    debug_interval: parseInt(els.debugInterval.value, 10),
    max_frames: parseInt(els.maxFrames.value, 10) || 0,
    device: els.device.value,
    io_backend: els.ioBackend.value,
    decode: els.decode.value,
    encode: els.encode.value,
    track_backend: els.trackBackend.value,
    keyframe_mode: els.keyframeMode.value,
    per_frame_batch: Math.max(1, parseInt(els.perFrameBatch.value, 10) || 1),
    per_frame_pipeline: Math.max(1, parseInt(els.perFramePipeline.value, 10) || 1),
    cache_per_frame: Boolean(els.perFrameCache.checked),
    clear_cache_on_exit: Boolean(els.clearCacheOnExit.checked),
    amp: Boolean(els.amp.checked),
  };
}

function appendLog(text) {
  if (!logBox) return;
  logBox.textContent += `${text}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}

function setStatus(state) {
  if (!statusPill) return;
  statusPill.textContent = state;
}

function setProgress(percent, frame, total) {
  if (progressBar) {
    progressBar.style.width = percent >= 0 ? `${percent}%` : "0%";
  }
  if (progressInfo) {
    if (percent < 0 || !total) {
      progressInfo.textContent = "处理中...";
    } else {
      progressInfo.textContent = `进度 ${percent.toFixed(1)}% (${frame}/${total})`;
    }
  }
}

function updateUI() {
  const maxDisp = parseFloat(els.maxDisp.value);
  const baseline = parseFloat(els.baseline.value);
  dispValue.textContent = Number.isNaN(maxDisp) ? "-" : maxDisp.toFixed(0);
  baselineValue.textContent = Number.isNaN(baseline) ? "-" : baseline.toFixed(3);
  comfortLabel.textContent = updateComfortLabel(maxDisp || defaults.maxDisp);

  if (videoHint && !els.videoPath.value.trim()) {
    videoHint.textContent = "未选择文件";
  }
  if (outHint && !els.outPath.value.trim()) {
    outHint.textContent = "请输入输出文件名";
  }

  if (runBtn) {
    const canRun = Boolean(els.videoPath.value.trim() && els.outPath.value.trim());
    runBtn.disabled = !backendOk || isRunning || !canRun;
  }
  if (downloadBtn) {
    downloadBtn.disabled = !lastOutputName;
  }
}

async function startRun() {
  if (!backendOk) {
    appendLog("后端未连接，请先启动 python tools/web_server.py");
    setStatus("未连接");
    return;
  }
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;
  setStatus("启动中");
  setProgress(0, 0, 0);
  if (logBox) logBox.textContent = "";

  let data;
  try {
    const payload = buildPayload();
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    data = await res.json();
  } catch (err) {
    appendLog("无法连接后端，请先启动本地服务。");
    setStatus("未连接");
    if (runBtn) runBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    return;
  }

  if (!data.ok) {
    appendLog(data.error || "启动失败");
    setStatus("失败");
    if (runBtn) runBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    return;
  }

  setStatus("运行中");
  isRunning = true;
  lastOutputName = "";
  updateUI();

  if (stream) stream.close();
  stream = new EventSource("/api/stream");
  stream.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "log") {
      appendLog(msg.line);
    }
    if (msg.type === "progress") {
      setProgress(msg.percent, msg.frame, msg.total);
    }
    if (msg.type === "status") {
      setStatus(msg.state);
      if (msg.state !== "运行中") {
        isRunning = false;
        if (msg.state === "完成") {
          lastOutputName = els.outPath.value.trim();
        }
        if (runBtn) runBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        stream.close();
        updateUI();
      }
    }
  };
  stream.onerror = () => {
    setStatus("连接中断");
    isRunning = false;
    if (runBtn) runBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    stream.close();
    updateUI();
  };
}

async function stopRun() {
  try {
    await fetch("/api/stop", { method: "POST" });
    setStatus("停止中");
  } catch (err) {
    setStatus("未连接");
  }
}

async function pingBackend() {
  try {
    const res = await fetch("/api/ping");
    const data = await res.json();
    backendOk = Boolean(data.ok);
  } catch (err) {
    backendOk = false;
  }

  if (!backendOk) {
    setStatus("未连接");
    appendLog("未检测到后端服务，请在终端运行 python tools/web_server.py");
    if (runBtn) runBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
  } else {
    setStatus("空闲");
    updateUI();
  }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file, file.name);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && videoHint) {
        const percent = (event.loaded / event.total) * 100;
        videoHint.textContent = `上传中 ${percent.toFixed(1)}%`;
      }
    };
    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText);
        resolve(data);
      } catch (err) {
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.send(form);
  });
}

function pickInputFile() {
  if (videoPicker) {
    videoPicker.click();
  }
}

if (videoPickBtn) {
  videoPickBtn.addEventListener("click", () => pickInputFile());
}

if (videoPicker) {
  videoPicker.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    if (videoHint) {
      videoHint.textContent = `准备上传 ${file.name}`;
    }
    uploadFile(file)
      .then((data) => {
        if (!data.ok) {
          appendLog(data.error || "上传失败");
          if (videoHint) {
            videoHint.textContent = "上传失败";
          }
          return;
        }
        els.videoPath.value = data.path || "";
        if (data.suggested_output) {
          els.outPath.value = data.suggested_output;
        } else if (!els.outPath.value.trim()) {
          els.outPath.value = `output_${Date.now()}.mp4`;
        }
        if (videoHint) {
          videoHint.textContent = `已上传：${data.name || file.name}`;
        }
        updateUI();
      })
      .catch(() => {
        appendLog("上传失败，请重试。");
        if (videoHint) {
          videoHint.textContent = "上传失败";
        }
      });
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/download");
      if (!res.ok) {
        appendLog("下载失败：输出文件不存在");
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = lastOutputName || "output_sbs.mp4";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      appendLog("下载失败，请重试。");
    }
  });
}

ids.forEach((id) => {
  els[id].addEventListener("input", updateUI);
  els[id].addEventListener("change", updateUI);
});

Array.from(document.querySelectorAll(".preset")).forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = presets[btn.dataset.preset];
    if (!preset) return;
    Object.entries(preset).forEach(([key, value]) => {
      const el = els[key];
      if (el) {
        el.value = value;
      }
    });
    updateUI();
  });
});

const resetBtn = document.getElementById("resetBtn");
resetBtn.addEventListener("click", () => {
  els.videoPath.value = "";
  els.outPath.value = "";
  els.debugDir.value = "";
  Object.entries(defaults).forEach(([key, value]) => {
    if (els[key]) {
      if (els[key].type === "checkbox") {
        els[key].checked = Boolean(value);
      } else {
        els[key].value = value;
      }
    }
  });
  els.maxFrames.value = "";
  lastOutputName = "";
  updateUI();
});

if (runBtn) runBtn.addEventListener("click", () => startRun().catch(() => {}));
if (stopBtn) stopBtn.addEventListener("click", () => stopRun().catch(() => {}));

updateUI();
if (stopBtn) stopBtn.disabled = true;
if (downloadBtn) downloadBtn.disabled = true;
appendLog(`前端已加载 ${BUILD_ID}`);

pingBackend();
