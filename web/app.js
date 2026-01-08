const BUILD_ID = "2025-12-26-9";

const defaults = {
  baseline: 0.064,
  baselineMin: 0,
  maxDisp: 60,
  fov: 60,
  cutThreshold: 0.9,
  minShot: 0.5,
  maxShot: 3.0,
  keyframeRefresh: 0,
  targetFps: 24,
  minInliers: 70,
  maxReproj: 2,
  mode: "video",
  batchMode: false,
  segmentFrames: 2000,
  crf: 10,
  preset: "slow",
  inpaint: 0,
  copyAudio: true,
  audioCodec: "aac",
  debugInterval: 24,
  logInterval: 100,
  device: "cuda",
  ioBackend: "ffmpeg",
  decode: "nvdec",
  encode: "hevc_nvenc",
  trackBackend: "opencv_cuda",
  renderBackend: "cuda",
  twoPass: false,
  gpuAssist: true,
  bufferFrames: 8,
  keyframeMode: "per_frame",
  perFrameBatch: 2,
  perFramePipeline: 4,
  perFrameCache: false,
  clearCacheOnExit: true,
  amp: true,
  pinMemory: true,
  eigBackend: "cuda",
  perfInterval: 100,
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
  "keyframeRefresh",
  "targetFps",
  "minInliers",
  "maxReproj",
  "mode",
  "batchMode",
  "segmentFrames",
  "crf",
  "preset",
  "inpaint",
  "copyAudio",
  "audioCodec",
  "debugInterval",
  "logInterval",
  "maxFrames",
  "device",
  "ioBackend",
  "decode",
  "encode",
  "trackBackend",
  "renderBackend",
  "twoPass",
  "gpuAssist",
  "bufferFrames",
  "keyframeMode",
  "perFrameBatch",
  "perFramePipeline",
  "perFrameCache",
  "clearCacheOnExit",
  "amp",
  "pinMemory",
  "eigBackend",
  "perfInterval",
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
const precompileBtn = document.getElementById("precompileBtn");
const videoHint = document.getElementById("videoHint");
const outHint = document.getElementById("outHint");
const batchPanel = document.getElementById("batchPanel");
const batchExample = document.getElementById("batchExample");
const dropZone = document.getElementById("dropZone");
const dropTitle = document.getElementById("dropTitle");
const dropSub = document.getElementById("dropSub");
const thumbGrid = document.getElementById("thumbGrid");
const batchPicker = document.getElementById("batchPicker");
const modeTabs = Array.from(document.querySelectorAll(".mode-tab"));
const batchRememberOut = document.getElementById("batchRememberOut");

let stream = null;
let streamRetryTimer = null;
let backendOk = false;
let isRunning = false;
let lastOutputName = "";
let currentJob = "";
const BATCH_REMEMBER_KEY = "sbs_batch_out_remember";
const BATCH_OUT_DIR_KEY = "sbs_batch_out_dir";
const runningStates = new Set(["运行中", "预编译中"]);
let frameProgress = { percent: -1, frame: 0, total: 0 };
let batchProgress = { index: 0, total: 0 };
let activeMode = defaults.batchMode ? "batch" : defaults.mode;
let batchKind = "video";
let thumbItems = [];
let batchOrder = [];
let batchCurrentKey = "";
let batchDoneKeys = new Set();
const videoExts = new Set([".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]);
const imageExts = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]);

function scheduleStreamReconnect() {
  if (streamRetryTimer) return;
  streamRetryTimer = window.setTimeout(() => {
    streamRetryTimer = null;
    pingBackend();
  }, 1000);
}

function updateComfortLabel(maxDisp) {
  if (maxDisp <= 28) return "舒适";
  if (maxDisp <= 35) return "偏强";
  return "强烈";
}

function getStoredBatchRemember() {
  return localStorage.getItem(BATCH_REMEMBER_KEY) === "1";
}

function getStoredBatchOutDir() {
  const stored = localStorage.getItem(BATCH_OUT_DIR_KEY);
  return stored ? stored.trim() : "";
}

function clearBatchRememberStorage() {
  localStorage.removeItem(BATCH_REMEMBER_KEY);
  localStorage.removeItem(BATCH_OUT_DIR_KEY);
}

function applyBatchRememberState(isBatch) {
  if (!batchRememberOut || !els.outPath) return;
  const rememberOn = getStoredBatchRemember();
  batchRememberOut.checked = rememberOn;
  if (!isBatch || els.outPath.value.trim() || !rememberOn) return;
  const storedDir = getStoredBatchOutDir();
  if (storedDir) {
    els.outPath.value = storedDir;
  }
}

function syncBatchRememberStorage() {
  if (activeMode !== "batch") return;
  if (!batchRememberOut || !els.outPath) return;
  const outValue = els.outPath.value.trim();
  if (batchRememberOut.checked && outValue) {
    localStorage.setItem(BATCH_REMEMBER_KEY, "1");
    localStorage.setItem(BATCH_OUT_DIR_KEY, outValue);
  } else {
    clearBatchRememberStorage();
  }
}

function setMode(mode) {
  const prevMode = activeMode;
  activeMode = mode;
  if (prevMode && prevMode !== mode) {
    clearThumbs();
    resetProgress();
  }
  const isBatch = mode === "batch";
  const isImage = mode === "image";
  if (els.mode) {
    if (isBatch) {
      els.mode.value = batchKind === "image" ? "image" : "video";
    } else {
      els.mode.value = isImage ? "image" : "video";
    }
  }
  if (els.batchMode) {
    els.batchMode.checked = isBatch;
  }
  modeTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  applyBatchRememberState(isBatch);
  updateUI();
}

function getExt(name) {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

function getBasename(path) {
  if (!path) return "";
  const normalized = String(path).replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || "";
}

function isVideoFile(file) {
  return file.type.startsWith("video/") || videoExts.has(getExt(file.name));
}

function isImageFile(file) {
  return file.type.startsWith("image/") || imageExts.has(getExt(file.name));
}

function clearThumbs() {
  thumbItems = [];
  batchOrder = [];
  batchCurrentKey = "";
  batchDoneKeys = new Set();
  if (thumbGrid) thumbGrid.textContent = "";
}

function createThumbItem({ name, kind, url, placeholder = false, key = "" }) {
  if (!thumbGrid) return null;
  const card = document.createElement("div");
  card.className = "thumb-item";

  let media;
  if (placeholder) {
    media = document.createElement("div");
    media.className = "thumb-media thumb-placeholder";
    media.textContent = kind === "image" ? "图片" : "视频";
  } else if (kind === "image") {
    media = document.createElement("img");
    media.className = "thumb-media";
    media.src = url;
  } else {
    media = document.createElement("video");
    media.className = "thumb-media";
    media.src = url;
    media.muted = true;
    media.playsInline = true;
    media.preload = "metadata";
    media.addEventListener("loadedmetadata", () => {
      media.currentTime = Math.min(0.1, media.duration || 0.1);
    });
  }

  const meta = document.createElement("div");
  meta.className = "thumb-meta";
  meta.textContent = name;

  const status = document.createElement("div");
  status.className = "thumb-status";
  status.textContent = "等待";

  const bar = document.createElement("div");
  bar.className = "thumb-progress";
  const barFill = document.createElement("div");
  barFill.className = "thumb-progress-bar";
  bar.appendChild(barFill);

  card.appendChild(media);
  card.appendChild(meta);
  card.appendChild(status);
  card.appendChild(bar);
  thumbGrid.appendChild(card);

  return { name, key: key || name, kind, status, barFill };
}

function setThumbStatus(item, label, percent) {
  if (!item) return;
  item.status.textContent = label;
  if (item.barFill) {
    item.barFill.style.width = percent > 0 ? `${percent}%` : "0%";
  }
}

function renderThumbItems(items) {
  clearThumbs();
  items.forEach((item) => {
    const entry = createThumbItem(item);
    if (entry) thumbItems.push(entry);
  });
}

function ensureBatchPlaceholders(total, kind = "video") {
  if (!total || thumbItems.length) return;
  const placeholders = Array.from({ length: total }, (_, idx) => ({
    name: `视频 ${idx + 1}`,
    key: `视频 ${idx + 1}`,
    kind,
    url: "",
    placeholder: true,
  }));
  renderThumbItems(placeholders);
  batchOrder = placeholders.map((item) => item.key);
}

function seedBatchOrder(names, kind = "video") {
  if (!Array.isArray(names) || !names.length) return;
  batchOrder = names.slice();
  if (!thumbItems.length) {
    const placeholders = names.map((name) => ({
      name,
      key: name,
      kind,
      url: "",
      placeholder: true,
    }));
    renderThumbItems(placeholders);
  }
}

function applyBatchStatuses() {
  if (!thumbItems.length) return;
  const current = batchProgress.index || 0;
  const total = batchProgress.total || 0;
  if (batchCurrentKey || batchDoneKeys.size) {
    thumbItems.forEach((item) => {
      const key = item.key || item.name;
      if (batchDoneKeys.has(key)) {
        setThumbStatus(item, "完成", 100);
      } else if (key === batchCurrentKey) {
        const percent = frameProgress.percent >= 0 ? frameProgress.percent : 0;
        setThumbStatus(item, "处理中", percent);
      } else {
        setThumbStatus(item, "等待", 0);
      }
    });
    return;
  }
  thumbItems.forEach((item, idx) => {
    if (total && idx < current - 1) {
      setThumbStatus(item, "完成", 100);
    } else if (idx === current - 1) {
      const percent = frameProgress.percent >= 0 ? frameProgress.percent : 0;
      setThumbStatus(item, "处理中", percent);
    } else {
      setThumbStatus(item, "等待", 0);
    }
  });
}

function markAllDone() {
  if (!thumbItems.length) return;
  batchDoneKeys = new Set(thumbItems.map((item) => item.key || item.name));
  batchCurrentKey = "";
  thumbItems.forEach((item) => setThumbStatus(item, "完成", 100));
}

function buildPayload() {
  const batchMode = Boolean(els.batchMode?.checked);
  const outValue = els.outPath.value.trim();
  return {
    video: els.videoPath.value.trim(),
    out: outValue,
    batch_mode: batchMode,
    debug_dir: els.debugDir.value.trim(),
    baseline_m: parseFloat(els.baseline.value),
    baseline_min_m: parseFloat(els.baselineMin.value),
    max_disp_px: parseFloat(els.maxDisp.value),
    fov_deg: parseFloat(els.fov.value),
    cut_threshold: parseFloat(els.cutThreshold.value),
    min_shot_len: parseFloat(els.minShot.value),
    max_shot_len: parseFloat(els.maxShot.value),
    keyframe_refresh_s: parseFloat(els.keyframeRefresh.value) || 0,
    target_fps: parseFloat(els.targetFps.value) || 0,
    min_inliers: parseFloat(els.minInliers.value),
    max_reproj: parseFloat(els.maxReproj.value),
    mode: els.mode.value,
    segment_frames: parseInt(els.segmentFrames.value, 10) || 0,
    ffmpeg_crf: parseInt(els.crf.value, 10),
    ffmpeg_preset: els.preset.value,
    inpaint_radius: parseInt(els.inpaint.value, 10),
    copy_audio: Boolean(els.copyAudio.checked),
    audio_codec: els.audioCodec.value,
    debug_interval: parseInt(els.debugInterval.value, 10),
    log_interval: Math.max(1, parseInt(els.logInterval.value, 10) || 1),
    max_frames: parseInt(els.maxFrames.value, 10) || 0,
    device: els.device.value,
    io_backend: els.ioBackend.value,
    decode: els.decode.value,
    encode: els.encode.value,
    track_backend: els.trackBackend.value,
    render_backend: els.renderBackend.value,
    two_pass: Boolean(els.twoPass.checked),
    gpu_assist: Boolean(els.gpuAssist.checked),
    buffer_frames: parseInt(els.bufferFrames.value, 10) || 0,
    keyframe_mode: els.keyframeMode.value,
    per_frame_batch: Math.max(1, parseInt(els.perFrameBatch.value, 10) || 1),
    per_frame_pipeline: Math.max(1, parseInt(els.perFramePipeline.value, 10) || 1),
    cache_per_frame: Boolean(els.perFrameCache.checked),
    clear_cache_on_exit: Boolean(els.clearCacheOnExit.checked),
    amp: Boolean(els.amp.checked),
    pin_memory: Boolean(els.pinMemory.checked),
    eig_backend: els.eigBackend.value,
    perf_interval: Math.max(0, parseInt(els.perfInterval.value, 10) || 0),
  };
}


function buildPrecompilePayload() {
  return {
    device: els.device.value,
    amp: Boolean(els.amp.checked),
    pin_memory: Boolean(els.pinMemory.checked),
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

function updateProgressInfo() {
  const hasFrame = frameProgress.total > 0 && frameProgress.percent >= 0;
  const hasBatch = batchProgress.total > 0;
  let barPercent = 0;
  if (hasFrame) {
    barPercent = frameProgress.percent;
  } else if (hasBatch) {
    barPercent = (batchProgress.index / batchProgress.total) * 100;
  }
  if (progressBar) {
    progressBar.style.width = barPercent > 0 ? `${barPercent}%` : "0%";
  }
  if (!progressInfo) return;
  const parts = [];
  if (hasBatch) {
    parts.push(`批处理 ${batchProgress.index}/${batchProgress.total}`);
  }
  if (hasFrame) {
    parts.push(`帧 ${frameProgress.frame}/${frameProgress.total} (${frameProgress.percent.toFixed(1)}%)`);
  }
  progressInfo.textContent = parts.length ? parts.join(" | ") : "处理中...";
}

function setProgress(percent, frame, total) {
  frameProgress = {
    percent: Number.isFinite(percent) ? percent : -1,
    frame: frame || 0,
    total: total || 0,
  };
  updateProgressInfo();
  applyBatchStatuses();
}

function setBatchProgress(index, total, item) {
  const nextIndex = index || 0;
  if (batchProgress.index && nextIndex !== batchProgress.index) {
    frameProgress = { percent: -1, frame: 0, total: 0 };
  }
  if (item) {
    if (batchCurrentKey && batchCurrentKey !== item) {
      batchDoneKeys.add(batchCurrentKey);
    }
    batchCurrentKey = item;
  }
  batchProgress = {
    index: nextIndex,
    total: total || 0,
  };
  if (!item && batchOrder.length && nextIndex && batchDoneKeys.size === 0 && !batchCurrentKey) {
    batchDoneKeys = new Set(batchOrder.slice(0, Math.max(0, nextIndex - 1)));
    batchCurrentKey = batchOrder[nextIndex - 1] || "";
  }
  if (batchProgress.total) {
    const kind = batchKind === "image" ? "image" : "video";
    ensureBatchPlaceholders(batchProgress.total, kind);
  }
  updateProgressInfo();
  applyBatchStatuses();
}

function resetProgress() {
  frameProgress = { percent: -1, frame: 0, total: 0 };
  batchProgress = { index: 0, total: 0 };
  batchOrder = [];
  batchCurrentKey = "";
  batchDoneKeys = new Set();
  updateProgressInfo();
  applyBatchStatuses();
}

function updateUI() {
  const mode = activeMode || (els.mode?.value || defaults.mode);
  const isBatch = mode === "batch";
  const isImage = mode === "image";
  const batchMode = isBatch;
  const twoPass = Boolean(els.twoPass?.checked);
  if (els.gpuAssist) {
    els.gpuAssist.disabled = !twoPass;
  }
  if (batchPanel) {
    batchPanel.hidden = !batchMode;
  }
  if (batchExample) {
    const ext = batchMode ? (batchKind === "image" ? "png" : "mp4") : (isImage ? "png" : "mp4");
    batchExample.textContent = `示例：clip.${ext} → clip_sbs.${ext}`;
  }
  const maxDisp = parseFloat(els.maxDisp.value);
  const baseline = parseFloat(els.baseline.value);
  dispValue.textContent = Number.isNaN(maxDisp) ? "-" : maxDisp.toFixed(0);
  baselineValue.textContent = Number.isNaN(baseline) ? "-" : baseline.toFixed(3);
  comfortLabel.textContent = updateComfortLabel(maxDisp || defaults.maxDisp);

  if (videoPicker) {
    videoPicker.accept = isImage ? "image/*" : "video/*";
  }
  if (batchPicker) {
    batchPicker.accept = batchMode ? "video/*,image/*" : "video/*";
  }
  if (videoPickBtn) {
    videoPickBtn.disabled = false;
    if (batchMode) {
      videoPickBtn.textContent = "选择多个视频/图片";
    } else {
      videoPickBtn.textContent = isImage ? "选择图片" : "选择视频";
    }
  }
  if (els.outPath) {
    els.outPath.disabled = false;
    els.outPath.placeholder = isImage ? "output_sbs.png" : "output_sbs.mp4";
  }

  if (videoHint) {
    if (batchMode) {
      videoHint.textContent = "批处理：可拖拽多个视频/图片或填写目录路径";
    } else if (!els.videoPath.value.trim()) {
      videoHint.textContent = isImage ? "支持拖拽图片或手动填写路径" : "支持拖拽视频或手动填写路径";
    } else {
      videoHint.textContent = isImage ? "已选择图片输入" : "已选择视频输入";
    }
  }
  if (outHint) {
    if (batchMode) {
      outHint.textContent = "批处理输出目录，可留空";
    } else if (!els.outPath.value.trim()) {
      outHint.textContent = isImage ? "可填写 PNG 文件名或输出目录" : "可填写 MP4 文件名或输出目录";
    } else {
      outHint.textContent = "已指定输出路径";
    }
  }

  if (runBtn) {
    const canRun = Boolean(els.videoPath.value.trim()) && (batchMode || Boolean(els.outPath.value.trim()));
    runBtn.disabled = !backendOk || isRunning || !canRun;
  }
  if (downloadBtn) {
    downloadBtn.disabled = batchMode || !lastOutputName;
  }
  if (precompileBtn) {
    precompileBtn.disabled = !backendOk || isRunning;
  }

  if (dropTitle && dropSub) {
    if (batchMode) {
      dropTitle.textContent = "拖拽多个视频或图片到这里";
      dropSub.textContent = "或点击选择多个视频/图片";
    } else if (isImage) {
      dropTitle.textContent = "拖拽图片到这里";
      dropSub.textContent = "或点击选择图片";
    } else {
      dropTitle.textContent = "拖拽视频到这里";
      dropSub.textContent = "或点击选择视频";
    }
  }
}

function buildBatchPreviewItems(files, serverItems, kind) {
  const ordered = serverItems.length
    ? [...serverItems].sort((a, b) => {
        const left = a.filename || a.name || "";
        const right = b.filename || b.name || "";
        return left.localeCompare(right);
      })
    : files.map((file) => ({ name: file.name, filename: file.name }));
  const map = new Map();
  files.forEach((file) => {
    const list = map.get(file.name) || [];
    list.push(file);
    map.set(file.name, list);
  });
  const mediaKind = kind || "video";
  return ordered.map((item) => {
    const list = map.get(item.name) || [];
    const file = list.shift();
    const key = item.filename || item.name || "";
    if (file) {
      return {
        name: item.name || item.filename,
        key,
        kind: mediaKind,
        url: URL.createObjectURL(file),
        placeholder: false,
      };
    }
    return {
      name: item.name || item.filename,
      key,
      kind: mediaKind,
      url: "",
      placeholder: true,
    };
  });
}

async function handleSingleFile(file) {
  const isImage = activeMode === "image";
  if (isImage && !isImageFile(file)) {
    appendLog("请选择图片文件。");
    return;
  }
  if (!isImage && !isVideoFile(file)) {
    appendLog("请选择视频文件。");
    return;
  }
  renderThumbItems([
    {
      name: file.name,
      kind: isImage ? "image" : "video",
      url: URL.createObjectURL(file),
    },
  ]);
  if (videoHint) {
    videoHint.textContent = `准备上传 ${file.name}`;
  }
  try {
    const data = await uploadFile(file);
    if (!data.ok) {
      appendLog(data.error || "上传失败");
      return;
    }
    els.videoPath.value = data.path || "";
    if (!els.batchMode.checked) {
      if (data.suggested_output) {
        els.outPath.value = data.suggested_output;
      } else if (!els.outPath.value.trim()) {
        els.outPath.value = `output_${Date.now()}.${isImage ? "png" : "mp4"}`;
      }
    }
    if (videoHint) {
      videoHint.textContent = `已上传：${data.name || file.name}`;
    }
    updateUI();
  } catch (err) {
    appendLog("上传失败，请重试。");
    if (videoHint) {
      videoHint.textContent = "上传失败";
    }
  }
}

async function uploadBatch(files) {
  const form = new FormData();
  files.forEach((file) => {
    form.append("files", file, file.name);
  });
  const res = await fetch("/api/upload_batch", { method: "POST", body: form });
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!res.ok) {
    if (res.status === 405) {
      return { ok: false, error: "后端未更新，请重启 tools/web_server.py" };
    }
    return { ok: false, error: data?.error || `批处理上传失败 (${res.status})` };
  }
  return data || { ok: false, error: "批处理上传失败" };
}

async function handleBatchFiles(files) {
  const videoFiles = files.filter(isVideoFile);
  const imageFiles = files.filter(isImageFile);
  if (videoFiles.length && imageFiles.length) {
    appendLog("批处理请不要混合视频和图片。");
    return;
  }
  const batchFiles = videoFiles.length ? videoFiles : imageFiles;
  const kind = videoFiles.length ? "video" : "image";
  if (!batchFiles.length) {
    appendLog("批处理仅支持视频或图片文件。");
    return;
  }
  batchKind = kind;
  if (els.mode) {
    els.mode.value = kind === "image" ? "image" : "video";
  }
  if (videoHint) {
    videoHint.textContent = `准备上传 ${batchFiles.length} 个${kind === "image" ? "图片" : "视频"}`;
  }
  try {
    const data = await uploadBatch(batchFiles);
    if (!data.ok) {
      appendLog(data.error || "批处理上传失败");
      return;
    }
    const serverItems = Array.isArray(data.items) ? data.items : [];
    const previewItems = buildBatchPreviewItems(batchFiles, serverItems, kind);
    renderThumbItems(previewItems);
    batchOrder = previewItems.map((item) => item.key || item.name);
    batchCurrentKey = "";
    batchDoneKeys = new Set();
    els.videoPath.value = data.path || "";
    setBatchProgress(0, previewItems.length);
    if (videoHint) {
      videoHint.textContent = `已上传 ${previewItems.length} 个${kind === "image" ? "图片" : "视频"}`;
    }
    updateUI();
  } catch (err) {
    appendLog("批处理上传失败，请重试。");
    if (videoHint) {
      videoHint.textContent = "批处理上传失败";
    }
  }
}

function handleFiles(files) {
  if (!files || !files.length) return;
  if (activeMode === "batch") {
    handleBatchFiles(files);
  } else {
    handleSingleFile(files[0]);
  }
}

async function refreshLastOutput() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data && data.last_output) {
      lastOutputName = getBasename(data.last_output);
      updateUI();
    }
  } catch (err) {
    // ignore
  }
}

function startStreamListener() {
  if (stream) stream.close();
  if (streamRetryTimer) {
    window.clearTimeout(streamRetryTimer);
    streamRetryTimer = null;
  }
  stream = new EventSource("/api/stream");
  stream.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "log") {
      appendLog(msg.line);
    }
    if (msg.type === "batch") {
      setBatchProgress(msg.index, msg.total, msg.item);
    }
    if (msg.type === "progress") {
      setProgress(msg.percent, msg.frame, msg.total);
    }
    if (msg.type === "status") {
      setStatus(msg.state);
      if (!runningStates.has(msg.state)) {
        isRunning = false;
        if (msg.state === "完成" && currentJob === "run") {
          lastOutputName = els.outPath.value.trim();
          if (!lastOutputName) {
            refreshLastOutput();
          }
        }
        if (msg.state === "完成") {
          markAllDone();
        }
        currentJob = "";
        if (runBtn) runBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        stream.close();
        updateUI();
      }
    }
  };
  stream.onerror = () => {
    if (stream) stream.close();
    if (!isRunning && !currentJob) {
      setStatus("空闲");
      if (runBtn) runBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      updateUI();
      return;
    }
    setStatus("连接中断");
    scheduleStreamReconnect();
  };
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
  resetProgress();
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
  currentJob = "run";
  lastOutputName = "";
  updateUI();

  startStreamListener();
}


async function startPrecompile() {
  if (!backendOk) {
    appendLog("后端未连接，请先启动 python tools/web_server.py");
    setStatus("未连接");
    return;
  }
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;
  setStatus("预编译中");
  resetProgress();
  if (logBox) logBox.textContent = "";

  let data;
  try {
    const payload = buildPrecompilePayload();
    const res = await fetch("/api/precompile", {
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
    appendLog(data.error || "预编译启动失败");
    setStatus("失败");
    if (runBtn) runBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    return;
  }

  isRunning = true;
  currentJob = "precompile";
  lastOutputName = "";
  updateUI();

  startStreamListener();
}

async function stopRun() {
  try {
    await fetch("/api/stop", { method: "POST" });
    setStatus("停止中");
  } catch (err) {
    setStatus("未连接");
  }
}

function applyStatusSnapshot(data) {
  if (!data) return;
  if (data.running) {
    if (data.batch_total && data.batch_total > 0) {
      setMode("batch");
      if (data.mode) {
        batchKind = data.mode === "image" ? "image" : "video";
      }
    }
    setStatus(data.status || "运行中");
    isRunning = true;
    currentJob = data.job || (data.status === "预编译中" ? "precompile" : "run");
    if (runBtn) runBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (precompileBtn) precompileBtn.disabled = true;
    setProgress(data.percent ?? -1, data.frame || 0, data.total || 0);
    if (Array.isArray(data.batch_items) && data.batch_items.length) {
      seedBatchOrder(data.batch_items, batchKind === "image" ? "image" : "video");
      if (data.batch_index) {
        const currentIdx = Math.max(0, data.batch_index - 1);
        batchDoneKeys = new Set(data.batch_items.slice(0, currentIdx));
        batchCurrentKey = data.batch_items[currentIdx] || "";
      }
    }
    if (data.batch_total) {
      setBatchProgress(data.batch_index || 0, data.batch_total || 0, data.batch_current || batchCurrentKey);
    } else {
      setBatchProgress(0, 0);
    }
    startStreamListener();
    updateUI();
    appendLog("已检测到正在运行的任务，进度已恢复。");
  } else {
    isRunning = false;
    currentJob = "";
    if (data.last_output) {
      lastOutputName = getBasename(data.last_output);
    } else {
      lastOutputName = "";
    }
    setStatus("空闲");
    updateUI();
  }
}

async function pingBackend() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    backendOk = Boolean(data.ok);
    if (backendOk) {
      applyStatusSnapshot(data);
    }
  } catch (err) {
    backendOk = false;
  }

  if (!backendOk) {
    setStatus("未连接");
    appendLog("未检测到后端服务，请在终端运行 python tools/web_server.py");
    if (runBtn) runBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    if (precompileBtn) precompileBtn.disabled = true;
  } else if (!isRunning) {
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
  if (activeMode === "batch") {
    if (batchPicker) batchPicker.click();
  } else if (videoPicker) {
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
    handleSingleFile(file);
    videoPicker.value = "";
  });
}

if (batchPicker) {
  batchPicker.addEventListener("change", (event) => {
    const files = event.target.files ? Array.from(event.target.files) : [];
    handleBatchFiles(files);
    batchPicker.value = "";
  });
}

if (dropZone) {
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const files = event.dataTransfer ? Array.from(event.dataTransfer.files) : [];
    handleFiles(files);
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

if (els.outPath) {
  els.outPath.addEventListener("input", syncBatchRememberStorage);
  els.outPath.addEventListener("change", syncBatchRememberStorage);
}

if (batchRememberOut) {
  batchRememberOut.addEventListener("change", syncBatchRememberStorage);
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setMode(tab.dataset.mode || "video");
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
  syncBatchRememberStorage();
});

if (runBtn) runBtn.addEventListener("click", () => startRun().catch(() => {}));
if (stopBtn) stopBtn.addEventListener("click", () => stopRun().catch(() => {}));
if (precompileBtn) precompileBtn.addEventListener("click", () => startPrecompile().catch(() => {}));

setMode(activeMode);
if (stopBtn) stopBtn.disabled = true;
if (downloadBtn) downloadBtn.disabled = true;
appendLog(`前端已加载 ${BUILD_ID}`);
pingBackend();
