import json
import os
import hashlib
import logging
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
LOG_PATH = ROOT / "web_server.log"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
QUEUE_MAX = 2000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger("web_server")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
_CUDA_BIN = _CUDA_ROOT / "bin" / "x64"
if not _CUDA_BIN.exists():
    _CUDA_BIN = _CUDA_ROOT / "bin"
if _CUDA_BIN.exists():
    os.environ.setdefault("CUDA_PATH", str(_CUDA_ROOT))
    os.environ.setdefault("CUDA_HOME", str(_CUDA_ROOT))
    os.environ.setdefault("CUDA_ROOT", str(_CUDA_ROOT))
    os.environ["PATH"] = f"{_CUDA_BIN};{os.environ.get('PATH', '')}"
    os.environ.setdefault("GSPLAT_CUDA_HOME", str(_CUDA_ROOT))

_MSVC_ROOT = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
if _MSVC_ROOT.exists():
    cl_candidates = list(_MSVC_ROOT.glob("**/bin/Hostx64/x64/cl.exe"))
    if cl_candidates:
        cl_bin = cl_candidates[0].parent
        os.environ["PATH"] = f"{cl_bin};{os.environ.get('PATH', '')}"

def _add_dll_dir(path: Path) -> None:
    try:
        os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        return

_OPENCV_BIN = Path(os.environ.get("OPENCV_BIN", ""))
if not _OPENCV_BIN.exists():
    candidate = Path(r"F:\build\opencv_cuda\build_sm120\install\x64\vc17\bin")
    if candidate.exists():
        _OPENCV_BIN = candidate
if _OPENCV_BIN.exists():
    os.environ["PATH"] = f"{_OPENCV_BIN};{os.environ.get('PATH', '')}"
    _add_dll_dir(_OPENCV_BIN)
if _CUDA_BIN.exists():
    _add_dll_dir(_CUDA_BIN)

import cv2

def safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    ext = Path(name).suffix
    if not cleaned:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"upload_{digest}{ext}"
    if cleaned != name:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        base, cleaned_ext = os.path.splitext(cleaned)
        if not cleaned_ext and ext:
            cleaned_ext = ext
        return f"{base}_{digest}{cleaned_ext}"
    return cleaned

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def _parse_frame_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _count_batch_inputs(input_path: Path, mode: str | None) -> int:
    if not input_path.is_dir():
        return 1
    exts = IMAGE_EXTS if mode == "image" else VIDEO_EXTS
    return sum(
        1
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in exts
    )


def _list_batch_inputs(input_path: Path, mode: str | None) -> list[str]:
    if not input_path.is_dir():
        return [input_path.name]
    exts = IMAGE_EXTS if mode == "image" else VIDEO_EXTS
    return [
        path.name
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in exts
    ]


def _probe_total_frames(video_path: Path, target_fps: float | None = None) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 0
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return 0
        stream = streams[0]
        nb_frames = stream.get("nb_frames")
        fps = _parse_frame_rate(stream.get("avg_frame_rate") or "") or _parse_frame_rate(
            stream.get("r_frame_rate") or ""
        )
        fmt = payload.get("format") or {}
        duration = float(fmt.get("duration") or 0.0)
        if target_fps and target_fps > 0:
            if duration > 0:
                return int(duration * target_fps + 0.5)
            if isinstance(nb_frames, str) and nb_frames.isdigit() and fps > 0:
                return int(int(nb_frames) * (target_fps / fps) + 0.5)
        if isinstance(nb_frames, str) and nb_frames.isdigit():
            return int(nb_frames)
        if duration > 0 and fps > 0:
            return int(duration * fps + 0.5)
    except Exception:
        return 0
    return 0


class Runner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc = None
        self.thread = None
        self.heartbeat_thread = None
        self.queue = queue.Queue(maxsize=QUEUE_MAX)
        self.total_frames = 0
        self.current_frame = 0
        self.running = False
        self.stop_requested = False
        self.last_output_path = ""
        self.start_time = 0.0
        self.last_output_time = 0.0
        self.last_heartbeat = 0.0
        self.activity_label = "运行中"
        self.status = "空闲"
        self.current_job = ""
        self.batch_index = 0
        self.batch_total = 0
        self.batch_mode = False
        self.batch_items = []
        self.batch_current = ""
        self.batch_root = None
        self.io_backend = None
        self.target_fps = 0.0
        self.mode = "video"

    def start(self, payload: dict) -> dict:
        with self.lock:
            if self.running:
                return {"ok": False, "error": "任务正在运行"}

            video = payload.get("video")
            out = payload.get("out")
            mode = payload.get("mode", "video")
            if not video:
                return {"ok": False, "error": "请先选择输入路径"}

            video_path = Path(video)
            if not video_path.exists():
                return {"ok": False, "error": "输入路径不存在"}

            is_batch = bool(payload.get("batch_mode")) or video_path.is_dir()
            if mode == "image" or is_batch:
                self.total_frames = 0
            else:
                self.total_frames = self._get_total_frames(
                    video_path,
                    payload.get("io_backend"),
                    float(payload.get("target_fps") or 0),
                )
            if is_batch:
                self.batch_total = _count_batch_inputs(video_path, mode)
                self.batch_items = _list_batch_inputs(video_path, mode)
                self.batch_root = video_path if video_path.is_dir() else None
            else:
                self.batch_total = 0
                self.batch_items = []
                self.batch_root = None
            self.current_frame = 0
            self.queue = queue.Queue(maxsize=QUEUE_MAX)
            self.running = True
            self.stop_requested = False
            self.start_time = time.time()
            self.last_output_time = self.start_time
            self.last_heartbeat = 0.0
            self.activity_label = "运行中"
            self.status = "运行中"
            self.current_job = "run"
            self.batch_mode = is_batch
            self.io_backend = payload.get("io_backend")
            self.target_fps = float(payload.get("target_fps") or 0)
            self.mode = mode

            self.batch_index = 0
            self.batch_current = ""

            if is_batch:
                output_path = self._resolve_batch_output(out, video_path)
                self.last_output_path = ""
            else:
                output_path = self._resolve_output(out, video_path, mode)
                self.last_output_path = output_path
            payload["out"] = output_path
            log_path = OUTPUTS_DIR / f"run_{int(time.time())}.log"
            payload["log_path"] = str(log_path)
            cmd = self._build_command(payload)

            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.heartbeat_thread.start()
            start_line = f"任务已启动: input={video_path} out={output_path}"
            if self.total_frames:
                start_line += f" frames={self.total_frames}"
            self._enqueue({"type": "log", "line": start_line})
            self._enqueue({"type": "status", "state": "运行中"})
            return {"ok": True}

    def start_precompile(self, payload: dict) -> dict:
        with self.lock:
            if self.running:
                return {"ok": False, "error": "任务正在运行"}

            self.total_frames = 0
            self.current_frame = 0
            self.queue = queue.Queue(maxsize=QUEUE_MAX)
            self.running = True
            self.stop_requested = False
            self.start_time = time.time()
            self.last_output_time = self.start_time
            self.last_heartbeat = 0.0
            self.last_output_path = ""
            self.activity_label = "预编译中"
            self.status = "预编译中"
            self.current_job = "precompile"
            self.batch_mode = False
            self.batch_index = 0
            self.batch_total = 0
            self.batch_items = []
            self.batch_current = ""
            self.batch_root = None

            cmd = self._build_precompile_command(payload)
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.heartbeat_thread.start()
            self._enqueue({"type": "log", "line": "预编译已启动"})
            self._enqueue({"type": "status", "state": "预编译中"})
            return {"ok": True}

    def stop(self) -> dict:
        with self.lock:
            if not self.running or self.proc is None:
                return {"ok": False, "error": "没有正在运行的任务"}
            self.stop_requested = True
            pid = self.proc.pid
            try:
                self.proc.terminate()
            except Exception:
                pass
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except Exception:
                    pass
            else:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.status = "停止中"
            return {"ok": True}

    def get_status(self) -> dict:
        with self.lock:
            running = self.running
            status = self.status or ("运行中" if running else "空闲")
            return {
                "ok": True,
                "running": running,
                "status": status,
                "job": self.current_job,
                "frame": self.current_frame,
                "total": self.total_frames,
                "percent": self._progress_percent(),
                "batch_index": self.batch_index,
                "batch_total": self.batch_total,
                "batch_items": list(self.batch_items) if self.batch_mode else [],
                "batch_current": self.batch_current,
                "batch_mode": self.batch_mode,
                "mode": self.mode,
                "last_output": self.last_output_path,
            }

    def stream(self):
        while True:
            try:
                msg = self.queue.get(timeout=1)
            except queue.Empty:
                if self.running and (time.time() - self.last_heartbeat) > 5:
                    elapsed = int(time.time() - self.start_time)
                    self.last_heartbeat = time.time()
                    label = self.activity_label or "运行中"
                    yield f"data: {json.dumps({'type': 'log', 'line': f'{label}... 已用时 {elapsed}s'}, ensure_ascii=False)}\n\n"
                else:
                    yield ": ping\n\n"
                continue

            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg.get("type") == "status" and msg.get("state") in {"完成", "失败", "已停止"}:
                break

    def _reader(self) -> None:
        assert self.proc is not None
        frame_re = re.compile(r"frame=(\d+)")
        batch_re = re.compile(r"\[(\d+)/(\d+)\]")
        batch_item_re = re.compile(r"BATCH_ITEM\s+(\d+)/(\d+)\s+(.+)$")
        for line in self.proc.stdout:
            line = line.rstrip()
            self._enqueue({"type": "log", "line": line})
            self.last_output_time = time.time()
            item_match = batch_item_re.search(line)
            if item_match:
                self.batch_index = int(item_match.group(1))
                self.batch_total = int(item_match.group(2))
                item_name = item_match.group(3).strip()
                if item_name:
                    self.batch_current = item_name
                if self.mode != "image" and self.batch_root is not None and item_name:
                    path = self.batch_root / item_name
                    if path.exists():
                        try:
                            self.total_frames = self._get_total_frames(path, self.io_backend, self.target_fps)
                            self.current_frame = 0
                            percent = self._progress_percent()
                            self._enqueue(
                                {
                                    "type": "progress",
                                    "frame": self.current_frame,
                                    "total": self.total_frames,
                                    "percent": percent,
                                }
                            )
                        except Exception:
                            pass
                self._enqueue(
                    {"type": "batch", "index": self.batch_index, "total": self.batch_total, "item": self.batch_current}
                )
                continue
            batch_match = batch_re.search(line)
            if batch_match:
                self.batch_index = int(batch_match.group(1))
                self.batch_total = int(batch_match.group(2))
                remainder = line[batch_match.end():].strip()
                input_path = remainder
                if "->" in remainder:
                    input_path = remainder.split("->", 1)[0].strip()
                item_name = ""
                path = None
                if input_path:
                    try:
                        path = Path(input_path)
                        item_name = path.name
                    except Exception:
                        path = None
                if item_name:
                    self.batch_current = item_name
                if self.mode != "image" and path is not None and path.exists():
                    try:
                        self.total_frames = self._get_total_frames(path, self.io_backend, self.target_fps)
                        self.current_frame = 0
                        percent = self._progress_percent()
                        self._enqueue(
                            {
                                "type": "progress",
                                "frame": self.current_frame,
                                "total": self.total_frames,
                                "percent": percent,
                            }
                        )
                    except Exception:
                        pass
                self._enqueue(
                    {"type": "batch", "index": self.batch_index, "total": self.batch_total, "item": self.batch_current}
                )
            match = frame_re.search(line)
            if match:
                self.current_frame = int(match.group(1))
                percent = self._progress_percent()
                self._enqueue(
                    {
                        "type": "progress",
                        "frame": self.current_frame,
                        "total": self.total_frames,
                        "percent": percent,
                    }
                )

        ret = self.proc.wait()
        with self.lock:
            self.running = False
            self.proc = None

        if self.stop_requested:
            self.status = "已停止"
            self._enqueue({"type": "status", "state": "已停止"})
        elif ret == 0:
            self.status = "完成"
            self._enqueue({"type": "status", "state": "完成"})
        else:
            self.status = "失败"
            self._enqueue({"type": "status", "state": "失败"})

    def _heartbeat(self) -> None:
        while True:
            time.sleep(2)
            if not self.running:
                return

    def _progress_percent(self) -> float:
        if not self.total_frames:
            return -1.0
        return min(100.0, (self.current_frame / max(1, self.total_frames)) * 100.0)

    def _enqueue(self, msg: dict) -> None:
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                pass

    def _build_command(self, payload: dict) -> list[str]:
        cmd = [sys.executable, "-u", "tools/video_to_sbs.py"]
        cmd += ["--video", payload["video"]]
        cmd += ["--out", payload["out"]]

        def add(flag: str, value) -> None:
            if value is None:
                return
            if isinstance(value, str) and not value:
                return
            cmd.extend([flag, str(value)])

        add("--debug_dir", payload.get("debug_dir"))
        add("--baseline_m", payload.get("baseline_m"))
        add("--baseline_min_m", payload.get("baseline_min_m"))
        add("--max_disp_px", payload.get("max_disp_px"))
        add("--fov_deg", payload.get("fov_deg"))
        add("--cut_threshold", payload.get("cut_threshold"))
        add("--min_shot_len", payload.get("min_shot_len"))
        add("--max_shot_len", payload.get("max_shot_len"))
        add("--keyframe_refresh_s", payload.get("keyframe_refresh_s"))
        add("--target_fps", payload.get("target_fps"))
        add("--min_inliers", payload.get("min_inliers"))
        add("--max_reproj", payload.get("max_reproj"))
        add("--ffmpeg_crf", payload.get("ffmpeg_crf"))
        add("--ffmpeg_preset", payload.get("ffmpeg_preset"))
        add("--inpaint_radius", payload.get("inpaint_radius"))
        add("--debug_interval", payload.get("debug_interval"))
        add("--log_interval", payload.get("log_interval"))
        add("--perf_interval", payload.get("perf_interval"))
        add("--mode", payload.get("mode"))
        add("--segment_frames", payload.get("segment_frames"))
        add("--buffer_frames", payload.get("buffer_frames"))
        if payload.get("two_pass"):
            cmd.append("--two_pass")
        gpu_assist = payload.get("gpu_assist")
        if gpu_assist is True:
            cmd.append("--gpu_assist")
        elif gpu_assist is False:
            cmd.append("--no-gpu_assist")
        if payload.get("max_frames"):
            add("--max_frames", payload.get("max_frames"))
        add("--log_path", payload.get("log_path"))
        add("--device", payload.get("device"))
        add("--io_backend", payload.get("io_backend"))
        add("--decode", payload.get("decode"))
        add("--encode", payload.get("encode"))
        add("--audio_codec", payload.get("audio_codec"))
        add("--track_backend", payload.get("track_backend"))
        add("--render_backend", payload.get("render_backend"))
        add("--keyframe_mode", payload.get("keyframe_mode"))
        add("--eig_backend", payload.get("eig_backend"))
        add("--per_frame_batch", payload.get("per_frame_batch"))
        add("--per_frame_pipeline", payload.get("per_frame_pipeline"))
        if payload.get("cache_per_frame"):
            cmd.append("--cache_per_frame")
        if payload.get("clear_cache_on_exit"):
            cmd.append("--clear_cache_on_exit")
        if payload.get("keep_segments"):
            cmd.append("--keep_segments")
        amp = payload.get("amp")
        if amp is True:
            cmd.append("--amp")
        elif amp is False:
            cmd.append("--no-amp")
        pin_memory = payload.get("pin_memory")
        if pin_memory is True:
            cmd.append("--pin_memory")
        elif pin_memory is False:
            cmd.append("--no-pin_memory")
        if payload.get("copy_audio"):
            cmd.append("--copy_audio")
        return cmd

    def _build_precompile_command(self, payload: dict) -> list[str]:
        cmd = [sys.executable, "-u", "tools/precompile.py"]

        def add(flag: str, value) -> None:
            if value is None:
                return
            if isinstance(value, str) and not value:
                return
            cmd.extend([flag, str(value)])

        add("--device", payload.get("device"))
        add("--sharp_ckpt", payload.get("sharp_ckpt"))
        amp = payload.get("amp")
        if amp is True:
            cmd.append("--amp")
        elif amp is False:
            cmd.append("--no-amp")
        pin_memory = payload.get("pin_memory")
        if pin_memory is True:
            cmd.append("--pin_memory")
        elif pin_memory is False:
            cmd.append("--no-pin_memory")
        return cmd

    def _resolve_batch_output(self, out_value: str | None, input_path: Path) -> str:
        if out_value and str(out_value).strip():
            return str(out_value)
        if input_path.is_dir():
            return str(input_path)
        return str(input_path.parent)

    def _resolve_output(self, out_value: str | None, input_path: Path, mode: str | None) -> str:
        ext = ".png" if mode == "image" else ".mp4"
        if not out_value:
            out_value = f"output_{int(time.time())}{ext}"
        out_path = Path(out_value)
        if out_path.exists() and out_path.is_dir():
            return str(out_path / f"{safe_filename(input_path.stem)}_sbs{ext}")
        if out_path.suffix:
            if mode == "image" and out_path.suffix.lower() not in IMAGE_EXTS:
                out_path = out_path.with_suffix(".png")
        else:
            out_path = out_path.with_suffix(ext)
        if out_path.is_absolute() or ":" in out_value or out_value.startswith("\\\\"):
            return str(out_path)
        return str(OUTPUTS_DIR / out_path.name)

    @staticmethod
    def _get_total_frames(video_path: Path, io_backend: str | None, target_fps: float | None) -> int:
        if io_backend == "ffmpeg":
            total = _probe_total_frames(video_path, target_fps)
            if total > 0:
                return total
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if target_fps and target_fps > 0 and total > 0:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if src_fps > 0:
                total = int(total * (target_fps / src_fps) + 0.5)
        cap.release()
        return total


runner = Runner()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/precompile", methods=["POST"])
def api_precompile():
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify(runner.start_precompile(payload))


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify(runner.start(payload))


@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify(runner.stop())


@app.route("/api/stream")
def api_stream():
    return Response(runner.stream(), mimetype="text/event-stream")

@app.route("/api/ping")
def api_ping():
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(runner.get_status())


def _pick_file(save: bool) -> tuple[str, str]:
    LOGGER.info("File picker start. save=%s", save)
    ps_script = [
        "Add-Type -AssemblyName System.Windows.Forms;",
    ]
    if save:
        ps_script.append("$dlg = New-Object System.Windows.Forms.SaveFileDialog;")
        ps_script.append("$dlg.Filter = 'MP4 (*.mp4)|*.mp4|All files (*.*)|*.*';")
        ps_script.append("$dlg.DefaultExt = 'mp4';")
    else:
        ps_script.append("$dlg = New-Object System.Windows.Forms.OpenFileDialog;")
        ps_script.append(
            "$dlg.Filter = 'Video (*.mp4;*.mov;*.mkv)|*.mp4;*.mov;*.mkv|All files (*.*)|*.*';"
        )
        ps_script.append("$dlg.Multiselect = $false;")
    ps_script.append("$null = $dlg.ShowDialog();")
    ps_script.append("if ($dlg.FileName) { Write-Output $dlg.FileName }")

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", " ".join(ps_script)],
            capture_output=True,
            text=True,
            check=False,
        )
        path = result.stdout.strip()
        LOGGER.info(
            "PS picker exit=%s stdout=%s stderr=%s",
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )
        if path:
            return path, ""
    except Exception as exc:
        LOGGER.exception("PS picker failed: %s", exc)
        return "", f"PS picker failed: {exc}"

    script = [
        "import tkinter as tk",
        "from tkinter import filedialog",
        "root = tk.Tk()",
        "root.withdraw()",
        "root.attributes('-topmost', True)",
    ]
    if save:
        script.append(
            "path = filedialog.asksaveasfilename("
            "defaultextension='.mp4', "
            "filetypes=[('MP4', '*.mp4'), ('All', '*.*')])"
        )
    else:
        script.append(
            "path = filedialog.askopenfilename("
            "filetypes=[('Video', '*.mp4;*.mov;*.mkv'), ('All', '*.*')])"
        )
    script.append("root.destroy()")
    script.append("print(path or '')")
    try:
        result = subprocess.run(
            [sys.executable, "-c", ";".join(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        path = result.stdout.strip()
        LOGGER.info(
            "Tk picker exit=%s stdout=%s stderr=%s",
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )
        if path:
            return path, ""
        return "", result.stderr.strip() or "Tk picker returned empty path."
    except Exception as exc:
        LOGGER.exception("Tk picker failed: %s", exc)
        return "", f"Tk picker failed: {exc}"


@app.route("/api/pick_input", methods=["POST"])
def api_pick_input():
    path, error = _pick_file(save=False)
    LOGGER.info("Pick input result ok=%s path=%s error=%s", bool(path), path, error)
    return jsonify({"path": path, "ok": bool(path), "error": error})


@app.route("/api/pick_output", methods=["POST"])
def api_pick_output():
    path, error = _pick_file(save=True)
    LOGGER.info("Pick output result ok=%s path=%s error=%s", bool(path), path, error)
    return jsonify({"path": path, "ok": bool(path), "error": error})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未收到文件"})
    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "error": "文件名为空"})
    name = safe_filename(file.filename)
    stamp = int(time.time())
    save_path = UPLOADS_DIR / f"{stamp}_{name}"
    file.save(save_path)
    ext = Path(name).suffix.lower()
    out_ext = ".png" if ext in IMAGE_EXTS else ".mp4"
    suggested_output = Path(name).stem + f"_sbs{out_ext}"
    LOGGER.info("Uploaded file %s -> %s", file.filename, save_path)
    return jsonify(
        {
            "ok": True,
            "path": str(save_path),
            "name": file.filename,
            "suggested_output": suggested_output,
        }
    )


@app.route("/api/upload_batch", methods=["POST"])
def api_upload_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "未收到文件"})
    stamp = int(time.time())
    batch_dir = UPLOADS_DIR / f"batch_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for file in files:
        if not file.filename:
            continue
        safe_name = safe_filename(Path(file.filename).name)
        if not safe_name:
            continue
        save_path = batch_dir / safe_name
        counter = 1
        while save_path.exists():
            save_path = batch_dir / f"{save_path.stem}_{counter}{save_path.suffix}"
            counter += 1
        file.save(save_path)
        items.append(
            {
                "name": file.filename,
                "path": str(save_path),
                "filename": save_path.name,
            }
        )
    if not items:
        return jsonify({"ok": False, "error": "没有可用文件"}), 400
    LOGGER.info("Batch uploaded %d files -> %s", len(items), batch_dir)
    return jsonify({"ok": True, "path": str(batch_dir), "items": items})


@app.route("/api/download")
def api_download():
    path = Path(runner.last_output_path) if runner.last_output_path else None
    if path is None or not path.exists():
        return ("", 404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route("/<path:path>")
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def main() -> int:
    port = int(os.environ.get("WEB_PORT", "7870"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
