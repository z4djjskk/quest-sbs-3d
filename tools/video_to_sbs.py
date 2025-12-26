import argparse
import copy
import json
import logging
import os
import queue
from collections import deque, defaultdict
import shutil
import subprocess
from functools import lru_cache
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def _add_dll_dir(path: Path) -> None:
    try:
        os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        return

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
_tmp_root = Path(r"C:\temp")
_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_tmp_root))
os.environ.setdefault("TEMP", str(_tmp_root))
os.environ.setdefault("TMPDIR", str(_tmp_root))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_tmp_root / "torch_extensions"))
_cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
_cuda_bin = _cuda_root / "bin" / "x64"
if not _cuda_bin.exists():
    _cuda_bin = _cuda_root / "bin"
if _cuda_bin.exists():
    os.environ.setdefault("CUDA_PATH", str(_cuda_root))
    os.environ.setdefault("CUDA_HOME", str(_cuda_root))
    os.environ.setdefault("CUDA_ROOT", str(_cuda_root))
    os.environ["PATH"] = f"{_cuda_bin};{os.environ.get('PATH', '')}"
    os.environ.setdefault("GSPLAT_CUDA_HOME", str(_cuda_root))
    _add_dll_dir(_cuda_bin)

_opencv_bin = Path(os.environ.get("OPENCV_BIN", ""))
if not _opencv_bin.exists():
    candidate = Path(r"F:\build\opencv_cuda\build_sm120\install\x64\vc17\bin")
    if candidate.exists():
        _opencv_bin = candidate
if _opencv_bin.exists():
    os.environ["PATH"] = f"{_opencv_bin};{os.environ.get('PATH', '')}"
    _add_dll_dir(_opencv_bin)

_msvc_root = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
if _msvc_root.exists():
    cl_candidates = list(_msvc_root.glob("**/bin/Hostx64/x64/cl.exe"))
    if cl_candidates:
        cl_bin = cl_candidates[0].parent
        os.environ["PATH"] = f"{cl_bin};{os.environ.get('PATH', '')}"

import cv2
import numpy as np
import torch
from torch.utils import cpp_extension

cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

from sbs.ffmpeg_writer import FFmpegWriter
from sbs.rendering import (
    build_intrinsics,
    clamp_baseline,
    compute_stereo_extrinsics,
    inpaint_holes,
)
from sharp.utils.gaussians import Gaussians3D, SceneMetaData
from sbs.ply_utils import save_ply_fast
from sbs.sharp_backend import SharpPredictor, SharpRenderer, load_or_predict_ply
from sbs.shots import ShotDetector
from sbs.tracking import (
    filter_fundamental,
    select_keypoints,
    smooth_pose,
    solve_pnp,
    track_keypoints,
)
from sbs.utils import (
    Timer,
    ensure_dir,
    pad_even,
    safe_filename,
    save_depth_png,
    save_json,
    setup_logging,
    video_cache_key,
)


@dataclass
class KeyframeState:
    frame_idx: int
    segment_id: int
    gaussians: object
    gaussians_cpu: object | None
    metadata: object
    renderer: SharpRenderer
    depth_map: np.ndarray
    median_depth: float
    intrinsics: np.ndarray
    k_mat: np.ndarray
    intrinsics_t: torch.Tensor
    gray: np.ndarray
    key_pts_2d: np.ndarray
    key_pts_3d: np.ndarray


@dataclass
class GaussianCacheItem:
    frame_idx: int
    width: int
    height: int
    path: Path


@dataclass
class RenderResult:
    frame_idx: int
    sbs: np.ndarray
    render_ms: float
    baseline: float
    backend: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D video to SBS 3D using SHARP keyframes")
    parser.add_argument(
        "--mode",
        default="video",
        choices=["video", "image"],
        help="Processing mode (video|image)",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input path (file or directory for batch)",
    )
    parser.add_argument("--out", required=True, help="Output file or directory")
    parser.add_argument("--cache_dir", default=".cache_sharp", help="Cache directory")
    parser.add_argument("--debug_dir", default=None, help="Optional debug output dir")
    parser.add_argument("--log_path", default=None, help="Optional log file path")

    parser.add_argument("--fov_deg", type=float, default=60.0, help="Assumed horizontal FOV")
    parser.add_argument("--focal_px", type=float, default=None, help="Override focal length in pixels")

    parser.add_argument("--target_fps", type=float, default=24.0, help="Resample input to target FPS (0=use source)")

    parser.add_argument("--baseline_m", type=float, default=0.064, help="Stereo baseline in meters")
    parser.add_argument("--baseline_min_m", type=float, default=0.0, help="Minimum baseline clamp")
    parser.add_argument("--max_disp_px", type=float, default=60.0, help="Disparity clamp in pixels")

    parser.add_argument("--cut_threshold", type=float, default=0.9, help="Histogram cut threshold")
    parser.add_argument("--min_shot_len", type=float, default=0.5, help="Min shot length seconds")
    parser.add_argument("--max_shot_len", type=float, default=3.0, help="Max shot length seconds")
    parser.add_argument("--keyframe_refresh_s", type=float, default=0.0, help="Force new keyframe every N seconds (0=disable)")

    parser.add_argument("--max_features", type=int, default=2000, help="Max keyframe features")
    parser.add_argument("--feature_quality", type=float, default=0.01, help="GFTT quality level")
    parser.add_argument("--feature_min_dist", type=int, default=7, help="GFTT min distance")

    parser.add_argument("--flow_fb_thresh", type=float, default=1.0, help="LK forward-backward threshold")
    parser.add_argument("--fundamental_thresh", type=float, default=1.0, help="F-matrix RANSAC threshold")

    parser.add_argument("--pnp_ransac_iters", type=int, default=200, help="PnP RANSAC iterations")
    parser.add_argument("--pnp_reproj", type=float, default=3.0, help="PnP reprojection error")
    parser.add_argument("--min_inliers", type=int, default=70, help="PnP min inliers")
    parser.add_argument("--max_reproj", type=float, default=2.0, help="PnP max reprojection error")
    parser.add_argument("--pose_smooth", type=float, default=0.25, help="Pose smoothing alpha")

    parser.add_argument("--depth_min", type=float, default=0.0, help="Min depth clamp")
    parser.add_argument("--depth_max", type=float, default=0.0, help="Max depth clamp")
    parser.add_argument("--depth_q_low", type=float, default=0.02, help="Depth quantile low")
    parser.add_argument("--depth_q_high", type=float, default=0.98, help="Depth quantile high")

    parser.add_argument("--ffmpeg_crf", type=int, default=0, help="FFmpeg CRF")
    parser.add_argument("--ffmpeg_preset", default="slow", help="FFmpeg preset")

    parser.add_argument("--inpaint_radius", type=int, default=2, help="Inpaint radius for holes")
    parser.add_argument("--debug_interval", type=int, default=24, help="Debug frame interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N frames (min 1)")
    parser.add_argument("--perf_interval", type=int, default=100, help="Log perf summary every N frames (0=disable)")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames for debug")
    parser.add_argument(
        "--segment_frames",
        type=int,
        default=2000,
        help="Auto split output every N frames (0=disable)",
    )
    parser.add_argument(
        "--keep_segments",
        action="store_true",
        help="Keep intermediate segment files after concat",
    )

    parser.add_argument("--sharp_ckpt", default=None, help="Path to SHARP checkpoint")
    parser.add_argument("--device", default="cuda", help="Torch device (auto|cuda|cuda:N|cpu)")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FP16 autocast for SHARP/GSplat",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned memory for H2D copies",
    )
    parser.add_argument("--eig_backend", default="cuda", choices=["cpu", "cuda"], help="Eigen decomposition backend (cpu|cuda)")
    parser.add_argument(
        "--io_backend",
        default="ffmpeg",
        choices=["opencv", "ffmpeg"],
        help="Frame IO backend",
    )
    parser.add_argument(
        "--decode",
        default="nvdec",
        choices=["cpu", "nvdec"],
        help="Decode backend when using ffmpeg IO",
    )
    parser.add_argument(
        "--encode",
        default="x264",
        choices=["x264", "nvenc", "hevc_nvenc"],
        help="Encoder for output video",
    )
    parser.add_argument(
        "--copy_audio",
        action="store_true",
        help="Copy audio stream from input video if available",
    )
    parser.add_argument(
        "--audio_codec",
        default="aac",
        choices=["copy", "aac"],
        help="Audio codec when copy_audio is enabled",
    )
    parser.add_argument(
        "--track_backend",
        default="opencv_cuda",
        choices=["cpu", "opencv_cuda"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--render_backend",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Render backend (cuda|cpu). CPU is experimental and slower.",
    )
    parser.add_argument(
        "--keyframe_mode",
        default="per_frame",
        choices=["normal", "freeze", "cache_only", "per_frame"],
        help="Keyframe mode (normal|freeze|cache_only|per_frame)",
    )
    parser.add_argument(
        "--cache_per_frame",
        action="store_true",
        help="Cache PLY for per_frame mode",
    )
    parser.add_argument(
        "--per_frame_batch",
        type=int,
        default=2,
        help="Batch size for per_frame mode",
    )
    parser.add_argument(
        "--per_frame_pipeline",
        type=int,
        default=4,
        help="Queue size for per_frame pipeline",
    )
    parser.add_argument(
        "--async_write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write frames asynchronously to overlap with GPU work",
    )
    parser.add_argument(
        "--write_queue",
        type=int,
        default=2,
        help="Queue size for async write",
    )
    parser.add_argument(
        "--two_pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Streaming two-pass: GPU caches gaussians, CPU renders/encodes",
    )
    parser.add_argument(
        "--buffer_frames",
        type=int,
        default=8,
        help="Frames to buffer before CPU starts rendering in two-pass mode",
    )
    parser.add_argument(
        "--gpu_assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow GPU assist after pass1 in two-pass mode",
    )
    parser.add_argument(
        "--eig_chunk_size",
        type=int,
        default=262144,
        help="Chunk size for covariance decomposition (0=all in one batch)",
    )
    parser.add_argument(
        "--eig_chunk_size_max",
        type=int,
        default=0,
        help="Max chunk size for covariance decomposition (0=auto)",
    )
    parser.add_argument(
        "--min_free_ram_gb",
        type=float,
        default=0.0,
        help="If >0, reduce batch/pipeline when free RAM drops below this threshold",
    )
    parser.add_argument(
        "--clear_cache_on_exit",
        action="store_true",
        help="Remove cache_dir after run finishes",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def compute_focal_px(width: int, fov_deg: float, focal_override: float | None) -> float:
    if focal_override and focal_override > 0:
        return focal_override
    fov_rad = np.deg2rad(fov_deg)
    return 0.5 * width / np.tan(0.5 * fov_rad)


def resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device(req if ":" in req else "cuda:0")
    return torch.device("cpu")


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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


def _available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
            return None
        return int(stat.ullAvailPhys)
    except Exception:
        return None


def _color_to_uint8(
    color: np.ndarray,
    scratch: np.ndarray | None = None,
    out_uint8: np.ndarray | None = None,
) -> np.ndarray:
    tmp = scratch
    if tmp is None or tmp.shape != color.shape or tmp.dtype != np.float32:
        tmp = np.empty_like(color, dtype=np.float32)
    np.multiply(color, 255.0, out=tmp)
    np.clip(tmp, 0.0, 255.0, out=tmp)
    if out_uint8 is None or out_uint8.shape != color.shape or out_uint8.dtype != np.uint8:
        out_uint8 = np.empty_like(color, dtype=np.uint8)
    out_uint8[...] = tmp
    return out_uint8


def probe_video_info_ffprobe(video_path: Path) -> tuple[int, int, float, int, str | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe did not return stream info")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_frame_rate(stream.get("avg_frame_rate") or "") or _parse_frame_rate(
        stream.get("r_frame_rate") or ""
    )
    if fps <= 0:
        fps = 30.0
    nb_frames = stream.get("nb_frames")
    total_frames = int(nb_frames) if isinstance(nb_frames, str) and nb_frames.isdigit() else 0
    if total_frames <= 0:
        fmt = payload.get("format") or {}
        duration = float(fmt.get("duration") or 0.0)
        if duration > 0:
            total_frames = int(duration * fps + 0.5)
    codec_name = stream.get("codec_name")
    return width, height, fps, total_frames, codec_name


@lru_cache(maxsize=1)
def _ffmpeg_decoders() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout or ""
    except Exception:
        return ""


def _has_decoder(name: str) -> bool:
    return name in _ffmpeg_decoders()


def _pick_cuvid_decoder(codec_name: str | None) -> str | None:
    if not codec_name:
        return None
    name = str(codec_name).lower()
    mapping = {
        "h264": "h264_cuvid",
        "avc1": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "h265": "hevc_cuvid",
        "av1": "av1_cuvid",
        "vp9": "vp9_cuvid",
    }
    candidate = mapping.get(name)
    if not candidate:
        return None
    return candidate if _has_decoder(candidate) else None


class FFmpegReader:
    def __init__(
        self,
        video_path: Path,
        width: int,
        height: int,
        use_hwaccel: bool,
        target_fps: float,
        output_rgb: bool = False,
        codec_name: str | None = None,
        reuse_buffer: bool = False,
    ) -> None:

        cmd = ["ffmpeg", "-v", "error"]
        filters = []
        pix_fmt = "rgb24" if output_rgb else "bgr24"
        if use_hwaccel:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "8"]
            decoder = _pick_cuvid_decoder(codec_name)
            if decoder:
                cmd += ["-c:v", decoder]
                logging.info("FFmpeg reader: using decoder %s", decoder)
            else:
                logging.warning("FFmpeg reader: NVDEC requested but no cuvid decoder found; using default decode.")
            filters.append("hwdownload")
            filters.append("format=nv12")
            filters.append(f"format={pix_fmt}")
        cmd += [
            "-i",
            str(video_path),
        ]
        if target_fps and target_fps > 0:
            filters.append(f"fps={target_fps}")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
        logging.info("FFmpeg reader: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.output_rgb = output_rgb
        self.reuse_buffer = bool(reuse_buffer)
        self._raw_buffer: bytearray | None = None
        self._raw_view: memoryview | None = None
        self._stderr_lines = deque(maxlen=40)
        self._stderr_thread = None
        self._start_stderr_thread()

    def _start_stderr_thread(self) -> None:
        if self.proc.stderr is None:
            return
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_lines.append(line)
                logging.warning("FFmpeg reader stderr: %s", line)
        except ValueError:
            return

    def read(self) -> np.ndarray | None:
        if self.proc.stdout is None:
            return None
        if self.reuse_buffer:
            if self._raw_buffer is None:
                self._raw_buffer = bytearray(self.frame_size)
                self._raw_view = memoryview(self._raw_buffer)
            assert self._raw_view is not None
            total = 0
            while total < self.frame_size:
                n = self.proc.stdout.readinto(self._raw_view[total:])
                if not n:
                    break
                total += n
            if total != self.frame_size:
                ret = self.proc.poll()
                if ret is not None and ret != 0:
                    if self._stderr_lines:
                        logging.error(
                            "FFmpeg reader exited with %s; last stderr: %s",
                            ret,
                            " | ".join(self._stderr_lines),
                        )
                    else:
                        logging.error("FFmpeg reader exited with %s.", ret)
                return None
            frame = np.frombuffer(self._raw_buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
            return frame

        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            ret = self.proc.poll()
            if ret is not None and ret != 0:
                if self._stderr_lines:
                    logging.error(
                        "FFmpeg reader exited with %s; last stderr: %s",
                        ret,
                        " | ".join(self._stderr_lines),
                    )
                else:
                    logging.error("FFmpeg reader exited with %s.", ret)
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            ret = self.proc.poll()
            if ret is not None and ret != 0:
                if self._stderr_lines:
                    logging.error(
                        "FFmpeg reader exited with %s; last stderr: %s",
                        ret,
                        " | ".join(self._stderr_lines),
                    )
                else:
                    logging.error("FFmpeg reader exited with %s.", ret)
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self) -> None:
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        self.proc.terminate()
        self.proc.wait()


def _run_ffmpeg(cmd: list[str], label: str) -> None:
    logging.info("%s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"{label} failed ({result.returncode}): {stderr}")


def _write_concat_list(list_path: Path, segment_paths: list[Path]) -> None:
    lines = []
    for path in segment_paths:
        try:
            rel_path = path.relative_to(list_path.parent)
        except ValueError:
            rel_path = path
        escaped = str(rel_path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines), encoding="utf-8")


class SegmentWriter:
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        encoder: str,
        audio_path: Path | None,
        audio_codec: str,
        segment_frames: int,
        keep_segments: bool,
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.encoder = encoder
        self.audio_path = audio_path if audio_path and audio_path.exists() else None
        self.audio_codec = audio_codec
        self.segment_frames = max(0, int(segment_frames))
        self.keep_segments = bool(keep_segments)
        self.writer: FFmpegWriter | None = None
        self.segment_paths: list[Path] = []
        self.segment_dir: Path | None = None
        self.segment_index = 0
        self.frames_in_segment = 0
        self.use_segments = self.segment_frames > 0

    def _open_segment(self) -> None:
        if self.use_segments:
            if self.segment_dir is None:
                self.segment_dir = self.output_path.parent / f"{self.output_path.stem}_segments"
                ensure_dir(self.segment_dir)
            segment_path = self.segment_dir / f"{self.output_path.stem}_seg{self.segment_index:04d}.mp4"
            self.writer = FFmpegWriter(
                segment_path,
                self.width,
                self.height,
                self.fps,
                self.crf,
                self.preset,
                encoder=self.encoder,
                audio_path=None,
                audio_codec=self.audio_codec,
            )
            self.segment_paths.append(segment_path)
        else:
            self.writer = FFmpegWriter(
                self.output_path,
                self.width,
                self.height,
                self.fps,
                self.crf,
                self.preset,
                encoder=self.encoder,
                audio_path=self.audio_path,
                audio_codec=self.audio_codec,
            )

    def write(self, frame_rgb: bytes) -> None:
        if self.writer is None:
            self._open_segment()
        elif self.use_segments and self.frames_in_segment >= self.segment_frames:
            self._close_segment()
            self.segment_index += 1
            self.frames_in_segment = 0
            self._open_segment()
        assert self.writer is not None
        self.writer.write(frame_rgb)
        self.frames_in_segment += 1

    def _close_segment(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None

    def close(self) -> None:
        self._close_segment()
        if not self.use_segments:
            return
        if not self.segment_paths:
            return
        assert self.segment_dir is not None
        list_path = self.segment_dir / "concat.txt"
        concat_path = self.segment_dir / f"{self.output_path.stem}_concat.mp4"
        _write_concat_list(list_path, self.segment_paths)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(concat_path),
            ],
            "FFmpeg concat",
        )
        if self.audio_path:
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(concat_path),
                    "-i",
                    str(self.audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    self.audio_codec,
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(self.output_path),
                ],
                "FFmpeg mux audio",
            )
        else:
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(concat_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(self.output_path),
                ],
                "FFmpeg finalize",
            )
        if not self.keep_segments:
            for path in self.segment_paths:
                path.unlink(missing_ok=True)
            list_path.unlink(missing_ok=True)
            concat_path.unlink(missing_ok=True)
            try:
                self.segment_dir.rmdir()
            except OSError:
                pass


def build_3d_points(
    pts_2d: np.ndarray,
    depth_map: np.ndarray,
    k_mat: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pts_2d) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    fx = k_mat[0, 0]
    fy = k_mat[1, 1]
    cx = k_mat[0, 2]
    cy = k_mat[1, 2]

    xs = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, depth_map.shape[1] - 1)
    ys = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, depth_map.shape[0] - 1)
    zs = depth_map[ys, xs]

    valid = np.isfinite(zs) & (zs > 0)
    if depth_min > 0:
        valid &= zs >= depth_min
    if depth_max > 0:
        valid &= zs <= depth_max

    pts_2d = pts_2d[valid]
    zs = zs[valid]
    if len(pts_2d) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    xs = (pts_2d[:, 0] - cx) * zs / fx
    ys = (pts_2d[:, 1] - cy) * zs / fy
    pts_3d = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    return pts_2d.astype(np.float32), pts_3d


def compute_depth_range(depth_map: np.ndarray, q_low: float, q_high: float) -> tuple[float, float, float]:
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    depth_vals = depth_map[valid].astype(np.float32, copy=False)
    count = depth_vals.size
    if count == 0:
        return 0.0, 0.0, 0.0
    if count == 1:
        val = float(depth_vals[0])
        return val, val, val

    q_list = [q_low, q_high, 0.5]
    positions = []
    indices = set()
    for q in q_list:
        q_clamped = float(min(max(q, 0.0), 1.0))
        pos = (count - 1) * q_clamped
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        positions.append((pos, lo, hi))
        indices.add(lo)
        indices.add(hi)

    sorted_indices = sorted(indices)
    depth_vals.partition(sorted_indices)

    def _interp(pos: float, lo: int, hi: int) -> float:
        if lo == hi:
            return float(depth_vals[lo])
        lo_val = depth_vals[lo]
        hi_val = depth_vals[hi]
        return float(lo_val + (hi_val - lo_val) * (pos - lo))

    low = _interp(*positions[0])
    high = _interp(*positions[1])
    median = _interp(*positions[2])
    return low, high, median


def _cleanup_cache_dir(cache_dir: Path) -> None:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return
    try:
        resolved = cache_dir.resolve()
    except OSError:
        resolved = cache_dir
    allow = False
    try:
        resolved.relative_to(ROOT_DIR)
        allow = True
    except ValueError:
        pass
    if not allow and resolved.name != ".cache_sharp":
        logging.warning("Skip cache cleanup for non-default path: %s", cache_dir)
        return
    logging.info("Clearing cache_dir: %s", resolved)
    shutil.rmtree(resolved, ignore_errors=True)


def build_keyframe_state(
    frame_bgr: np.ndarray,
    frame_idx: int,
    segment_id: int,
    gaussians,
    metadata,
    device: torch.device,
    args: argparse.Namespace,
    debug_root: Path | None,
    timer: Timer | None = None,
    log_prefix: str = "Keyframe",
    skip_keypoints: bool = False,
    renderer: SharpRenderer | None = None,
    intrinsics: np.ndarray | None = None,
    intrinsics_t: torch.Tensor | None = None,
    frame_is_rgb: bool = False,
) -> KeyframeState:
    if timer is None:
        timer = Timer()
    gaussians = gaussians.to(device)
    gaussians_cpu = None
    if args.render_backend == "cpu":
        gaussians_cpu = gaussians.to(torch.device("cpu"))
    if renderer is None:
        renderer = SharpRenderer(metadata.color_space, use_amp=args.amp)

    if intrinsics is None:
        intrinsics = build_intrinsics(
            metadata.focal_length_px,
            metadata.resolution_px[0],
            metadata.resolution_px[1],
        )
    if intrinsics_t is None:
        intrinsics_t = (
            torch.from_numpy(intrinsics)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
        )
    extrinsics = np.eye(4, dtype=np.float32)
    if args.render_backend == "cpu":
        gaussians_for_depth = gaussians_cpu or gaussians.to(torch.device("cpu"))
        _, depth, _ = render_batch_cpu(
            gaussians_for_depth,
            intrinsics,
            extrinsics[None, ...],
            metadata.resolution_px[0],
            metadata.resolution_px[1],
            need_depth=True,
            need_alpha=False,
        )
        depth = depth[0] if depth is not None else np.zeros(
            (metadata.resolution_px[1], metadata.resolution_px[0]),
            dtype=np.float32,
        )
    else:
        _, depth, _ = renderer.render(
            gaussians,
            intrinsics,
            extrinsics,
            metadata.resolution_px[0],
            metadata.resolution_px[1],
            device,
            need_alpha=False,
        )

    low, high, median = compute_depth_range(depth, args.depth_q_low, args.depth_q_high)
    depth_min = max(low, args.depth_min) if low > 0 else args.depth_min
    depth_max = min(high, args.depth_max) if args.depth_max > 0 and high > 0 else (high if high > 0 else args.depth_max)

    if skip_keypoints:
        gray = np.empty((0, 0), dtype=np.uint8)
        key_pts_2d = np.empty((0, 2), dtype=np.float32)
        key_pts_3d = np.empty((0, 3), dtype=np.float32)
    else:
        code = cv2.COLOR_RGB2GRAY if frame_is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame_bgr, code)
        key_pts_2d = select_keypoints(
            gray,
            args.max_features,
            args.feature_quality,
            args.feature_min_dist,
            backend=args.track_backend,
        )
        key_pts_2d, key_pts_3d = build_3d_points(
            key_pts_2d, depth, intrinsics[:3, :3], depth_min, depth_max
        )

    if debug_root:
        seg_dir = debug_root / f"segment_{segment_id:03d}"
        ensure_dir(seg_dir)
        debug_frame = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            if frame_is_rgb
            else frame_bgr
        )
        cv2.imwrite(str(seg_dir / f"keyframe_{frame_idx:06d}.png"), debug_frame)
        save_depth_png(seg_dir / f"keyframe_{frame_idx:06d}.depth.png", depth)

    logging.info(
        "%s build done: seg=%d frame=%d pts=%d depth=%.3f ms=%.1f",
        log_prefix,
        segment_id,
        frame_idx,
        len(key_pts_2d),
        median,
        timer.elapsed_ms(),
    )

    return KeyframeState(
        frame_idx=frame_idx,
        segment_id=segment_id,
        gaussians=gaussians,
        gaussians_cpu=gaussians_cpu,
        metadata=metadata,
        renderer=renderer,
        depth_map=depth,
        median_depth=median,
        intrinsics=intrinsics,
        k_mat=intrinsics[:3, :3],
        intrinsics_t=intrinsics_t,
        gray=gray,
        key_pts_2d=key_pts_2d,
        key_pts_3d=key_pts_3d,
    )


def split_gaussians_batch(gaussians: Gaussians3D) -> list[Gaussians3D]:
    batch = gaussians.mean_vectors.shape[0]
    items = []
    for i in range(batch):
        items.append(
            Gaussians3D(
                mean_vectors=gaussians.mean_vectors[i : i + 1],
                singular_values=gaussians.singular_values[i : i + 1],
                quaternions=gaussians.quaternions[i : i + 1],
                colors=gaussians.colors[i : i + 1],
                opacities=gaussians.opacities[i : i + 1],
            )
        )
    return items


def _gaussians_to_cpu(gaussians: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors.detach().cpu(),
        singular_values=gaussians.singular_values.detach().cpu(),
        quaternions=gaussians.quaternions.detach().cpu(),
        colors=gaussians.colors.detach().cpu(),
        opacities=gaussians.opacities.detach().cpu(),
    )


def read_next_frame(cap: cv2.VideoCapture | None, reader) -> np.ndarray | None:
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    if reader is None:
        return None
    return reader.read()


def write_sbs_frame(writer, sbs: np.ndarray) -> bool:
    try:
        if not sbs.flags["C_CONTIGUOUS"]:
            sbs = np.ascontiguousarray(sbs)
        writer.write(memoryview(sbs))
        return True
    except Exception as exc:
        logging.error("FFmpeg write failed: %s", exc)
        return False



class AsyncSBSWriter:
    def __init__(self, writer: SegmentWriter, queue_size: int) -> None:
        self.writer = writer
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, sbs: np.ndarray) -> bool:
        if self.error is not None:
            return False
        while True:
            try:
                self.queue.put(sbs, timeout=0.1)
                return True
            except queue.Full:
                if self.error is not None:
                    return False
                continue

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                break
            if not write_sbs_frame(self.writer, item):
                self.error = RuntimeError("FFmpeg write failed")
                break

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=1)
        except queue.Full:
            pass
        self.thread.join(timeout=5)
        if self.error is not None:
            raise self.error
        self.writer.close()


def submit_sbs_frame(
    writer: SegmentWriter,
    writer_worker: AsyncSBSWriter | None,
    sbs: np.ndarray,
) -> bool:
    if writer_worker is None:
        return write_sbs_frame(writer, sbs)
    return writer_worker.submit(sbs.copy())


def should_async_write(width: int, height: int, args: argparse.Namespace) -> bool:
    if not args.async_write:
        return False
    frame_bytes = width * height * 2 * 3
    threshold_mb = int(os.environ.get("SBS_ASYNC_MAX_MB", "32"))
    return frame_bytes <= threshold_mb * 1024 * 1024

def build_sbs(left_img: np.ndarray, right_img: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    height, width, channels = left_img.shape
    if (
        out is None
        or out.shape != (height, width * 2, channels)
        or out.dtype != left_img.dtype
    ):
        out = np.empty((height, width * 2, channels), dtype=left_img.dtype)
    out[:, :width] = left_img
    out[:, width:] = right_img
    return out



_CPU_KERNEL = np.array(
    [
        [0.07511361, 0.1238414, 0.07511361],
        [0.1238414, 0.20417996, 0.1238414],
        [0.07511361, 0.1238414, 0.07511361],
    ],
    dtype=np.float32,
)


def _linear_to_srgb_np(values: np.ndarray) -> np.ndarray:
    threshold = 0.0031308
    below = values <= threshold
    out = np.empty_like(values, dtype=np.float32)
    out[below] = values[below] * 12.92
    above = ~below
    out[above] = 1.055 * np.power(values[above], 1.0 / 2.4) - 0.055
    return out


def _gaussians_to_numpy(gaussians: Gaussians3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = gaussians.mean_vectors.detach().cpu().numpy()
    colors = gaussians.colors.detach().cpu().numpy()
    opacities = gaussians.opacities.detach().cpu().numpy()
    if means.ndim == 3:
        means = means[0]
    if colors.ndim == 3:
        colors = colors[0]
    if opacities.ndim == 2:
        opacities = opacities[0]
    return means.astype(np.float32), colors.astype(np.float32), opacities.astype(np.float32)


def _render_eye_cpu(
    means: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    need_depth: bool,
    need_alpha: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    rmat = extrinsics[:3, :3]
    tvec = extrinsics[:3, 3]
    pts_cam = (means @ rmat.T) + tvec
    zs = pts_cam[:, 2]
    valid = zs > 1e-6
    if not np.any(valid):
        color = np.zeros((height, width, 3), dtype=np.float32)
        depth = np.zeros((height, width), dtype=np.float32) if need_depth else None
        alpha = np.zeros((height, width), dtype=np.float32) if need_alpha else None
        return color, depth, alpha
    pts_cam = pts_cam[valid]
    zs = zs[valid]
    cols = colors[valid]
    ops = opacities[valid]
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    u0 = np.rint(us).astype(np.int32)
    v0 = np.rint(vs).astype(np.int32)
    color_acc = np.zeros((height, width, 3), dtype=np.float32)
    alpha_acc = np.zeros((height, width), dtype=np.float32)
    depth_acc = np.zeros((height, width), dtype=np.float32) if need_depth else None

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            weight = _CPU_KERNEL[dy + 1, dx + 1]
            xs = u0 + dx
            ys = v0 + dy
            in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not np.any(in_bounds):
                continue
            xs = xs[in_bounds]
            ys = ys[in_bounds]
            w = ops[in_bounds] * weight
            np.add.at(alpha_acc, (ys, xs), w)
            for c in range(3):
                np.add.at(color_acc[..., c], (ys, xs), w * cols[in_bounds, c])
            if need_depth and depth_acc is not None:
                np.add.at(depth_acc, (ys, xs), w * zs[in_bounds])

    if need_alpha:
        alpha = np.clip(alpha_acc, 0.0, 1.0)
    else:
        alpha = None
    denom = np.maximum(alpha_acc[..., None], 1e-6)
    color = color_acc / denom
    color = np.clip(color, 0.0, 1.0)
    color = _linear_to_srgb_np(color)
    if need_depth and depth_acc is not None:
        depth = depth_acc / np.maximum(alpha_acc, 1e-6)
    else:
        depth = None
    return color, depth, alpha


def render_batch_cpu(
    gaussians: Gaussians3D,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    need_depth: bool,
    need_alpha: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    means, colors, opacities = _gaussians_to_numpy(gaussians)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError("extrinsics must be shaped as (N, 4, 4)")
    outputs = []
    depth_list = [] if need_depth else None
    alpha_list = [] if need_alpha else None
    for idx in range(extrinsics.shape[0]):
        color, depth, alpha = _render_eye_cpu(
            means,
            colors,
            opacities,
            intrinsics,
            extrinsics[idx],
            width,
            height,
            need_depth=need_depth,
            need_alpha=need_alpha,
        )
        outputs.append(color)
        if need_depth and depth_list is not None:
            depth_list.append(depth)
        if need_alpha and alpha_list is not None:
            alpha_list.append(alpha)
    colors_out = np.stack(outputs, axis=0)
    depths_out = np.stack(depth_list, axis=0) if need_depth and depth_list is not None else None
    alphas_out = np.stack(alpha_list, axis=0) if need_alpha and alpha_list is not None else None
    return colors_out, depths_out, alphas_out


def render_batch_dispatch(
    args: argparse.Namespace,
    keyframe: KeyframeState,
    gaussians: Gaussians3D,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
    intrinsics_t: torch.Tensor | None = None,
    need_depth: bool = True,
    need_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if args.render_backend == "cpu":
        gaussians_cpu = keyframe.gaussians_cpu
        if gaussians_cpu is None:
            gaussians_cpu = gaussians.to(torch.device("cpu"))
        return render_batch_cpu(
            gaussians_cpu,
            intrinsics,
            extrinsics,
            width,
            height,
            need_depth=need_depth,
            need_alpha=need_alpha,
        )
    return keyframe.renderer.render_batch(
        gaussians,
        intrinsics,
        extrinsics,
        width,
        height,
        device,
        intrinsics_t=intrinsics_t,
        need_depth=need_depth,
        need_alpha=need_alpha,
    )


class PerfStats:
    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))
        self.count = 0
        self.totals = defaultdict(float)
        self.window_start = time.perf_counter()

    def add(self, key: str, ms: float) -> None:
        self.totals[key] += ms

    def tick(self) -> None:
        self.count += 1
        if self.count % self.interval != 0:
            return
        elapsed_ms = (time.perf_counter() - self.window_start) * 1000.0
        pieces = []
        for key in sorted(self.totals.keys()):
            avg = self.totals[key] / max(1, self.interval)
            pieces.append(f"{key}={avg:.1f}ms")
        fps = (self.interval * 1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0
        logging.info("Perf avg: fps=%.2f %s", fps, " ".join(pieces))
        self.totals.clear()
        self.window_start = time.perf_counter()


def create_keyframe(
    frame_bgr: np.ndarray,
    frame_idx: int,
    segment_id: int,
    predictor: SharpPredictor,
    device: torch.device,
    cache_dir: Path,
    f_px: float,
    args: argparse.Namespace,
    debug_root: Path | None,
    allow_predict: bool = True,
    use_cache: bool = True,
    skip_keypoints: bool = False,
) -> KeyframeState | None:
    try:
        timer = Timer()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ply_path = cache_dir / f"seg{segment_id:03d}_f{frame_idx:06d}.ply"
        cached = ply_path.exists() if use_cache else False
        if not cached and not allow_predict:
            logging.error("Keyframe cache_only: missing ply %s", ply_path)
            return None
        logging.info(
            "Keyframe build start: seg=%d frame=%d ply=%s",
            segment_id,
            frame_idx,
            "cache" if cached else "predict",
        )

        if use_cache:
            gaussians, metadata = load_or_predict_ply(predictor, frame_rgb, f_px, ply_path)
        else:
            gaussians = predictor.predict_gaussians(frame_rgb, f_px)
            metadata = SceneMetaData(f_px, (frame_rgb.shape[1], frame_rgb.shape[0]), "linearRGB")

        return build_keyframe_state(
            frame_bgr,
            frame_idx,
            segment_id,
            gaussians,
            metadata,
            device,
            args,
            debug_root,
            timer=timer,
            log_prefix="Keyframe",
            skip_keypoints=skip_keypoints,
        )
    except Exception:
        logging.exception("Keyframe creation failed at frame %d", frame_idx)
        return None


def _collect_inputs(input_path: Path, mode: str) -> list[Path]:
    if input_path.is_dir():
        exts = VIDEO_EXTS if mode == "video" else IMAGE_EXTS
        items = [
            path
            for path in sorted(input_path.iterdir())
            if path.is_file() and path.suffix.lower() in exts
        ]
        return items
    return [input_path]


def _resolve_batch_output_dir(input_path: Path, out_arg: Path | None) -> Path | None:
    if out_arg is None:
        return None
    out_raw = str(out_arg).strip()
    if not out_raw or out_raw == ".":
        return None

    base_dir = input_path if input_path.is_dir() else input_path.parent
    candidate = out_arg
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if candidate.suffix and not candidate.is_dir():
        logging.warning("Batch mode: --out looks like a file, using its parent: %s", candidate)
        candidate = candidate.parent
    ensure_dir(candidate)
    return candidate


def _resolve_output_path(
    input_path: Path,
    out_arg: Path | None,
    mode: str,
    is_batch: bool,
    batch_out_dir: Path | None = None,
) -> Path:
    ext = ".mp4" if mode == "video" else ".png"
    if is_batch:
        out_dir = batch_out_dir or input_path.parent
        ensure_dir(out_dir)
        return out_dir / f"{safe_filename(input_path.stem)}_sbs{ext}"
    if out_arg is None:
        out_arg = Path(f"output_{int(time.time())}{ext}")
    if out_arg.exists() and out_arg.is_dir():
        ensure_dir(out_arg)
        return out_arg / f"{safe_filename(input_path.stem)}_sbs{ext}"
    if out_arg.suffix:
        return out_arg
    return out_arg.with_suffix(ext)


def _load_image_bgr(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def process_image(
    args: argparse.Namespace,
    image_path: Path,
    out_path: Path,
    device: torch.device,
    predictor: SharpPredictor,
) -> int:
    frame_bgr = _load_image_bgr(image_path)
    if frame_bgr is None:
        logging.error("Failed to read image: %s", image_path)
        return 2
    frame_bgr, (width_even, height_even) = pad_even(frame_bgr)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    f_px = compute_focal_px(width_even, args.fov_deg, args.focal_px)

    gaussians = predictor.predict_gaussians(frame_rgb, f_px)
    metadata = SceneMetaData(f_px, (frame_rgb.shape[1], frame_rgb.shape[0]), "linearRGB")
    keyframe = build_keyframe_state(
        frame_bgr,
        0,
        0,
        gaussians,
        metadata,
        device,
        args,
        debug_root=None,
        log_prefix="Image",
        skip_keypoints=True,
        renderer=None,
        intrinsics=None,
        intrinsics_t=None,
        frame_is_rgb=False,
    )

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    rmat, _ = cv2.Rodrigues(rvec)
    baseline = clamp_baseline(
        args.baseline_m,
        args.max_disp_px,
        keyframe.median_depth,
        keyframe.k_mat[0, 0],
        args.baseline_min_m,
    )
    left_extr, right_extr = compute_stereo_extrinsics(rmat, tvec, baseline)
    extrinsics = np.stack([left_extr, right_extr], axis=0)
    need_alpha = args.inpaint_radius > 0
    colors, depths, alphas = render_batch_dispatch(
        args,
        keyframe,
        keyframe.gaussians,
        keyframe.intrinsics,
        extrinsics,
        keyframe.metadata.resolution_px[0],
        keyframe.metadata.resolution_px[1],
        device,
        intrinsics_t=keyframe.intrinsics_t,
        need_depth=False,
        need_alpha=need_alpha,
    )

    scratch = np.empty_like(colors[0], dtype=np.float32)
    left_u8 = np.empty_like(colors[0], dtype=np.uint8)
    right_u8 = np.empty_like(colors[0], dtype=np.uint8)
    left_img = _color_to_uint8(colors[0], scratch, left_u8)
    right_img = _color_to_uint8(colors[1], scratch, right_u8)
    if need_alpha and alphas is not None:
        left_img = inpaint_holes(left_img, alphas[0], args.inpaint_radius)
        right_img = inpaint_holes(right_img, alphas[1], args.inpaint_radius)

    sbs = build_sbs(left_img, right_img, None)
    ensure_dir(out_path.parent)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR)):
        logging.error("Failed to write image: %s", out_path)
        return 2
    logging.info("Image processed: %s -> %s", image_path, out_path)
    return 0




def run_two_pass_streaming(
    args: argparse.Namespace,
    cap: cv2.VideoCapture | None,
    reader,
    first_frame: np.ndarray | None,
    video_path: Path,
    out_path: Path,
    fps: float,
    f_px: float,
    cache_root: Path,
    predictor: SharpPredictor,
    segment_frames: int,
) -> int:
    if args.keyframe_mode != "per_frame":
        logging.error("two_pass requires keyframe_mode=per_frame")
        return 2
    if args.debug_dir:
        logging.warning("two_pass ignores debug_dir to avoid extra I/O")
    buffer_frames = max(0, int(args.buffer_frames))
    cache_dir = cache_root / "gaussians_cache"
    ensure_dir(cache_dir)

    queue_items: queue.Queue[GaussianCacheItem] = queue.Queue()
    producer_done = threading.Event()
    producer_error: Exception | None = None
    render_error: Exception | None = None
    produced_count = 0
    produced_lock = threading.Lock()
    frame_is_rgb = bool(reader is not None and getattr(reader, "output_rgb", False))

    pending_frames = deque()
    if first_frame is not None:
        pending_frames.append(first_frame)

    def _next_frame() -> np.ndarray | None:
        if pending_frames:
            return pending_frames.popleft()
        return read_next_frame(cap, reader)

    def _producer() -> None:
        nonlocal producer_error, produced_count
        frame_idx_local = 0
        try:
            batch_size = max(1, int(args.per_frame_batch))
            while True:
                batch_frames = []
                while len(batch_frames) < batch_size:
                    frame = _next_frame()
                    if frame is None:
                        break
                    batch_frames.append(frame)
                if not batch_frames:
                    break

                padded_frames = []
                for frame in batch_frames:
                    frame, _ = pad_even(frame)
                    padded_frames.append(frame)

                if len(padded_frames) == 1:
                    rgb = (
                        padded_frames[0]
                        if frame_is_rgb
                        else cv2.cvtColor(padded_frames[0], cv2.COLOR_BGR2RGB)
                    )
                    gaussians_list = [predictor.predict_gaussians(rgb, f_px)]
                else:
                    if frame_is_rgb:
                        rgb_frames = list(padded_frames)
                    else:
                        rgb_frames = [
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            for frame in padded_frames
                        ]
                    rgb_batch = np.stack(rgb_frames, axis=0)
                    try:
                        gaussians_batch = predictor.predict_gaussians_batch(rgb_batch, f_px)
                        gaussians_list = split_gaussians_batch(gaussians_batch)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        logging.warning("Per-frame batch OOM, falling back to batch=1: %s", exc)
                        gaussians_list = [
                            predictor.predict_gaussians(rgb, f_px)
                            for rgb in rgb_frames
                        ]

                for gaussians, frame in zip(gaussians_list, padded_frames):
                    gaussians_cpu = _gaussians_to_cpu(gaussians)
                    cache_path = cache_dir / f"frame_{frame_idx_local:06d}.pt"
                    tmp_path = cache_path.with_suffix(".pt.tmp")
                    torch.save(gaussians_cpu, tmp_path)
                    os.replace(tmp_path, cache_path)
                    queue_items.put(
                        GaussianCacheItem(
                            frame_idx=frame_idx_local,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            path=cache_path,
                        )
                    )
                    frame_idx_local += 1
                    with produced_lock:
                        produced_count = frame_idx_local
                    if args.max_frames and frame_idx_local >= args.max_frames:
                        raise StopIteration
        except StopIteration:
            pass
        except Exception as exc:
            producer_error = exc
        finally:
            producer_done.set()

    producer = threading.Thread(target=_producer, daemon=True)
    producer.start()

    args_pass2_cpu = copy.copy(args)
    if args_pass2_cpu.render_backend != "cpu":
        logging.info("two_pass forces render_backend=cpu for pass2")
    args_pass2_cpu.render_backend = "cpu"

    args_pass2_gpu = copy.copy(args)
    args_pass2_gpu.render_backend = "cuda"

    gpu_assist = bool(args.gpu_assist)
    device_cpu = torch.device("cpu")
    device_gpu = None
    if gpu_assist:
        try:
            device_gpu = resolve_device(args.device)
        except RuntimeError as exc:
            logging.warning("GPU assist disabled: %s", exc)
            gpu_assist = False
        else:
            if device_gpu.type != "cuda":
                logging.warning("GPU assist disabled: non-CUDA device %s", device_gpu)
                gpu_assist = False
    if gpu_assist:
        logging.info("two_pass GPU assist after pass1: enabled")

    while buffer_frames > 0 and queue_items.qsize() < buffer_frames:
        if producer_done.is_set():
            break
        time.sleep(0.05)

    writer = None
    writer_worker: AsyncSBSWriter | None = None
    log_interval = max(1, int(args.log_interval))
    perf = PerfStats(args.perf_interval) if args.perf_interval and args.perf_interval > 0 else None

    results: dict[int, RenderResult] = {}
    result_cond = threading.Condition()
    active_workers = 0

    def _load_gaussians(item: GaussianCacheItem):
        gaussians = None
        for _ in range(5):
            try:
                try:
                    gaussians = torch.load(item.path, map_location="cpu", weights_only=False)
                except TypeError:
                    gaussians = torch.load(item.path, map_location="cpu")
                break
            except PermissionError:
                time.sleep(0.05)
        if gaussians is None:
            raise RuntimeError(f"Failed to read cache: {item.path}")
        item.path.unlink(missing_ok=True)
        return gaussians

    def _render_worker(name: str, args_render: argparse.Namespace, device: torch.device, backend: str) -> None:
        nonlocal render_error, active_workers
        dummy_cache: dict[tuple[int, int], np.ndarray] = {}
        intrinsics_cache: dict[tuple[int, int], np.ndarray] = {}
        intrinsics_t_cache: dict[tuple[int, int], torch.Tensor] = {}
        renderer = None
        scratch = None
        left_u8 = None
        right_u8 = None
        rmat = np.eye(3, dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        while True:
            if render_error is not None or producer_error is not None:
                break
            try:
                item = queue_items.get(timeout=0.1)
            except queue.Empty:
                if producer_done.is_set():
                    break
                continue
            try:
                gaussians = _load_gaussians(item)
                key = (item.height, item.width)
                frame_bgr = dummy_cache.get(key)
                if frame_bgr is None:
                    frame_bgr = np.zeros((item.height, item.width, 3), dtype=np.uint8)
                    dummy_cache[key] = frame_bgr

                metadata = SceneMetaData(f_px, (item.width, item.height), "linearRGB")
                intrinsics = intrinsics_cache.get(key)
                if intrinsics is None:
                    intrinsics = build_intrinsics(
                        metadata.focal_length_px,
                        metadata.resolution_px[0],
                        metadata.resolution_px[1],
                    )
                    intrinsics_cache[key] = intrinsics

                intrinsics_t = None
                if args_render.render_backend == "cuda":
                    cached = intrinsics_t_cache.get(key)
                    if cached is None or cached.device != device:
                        cached = (
                            torch.from_numpy(intrinsics)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        intrinsics_t_cache[key] = cached
                    intrinsics_t = cached

                keyframe = build_keyframe_state(
                    frame_bgr,
                    item.frame_idx,
                    0,
                    gaussians,
                    metadata,
                    device,
                    args_render,
                    debug_root=None,
                    log_prefix="Per-frame",
                    skip_keypoints=True,
                    renderer=renderer,
                    intrinsics=intrinsics,
                    intrinsics_t=intrinsics_t,
                    frame_is_rgb=False,
                )
                renderer = keyframe.renderer

                baseline = clamp_baseline(
                    args.baseline_m,
                    args.max_disp_px,
                    keyframe.median_depth,
                    keyframe.k_mat[0, 0],
                    args.baseline_min_m,
                )

                render_timer = Timer()
                left_extr, right_extr = compute_stereo_extrinsics(rmat, tvec, baseline)
                extrinsics = np.stack([left_extr, right_extr], axis=0)
                need_alpha = args.inpaint_radius > 0
                colors, _, alphas = render_batch_dispatch(
                    args_render,
                    keyframe,
                    keyframe.gaussians,
                    keyframe.intrinsics,
                    extrinsics,
                    keyframe.metadata.resolution_px[0],
                    keyframe.metadata.resolution_px[1],
                    device,
                    intrinsics_t=intrinsics_t,
                    need_depth=False,
                    need_alpha=need_alpha,
                )
                render_ms = render_timer.elapsed_ms()

                if scratch is None or scratch.shape != colors[0].shape:
                    scratch = np.empty_like(colors[0], dtype=np.float32)
                    left_u8 = None
                    right_u8 = None
                if left_u8 is None or left_u8.shape != colors[0].shape:
                    left_u8 = np.empty_like(colors[0], dtype=np.uint8)
                if right_u8 is None or right_u8.shape != colors[0].shape:
                    right_u8 = np.empty_like(colors[0], dtype=np.uint8)
                left_img = _color_to_uint8(colors[0], scratch, left_u8)
                right_img = _color_to_uint8(colors[1], scratch, right_u8)
                if need_alpha and alphas is not None:
                    left_img = inpaint_holes(left_img, alphas[0], args.inpaint_radius)
                    right_img = inpaint_holes(right_img, alphas[1], args.inpaint_radius)

                sbs = build_sbs(left_img, right_img, None)
                result = RenderResult(
                    frame_idx=item.frame_idx,
                    sbs=sbs,
                    render_ms=render_ms,
                    baseline=baseline,
                    backend=backend,
                    width=item.width,
                    height=item.height,
                )
            except Exception as exc:
                render_error = exc
                with result_cond:
                    result_cond.notify_all()
                break

            with result_cond:
                results[item.frame_idx] = result
                result_cond.notify_all()

        with result_cond:
            active_workers -= 1
            result_cond.notify_all()

    def _start_worker(name: str, args_render: argparse.Namespace, device: torch.device, backend: str) -> threading.Thread:
        nonlocal active_workers
        worker = threading.Thread(
            target=_render_worker,
            args=(name, args_render, device, backend),
            daemon=True,
        )
        with result_cond:
            active_workers += 1
            result_cond.notify_all()
        worker.start()
        return worker

    workers: list[threading.Thread] = []
    workers.append(_start_worker("cpu", args_pass2_cpu, device_cpu, "cpu"))
    gpu_worker_started = False

    return_code = 0
    next_write_idx = 0
    try:
        while True:
            if gpu_assist and not gpu_worker_started and producer_done.is_set():
                if device_gpu is not None:
                    workers.append(_start_worker("gpu", args_pass2_gpu, device_gpu, "cuda"))
                    gpu_worker_started = True

            with result_cond:
                while next_write_idx not in results:
                    if render_error is not None:
                        raise render_error
                    if producer_error is not None:
                        raise producer_error
                    if producer_done.is_set() and active_workers == 0:
                        with produced_lock:
                            total_frames = produced_count
                        if next_write_idx >= total_frames and not results:
                            return_code = 0
                            break
                    result_cond.wait(0.05)
                if next_write_idx not in results:
                    if producer_done.is_set() and active_workers == 0:
                        break
                    continue
                result = results.pop(next_write_idx)

            if writer is None:
                writer = SegmentWriter(
                    out_path,
                    result.width * 2,
                    result.height,
                    fps,
                    args.ffmpeg_crf,
                    args.ffmpeg_preset,
                    args.encode,
                    video_path if args.copy_audio else None,
                    args.audio_codec,
                    segment_frames,
                    args.keep_segments,
                )
                if should_async_write(result.width, result.height, args):
                    writer_worker = AsyncSBSWriter(writer, args.write_queue)
                else:
                    writer_worker = None
                    if args.async_write:
                        logging.info(
                            "Async write disabled for large frame %sx%s",
                            result.width * 2,
                            result.height,
                        )

            write_timer = Timer()
            ok = submit_sbs_frame(writer, writer_worker, result.sbs)
            write_ms = write_timer.elapsed_ms()
            if not ok:
                return_code = 2
                break

            if result.frame_idx % log_interval == 0:
                logging.info(
                    "seg=%d frame=%d inliers=%d reproj=%.3f new_keyframe=%s pnp_ms=%.2f render_ms=%.2f baseline=%.4f backend=%s",
                    0,
                    result.frame_idx,
                    0,
                    0.0,
                    "yes",
                    0.0,
                    result.render_ms,
                    result.baseline,
                    result.backend,
                )

            if perf is not None:
                perf.add("render", result.render_ms)
                perf.add(f"render_{result.backend}", result.render_ms)
                perf.add("write", write_ms)
                perf.tick()

            next_write_idx += 1
            if args.max_frames and next_write_idx >= args.max_frames:
                break
    except Exception as exc:
        logging.error("two_pass failed: %s", exc)
        return_code = 2
    finally:
        producer_done.set()
        for worker in workers:
            worker.join(timeout=1)
        if writer_worker is not None:
            try:
                writer_worker.close()
            except Exception as exc:
                logging.error("two_pass cleanup failed: %s", exc)
                return_code = 2
        elif writer is not None:
            writer.close()

    return return_code


def run_video(
    args: argparse.Namespace,
    video_path: Path,
    out_path: Path,
    device: torch.device,
    predictor: SharpPredictor,
) -> int:
    low_ram_active = False

    def _apply_low_ram_fallback() -> None:
        nonlocal low_ram_active
        if low_ram_active:
            return
        low_ram_active = True
        logging.warning(
            "Low free RAM. Forcing per_frame_batch=1, per_frame_pipeline=1, cache_per_frame=false.",
        )
        args.per_frame_batch = 1
        args.per_frame_pipeline = 1
        args.cache_per_frame = False
        os.environ["SHARP_EIG_CHUNK_SIZE"] = str(
            min(int(os.environ.get("SHARP_EIG_CHUNK_SIZE", "262144")), 65536)
        )

    if args.min_free_ram_gb and args.min_free_ram_gb > 0:
        avail = _available_memory_bytes()
        if avail is not None and avail < args.min_free_ram_gb * (1024**3):
            _apply_low_ram_fallback()

    if not video_path.exists():
        logging.error("Input video not found: %s", video_path)
        return 2

    cap = None
    reader = None
    total_frames = 0
    codec_name = None

    if args.io_backend == "ffmpeg":
        try:
            width, height, fps, total_frames, codec_name = probe_video_info_ffprobe(video_path)
            if width <= 0 or height <= 0:
                raise RuntimeError("ffprobe returned invalid width/height")
            if args.target_fps and args.target_fps > 0:
                if fps > 0 and total_frames > 0:
                    total_frames = int(total_frames * (args.target_fps / fps) + 0.5)
                fps = float(args.target_fps)
        except Exception as exc:
            logging.error("FFmpeg IO init failed: %s", exc)
            return 2
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error("Failed to open video: %s", video_path)
            return 2

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if args.target_fps and args.target_fps > 0:
            logging.warning("target_fps is only supported with ffmpeg io_backend; ignoring for opencv.")
    width_even = width + (width % 2)
    height_even = height + (height % 2)
    large_frame_area = int(os.environ.get("SBS_LARGE_FRAME_AREA", "8000000"))
    if width_even * height_even >= large_frame_area and not low_ram_active:
        low_ram_active = True
        logging.warning(
            "Large frame %dx%d; forcing low-memory settings.",
            width_even * 2,
            height_even,
        )
        args.per_frame_batch = min(args.per_frame_batch, 1)
        args.per_frame_pipeline = min(args.per_frame_pipeline, 1)
        args.write_queue = min(args.write_queue, 1)
        args.buffer_frames = min(args.buffer_frames, 2)
        args.cache_per_frame = False
    logging.info(
        "Video opened: path=%s fps=%.2f size=%dx%d io=%s decode=%s frames=%s",
        video_path,
        fps,
        width,
        height,
        args.io_backend,
        args.decode,
        total_frames if total_frames > 0 else "unknown",
    )
    segment_frames = 0
    if args.segment_frames and args.segment_frames > 0:
        if total_frames <= 0 or total_frames > args.segment_frames:
            segment_frames = int(args.segment_frames)
            logging.info("Auto segment: %d frames per segment.", segment_frames)

    f_px = compute_focal_px(width_even, args.fov_deg, args.focal_px)

    cache_root = Path(args.cache_dir) / safe_filename(video_path.stem) / video_cache_key(video_path)
    ensure_dir(cache_root)

    debug_root = Path(args.debug_dir) if args.debug_dir else None
    if debug_root:
        ensure_dir(debug_root)

    if device.type == "cuda":
        device_index = device.index or 0
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(device_index))
        logging.info("TF32 enabled: %s", torch.backends.cuda.matmul.allow_tf32)
    else:
        logging.warning("Using CPU device. This is very slow and may not be supported by GSplat.")
    logging.info("Tracking backend: %s", args.track_backend)
    logging.info("Render backend: %s", args.render_backend)
    if args.render_backend == "cpu":
        logging.warning("CPU render backend is experimental and slower.")
    logging.info("Keyframe mode: %s", args.keyframe_mode)
    logging.info("Per-frame cache: %s", args.cache_per_frame)
    logging.info("Per-frame batch: %s", args.per_frame_batch)
    logging.info("Per-frame pipeline: %s", args.per_frame_pipeline)
    logging.info("Async write: %s", args.async_write)
    logging.info("Write queue: %s", args.write_queue)
    logging.info("Two-pass: %s", args.two_pass)
    logging.info("Buffer frames: %s", args.buffer_frames)
    logging.info("GPU assist: %s", args.gpu_assist)
    logging.info("Clear cache on exit: %s", args.clear_cache_on_exit)
    logging.info("AMP enabled: %s", args.amp)
    logging.info("Pin memory: %s", args.pin_memory)
    logging.info("EIG backend: %s", args.eig_backend)
    logging.info("Perf interval: %s", args.perf_interval)
    logging.info("Copy audio: %s", args.copy_audio)
    logging.info("Target FPS: %s", args.target_fps)
    logging.info("Audio codec: %s", args.audio_codec)
    logging.info("Using SHARP predictor.")
    if args.io_backend == "ffmpeg":
        try:
            reuse_reader_buffer = (not args.two_pass) and (
                args.keyframe_mode != "per_frame"
                or (args.per_frame_pipeline <= 1 and args.per_frame_batch <= 1)
            )
            reader = FFmpegReader(
                video_path,
                width,
                height,
                use_hwaccel=args.decode == "nvdec",
                target_fps=float(args.target_fps or 0),
                output_rgb=args.keyframe_mode == "per_frame",
                codec_name=codec_name,
                reuse_buffer=reuse_reader_buffer,
            )
            if reuse_reader_buffer:
                logging.info("FFmpeg reader: reuse buffer enabled")
        except Exception as exc:
            logging.error("FFmpeg reader init failed: %s", exc)
            return 2

    min_len_frames = max(1, int(args.min_shot_len * fps))
    max_len_frames = max(1, int(args.max_shot_len * fps))
    if max_len_frames < min_len_frames:
        logging.warning(
            "max_shot_len (%.3fs) < min_shot_len (%.3fs); clamping max to min.",
            args.max_shot_len,
            args.min_shot_len,
        )
        max_len_frames = min_len_frames

    shot_detector = ShotDetector(
        cut_threshold=args.cut_threshold,
        min_len_frames=min_len_frames,
        max_len_frames=max_len_frames,
    )

    writer = None
    writer_worker: AsyncSBSWriter | None = None
    keyframe: KeyframeState | None = None
    prev_rvec = None
    prev_tvec = None
    frame_idx = 0
    segment_id = 0
    log_interval = max(1, int(args.log_interval))
    perf = PerfStats(args.perf_interval) if args.perf_interval and args.perf_interval > 0 else None
    pose_log = [] if debug_root else None
    per_frame_sharp = args.keyframe_mode == "per_frame"
    freeze_keyframes = args.keyframe_mode in {"freeze", "cache_only"}
    allow_predict = args.keyframe_mode != "cache_only"
    use_cache = (not per_frame_sharp) or args.cache_per_frame
    refresh_frames = 0
    if args.keyframe_refresh_s and args.keyframe_refresh_s > 0:
        refresh_frames = max(1, int(args.keyframe_refresh_s * fps))
        if per_frame_sharp:
            logging.warning("keyframe_refresh_s ignored in per_frame mode.")
        else:
            logging.info(
                "Keyframe refresh: every %.3fs (~%d frames)",
                args.keyframe_refresh_s,
                refresh_frames,
            )
    first_frame = None
    if reader is not None:
        first_frame = reader.read()
        if first_frame is None:
            logging.error("FFmpeg reader returned no frames for %s", video_path)
            return 2
    fatal_error = False
    return_code = 0

    try:
        if per_frame_sharp and args.two_pass:
            return_code = run_two_pass_streaming(
                args,
                cap,
                reader,
                first_frame,
                video_path,
                out_path,
                fps,
                f_px,
                cache_root,
                predictor,
                segment_frames,
            )
            return return_code
        if per_frame_sharp:
            batch_size = max(1, int(args.per_frame_batch))
            queue_size = max(1, int(args.per_frame_pipeline))
            use_pipeline = queue_size > 1
            stop_event = threading.Event()
            frame_queue = queue.Queue(maxsize=queue_size) if use_pipeline else None
            producer = None

            if use_pipeline:
                def _producer() -> None:
                    while True:
                        if stop_event.is_set():
                            break
                        frame = read_next_frame(cap, reader)
                        if frame is None:
                            break
                        while True:
                            try:
                                frame_queue.put(frame, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                        if stop_event.is_set():
                            break
                    if stop_event.is_set():
                        return
                    while True:
                        try:
                            frame_queue.put(None, timeout=0.1)
                            break
                        except queue.Full:
                            continue

                producer = threading.Thread(target=_producer, daemon=True)
                producer.start()

            pending_frames = deque()
            if first_frame is not None:
                pending_frames.append(first_frame)
                first_frame = None
            pending_eof = False

            def _get_frame():
                nonlocal pending_eof
                if pending_frames:
                    return pending_frames.popleft()
                if pending_eof:
                    return None
                if use_pipeline and frame_queue is not None:
                    item = frame_queue.get()
                    if item is None:
                        pending_eof = True
                        return None
                    return item
                return read_next_frame(cap, reader)

            def _disable_pipeline() -> None:
                nonlocal use_pipeline, frame_queue, producer, batch_size, pending_eof
                if not use_pipeline:
                    return
                if frame_queue is not None:
                    try:
                        while True:
                            item = frame_queue.get_nowait()
                            if item is None:
                                pending_eof = True
                            else:
                                pending_frames.append(item)
                    except queue.Empty:
                        pass
                stop_event.set()
                if producer is not None:
                    producer.join(timeout=2)
                if frame_queue is not None:
                    try:
                        while True:
                            item = frame_queue.get_nowait()
                            if item is None:
                                pending_eof = True
                            else:
                                pending_frames.append(item)
                    except queue.Empty:
                        pass
                use_pipeline = False
                frame_queue = None
                batch_size = 1

            scratch = None
            left_u8 = None
            right_u8 = None
            sbs_buf = None
            per_frame_renderer = None
            per_frame_intrinsics = None
            per_frame_intrinsics_t = None
            per_frame_shape = None
            frame_is_rgb = bool(reader is not None and getattr(reader, "output_rgb", False))
            low_ram_check_interval = 30
            last_low_ram_frame = -1
            write_failed = False
            while True:
                if (
                    args.min_free_ram_gb
                    and frame_idx % low_ram_check_interval == 0
                    and frame_idx != last_low_ram_frame
                ):
                    avail = _available_memory_bytes()
                    last_low_ram_frame = frame_idx
                    if avail is not None and avail < args.min_free_ram_gb * (1024**3):
                        _apply_low_ram_fallback()
                        _disable_pipeline()
                if low_ram_active and use_pipeline:
                    _disable_pipeline()
                if low_ram_active:
                    batch_size = 1
                fetch_timer = Timer()
                batch_frames = []
                while len(batch_frames) < batch_size:
                    frame = _get_frame()
                    if frame is None:
                        break
                    batch_frames.append(frame)

                if not batch_frames:
                    break

                if perf is not None:
                    fetch_ms = fetch_timer.elapsed_ms()
                    perf.add("fetch", fetch_ms / max(1, len(batch_frames)))

                padded_frames = []
                for frame in batch_frames:
                    frame, (width_even, height_even) = pad_even(frame)
                    padded_frames.append(frame)

                if writer is None:
                    writer = SegmentWriter(
                        out_path,
                        width_even * 2,
                        height_even,
                        fps,
                        args.ffmpeg_crf,
                        args.ffmpeg_preset,
                        args.encode,
                        video_path if args.copy_audio else None,
                        args.audio_codec,
                        segment_frames,
                        args.keep_segments,
                    )

                    if should_async_write(width_even, height_even, args):
                        writer_worker = AsyncSBSWriter(writer, args.write_queue)
                    else:
                        writer_worker = None
                        if args.async_write:
                            logging.info(
                                "Async write disabled for large frame %sx%s",
                                width_even * 2,
                                height_even,
                            )

                if len(padded_frames) == 1:
                    rgb = (
                        padded_frames[0]
                        if frame_is_rgb
                        else cv2.cvtColor(padded_frames[0], cv2.COLOR_BGR2RGB)
                    )
                    batch_timer = Timer()
                    gaussians_list = [predictor.predict_gaussians(rgb, f_px)]
                    predict_ms = batch_timer.elapsed_ms()
                    logging.info(
                        "Per-frame predict: frames=1 ms=%.1f",
                        predict_ms,
                    )
                else:
                    if frame_is_rgb:
                        rgb_frames = list(padded_frames)
                    else:
                        rgb_frames = [
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            for frame in padded_frames
                        ]
                    rgb_batch = np.stack(rgb_frames, axis=0)
                    try:
                        batch_timer = Timer()
                        gaussians_batch = predictor.predict_gaussians_batch(rgb_batch, f_px)
                        predict_ms = batch_timer.elapsed_ms()
                        logging.info(
                            "Per-frame batch predict: frames=%d ms=%.1f",
                            len(padded_frames),
                            predict_ms,
                        )
                        gaussians_list = split_gaussians_batch(gaussians_batch)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        logging.warning("Per-frame batch OOM, falling back to batch=1: %s", exc)
                        fallback_timer = Timer()
                        gaussians_list = [predictor.predict_gaussians(rgb, f_px) for rgb in rgb_frames]
                        predict_ms = fallback_timer.elapsed_ms()
                        logging.info(
                            "Per-frame batch fallback: frames=%d ms=%.1f",
                            len(padded_frames),
                            predict_ms,
                        )

                predict_ms_per_frame = predict_ms / max(1, len(padded_frames))

                for gaussians, frame in zip(gaussians_list, padded_frames):
                    if (
                        args.min_free_ram_gb
                        and frame_idx % low_ram_check_interval == 0
                        and frame_idx != last_low_ram_frame
                    ):
                        avail = _available_memory_bytes()
                        last_low_ram_frame = frame_idx
                        if avail is not None and avail < args.min_free_ram_gb * (1024**3):
                            _apply_low_ram_fallback()
                            _disable_pipeline()
                    if args.cache_per_frame:
                        ply_path = cache_root / f"seg000_f{frame_idx:06d}.ply"
                        if not ply_path.exists():
                            save_ply_fast(
                                gaussians,
                                f_px,
                                (frame.shape[0], frame.shape[1]),
                                ply_path,
                            )

                    shape_hw = frame.shape[:2]
                    if per_frame_shape != shape_hw:
                        per_frame_shape = shape_hw
                        per_frame_intrinsics = build_intrinsics(
                            f_px,
                            frame.shape[1],
                            frame.shape[0],
                        )
                        per_frame_intrinsics_t = (
                            torch.from_numpy(per_frame_intrinsics)
                            .to(device=device, dtype=torch.float32)
                            .unsqueeze(0)
                        )
                        per_frame_renderer = SharpRenderer("linearRGB", use_amp=args.amp)

                    metadata = SceneMetaData(f_px, (frame.shape[1], frame.shape[0]), "linearRGB")
                    keyframe = build_keyframe_state(
                        frame,
                        frame_idx,
                        0,
                        gaussians,
                        metadata,
                        device,
                        args,
                        debug_root,
                        log_prefix="Per-frame",
                        skip_keypoints=True,
                        renderer=per_frame_renderer,
                        intrinsics=per_frame_intrinsics,
                        intrinsics_t=per_frame_intrinsics_t,
                        frame_is_rgb=frame_is_rgb,
                    )

                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)
                    pnp_result = None

                    rmat, _ = cv2.Rodrigues(rvec)
                    baseline = clamp_baseline(
                        args.baseline_m,
                        args.max_disp_px,
                        keyframe.median_depth,
                        keyframe.k_mat[0, 0],
                        args.baseline_min_m,
                    )

                    render_timer = Timer()
                    left_extr, right_extr = compute_stereo_extrinsics(rmat, tvec, baseline)
                    extrinsics = np.stack([left_extr, right_extr], axis=0)
                    need_alpha = args.inpaint_radius > 0
                    need_depth = bool(debug_root and frame_idx % args.debug_interval == 0)
                    colors, depths, alphas = render_batch_dispatch(
                        args,
                        keyframe,
                        keyframe.gaussians,
                        keyframe.intrinsics,
                        extrinsics,
                        keyframe.metadata.resolution_px[0],
                        keyframe.metadata.resolution_px[1],
                        device,
                        intrinsics_t=keyframe.intrinsics_t,
                        need_depth=need_depth,
                        need_alpha=need_alpha,
                    )

                    if scratch is None or scratch.shape != colors[0].shape:
                        scratch = np.empty_like(colors[0], dtype=np.float32)
                        left_u8 = None
                        right_u8 = None
                    if left_u8 is None or left_u8.shape != colors[0].shape:
                        left_u8 = np.empty_like(colors[0], dtype=np.uint8)
                    if right_u8 is None or right_u8.shape != colors[0].shape:
                        right_u8 = np.empty_like(colors[0], dtype=np.uint8)
                    left_img = _color_to_uint8(colors[0], scratch, left_u8)
                    right_img = _color_to_uint8(colors[1], scratch, right_u8)

                    if need_alpha and alphas is not None:
                        left_img = inpaint_holes(left_img, alphas[0], args.inpaint_radius)
                        right_img = inpaint_holes(right_img, alphas[1], args.inpaint_radius)

                    sbs_buf = build_sbs(left_img, right_img, sbs_buf)
                    sbs = sbs_buf
                    write_timer = Timer()
                    ok = submit_sbs_frame(writer, writer_worker, sbs)
                    write_ms = write_timer.elapsed_ms()
                    if not ok:
                        write_failed = True
                        fatal_error = True
                        return_code = 2
                        stop_event.set()
                        break

                    if debug_root and frame_idx % args.debug_interval == 0:
                        seg_dir = debug_root / "segment_000"
                        ensure_dir(seg_dir)
                        cv2.imwrite(
                            str(seg_dir / f"sbs_{frame_idx:06d}.jpg"),
                            cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
                        )
                        if depths is not None:
                            save_depth_png(seg_dir / f"depth_{frame_idx:06d}.png", depths[0])

                    render_ms = render_timer.elapsed_ms()
                    if frame_idx % log_interval == 0:
                        logging.info(
                            "seg=%d frame=%d inliers=%d reproj=%.3f new_keyframe=%s pnp_ms=%.2f render_ms=%.2f baseline=%.4f",
                            0,
                            frame_idx,
                            0,
                            0.0,
                            "yes",
                            0.0,
                            render_ms,
                            baseline,
                        )

                    if perf is not None:
                        perf.add("predict", predict_ms_per_frame)
                        perf.add("render", render_ms)
                        perf.add("write", write_ms)
                        perf.tick()

                    if pose_log is not None:
                        pose_log.append(
                            {
                                "frame": frame_idx,
                                "segment": 0,
                                "inliers": 0,
                                "reproj": 0.0,
                                "baseline": baseline,
                                "new_keyframe": True,
                                "pnp_ms": 0.0,
                                "render_ms": render_ms,
                            }
                        )
                    frame_idx += 1
                    if args.max_frames and frame_idx >= args.max_frames:
                        stop_event.set()
                        break

                if write_failed:
                    break
                if args.max_frames and frame_idx >= args.max_frames:
                    break

            if producer is not None:
                stop_event.set()
                producer.join(timeout=2)
        else:
            scratch = None
            left_u8 = None
            right_u8 = None
            sbs_buf = None
            low_ram_check_interval = 30
            last_low_ram_frame = -1
            write_failed = False
            pending_frame = first_frame
            first_frame = None
            while True:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    decode_timer = Timer()
                    frame = read_next_frame(cap, reader)
                    decode_ms = decode_timer.elapsed_ms()
                    if perf is not None:
                        perf.add("decode", decode_ms)
                if frame is None:
                    break

                frame, (width_even, height_even) = pad_even(frame)
                if (
                    args.min_free_ram_gb
                    and frame_idx % low_ram_check_interval == 0
                    and frame_idx != last_low_ram_frame
                ):
                    avail = _available_memory_bytes()
                    last_low_ram_frame = frame_idx
                    if avail is not None and avail < args.min_free_ram_gb * (1024**3):
                        _apply_low_ram_fallback()
                if writer is None:
                    writer = SegmentWriter(
                        out_path,
                        width_even * 2,
                        height_even,
                        fps,
                        args.ffmpeg_crf,
                        args.ffmpeg_preset,
                        args.encode,
                        video_path if args.copy_audio else None,
                        args.audio_codec,
                        segment_frames,
                        args.keep_segments,
                    )

                    if should_async_write(width_even, height_even, args):
                        writer_worker = AsyncSBSWriter(writer, args.write_queue)
                    else:
                        writer_worker = None
                        if args.async_write:
                            logging.info(
                                "Async write disabled for large frame %sx%s",
                                width_even * 2,
                                height_even,
                            )

                cut = shot_detector.update(frame)
                if cut and freeze_keyframes and keyframe is not None:
                    logging.info(
                        "Shot cut detected but keyframe_mode=%s; reusing keyframe.",
                        args.keyframe_mode,
                    )
                refresh_due = (
                    not freeze_keyframes
                    and refresh_frames > 0
                    and keyframe is not None
                    and (frame_idx - keyframe.frame_idx) >= refresh_frames
                )
                need_keyframe = keyframe is None or (cut and not freeze_keyframes) or refresh_due

                new_keyframe_reason = None
                if need_keyframe:
                    if keyframe is None:
                        new_keyframe_reason = "init"
                    elif cut and not freeze_keyframes:
                        new_keyframe_reason = "cut"
                    elif refresh_due:
                        new_keyframe_reason = "refresh"
                    keyframe = create_keyframe(
                        frame,
                        frame_idx,
                        segment_id,
                        predictor,
                        device,
                        cache_root,
                        f_px,
                        args,
                        debug_root,
                        allow_predict=allow_predict,
                        use_cache=use_cache,
                    )
                    if keyframe is None:
                        if args.keyframe_mode == "cache_only":
                            fatal_error = True
                            return_code = 2
                            break
                        shot_detector.reset()
                    else:
                        segment_id += 1
                        shot_detector.reset()
                        prev_rvec = None
                        prev_tvec = None
                        logging.info(
                            "Keyframe created: seg=%d frame=%d reason=%s pts=%d",
                            keyframe.segment_id,
                            keyframe.frame_idx,
                            new_keyframe_reason,
                            len(keyframe.key_pts_2d),
                        )

                if keyframe is None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sbs_buf = build_sbs(frame_rgb, frame_rgb, sbs_buf)
                    sbs = sbs_buf
                    write_timer = Timer()
                    ok = submit_sbs_frame(writer, writer_worker, sbs)
                    write_ms = write_timer.elapsed_ms()
                    if perf is not None:
                        perf.add("write", write_ms)
                        perf.tick()
                    if not ok:
                        write_failed = True
                        fatal_error = True
                        return_code = 2
                        break
                    frame_idx += 1
                    if args.max_frames and frame_idx >= args.max_frames:
                        break
                    continue

                tracking_timer = Timer()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if new_keyframe_reason is None:
                    prev_pts, curr_pts, indices = track_keypoints(
                        keyframe.gray,
                        gray,
                        keyframe.key_pts_2d,
                        args.flow_fb_thresh,
                        backend=args.track_backend,
                    )
                    prev_pts, curr_pts, indices = filter_fundamental(
                        prev_pts,
                        curr_pts,
                        indices,
                        args.fundamental_thresh,
                    )
                    obj_pts = keyframe.key_pts_3d[indices]
                    pnp_result = solve_pnp(
                        obj_pts,
                        curr_pts,
                        keyframe.k_mat,
                        args.pnp_ransac_iters,
                        args.pnp_reproj,
                    )

                    if (
                        (not pnp_result.success)
                        or (pnp_result.inliers is None)
                        or (len(pnp_result.inliers) < args.min_inliers)
                        or (pnp_result.reproj_error > args.max_reproj)
                    ):
                        if freeze_keyframes:
                            logging.warning(
                                "PnP failed: inliers=%s reproj=%.3f (keyframe_mode=%s)",
                                0 if pnp_result.inliers is None else len(pnp_result.inliers),
                                pnp_result.reproj_error,
                                args.keyframe_mode,
                            )
                        else:
                            logging.warning(
                                "PnP failed: inliers=%s reproj=%.3f -> new keyframe",
                                0 if pnp_result.inliers is None else len(pnp_result.inliers),
                                pnp_result.reproj_error,
                            )
                        if freeze_keyframes:
                            logging.warning(
                                "Keyframe mode %s: skipping rebuild, reusing last pose.",
                                args.keyframe_mode,
                            )
                            if prev_rvec is not None and prev_tvec is not None:
                                rvec, tvec = prev_rvec, prev_tvec
                            else:
                                rvec = np.zeros((3, 1), dtype=np.float32)
                                tvec = np.zeros((3, 1), dtype=np.float32)
                            pnp_result = None
                        else:
                            new_keyframe_reason = "pnp_fail"
                            keyframe = create_keyframe(
                                frame,
                                frame_idx,
                                segment_id,
                                predictor,
                                device,
                                cache_root,
                                f_px,
                                args,
                                debug_root,
                                allow_predict=allow_predict,
                                use_cache=use_cache,
                            )
                            if keyframe is None:
                                shot_detector.reset()
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                sbs_buf = build_sbs(frame_rgb, frame_rgb, sbs_buf)
                                sbs = sbs_buf
                                write_timer = Timer()
                                ok = submit_sbs_frame(writer, writer_worker, sbs)
                                write_ms = write_timer.elapsed_ms()
                                if perf is not None:
                                    perf.add("write", write_ms)
                                    perf.tick()
                                if not ok:
                                    write_failed = True
                                    fatal_error = True
                                    return_code = 2
                                    break
                                frame_idx += 1
                                if args.max_frames and frame_idx >= args.max_frames:
                                    break
                                continue
                            segment_id += 1
                            shot_detector.reset()
                            prev_rvec = None
                            prev_tvec = None
                            logging.info(
                                "Keyframe created: seg=%d frame=%d reason=%s pts=%d",
                                keyframe.segment_id,
                                keyframe.frame_idx,
                                new_keyframe_reason,
                                len(keyframe.key_pts_2d),
                            )
                            rvec = np.zeros((3, 1), dtype=np.float32)
                            tvec = np.zeros((3, 1), dtype=np.float32)
                            pnp_result = None
                    else:
                        rvec, tvec = pnp_result.rvec, pnp_result.tvec
                        if prev_rvec is not None and prev_tvec is not None:
                            rvec, tvec = smooth_pose(prev_rvec, prev_tvec, rvec, tvec, args.pose_smooth)

                        prev_rvec, prev_tvec = rvec, tvec
                else:
                    rvec = np.zeros((3, 1), dtype=np.float32)
                    tvec = np.zeros((3, 1), dtype=np.float32)
                    pnp_result = None

                rmat, _ = cv2.Rodrigues(rvec)
                baseline = clamp_baseline(
                    args.baseline_m,
                    args.max_disp_px,
                    keyframe.median_depth,
                    keyframe.k_mat[0, 0],
                    args.baseline_min_m,
                )

                render_timer = Timer()
                left_extr, right_extr = compute_stereo_extrinsics(rmat, tvec, baseline)
                extrinsics = np.stack([left_extr, right_extr], axis=0)
                need_alpha = args.inpaint_radius > 0
                need_depth = bool(
                    debug_root and frame_idx % args.debug_interval == 0 and new_keyframe_reason is None
                )
                colors, depths, alphas = render_batch_dispatch(
                    args,
                    keyframe,
                    keyframe.gaussians,
                    keyframe.intrinsics,
                    extrinsics,
                    keyframe.metadata.resolution_px[0],
                    keyframe.metadata.resolution_px[1],
                    device,
                    intrinsics_t=keyframe.intrinsics_t,
                    need_depth=need_depth,
                    need_alpha=need_alpha,
                )

                if scratch is None or scratch.shape != colors[0].shape:
                    scratch = np.empty_like(colors[0], dtype=np.float32)
                    left_u8 = None
                    right_u8 = None
                if left_u8 is None or left_u8.shape != colors[0].shape:
                    left_u8 = np.empty_like(colors[0], dtype=np.uint8)
                if right_u8 is None or right_u8.shape != colors[0].shape:
                    right_u8 = np.empty_like(colors[0], dtype=np.uint8)
                left_img = _color_to_uint8(colors[0], scratch, left_u8)
                right_img = _color_to_uint8(colors[1], scratch, right_u8)

                if need_alpha and alphas is not None:
                    left_img = inpaint_holes(left_img, alphas[0], args.inpaint_radius)
                    right_img = inpaint_holes(right_img, alphas[1], args.inpaint_radius)

                sbs_buf = build_sbs(left_img, right_img, sbs_buf)
                sbs = sbs_buf
                write_timer = Timer()
                ok = submit_sbs_frame(writer, writer_worker, sbs)
                write_ms = write_timer.elapsed_ms()
                if not ok:
                    write_failed = True
                    fatal_error = True
                    return_code = 2
                    break

                if debug_root and frame_idx % args.debug_interval == 0:
                    seg_dir = debug_root / f"segment_{keyframe.segment_id:03d}"
                    ensure_dir(seg_dir)
                    cv2.imwrite(
                        str(seg_dir / f"sbs_{frame_idx:06d}.jpg"),
                        cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
                    )
                    if new_keyframe_reason is None and depths is not None:
                        save_depth_png(seg_dir / f"depth_{frame_idx:06d}.png", depths[0])

                tracking_ms = tracking_timer.elapsed_ms()
                render_ms = render_timer.elapsed_ms()
                inliers = 0
                reproj = 0.0
                if pnp_result is not None and pnp_result.inliers is not None:
                    inliers = len(pnp_result.inliers)
                    reproj = pnp_result.reproj_error

                if frame_idx % log_interval == 0:
                    logging.info(
                        "seg=%d frame=%d inliers=%d reproj=%.3f new_keyframe=%s pnp_ms=%.2f render_ms=%.2f baseline=%.4f",
                        keyframe.segment_id,
                        frame_idx,
                        inliers,
                        reproj,
                        "yes" if new_keyframe_reason else "no",
                        tracking_ms,
                        render_ms,
                        baseline,
                    )

                if perf is not None:
                    perf.add("track", tracking_ms)
                    perf.add("render", render_ms)
                    perf.add("write", write_ms)
                    perf.tick()

                if pose_log is not None:
                    pose_log.append(
                        {
                            "frame": frame_idx,
                            "segment": keyframe.segment_id,
                            "inliers": inliers,
                            "reproj": reproj,
                            "baseline": baseline,
                            "new_keyframe": bool(new_keyframe_reason),
                            "pnp_ms": tracking_ms,
                            "render_ms": render_ms,
                        }
                    )
                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break

        if debug_root and pose_log is not None:
            save_json(debug_root / "pose_log.json", {"items": pose_log})
        if fatal_error:
            logging.error("Keyframe cache_only failed: missing required cache.")

    finally:
        if cap is not None:
            cap.release()
        if reader is not None:
            reader.close()
        if writer_worker is not None:
            try:
                writer_worker.close()
            except Exception as exc:
                logging.error("FFmpeg close failed: %s", exc)
                fatal_error = True
                return_code = 2
        elif writer is not None:
            try:
                writer.close()
            except Exception as exc:
                logging.error("FFmpeg close failed: %s", exc)
                fatal_error = True
                return_code = 2
        if return_code != 0 or fatal_error:
            logging.error("Exit with code=%s fatal_error=%s", return_code, fatal_error)
        if args.clear_cache_on_exit:
            _cleanup_cache_dir(Path(args.cache_dir))

    return return_code


def main() -> int:
    args = parse_args()
    setup_logging(args.log_path, args.verbose)
    if args.eig_chunk_size is not None:
        os.environ["SHARP_EIG_CHUNK_SIZE"] = str(int(args.eig_chunk_size))
    if args.eig_chunk_size_max is not None:
        os.environ["SHARP_EIG_CHUNK_SIZE_MAX"] = str(int(args.eig_chunk_size_max))
    os.environ["SHARP_EIG_BACKEND"] = str(args.eig_backend)

    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        logging.error("Device init failed: %s", exc)
        return 2
    if device.type == "cuda":
        enable_tf32()
        torch.cuda.set_device(device)
        device_index = device.index or 0
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(device_index))
        logging.info("TF32 enabled: %s", torch.backends.cuda.matmul.allow_tf32)
    else:
        logging.warning("Using CPU device. This is very slow and may not be supported by GSplat.")

    logging.info("Loading SHARP predictor...")
    predictor = SharpPredictor(
        device=str(device),
        checkpoint_path=args.sharp_ckpt,
        use_amp=args.amp,
        pin_memory=args.pin_memory,
    )
    logging.info("SHARP predictor ready.")

    input_path = Path(args.video)
    if not input_path.exists():
        logging.error("Input path not found: %s", input_path)
        return 2

    inputs = _collect_inputs(input_path, args.mode)
    if not inputs:
        logging.error("No inputs found for %s mode in %s", args.mode, input_path)
        return 2
    is_batch = input_path.is_dir() or len(inputs) > 1
    out_raw = (args.out or "").strip()
    out_arg = Path(out_raw) if out_raw else None
    batch_out_dir = _resolve_batch_output_dir(input_path, out_arg) if is_batch else None
    if is_batch:
        if batch_out_dir:
            logging.info("Batch mode: outputs -> %s", batch_out_dir)
        else:
            logging.info("Batch mode: outputs follow each input directory.")

    if args.mode == "image":
        logging.info("Image mode: %d item(s).", len(inputs))
    else:
        logging.info("Video mode: %d item(s).", len(inputs))

    failures = 0
    for idx, path in enumerate(inputs, start=1):
        out_path = _resolve_output_path(path, out_arg, args.mode, is_batch, batch_out_dir)
        logging.info("[%d/%d] %s -> %s", idx, len(inputs), path, out_path)
        if args.mode == "image":
            ret = process_image(args, path, out_path, device, predictor)
        else:
            ret = run_video(args, path, out_path, device, predictor)
        if ret != 0:
            failures += 1
            logging.error("Failed: %s (code=%d)", path, ret)

    if failures:
        logging.error("Completed with %d failure(s).", failures)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
