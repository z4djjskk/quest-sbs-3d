import argparse
import json
import logging
import os
import queue
from collections import deque
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D video to SBS 3D using SHARP keyframes")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Path to output SBS mp4")
    parser.add_argument("--cache_dir", default=".cache_sharp", help="Cache directory")
    parser.add_argument("--debug_dir", default=None, help="Optional debug output dir")
    parser.add_argument("--log_path", default=None, help="Optional log file path")

    parser.add_argument("--fov_deg", type=float, default=60.0, help="Assumed horizontal FOV")
    parser.add_argument("--focal_px", type=float, default=None, help="Override focal length in pixels")

    parser.add_argument("--baseline_m", type=float, default=0.064, help="Stereo baseline in meters")
    parser.add_argument("--baseline_min_m", type=float, default=0.0, help="Minimum baseline clamp")
    parser.add_argument("--max_disp_px", type=float, default=60.0, help="Disparity clamp in pixels")

    parser.add_argument("--cut_threshold", type=float, default=0.9, help="Histogram cut threshold")
    parser.add_argument("--min_shot_len", type=float, default=0.01, help="Min shot length seconds")
    parser.add_argument("--max_shot_len", type=float, default=0.05, help="Max shot length seconds")

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

    parser.add_argument("--ffmpeg_crf", type=int, default=18, help="FFmpeg CRF")
    parser.add_argument("--ffmpeg_preset", default="slow", help="FFmpeg preset")

    parser.add_argument("--inpaint_radius", type=int, default=2, help="Inpaint radius for holes")
    parser.add_argument("--debug_interval", type=int, default=30, help="Debug frame interval")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames for debug")

    parser.add_argument("--sharp_ckpt", default=None, help="Path to SHARP checkpoint")
    parser.add_argument("--device", default="auto", help="Torch device (auto|cuda|cuda:N|cpu)")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 autocast for SHARP/GSplat")
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
        default="hevc_nvenc",
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
        default="copy",
        choices=["copy", "aac"],
        help="Audio codec when copy_audio is enabled",
    )
    parser.add_argument(
        "--track_backend",
        default="cpu",
        choices=["cpu", "opencv_cuda"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--keyframe_mode",
        default="normal",
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
        default=1,
        help="Batch size for per_frame mode",
    )
    parser.add_argument(
        "--per_frame_pipeline",
        type=int,
        default=2,
        help="Queue size for per_frame pipeline",
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


def probe_video_info_ffprobe(video_path: Path) -> tuple[int, int, float, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
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
    return width, height, fps, total_frames


class FFmpegReader:
    def __init__(self, video_path: Path, width: int, height: int, use_hwaccel: bool) -> None:
        cmd = ["ffmpeg", "-v", "error"]
        if use_hwaccel:
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        logging.info("FFmpeg reader: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.width = width
        self.height = height
        self.frame_size = width * height * 3

    def read(self) -> np.ndarray | None:
        if self.proc.stdout is None:
            return None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self) -> None:
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        self.proc.terminate()
        self.proc.wait()


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
    depth_vals = depth_map[valid]
    low = float(np.quantile(depth_vals, q_low))
    high = float(np.quantile(depth_vals, q_high))
    median = float(np.median(depth_vals))
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
) -> KeyframeState:
    if timer is None:
        timer = Timer()
    gaussians = gaussians.to(device)
    renderer = SharpRenderer(metadata.color_space, use_amp=args.amp)

    intrinsics = build_intrinsics(metadata.focal_length_px, metadata.resolution_px[0], metadata.resolution_px[1])
    intrinsics_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32).unsqueeze(0)
    extrinsics = np.eye(4, dtype=np.float32)
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

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if skip_keypoints:
        key_pts_2d = np.empty((0, 2), dtype=np.float32)
        key_pts_3d = np.empty((0, 3), dtype=np.float32)
    else:
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
        cv2.imwrite(str(seg_dir / f"keyframe_{frame_idx:06d}.png"), frame_bgr)
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


def read_next_frame(cap: cv2.VideoCapture | None, reader) -> np.ndarray | None:
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    if reader is None:
        return None
    return reader.read()


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


def main() -> int:
    args = parse_args()
    setup_logging(args.log_path, args.verbose)
    if args.eig_chunk_size is not None:
        os.environ["SHARP_EIG_CHUNK_SIZE"] = str(int(args.eig_chunk_size))
    if args.eig_chunk_size_max is not None:
        os.environ["SHARP_EIG_CHUNK_SIZE_MAX"] = str(int(args.eig_chunk_size_max))
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

    video_path = Path(args.video)
    out_path = Path(args.out)

    if not video_path.exists():
        logging.error("Input video not found: %s", video_path)
        return 2

    cap = None
    reader = None
    total_frames = 0

    if args.io_backend == "ffmpeg":
        try:
            width, height, fps, total_frames = probe_video_info_ffprobe(video_path)
            if width <= 0 or height <= 0:
                raise RuntimeError("ffprobe returned invalid width/height")
            reader = FFmpegReader(video_path, width, height, use_hwaccel=args.decode == "nvdec")
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
    width_even = width + (width % 2)
    height_even = height + (height % 2)
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

    f_px = compute_focal_px(width_even, args.fov_deg, args.focal_px)

    cache_root = Path(args.cache_dir) / safe_filename(video_path.stem) / video_cache_key(video_path)
    ensure_dir(cache_root)

    debug_root = Path(args.debug_dir) if args.debug_dir else None
    if debug_root:
        ensure_dir(debug_root)

    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        logging.error("Device init failed: %s", exc)
        return 2
    if device.type == "cuda":
        torch.cuda.set_device(device)
        device_index = device.index or 0
        logging.info("Using CUDA device: %s", torch.cuda.get_device_name(device_index))
    else:
        logging.warning("Using CPU device. This is very slow and may not be supported by GSplat.")
    logging.info("Tracking backend: %s", args.track_backend)
    logging.info("Keyframe mode: %s", args.keyframe_mode)
    logging.info("Per-frame cache: %s", args.cache_per_frame)
    logging.info("Per-frame batch: %s", args.per_frame_batch)
    logging.info("Per-frame pipeline: %s", args.per_frame_pipeline)
    logging.info("Clear cache on exit: %s", args.clear_cache_on_exit)
    logging.info("AMP enabled: %s", args.amp)
    logging.info("Copy audio: %s", args.copy_audio)
    logging.info("Audio codec: %s", args.audio_codec)
    logging.info("Loading SHARP predictor...")
    predictor = SharpPredictor(device=str(device), checkpoint_path=args.sharp_ckpt, use_amp=args.amp)
    logging.info("SHARP predictor ready.")

    shot_detector = ShotDetector(
        cut_threshold=args.cut_threshold,
        min_len_frames=max(1, int(args.min_shot_len * fps)),
        max_len_frames=max(1, int(args.max_shot_len * fps)),
    )

    writer = None
    keyframe: KeyframeState | None = None
    prev_rvec = None
    prev_tvec = None
    frame_idx = 0
    segment_id = 0
    pose_log = []
    per_frame_sharp = args.keyframe_mode == "per_frame"
    freeze_keyframes = args.keyframe_mode in {"freeze", "cache_only"}
    allow_predict = args.keyframe_mode != "cache_only"
    use_cache = (not per_frame_sharp) or args.cache_per_frame
    fatal_error = False
    return_code = 0

    try:
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
            low_ram_check_interval = 30
            last_low_ram_frame = -1
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
                batch_frames = []
                while len(batch_frames) < batch_size:
                    frame = _get_frame()
                    if frame is None:
                        break
                    batch_frames.append(frame)

                if not batch_frames:
                    break

                padded_frames = []
                for frame in batch_frames:
                    frame, (width_even, height_even) = pad_even(frame)
                    padded_frames.append(frame)

                if writer is None:
                    writer = FFmpegWriter(
                        out_path,
                        width_even * 2,
                        height_even,
                        fps,
                        args.ffmpeg_crf,
                        args.ffmpeg_preset,
                        encoder=args.encode,
                        audio_path=video_path if args.copy_audio else None,
                        audio_codec=args.audio_codec,
                    )

                rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in padded_frames]
                rgb_batch = np.stack(rgb_frames, axis=0)
                try:
                    batch_timer = Timer()
                    gaussians_batch = predictor.predict_gaussians_batch(rgb_batch, f_px)
                    logging.info(
                        "Per-frame batch predict: frames=%d ms=%.1f",
                        len(padded_frames),
                        batch_timer.elapsed_ms(),
                    )
                    gaussians_list = split_gaussians_batch(gaussians_batch)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    logging.warning("Per-frame batch OOM, falling back to batch=1: %s", exc)
                    gaussians_list = [predictor.predict_gaussians(rgb, f_px) for rgb in rgb_frames]

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
                    colors, depths, alphas = keyframe.renderer.render_batch(
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

                    sbs = np.concatenate([left_img, right_img], axis=1)
                    writer.write(sbs.tobytes())

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

                if args.max_frames and frame_idx >= args.max_frames:
                    break

            if producer is not None:
                stop_event.set()
                producer.join(timeout=2)
        else:
            scratch = None
            left_u8 = None
            right_u8 = None
            low_ram_check_interval = 30
            last_low_ram_frame = -1
            while True:
                frame = read_next_frame(cap, reader)
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
                    writer = FFmpegWriter(
                        out_path,
                        width_even * 2,
                        height_even,
                        fps,
                        args.ffmpeg_crf,
                        args.ffmpeg_preset,
                        encoder=args.encode,
                        audio_path=video_path if args.copy_audio else None,
                        audio_codec=args.audio_codec,
                    )

                cut = shot_detector.update(frame)
                if cut and freeze_keyframes and keyframe is not None:
                    logging.info(
                        "Shot cut detected but keyframe_mode=%s; reusing keyframe.",
                        args.keyframe_mode,
                    )
                need_keyframe = keyframe is None or (cut and not freeze_keyframes)

                new_keyframe_reason = None
                if need_keyframe:
                    new_keyframe_reason = "init" if keyframe is None else "cut"
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
                    sbs = np.concatenate([frame_rgb, frame_rgb], axis=1)
                    writer.write(sbs.tobytes())
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
                                sbs = np.concatenate([frame_rgb, frame_rgb], axis=1)
                                writer.write(sbs.tobytes())
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
                colors, depths, alphas = keyframe.renderer.render_batch(
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

                sbs = np.concatenate([left_img, right_img], axis=1)
                writer.write(sbs.tobytes())

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

        if debug_root:
            save_json(debug_root / "pose_log.json", {"items": pose_log})
        if fatal_error:
            logging.error("Keyframe cache_only failed: missing required cache.")

    finally:
        if cap is not None:
            cap.release()
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        if args.clear_cache_on_exit:
            _cleanup_cache_dir(Path(args.cache_dir))

    return return_code


if __name__ == "__main__":
    sys.exit(main())
