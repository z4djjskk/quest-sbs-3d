import hashlib
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: str | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(file_handler)


def pad_even(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    pad_w = width % 2
    pad_h = height % 2
    if pad_w == 0 and pad_h == 0:
        return frame, (width, height)
    padded = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return padded, (width + pad_w, height + pad_h)


def compute_hist_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_depth_png(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return
    depth_vis = depth.copy()
    depth_vis[~valid] = 0
    min_val = float(np.percentile(depth_vis[valid], 1))
    max_val = float(np.percentile(depth_vis[valid], 99))
    if max_val <= min_val:
        max_val = min_val + 1e-3
    norm = (depth_vis - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    img = (norm * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    cleaned = cleaned.strip("._")
    if not cleaned:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"file_{digest}"
    if cleaned != name:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned}_{digest}"
    return cleaned


def video_cache_key(video_path: Path) -> str:
    stat = video_path.stat()
    payload = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def set_env_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
