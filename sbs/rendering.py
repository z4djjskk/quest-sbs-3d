from __future__ import annotations

import numpy as np


def build_intrinsics(f_px: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [f_px, 0.0, (width - 1) / 2.0, 0.0],
            [0.0, f_px, (height - 1) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def pose_to_extrinsics(rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = rmat
    extrinsics[:3, 3] = tvec.reshape(3)
    return extrinsics


def compute_stereo_extrinsics(
    rmat: np.ndarray,
    tvec: np.ndarray,
    baseline_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    cam_center = -rmat.T @ tvec.reshape(3)
    right_world = rmat.T @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

    half = baseline_m / 2.0
    left_center = cam_center - right_world * half
    right_center = cam_center + right_world * half

    t_left = -rmat @ left_center
    t_right = -rmat @ right_center

    left_extr = pose_to_extrinsics(rmat, t_left)
    right_extr = pose_to_extrinsics(rmat, t_right)
    return left_extr, right_extr


def clamp_baseline(
    baseline_m: float,
    max_disp_px: float,
    median_depth: float,
    f_px: float,
    min_baseline_m: float,
) -> float:
    if median_depth <= 0:
        return baseline_m
    max_baseline = max_disp_px * median_depth / max(f_px, 1e-6)
    baseline = min(baseline_m, max_baseline)
    return max(baseline, min_baseline_m)


def inpaint_holes(color: np.ndarray, alpha: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return color
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    mask = (alpha < 0.5).astype(np.uint8) * 255
    if mask.sum() == 0:
        return color
    import cv2

    return cv2.inpaint(color, mask, radius, cv2.INPAINT_TELEA)