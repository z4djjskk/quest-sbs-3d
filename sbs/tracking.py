from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class PnPResult:
    success: bool
    rvec: np.ndarray
    tvec: np.ndarray
    inliers: np.ndarray | None
    reproj_error: float


def cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _normalize_points(pts: np.ndarray | None) -> np.ndarray:
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(pts, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = arr.reshape(-1, 2)
    return np.ascontiguousarray(arr)


def _create_cuda_gftt_detector(
    max_features: int,
    quality: float,
    min_distance: int,
):
    if hasattr(cv2, "cuda_GoodFeaturesToTrackDetector"):
        return cv2.cuda_GoodFeaturesToTrackDetector.create(
            cv2.CV_8UC1,
            maxCorners=max_features,
            qualityLevel=quality,
            minDistance=min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )
    cuda_mod = getattr(cv2, "cuda", None)
    creator = getattr(cuda_mod, "createGoodFeaturesToTrackDetector", None) if cuda_mod else None
    if callable(creator):
        return creator(
            cv2.CV_8UC1,
            maxCorners=max_features,
            qualityLevel=quality,
            minDistance=min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )
    return None


def _select_keypoints_cpu(
    gray: np.ndarray,
    max_features: int,
    quality: float,
    min_distance: int,
) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_features,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False,
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return _normalize_points(pts)


def _select_keypoints_cuda(
    gray: np.ndarray,
    max_features: int,
    quality: float,
    min_distance: int,
) -> np.ndarray:
    if not cuda_available():
        LOGGER.warning("OpenCV CUDA not available, falling back to CPU keypoints.")
        return _select_keypoints_cpu(gray, max_features, quality, min_distance)
    try:
        detector = _create_cuda_gftt_detector(max_features, quality, min_distance)
        if detector is None:
            LOGGER.warning(
                "CUDA GFTT detector not available in Python bindings, falling back to CPU keypoints."
            )
            return _select_keypoints_cpu(gray, max_features, quality, min_distance)
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray)
        gpu_pts = detector.detect(gpu_gray)
        if gpu_pts is None:
            return np.empty((0, 2), dtype=np.float32)
        pts = gpu_pts.download()
        if pts is None or len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return _normalize_points(pts)
    except Exception as exc:
        LOGGER.warning("CUDA keypoint detect failed, falling back to CPU: %s", exc)
        return _select_keypoints_cpu(gray, max_features, quality, min_distance)


def select_keypoints(
    gray: np.ndarray,
    max_features: int,
    quality: float,
    min_distance: int,
    backend: str = "cpu",
) -> np.ndarray:
    if backend == "opencv_cuda":
        return _select_keypoints_cuda(gray, max_features, quality, min_distance)
    return _select_keypoints_cpu(gray, max_features, quality, min_distance)


def _track_keypoints_cpu(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
    fb_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_pts = _normalize_points(prev_pts)
    if len(prev_pts) == 0:
        return prev_pts, prev_pts, np.array([], dtype=np.int32)
    prev_pts_in = prev_pts.reshape(-1, 1, 2)
    curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts_in,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        curr_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if curr_pts is None or back_pts is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([], dtype=np.int32),
        )

    curr_pts = curr_pts.reshape(-1, 2)
    back_pts = back_pts.reshape(-1, 2)
    st = st.reshape(-1)
    st_back = st_back.reshape(-1)

    fb_err = np.linalg.norm(prev_pts - back_pts, axis=1)
    mask = (st == 1) & (st_back == 1) & (fb_err < fb_thresh)
    idx = np.arange(len(prev_pts), dtype=np.int32)[mask]
    return prev_pts[mask], curr_pts[mask], idx


def _track_keypoints_cuda(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
    fb_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_pts = _normalize_points(prev_pts)
    if len(prev_pts) == 0:
        return prev_pts, prev_pts, np.array([], dtype=np.int32)
    if not cuda_available():
        LOGGER.warning("OpenCV CUDA not available, falling back to CPU tracking.")
        return _track_keypoints_cpu(prev_gray, curr_gray, prev_pts, fb_thresh)

    try:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        gpu_curr.upload(curr_gray)

        gpu_prev_pts = cv2.cuda_GpuMat()
        prev_pts_cuda = np.ascontiguousarray(prev_pts.reshape(1, -1, 2))
        gpu_prev_pts.upload(prev_pts_cuda)

        lk = cv2.cuda_SparsePyrLKOpticalFlow.create(
            winSize=(21, 21),
            maxLevel=3,
            iters=30,
        )
        gpu_curr_pts, st, _ = lk.calc(gpu_prev, gpu_curr, gpu_prev_pts, None)
        gpu_back_pts, st_back, _ = lk.calc(gpu_curr, gpu_prev, gpu_curr_pts, None)

        if gpu_curr_pts is None or gpu_back_pts is None or st is None or st_back is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=np.int32),
            )
        curr_pts = gpu_curr_pts.download().reshape(-1, 2)
        back_pts = gpu_back_pts.download().reshape(-1, 2)
        st = st.download().reshape(-1)
        st_back = st_back.download().reshape(-1)

        fb_err = np.linalg.norm(prev_pts - back_pts, axis=1)
        mask = (st == 1) & (st_back == 1) & (fb_err < fb_thresh)
        idx = np.arange(len(prev_pts), dtype=np.int32)[mask]
        return prev_pts[mask], curr_pts[mask], idx
    except Exception as exc:
        LOGGER.warning("CUDA tracking failed, falling back to CPU: %s", exc)
        return _track_keypoints_cpu(prev_gray, curr_gray, prev_pts, fb_thresh)


def track_keypoints(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
    fb_thresh: float,
    backend: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "opencv_cuda":
        return _track_keypoints_cuda(prev_gray, curr_gray, prev_pts, fb_thresh)
    return _track_keypoints_cpu(prev_gray, curr_gray, prev_pts, fb_thresh)


def filter_fundamental(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    indices: np.ndarray,
    ransac_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prev_pts) < 8:
        return prev_pts, curr_pts, indices
    fmat, mask = cv2.findFundamentalMat(
        prev_pts,
        curr_pts,
        cv2.FM_RANSAC,
        ransac_thresh,
        0.99,
    )
    if mask is None:
        return prev_pts, curr_pts, indices
    mask = mask.ravel().astype(bool)
    return prev_pts[mask], curr_pts[mask], indices[mask]


def solve_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    k_mat: np.ndarray,
    ransac_iters: int,
    reproj_error: float,
) -> PnPResult:
    if len(obj_pts) < 6:
        return PnPResult(False, np.zeros((3, 1)), np.zeros((3, 1)), None, float("inf"))

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        k_mat,
        None,
        iterationsCount=ransac_iters,
        reprojectionError=reproj_error,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return PnPResult(False, np.zeros((3, 1)), np.zeros((3, 1)), None, float("inf"))

    if inliers is not None and len(inliers) >= 6:
        inlier_obj = obj_pts[inliers.ravel()]
        inlier_img = img_pts[inliers.ravel()]
        success_refine, rvec, tvec = cv2.solvePnP(
            inlier_obj,
            inlier_img,
            k_mat,
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success_refine:
            return PnPResult(False, rvec, tvec, inliers, float("inf"))

    reproj = compute_reprojection_error(obj_pts, img_pts, k_mat, rvec, tvec, inliers)
    return PnPResult(True, rvec, tvec, inliers, reproj)


def compute_reprojection_error(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    k_mat: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    inliers: np.ndarray | None,
) -> float:
    if inliers is not None:
        obj_pts = obj_pts[inliers.ravel()]
        img_pts = img_pts[inliers.ravel()]
    if len(obj_pts) == 0:
        return float("inf")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k_mat, None)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.mean(err))


def smooth_pose(
    prev_rvec: np.ndarray,
    prev_tvec: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if prev_rvec is None or prev_tvec is None:
        return rvec, tvec

    r_prev = cv2.Rodrigues(prev_rvec)[0]
    r_curr = cv2.Rodrigues(rvec)[0]

    q_prev = rotmat_to_quat(r_prev)
    q_curr = rotmat_to_quat(r_curr)
    q_smooth = slerp(q_prev, q_curr, alpha)

    r_smooth = quat_to_rotmat(q_smooth)
    rvec_smooth, _ = cv2.Rodrigues(r_smooth)
    t_smooth = (1.0 - alpha) * prev_tvec + alpha * tvec
    return rvec_smooth, t_smooth


def rotmat_to_quat(rmat: np.ndarray) -> np.ndarray:
    trace = np.trace(rmat)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rmat[2, 1] - rmat[1, 2]) * s
        y = (rmat[0, 2] - rmat[2, 0]) * s
        z = (rmat[1, 0] - rmat[0, 1]) * s
    else:
        if rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
            w = (rmat[2, 1] - rmat[1, 2]) / s
            x = 0.25 * s
            y = (rmat[0, 1] + rmat[1, 0]) / s
            z = (rmat[0, 2] + rmat[2, 0]) / s
        elif rmat[1, 1] > rmat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
            w = (rmat[0, 2] - rmat[2, 0]) / s
            x = (rmat[0, 1] + rmat[1, 0]) / s
            y = 0.25 * s
            z = (rmat[1, 2] + rmat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
            w = (rmat[1, 0] - rmat[0, 1]) / s
            x = (rmat[0, 2] + rmat[2, 0]) / s
            y = (rmat[1, 2] + rmat[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = (s0 * q0) + (s1 * q1)
    return result / np.linalg.norm(result)
