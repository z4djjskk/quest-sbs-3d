from __future__ import annotations

import logging
import os
import ctypes
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import requests

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
from sharp.utils import linalg
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    compose_covariance_matrices,
    load_ply,
)
from sharp.utils.gsplat import GSplatRenderer
from sbs.ply_utils import save_ply_fast

LOGGER = logging.getLogger(__name__)
_EIG_CHUNK_WARNED = False


def _available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
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


def _auto_chunk_cap() -> int | None:
    avail = _available_memory_bytes()
    if not avail or avail <= 0:
        return None
    target = int(avail * 0.25)
    bytes_per = 1024
    cap = max(1024, target // bytes_per)
    return cap if cap > 0 else None


@contextmanager
def _maybe_autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


class SharpPredictor:
    def __init__(self, device: str, checkpoint_path: str | None, use_amp: bool = False) -> None:
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.model = create_predictor(PredictorParams())
        self._load_weights(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    def _load_weights(self, checkpoint_path: str | None) -> None:
        if checkpoint_path:
            LOGGER.info("Loading SHARP checkpoint from %s", checkpoint_path)
            try:
                state_dict = torch.load(checkpoint_path, weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            LOGGER.info("Downloading SHARP checkpoint from %s", DEFAULT_MODEL_URL)
            try:
                state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
            except Exception as exc:
                LOGGER.warning("Default download failed: %s", exc)
                cached_path = _get_checkpoint_cache_path(DEFAULT_MODEL_URL)
                if not cached_path.exists():
                    _download_checkpoint_insecure(DEFAULT_MODEL_URL, cached_path)
                try:
                    state_dict = torch.load(cached_path, weights_only=True)
                except TypeError:
                    state_dict = torch.load(cached_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def predict_gaussians(
        self,
        image_rgb: np.ndarray,
        f_px: float,
    ) -> Gaussians3D:
        with torch.inference_mode():
            try:
                return _predict_image_amp_safe(
                    self.model,
                    image_rgb,
                    f_px,
                    self.device,
                    use_amp=self.use_amp,
                )
            except RuntimeError as exc:
                if not self.use_amp or "Low precision dtypes not supported" not in str(exc):
                    raise
                LOGGER.warning("AMP disabled for SHARP unprojection due to: %s", exc)
                return _predict_image_amp_safe(
                    self.model,
                    image_rgb,
                    f_px,
                    self.device,
                    use_amp=False,
                )

    def predict_gaussians_batch(
        self,
        images_rgb: np.ndarray,
        f_px: float,
    ) -> Gaussians3D:
        with torch.inference_mode():
            try:
                return _predict_batch_amp_safe(
                    self.model,
                    images_rgb,
                    f_px,
                    self.device,
                    use_amp=self.use_amp,
                )
            except RuntimeError as exc:
                if not self.use_amp or "Low precision dtypes not supported" not in str(exc):
                    raise
                LOGGER.warning("AMP disabled for SHARP unprojection due to: %s", exc)
                return _predict_batch_amp_safe(
                    self.model,
                    images_rgb,
                    f_px,
                    self.device,
                    use_amp=False,
                )


class SharpRenderer:
    def __init__(self, color_space: str, use_amp: bool = False) -> None:
        self.renderer = GSplatRenderer(color_space=color_space)
        self.use_amp = bool(use_amp)

    def render(
        self,
        gaussians: Gaussians3D,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        width: int,
        height: int,
        device: torch.device,
        need_depth: bool = True,
        need_alpha: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        intr = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32).unsqueeze(0)
        extr = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            with _maybe_autocast(self.use_amp, device):
                outputs = self.renderer(
                    gaussians,
                    extrinsics=extr,
                    intrinsics=intr,
                    image_width=width,
                    image_height=height,
                )
        color = outputs.color[0].permute(1, 2, 0).detach().cpu().numpy()
        depth = outputs.depth[0].squeeze(0).detach().cpu().numpy() if need_depth else None
        alpha = outputs.alpha[0].squeeze(0).detach().cpu().numpy() if need_alpha else None
        return color, depth, alpha

    def render_batch(
        self,
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
        if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
            raise ValueError("extrinsics must be shaped as (N, 4, 4)")
        batch = extrinsics.shape[0]
        gaussians = _repeat_gaussians(gaussians, batch)
        if intrinsics_t is not None:
            intr = intrinsics_t.to(device=device, dtype=torch.float32)
            if intr.dim() == 2:
                intr = intr.unsqueeze(0)
            if intr.shape[0] == 1:
                intr = intr.expand(batch, -1, -1)
        else:
            intr = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32).unsqueeze(0)
            intr = intr.expand(batch, -1, -1)
        extr = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            with _maybe_autocast(self.use_amp, device):
                outputs = self.renderer(
                    gaussians,
                    extrinsics=extr,
                    intrinsics=intr,
                    image_width=width,
                    image_height=height,
                )
        if outputs.color.shape[0] != batch:
            LOGGER.warning(
                "Renderer batch output=%d, expected=%d. Falling back to per-eye render.",
                outputs.color.shape[0],
                batch,
            )
            colors = []
            depths = [] if need_depth else None
            alphas = [] if need_alpha else None
            for i in range(batch):
                color, depth, alpha = self.render(
                    gaussians,
                    intrinsics,
                    extrinsics[i],
                    width,
                    height,
                    device,
                    need_depth=need_depth,
                    need_alpha=need_alpha,
                )
                colors.append(color)
                if need_depth and depths is not None:
                    depths.append(depth)
                if need_alpha and alphas is not None:
                    alphas.append(alpha)
            color = np.stack(colors, axis=0)
            depth = np.stack(depths, axis=0) if need_depth and depths is not None else None
            alpha = np.stack(alphas, axis=0) if need_alpha and alphas is not None else None
            return color, depth, alpha
        color = outputs.color.permute(0, 2, 3, 1).detach().cpu().numpy()
        depth = outputs.depth.squeeze(1).detach().cpu().numpy() if need_depth else None
        alpha = outputs.alpha.squeeze(1).detach().cpu().numpy() if need_alpha else None
        return color, depth, alpha


def _repeat_gaussians(gaussians: Gaussians3D, batch: int) -> Gaussians3D:
    if batch <= 1:
        return gaussians
    current = gaussians.mean_vectors.shape[0]
    if current == batch or current != 1:
        return gaussians

    def _repeat(tensor: torch.Tensor) -> torch.Tensor:
        reps = [batch] + [1] * (tensor.dim() - 1)
        return tensor.repeat(*reps)

    return Gaussians3D(
        mean_vectors=_repeat(gaussians.mean_vectors),
        singular_values=_repeat(gaussians.singular_values),
        quaternions=_repeat(gaussians.quaternions),
        colors=_repeat(gaussians.colors),
        opacities=_repeat(gaussians.opacities),
    )


def _predict_image_amp_safe(
    predictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_amp: bool,
) -> Gaussians3D:
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width], device=device, dtype=torch.float32)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    with _maybe_autocast(use_amp, device):
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = torch.tensor(
        [
            [f_px, 0.0, width / 2.0, 0.0],
            [0.0, f_px, height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return _unproject_gaussians_fp32(
                gaussians_ndc,
                torch.eye(4, device=device, dtype=torch.float32),
                intrinsics_resized,
                internal_shape,
            )
    return _unproject_gaussians_fp32(
        gaussians_ndc,
        torch.eye(4, device=device, dtype=torch.float32),
        intrinsics_resized,
        internal_shape,
    )


def _predict_batch_amp_safe(
    predictor,
    images: np.ndarray,
    f_px: float,
    device: torch.device,
    use_amp: bool,
) -> Gaussians3D:
    images_np = np.asarray(images)
    if images_np.ndim == 3:
        images_np = images_np[None, ...]
    if images_np.ndim != 4:
        raise ValueError("images must be shaped as (B, H, W, 3)")
    batch = images_np.shape[0]
    if batch == 1:
        return _predict_image_amp_safe(predictor, images_np[0], f_px, device, use_amp)

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(images_np.copy()).float().to(device) / 255.0
    image_pt = image_pt.permute(0, 3, 1, 2)
    _, _, height, width = image_pt.shape
    disparity = f_px / width
    disparity_factor = torch.full((batch,), disparity, device=device, dtype=torch.float32)

    image_resized_pt = F.interpolate(
        image_pt,
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    with _maybe_autocast(use_amp, device):
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = torch.tensor(
        [
            [f_px, 0.0, width / 2.0, 0.0],
            [0.0, f_px, height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return _unproject_gaussians_fp32(
                gaussians_ndc,
                torch.eye(4, device=device, dtype=torch.float32),
                intrinsics_resized,
                internal_shape,
            )
    return _unproject_gaussians_fp32(
        gaussians_ndc,
        torch.eye(4, device=device, dtype=torch.float32),
        intrinsics_resized,
        internal_shape,
    )


def _unproject_gaussians_fp32(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    unprojection_matrix = _get_unprojection_matrix_fp32(extrinsics, intrinsics, image_shape)
    return _apply_transform_fp32(gaussians_ndc, unprojection_matrix[:3])


def _get_unprojection_matrix_fp32(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    device = intrinsics.device
    image_width, image_height = image_shape
    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    mat = ndc_matrix @ intrinsics.to(dtype=torch.float32) @ extrinsics.to(dtype=torch.float32)
    return torch.linalg.inv(mat)


def _apply_transform_fp32(gaussians: Gaussians3D, transform: torch.Tensor) -> Gaussians3D:
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors.to(dtype=torch.float32)
    quaternions = gaussians.quaternions.to(dtype=torch.float32)
    singular_values = gaussians.singular_values.to(dtype=torch.float32)
    colors = gaussians.colors.to(dtype=torch.float32)
    opacities = gaussians.opacities.to(dtype=torch.float32)

    mean_vectors = mean_vectors @ transform_linear.transpose(-1, -2) + transform_offset
    covariance_matrices = compose_covariance_matrices(quaternions, singular_values)
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )
    quaternions_out, singular_values_out = _decompose_covariance_matrices_stable(
        covariance_matrices
    )

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values_out,
        quaternions=quaternions_out,
        colors=colors,
        opacities=opacities,
    )


def _decompose_covariance_matrices_stable(
    covariance_matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype

    cov_flat = covariance_matrices.reshape(-1, 3, 3)
    total = int(cov_flat.shape[0])
    if total == 0:
        out_shape = covariance_matrices.shape[:-2]
        empty_quat = torch.empty((*out_shape, 4), device=device, dtype=dtype)
        empty_sv = torch.empty((*out_shape, 3), device=device, dtype=dtype)
        return empty_quat, empty_sv

    chunk_size = int(os.environ.get("SHARP_EIG_CHUNK_SIZE", "262144"))
    max_chunk = int(os.environ.get("SHARP_EIG_CHUNK_SIZE_MAX", "0"))
    if chunk_size <= 0:
        chunk_size = total
    auto_cap = _auto_chunk_cap()
    effective_max = max_chunk if max_chunk > 0 else None
    if auto_cap:
        effective_max = auto_cap if effective_max is None else min(effective_max, auto_cap)
    if effective_max:
        if chunk_size > effective_max:
            global _EIG_CHUNK_WARNED
            if not _EIG_CHUNK_WARNED:
                LOGGER.warning(
                    "EIG chunk capped from %d to %d (available memory)",
                    chunk_size,
                    effective_max,
                )
                _EIG_CHUNK_WARNED = True
            chunk_size = effective_max
    chunk_size = min(chunk_size, total)

    quaternions_cpu = torch.empty((total, 4), device="cpu", dtype=torch.float32)
    singular_values_cpu = torch.empty((total, 3), device="cpu", dtype=torch.float32)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        cov_cpu = cov_flat[start:end].detach().cpu().to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
        det = torch.linalg.det(eigvecs)
        sign = torch.where(det < 0, -1.0, 1.0).to(eigvecs.dtype)
        eigvecs[..., :, -1] *= sign.unsqueeze(-1)

        eigvals = torch.clamp(eigvals, min=0.0)
        singular_values = eigvals.sqrt().to(torch.float32)
        eigvecs = eigvecs.to(torch.float32)
        quaternions = linalg.quaternions_from_rotation_matrices(eigvecs)
        quaternions_cpu[start:end] = quaternions.cpu()
        singular_values_cpu[start:end] = singular_values.cpu()

    quaternions = quaternions_cpu.to(dtype=dtype, device=device)
    singular_values = singular_values_cpu.to(dtype=dtype, device=device)
    out_shape = covariance_matrices.shape[:-2]
    return (
        quaternions.view(*out_shape, 4),
        singular_values.view(*out_shape, 3),
    )


def load_or_predict_ply(
    predictor: SharpPredictor,
    image_rgb: np.ndarray,
    f_px: float,
    ply_path: Path,
) -> tuple[Gaussians3D, SceneMetaData]:
    if ply_path.exists():
        gaussians, metadata = load_ply(ply_path)
        return gaussians, metadata

    gaussians = predictor.predict_gaussians(image_rgb, f_px)
    save_ply_fast(gaussians, f_px, (image_rgb.shape[0], image_rgb.shape[1]), ply_path)
    metadata = SceneMetaData(f_px, (image_rgb.shape[1], image_rgb.shape[0]), "linearRGB")
    return gaussians, metadata


def _get_checkpoint_cache_path(url: str) -> Path:
    filename = os.path.basename(url)
    hub_dir = Path(torch.hub.get_dir())
    cache_dir = hub_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _download_checkpoint_insecure(url: str, dst: Path) -> None:
    LOGGER.warning("Downloading checkpoint with SSL verification disabled.")
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with requests.get(url, stream=True, verify=False, timeout=60) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(dst)
