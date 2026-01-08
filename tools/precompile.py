import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _add_dll_dir(path: Path) -> None:
    try:
        os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        return

def _find_cl_exe() -> Path | None:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            install_path = result.stdout.strip().splitlines()[0] if result.stdout else ""
            if install_path:
                msvc_root = Path(install_path) / "VC" / "Tools" / "MSVC"
                if msvc_root.exists():
                    matches = list(msvc_root.glob("**/bin/Hostx64/x64/cl.exe"))
                    if matches:
                        return matches[0]
        except Exception:
            pass

    buildtools_root = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
    if buildtools_root.exists():
        matches = list(buildtools_root.glob("**/bin/Hostx64/x64/cl.exe"))
        if matches:
            return matches[0]
    return None


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

_cl_exe = _find_cl_exe()
if _cl_exe:
    cl_bin = _cl_exe.parent
    os.environ["PATH"] = f"{cl_bin};{os.environ.get('PATH', '')}"

from sbs.rendering import build_intrinsics
from sbs.sharp_backend import SharpPredictor, SharpRenderer
from sharp.utils.gaussians import Gaussians3D


def _resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device(req if ":" in req else "cuda:0")
    return torch.device("cpu")


def _make_dummy_gaussians(device: torch.device) -> Gaussians3D:
    mean_vectors = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    singular_values = torch.full((1, 1, 3), 0.01, device=device, dtype=torch.float32)
    quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    colors = torch.full((1, 1, 3), 0.5, device=device, dtype=torch.float32)
    opacities = torch.full((1, 1), 0.5, device=device, dtype=torch.float32)
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompile CUDA extensions and warm caches")
    parser.add_argument("--device", default="cuda", help="Torch device (auto|cuda|cuda:N|cpu)")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FP16 autocast for warmup",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned memory for SHARP",
    )
    parser.add_argument("--sharp_ckpt", default=None, help="Optional SHARP checkpoint path")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    try:
        device = _resolve_device(args.device)
    except RuntimeError as exc:
        logging.error("Device error: %s", exc)
        return 2

    logging.info("Precompile start. device=%s amp=%s", device, args.amp)

    predictor = SharpPredictor(
        device=str(device),
        checkpoint_path=args.sharp_ckpt,
        use_amp=args.amp,
        pin_memory=args.pin_memory,
    )
    logging.info("SHARP weights ready.")

    if device.type != "cuda":
        logging.warning("CUDA not available; skipping GSplat warmup.")
        return 0

    gaussians = _make_dummy_gaussians(device)
    renderer = SharpRenderer("linearRGB", use_amp=args.amp)
    width = 16
    height = 16
    intrinsics = build_intrinsics(16.0, width, height)
    extrinsics = np.eye(4, dtype=np.float32)

    logging.info("GSplat warmup...")
    renderer.render(gaussians, intrinsics, extrinsics, width, height, device, need_depth=False, need_alpha=False)
    torch.cuda.synchronize()
    logging.info("Precompile done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
