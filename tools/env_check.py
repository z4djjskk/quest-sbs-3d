import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _add_dll_dir(path: Path) -> None:
    try:
        os.add_dll_directory(str(path))
    except (AttributeError, OSError):
        return


_CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
_CUDA_BIN = _CUDA_ROOT / "bin" / "x64"
if not _CUDA_BIN.exists():
    _CUDA_BIN = _CUDA_ROOT / "bin"
_OPENCV_BIN = Path(os.environ.get("OPENCV_BIN", ""))
if not _OPENCV_BIN.exists():
    candidate = Path(r"F:\build\opencv_cuda\build_sm120\install\x64\vc17\bin")
    if candidate.exists():
        _OPENCV_BIN = candidate
if _OPENCV_BIN.exists():
    _add_dll_dir(_OPENCV_BIN)
if _CUDA_BIN.exists():
    _add_dll_dir(_CUDA_BIN)


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def check_ffmpeg() -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run([exe, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main() -> int:
    ok = True

    print("Python:", sys.version.split()[0])

    torch_ok = check_import("torch")
    print("torch:", "OK" if torch_ok else "MISSING")
    if torch_ok:
        import torch

        print("torch.cuda:", "OK" if torch.cuda.is_available() else "MISSING")
        print("torch.cuda.version:", torch.version.cuda)
        if not torch.cuda.is_available():
            ok = False
    else:
        ok = False

    cv2_ok = check_import("cv2")
    print("opencv:", "OK" if cv2_ok else "MISSING")
    if cv2_ok:
        import cv2

        cuda_ok = False
        try:
            cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            cuda_ok = False
        print("opencv.cuda:", "OK" if cuda_ok else "MISSING")
    ok = ok and cv2_ok

    numpy_ok = check_import("numpy")
    print("numpy:", "OK" if numpy_ok else "MISSING")
    ok = ok and numpy_ok

    sharp_ok = check_import("sharp")
    print("sharp:", "OK" if sharp_ok else "MISSING")
    ok = ok and sharp_ok

    gsplat_ok = check_import("gsplat")
    print("gsplat:", "OK" if gsplat_ok else "MISSING")
    ok = ok and gsplat_ok

    flask_ok = check_import("flask")
    print("flask:", "OK" if flask_ok else "MISSING")
    ok = ok and flask_ok

    ffmpeg_ok = check_ffmpeg()
    print("ffmpeg:", "OK" if ffmpeg_ok else "MISSING")
    ok = ok and ffmpeg_ok
    if ffmpeg_ok:
        exe = shutil.which("ffmpeg")
        try:
            encoders = subprocess.run(
                [exe, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            print("ffmpeg.nvenc:", "OK" if "h264_nvenc" in encoders else "MISSING")
            print("ffmpeg.hevc_nvenc:", "OK" if "hevc_nvenc" in encoders else "MISSING")
        except Exception:
            print("ffmpeg.nvenc:", "MISSING")
            print("ffmpeg.hevc_nvenc:", "MISSING")
        try:
            hwaccels = subprocess.run(
                [exe, "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            print("ffmpeg.cuda:", "OK" if "cuda" in hwaccels else "MISSING")
        except Exception:
            print("ffmpeg.cuda:", "MISSING")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
