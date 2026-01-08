import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _ffmpeg_encoders() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout or ""
    except Exception:
        return ""


def _has_encoder(name: str) -> bool:
    return name in _ffmpeg_encoders()


@lru_cache(maxsize=8)
def _encoder_help(encoder: str) -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "") + "\n" + (result.stderr or "")
    except Exception:
        return ""


def _encoder_pix_fmts(encoder: str) -> set[str]:
    payload = _encoder_help(encoder)
    for line in payload.splitlines():
        if "Supported pixel formats:" in line:
            parts = line.split("Supported pixel formats:", 1)[1]
            return set(parts.strip().split())
    return set()


def _supports_pix_fmt(encoder: str, pix_fmt: str) -> bool:
    return pix_fmt in _encoder_pix_fmts(encoder)


def _supports_option(encoder: str, option: str) -> bool:
    token = option.strip().lstrip('-')
    payload = _encoder_help(encoder)
    for line in payload.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(token + " " ) or stripped.startswith("-" + token + " " ):
            return True
    return False


class FFmpegWriter:
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        encoder: str = "x264",
        audio_path: Path | None = None,
        audio_codec: str = "copy",
        x264_lowmem: bool = False,
        nvenc_lowmem: bool = False,
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.encoder = encoder
        self.audio_path = audio_path
        self.audio_codec = audio_codec
        self.x264_lowmem = bool(x264_lowmem)
        self.nvenc_lowmem = bool(nvenc_lowmem)
        self.proc = self._start()

    def _start(self) -> subprocess.Popen:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "libx264"
        preset = self.preset
        extra = []
        extra_x264 = []
        lossless = self.crf <= 0
        pixel_count = self.width * self.height
        large_frame = pixel_count >= 12_000_000
        try:
            nvenc_lowmem_area = int(os.environ.get("SBS_NVENC_LOW_MEM_AREA", "4000000"))
        except ValueError:
            nvenc_lowmem_area = 4_000_000
        nvenc_lowmem = self.nvenc_lowmem or pixel_count >= nvenc_lowmem_area
        if self.encoder in ("nvenc", "hevc_nvenc"):
            preset_map = {"slow": "p7", "medium": "p4", "fast": "p2"}
            preset = preset_map.get(self.preset, "p4")
            target = "h264_nvenc" if self.encoder == "nvenc" else "hevc_nvenc"

            if self.encoder == "nvenc" and (self.width > 4096 or self.height > 4096):
                target = "hevc_nvenc"
                if _has_encoder(target):
                    logging.warning(
                        "NVENC H.264 max 4096 in width/height; using hevc_nvenc for %dx%d.",
                        self.width,
                        self.height,
                    )
                else:
                    logging.warning(
                        "NVENC H.264 max 4096 in width/height; hevc_nvenc missing, falling back to x264 for %dx%d.",
                        self.width,
                        self.height,
                    )
                    target = ""

            if target and _has_encoder(target):
                codec = target
                if lossless:
                    extra = ["-rc", "constqp", "-qp", "0"]
                else:
                    extra = ["-rc", "vbr", "-cq", str(self.crf)]
            elif target:
                logging.warning("FFmpeg encoder %s not available; falling back to x264.", target)
        elif self.encoder == "x264" and lossless and large_frame:
            logging.warning(
                "Large lossless encode at %dx%d; using low-memory x264 settings.",
                self.width,
                self.height,
            )
            preset = "ultrafast"
            extra_x264 = ["-tune", "zerolatency", "-x264-params", "rc-lookahead=0:bframes=0"]
        elif self.encoder == "x264" and self.x264_lowmem and large_frame:
            logging.warning(
                "x264 low-memory mode for %dx%d.",
                self.width,
                self.height,
            )
            preset = "ultrafast"
            extra_x264 = ["-tune", "zerolatency", "-x264-params", "rc-lookahead=0:bframes=0"]
        elif self.encoder != "x264":
            logging.warning("Unknown encoder %s, falling back to x264.", self.encoder)

        pix_fmt = "yuv420p"
        if lossless:
            if _supports_pix_fmt(codec, "yuv444p"):
                pix_fmt = "yuv444p"
            else:
                logging.warning(
                    "Lossless requested but %s lacks yuv444p; using yuv420p.",
                    codec,
                )

        if codec in ("h264_nvenc", "hevc_nvenc") and (
            nvenc_lowmem or self.width > 4096 or self.height > 4096
        ):
            low_mem_opts = []
            if _supports_option(codec, "surfaces"):
                low_mem_opts += ["-surfaces", "1"]
            if _supports_option(codec, "rc-lookahead"):
                low_mem_opts += ["-rc-lookahead", "0"]
            if _supports_option(codec, "bf"):
                low_mem_opts += ["-bf", "0"]
            if _supports_option(codec, "b_ref_mode"):
                low_mem_opts += ["-b_ref_mode", "disabled"]
            if low_mem_opts:
                extra += low_mem_opts
                logging.warning(
                    "NVENC low-memory mode for %dx%d: %s",
                    self.width,
                    self.height,
                    " ".join(low_mem_opts),
                )

        audio_path = self.audio_path
        if audio_path and not audio_path.exists():
            logging.warning("Audio source not found: %s; skipping audio copy.", audio_path)
            audio_path = None

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps}",
            "-i",
            "-",
        ]
        if audio_path:
            cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a?"]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v",
            codec,
            "-pix_fmt",
            pix_fmt,
            "-preset",
            preset,
        ]
        if codec == "libx264":
            cmd += ["-crf", str(self.crf)]
            cmd += extra_x264
        else:
            cmd += extra
        if audio_path:
            cmd += ["-c:a", self.audio_codec, "-shortest"]

        cmd += [
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        logging.info("FFmpeg: %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_rgb: bytes) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg stdin is closed")
        self.proc.stdin.write(frame_rgb)

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg exited with code {ret}")
