import logging
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
        elif self.encoder != "x264":
            logging.warning("Unknown encoder %s, falling back to x264.", self.encoder)

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
            "yuv420p",
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
