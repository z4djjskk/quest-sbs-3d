# Video to SBS (Quest-friendly 3D)
English | [Chinese](README.zh-CN.md)

Goal: Convert any 2D video into a Quest-friendly, high-quality 3D stereoscopic video (SBS side-by-side MP4).

## Install

Step 1 / Python dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Step 2 / Install ffmpeg

Make sure `ffmpeg` is available in PATH:

```bash
ffmpeg -version
```

Step 3 / Environment check

```bash
python tools/env_check.py
```

## One-click run

```bash
python tools/video_to_sbs.py --video input.mp4 --out output_sbs.mp4
```

## Optional web UI

Start the local UI (run button + progress bar):

```bash
python tools/web_server.py
```

Open `http://127.0.0.1:7860`, upload a video to local cache, and click "Download output" after it finishes.

## Key parameters (quality first)

- `--baseline_m`: default 0.03m, comfort first.
- `--max_disp_px`: max disparity in pixels, default 30. Larger = stronger depth but more motion sickness.
- `--fov_deg`: default 60 (horizontal FOV). If depth feels wrong, adjust this first.
- `--target_fps`: resample input to target FPS (default 24).
- `--cut_threshold` / `--max_shot_len`: shot cut stability.
- `--keyframe_refresh_s`: force SHARP keyframe refresh every N seconds (0=disable), useful for dynamic motion.
- `--debug_dir`: export keyframe depth, sampled SBS, and pose logs.
- `--log_interval`: log every N frames (default 1); increase to reduce logging overhead.

## Definition of Done (checked/logged in code)

1) Output quality (visual)
- Comfortable disparity: per-frame clamp with `--baseline_m` + `--max_disp_px`.
- Temporal stability: light PnP smoothing; per-frame `inliers/reproj` logs.
- Occlusion boundary handling: lightweight alpha-based hole filling (`--inpaint_radius`).

2) Robustness
- Auto shot cuts + recovery: histogram-based cuts + PnP failure triggers new keyframes.
- Dynamic subject isolation: LK optical flow + Fundamental RANSAC + PnP RANSAC.

3) Performance & engineering
- SHARP only on keyframes; 3DGS `.ply` cache.
- ffmpeg pipe streaming encode (no intermediate frames on disk).
- Diagnostics: per-frame logs (`inliers/reproj/new keyframe/render_ms`); use `--log_interval` to reduce log frequency; `--debug_dir` exports key results.

## Common errors

- CUDA not available: install CUDA-enabled PyTorch, check with `python tools/env_check.py`.
- CUDA version assumption: scripts default to CUDA v13.0 and `TORCH_CUDA_ARCH_LIST=12.0`; set `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT` to your install if different.
- `where cl` fails (MSVC not found): open "x64 Native Tools Command Prompt for VS 2022", then activate the venv and retry; ensure "Desktop development with C++", MSVC v143, and Windows SDK are installed.
- VS installed but still missing `cl.exe`: run `C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`, then call `"<install>\\Common7\\Tools\\VsDevCmd.bat" -arch=x64` or `"<install>\\VC\\Auxiliary\\Build\\vcvars64.bat"` and retry `where cl`.
- `ffmpeg` not in PATH: install ffmpeg and set environment variables.
- `opencv.cuda: MISSING`: pip OpenCV is CPU-only; use `--track_backend cpu` or install a CUDA-enabled OpenCV build and set `OPENCV_BIN`.
- SHARP model download fails: set a proxy or pass `--sharp_ckpt` with a local checkpoint.
- If SSL cert errors happen, set SHARP_ALLOW_INSECURE_DOWNLOAD=1 to allow an insecure fallback, or download manually and pass --sharp_ckpt.

## Dependencies

- Apple SHARP: https://github.com/apple/ml-sharp
