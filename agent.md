# Agent Guide (当前版本)

## 项目定位
- 目标：把 2D 视频/图片转换为可在 Quest 上舒适观看的 3D 立体 SBS（左右并排）。
- 核心：SHARP 生成高斯 + 立体渲染 + ffmpeg 管线输出。

## 入口与运行
- CLI（视频/图片/批处理统一入口）：
  - `python tools/video_to_sbs.py --mode video --video input.mp4 --out output_sbs.mp4`
  - `python tools/video_to_sbs.py --mode image --video input.png --out output_sbs.png`
  - 目录输入自动批处理（视频/图片）：`--video <dir>` + `--out <dir>`
- Web UI：`python tools/web_server.py` → `http://127.0.0.1:7860`
- 预编译/预热：`python tools/precompile.py`
- 环境检查：`python tools/env_check.py`

## 模式与输入输出
- `--mode`：`video` | `image`
- `--video`：文件或目录；目录=批处理
- `--out`：单文件输出路径或输出目录（批处理时为目录）
- 输出命名：`<输入名>_sbs.mp4` 或 `<输入名>_sbs.png`
- 分段输出：`--segment_frames`（默认 2000，0=禁用），超长视频自动分段后拼接；`--keep_segments` 保留分段文件

## 关键参数（按用途）
- 画面与深度：
  - `--baseline_m`（默认 0.064）
  - `--baseline_min_m`（默认 0.0）
  - `--max_disp_px`（默认 60）
  - `--fov_deg`（默认 60）
  - `--keyframe_refresh_s`（默认 0）
- 镜头/跟踪稳定性：
  - `--cut_threshold`、`--min_shot_len`、`--max_shot_len`
  - `--pnp_ransac_iters`、`--pnp_reproj`、`--min_inliers`、`--max_reproj`
  - `--pose_smooth`
- 性能/资源：
  - `--per_frame_batch`（默认 2）、`--per_frame_pipeline`（默认 4）
  - `--async_write`、`--write_queue`
  - `--amp`、`--pin_memory`
  - `--eig_backend`（默认 cuda）
  - `--two_pass`（流式两段式：Pass1 GPU 生成高斯，Pass2 CPU 渲染/编码）
  - `--buffer_frames`、`--gpu_assist`
- IO/编码：
  - `--io_backend`（默认 ffmpeg）、`--decode`（默认 nvdec）
  - `--encode`（默认 x264）、`--ffmpeg_crf`（默认 0）、`--ffmpeg_preset`（默认 slow）
  - `--copy_audio`、`--audio_codec`（默认 aac）
- 追踪/渲染：
  - `--track_backend`（默认 opencv_cuda；CPU 为兼容备选）
  - `--render_backend`（默认 cuda；CPU 为实验性）
- 调试/日志：
  - `--debug_dir`、`--debug_interval`（默认 24）
  - `--log_interval`（默认 100）、`--perf_interval`（默认 100）
  - `--max_frames`（调试限制帧数）

## 目录结构
- `tools/`：主入口与工具（`video_to_sbs.py`、`web_server.py`、`precompile.py`、`env_check.py`）
- `sbs/`：渲染/追踪/SHARP 封装与 ffmpeg 写出
- `web/`：前端页面与脚本
- `uploads/`、`outputs/`：Web UI 上传与输出目录
- `requirements.txt`、`README*.md`

## 依赖与注意事项
- 需要 `ffmpeg` 在 PATH 中可用
- CUDA + GPU 版 PyTorch 才能使用 GPU 加速
- SHARP 权重默认自动下载；可用 `--sharp_ckpt` 指定本地模型
- 如遇 SSL 证书问题，可设置 `SHARP_ALLOW_INSECURE_DOWNLOAD=1` 或手动下载
