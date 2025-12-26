# 使用说明（详细）

本文档基于当前版本代码与 README，面向日常使用与参数调优。

## 1. 环境准备

### 1.1 必备软件
- Python 3.10+（建议 3.11）
- ffmpeg（可在终端执行 `ffmpeg -version`）
- CUDA 驱动 + GPU 版 PyTorch（需要 GPU 加速时）

### 1.2 安装依赖
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 环境检查（建议）
```bash
python tools/env_check.py
```

### 1.4 预编译/预热（可选）
```bash
python tools/precompile.py
```
用途：预热 CUDA 扩展与渲染缓存，减少首次运行卡顿。

## 2. 快速开始

### 2.1 视频 → SBS
```bash
python tools/video_to_sbs.py --mode video --video input.mp4 --out output_sbs.mp4
```

### 2.2 图片 → SBS
```bash
python tools/video_to_sbs.py --mode image --video input.png --out output_sbs.png
```
> 注意：图片模式仍使用 `--video` 传入图片路径（与参数名一致）。

### 2.3 批处理（目录输入）
```bash
python tools/video_to_sbs.py --mode video --video ./inputs --out ./outputs
```
- `--video` 指向目录时自动批处理。
- 输出文件名自动追加 `_sbs`。

### 2.4 Web UI
```bash
python tools/web_server.py
```
浏览器访问 `http://127.0.0.1:7860`，上传/拖拽视频或图片并运行。

## 3. 输入/输出规则
- `--video`：文件或目录。
- `--out`：
  - 单文件模式：输出文件路径或目录。
  - 批处理模式：输出目录（每个文件独立输出）。
- 输出命名：`<输入名>_sbs.mp4` 或 `<输入名>_sbs.png`。
- 分段输出：`--segment_frames`（默认 2000，0=禁用）。超长视频会按段处理后自动拼接。

## 4. 模式与关键帧策略

### 4.1 `--keyframe_mode`
- `per_frame`（默认）：每帧都用 SHARP 预测高斯，质量最高，速度最慢。
- `normal`：关键帧 + 追踪；镜头切换会重新生成关键帧。
- `freeze`：固定关键帧，镜头切换也不更新；适合静态画面。
- `cache_only`：只读缓存关键帧（缺缓存会失败）。

### 4.2 `--keyframe_refresh_s`
- 每隔 N 秒强制刷新关键帧（0=禁用）。
- `per_frame` 模式下会被忽略。

## 5. 画面与立体感参数（重要）
- `--baseline_m`（默认 0.064）：双眼基线，越大越“立体”。
- `--baseline_min_m`（默认 0）：基线下限。
- `--max_disp_px`（默认 60）：视差上限，过大易眩晕。
- `--fov_deg`（默认 60）：水平 FOV，深度不对时优先调它。

## 6. 镜头切分与跟踪稳定
- `--cut_threshold`（默认 0.9）：镜头切分阈值。
- `--min_shot_len` / `--max_shot_len`：镜头最短/最长时长（秒）。
- `--max_features`（默认 2000）：关键点数量。
- `--feature_quality`（默认 0.01）：角点质量门槛。
- `--feature_min_dist`（默认 7）：角点最小间隔。
- `--flow_fb_thresh`（默认 1.0）：LK 前后向一致性阈值。
- `--fundamental_thresh`（默认 1.0）：基础矩阵 RANSAC 阈值。
- `--pnp_ransac_iters`（默认 200）：PnP RANSAC 次数。
- `--pnp_reproj`（默认 3.0）：PnP 重投影误差阈值。
- `--min_inliers`（默认 70）：PnP 最小内点数。
- `--max_reproj`（默认 2.0）：最大重投影误差。
- `--pose_smooth`（默认 0.25）：位姿平滑强度。

## 7. 深度范围控制
- `--depth_min` / `--depth_max`：硬裁剪深度范围（0=禁用）。
- `--depth_q_low` / `--depth_q_high`：深度分位裁剪（默认 0.02 / 0.98）。

## 8. 性能与资源
- `--per_frame_batch`（默认 2）：per_frame 模式批大小。
- `--per_frame_pipeline`（默认 4）：per_frame 管线队列。
- `--async_write` + `--write_queue`：异步写出与队列大小。
- `--amp`：启用 FP16 自动混合精度（默认开）。
- `--pin_memory`：Pinned memory 加速 H2D（默认开）。
- `--eig_backend`：`cuda` | `cpu`（默认 cuda）。
- `--eig_chunk_size`（默认 262144）：分块特征分解。
- `--min_free_ram_gb`：内存不足时自动降级。
- `--segment_frames`：超长视频分段阈值（默认 2000）。

### 两段式模式（实验）
- `--two_pass`：Pass1 GPU 生成高斯缓存，Pass2 CPU 渲染/编码。
- `--buffer_frames`：CPU 开始渲染前的缓存帧数。
- `--gpu_assist`：Pass1 完成后允许 GPU 协助渲染。
> 仅适用于 `keyframe_mode=per_frame`。

## 9. 编码与音频
- `--encode`：`x264` / `nvenc` / `hevc_nvenc`。
- `--ffmpeg_crf`（默认 0）：
  - x264：CRF 越低质量越高，`0` 为无损。
  - NVENC：内部使用 `-cq`。
- `--ffmpeg_preset`（默认 slow）：`slow`/`medium`/`fast`。
- `--copy_audio`：复制原音轨。
- `--audio_codec`：`aac` 或 `copy`。

## 10. 追踪/渲染后端
- `--track_backend`：`opencv_cuda` | `cpu`（默认 opencv_cuda）。
  - CUDA 追踪更快；CPU 作为兼容备选。
- `--render_backend`：`cuda` | `cpu`（默认 cuda，CPU 实验性更慢）。

## 11. 调试与日志
- `--debug_dir`：导出关键帧深度/采样图/位姿。
- `--debug_interval`（默认 24）：调试帧间隔。
- `--log_interval`（默认 100）：日志间隔（帧）。
- `--perf_interval`（默认 100）：性能统计间隔（帧）。
- `--max_frames`：限制处理帧数（调试用）。

## 12. 目录结构（简要）
- `tools/`：主入口与辅助工具
- `sbs/`：渲染/追踪/SHARP 封装与 ffmpeg 写出
- `web/`：前端
- `uploads/`、`outputs/`：Web UI 上传与输出

## 13. 常见问题
1) ffmpeg 不可用：确认 `ffmpeg -version` 成功。
2) CUDA 不可用：安装 CUDA 版 PyTorch；或改用 `--device cpu`（非常慢）。
3) SHARP 下载失败：手动下载后用 `--sharp_ckpt` 指定；或设置 `SHARP_ALLOW_INSECURE_DOWNLOAD=1`。
4) NVDEC 失败：改用 `--decode cpu`。
5) 显存/内存不足：降低 `--per_frame_batch`/`--per_frame_pipeline`，或使用分段 `--segment_frames`。

## 14. 示例组合

### 质量优先（慢）
```bash
python tools/video_to_sbs.py   --mode video   --video input.mp4   --out output_sbs.mp4   --keyframe_mode per_frame   --per_frame_batch 1   --per_frame_pipeline 2   --encode hevc_nvenc   --ffmpeg_crf 10   --copy_audio
```

### 性能优先（快）
```bash
python tools/video_to_sbs.py   --mode video   --video input.mp4   --out output_sbs.mp4   --keyframe_mode normal   --per_frame_pipeline 2   --encode nvenc   --ffmpeg_crf 18   --copy_audio
```
