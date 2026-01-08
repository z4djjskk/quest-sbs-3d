# Video to SBS (Quest-friendly 3D)
中文 | [English](README.md)

目标：把任意 2D 视频转换为 Quest 可舒适观看的高质量 3D 立体视频（SBS 左右并排 mp4）。

详细使用说明：见 [USAGE.zh-CN.md](USAGE.zh-CN.md)。

## 安装

步骤 1 / 安装 Python 依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

步骤 2 / 安装 ffmpeg

确保 `ffmpeg` 在 PATH 中可用：

```bash
ffmpeg -version
```

步骤 3 / 环境自检

```bash
python tools/env_check.py
```

## 一键运行

```bash
python tools/video_to_sbs.py --video input.mp4 --out output_sbs.mp4
```

## 可选前端

启动本地前端（带运行按钮与进度条）：

```bash
python tools/web_server.py
```

浏览器访问 `http://127.0.0.1:7860`，输入视频会自动上传到本地缓存，输出完成后点击“下载输出”。

## 关键参数（效果优先）

- `--baseline_m`：默认 0.03m，舒适优先。
- `--max_disp_px`：视差上限（像素），默认 30，越大越强烈但越容易眩晕。
- `--fov_deg`：默认 60（水平 FOV）。若画面深度不对，优先调这个。
- `--target_fps`：输入重采样到目标帧率（默认 24）。
- `--cut_threshold` / `--max_shot_len`：镜头切分稳定性。
- `--keyframe_refresh_s`：每隔 N 秒强制刷新一次 SHARP 关键帧（0=关闭），用于缓解动态主体卡顿。
- `--debug_dir`：导出关键帧深度、采样 SBS、位姿日志。
- `--log_interval`：日志间隔（帧），默认 1；调大可降低日志开销。

## Definition of Done（已在代码中检查/日志化）

1) 输出质量（视觉）
- 舒适视差：每帧基线做 disparity clamp（`--baseline_m` + `--max_disp_px`）。
- 时序稳定：PnP 位姿轻量平滑；每帧记录 `inliers/reproj`。
- 遮挡边界处理：基于 alpha 的轻量补洞（`--inpaint_radius`）。

2) 稳健性
- 自动镜头切分 + 失败恢复：直方图切分 + PnP 失败自动新关键帧。
- 动态主体防污染：LK 光流 + Fundamental RANSAC + PnP RANSAC。

3) 性能与工程质量
- SHARP 仅关键帧调用，3DGS `.ply` 缓存。
- ffmpeg pipe 流式编码（不落盘中间帧）。
- 输出可诊断：默认每帧 log（inliers/reproj/new keyframe/render_ms），可用 `--log_interval` 调整间隔，`--debug_dir` 导出关键结果。

## 常见报错

- CUDA 不可用：确保装了 CUDA 版 PyTorch，`python tools/env_check.py` 查看状态。
- CUDA 版本默认假设：脚本默认 CUDA v13.0 且 `TORCH_CUDA_ARCH_LIST=12.0`；若不同，请设置 `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT` 指向你的安装目录。
- `where cl`/`cl.exe` 找不到：请用 "x64 Native Tools Command Prompt for VS 2022" 打开，再激活 venv 运行；确认已安装 "Desktop development with C++"、MSVC v143 和 Windows SDK。
- 已安装 VS 但仍找不到：先运行 `C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath` 获取安装路径，再执行 `"<install>\\Common7\\Tools\\VsDevCmd.bat" -arch=x64` 或 `"<install>\\VC\\Auxiliary\\Build\\vcvars64.bat"` 后重试 `where cl`。
- `ffmpeg` 不在 PATH：安装 ffmpeg 并配置环境变量。
- `opencv.cuda: MISSING`：pip 安装的 OpenCV 仅 CPU 版；可用 `--track_backend cpu`，或自编译 CUDA 版并设置 `OPENCV_BIN`。
- SHARP 模型下载失败：设置代理或手动下载后用 `--sharp_ckpt` 指定。
- 如果遇到 SSL 证书错误，需要设置 SHARP_ALLOW_INSECURE_DOWNLOAD=1 才会启用不校验证书的下载兜底，或手动下载并用 --sharp_ckpt 指定。

## 反馈模板

请按以下模板反馈问题：

- 问题简述
- 运行命令（完整）
- 完整日志（不要截断）
- 环境信息：Windows 版本、Python 版本、GPU 型号、NVIDIA 驱动版本、CUDA 版本、PyTorch 版本
- 编译工具链（编译类报错必填）

建议执行以下命令并粘贴输出：

```bash
python --version
where nvcc
nvcc --version
where cl
cl
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```

## 依赖项目

- Apple SHARP: https://github.com/apple/ml-sharp
