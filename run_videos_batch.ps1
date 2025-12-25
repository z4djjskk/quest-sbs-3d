 = "C:\Users\Q\Downloads\videos"
 = Join-Path (Get-Location) "outputs"
 = Get-ChildItem -Path  -File | Where-Object { .Extension -ieq ".mp4" }
if (-not (Test-Path )) { New-Item -ItemType Directory -Path  | Out-Null }
foreach ( in ) {
     = Join-Path -Path  -ChildPath (.BaseName + "_sbs.mp4")
    Write-Host "Processing  -> "
    python tools/video_to_sbs.py \
        --video "" \
        --out "" \
        --baseline_m 0.064 \
        --baseline_min_m 0 \
        --max_disp_px 60 \
        --fov_deg 60 \
        --cut_threshold 0.9 \
        --min_shot_len 0.01 \
        --max_shot_len 0.05 \
        --min_inliers 70 \
        --max_reproj 2 \
        --ffmpeg_crf 10 \
        --ffmpeg_preset slow \
        --inpaint_radius 2 \
        --debug_interval 24 \
        --max_frames 2500 \
        --copy_audio \
        --audio_codec copy \
        --io_backend ffmpeg \
        --decode nvdec \
        --encode hevc_nvenc \
        --track_backend opencv_cuda \
        --keyframe_mode normal \
        --per_frame_batch 2 \
        --per_frame_pipeline 2 \
        --amp \
        --clear_cache_on_exit
    if ( -ne 0) { throw "Conversion failed for " }
}
