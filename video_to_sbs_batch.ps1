$sourceDir = 'C:\Users\Q\Downloads\videos'
$prefix = 0x90b1,0x65b0,0x51ef,0x7684,0x5947,0x5999,0x7a0b,0x5e8f
$suffix = 0x89c6,0x9891,0x8f6c,0x33,0x64,0x82f9,0x679c,0x7248
$leftName = ($prefix | ForEach-Object { [char]$_ }) -join ''
$rightName = ($suffix | ForEach-Object { [char]$_ }) -join ''
$repoDir = Join-Path 'F:\' (Join-Path $leftName $rightName)
$outputDir = Join-Path $repoDir 'outputs'
$videos = Get-ChildItem -Path $sourceDir -File -Filter *.mp4
if (-not $videos) {
    Write-Error 'No mp4 files found in ' + $sourceDir
    exit 1
}
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}
Set-Location $repoDir
foreach ($video in $videos) {
    $out = Join-Path -Path $outputDir -ChildPath ($video.BaseName + '_sbs.mp4')
    Write-Host "Processing $($video.Name) -> $out"
    python tools/video_to_sbs.py --video "$($video.FullName)" --out "$out" --baseline_m 0.064 --baseline_min_m 0 --max_disp_px 60 --fov_deg 60 --cut_threshold 0.9 --min_shot_len 0.01 --max_shot_len 0.05 --min_inliers 70 --max_reproj 2 --ffmpeg_crf 10 --ffmpeg_preset slow --inpaint_radius 2 --debug_interval 24 --max_frames 2500 --copy_audio --audio_codec copy --io_backend ffmpeg --decode nvdec --encode hevc_nvenc --track_backend opencv_cuda --keyframe_mode normal --per_frame_batch 2 --per_frame_pipeline 2 --amp --clear_cache_on_exit
    if ($LASTEXITCODE -ne 0) {
        throw "Conversion failed for $($video.Name)"
    }
}
