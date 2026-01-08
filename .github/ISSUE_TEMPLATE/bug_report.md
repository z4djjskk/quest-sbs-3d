---
name: Bug report
about: Report a bug with full logs and environment (AI-friendly)
title: "[Bug] "
labels: bug
assignees: ""
---

## Summary

## Steps to reproduce

## Expected vs actual

## Full logs (do not truncate)

```
PASTE FULL LOG HERE
```

## Environment

- Windows version:
- Python version (`python --version`):
- GPU model:
- NVIDIA driver version:
- CUDA version (`nvcc --version`):
- PyTorch version (`python - <<EOF ... EOF`):

## Toolchain (for build errors)

```
where nvcc
nvcc --version
where cl
cl
"C:\Program Files (x86)\Microsoft Visual Studio\Installerswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```

## AI prompt (optional)

Copy this prompt into your AI tool, then paste the AI output above:

```
You are helping format a bug report for a Windows-based CUDA/Python project.

Input:
- Problem summary:
- Command used (full):
- Full log output (do not truncate):
- Environment: Windows version, Python version, GPU model, NVIDIA driver, CUDA version, PyTorch version
- Toolchain outputs: where nvcc, nvcc --version, where cl, cl, vswhere output

Task:
1) Summarize the issue in 2-4 sentences.
2) List exact repro steps.
3) Highlight the first error line and 5-10 lines around it.
4) Present environment/toolchain details in bullet form.

Output format:
- Summary:
- Repro steps:
- Error excerpt:
- Environment:
- Toolchain:
```
