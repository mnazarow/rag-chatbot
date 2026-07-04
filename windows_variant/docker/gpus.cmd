@echo off
rem ============================================================================
rem  RAG in Docker on Windows - enable MULTIPLE NVIDIA GPUs and restart project.
rem  Double-click this file, or run from a terminal:
rem     gpus.cmd                 (use all detected GPUs for Ollama)
rem     gpus.cmd -Gpus 2         (use 2 GPUs)
rem     gpus.cmd -Gpus 3 -Cuda   (3 GPUs + run app container on GPU too)
rem     gpus.cmd -Machine        (set vars system-wide; run as Administrator; for Ollama service)
rem  Sets Ollama multi-GPU env vars, restarts Ollama, then restarts containers.
rem  Requires NVIDIA driver + Docker Desktop.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker - enable multiple GPUs ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0enable_gpus_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
