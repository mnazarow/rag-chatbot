@echo off
rem ============================================================================
rem  RAG in Docker on Windows - one-click UPDATE and RESTART.
rem  Double-click this file, or run from a terminal:
rem     update.cmd
rem     update.cmd -Cuda        (rebuild with NVIDIA GPU support)
rem     update.cmd -NoPull      (only rebuild/restart, do not pull from GitHub)
rem  Pulls latest code (if a git repo), rebuilds the app image and restarts the
rem  containers. Data is preserved (Qdrant index, model cache, settings, docs).
rem  Requires Docker Desktop.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker updater ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running updater...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
