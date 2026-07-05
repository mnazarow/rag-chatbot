@echo off
rem ============================================================================
rem  RAG in Docker on Windows - RESTART the whole project in one command.
rem  Double-click this file, or run from a terminal:
rem     restart.cmd            (restart Qdrant + app + Redis, no rebuild)
rem     restart.cmd -Cpu       (restart without the GPU override)
rem  App data is preserved. To update code use update.cmd; for GPU use gpus.cmd.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker - restart ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Restarting...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
