@echo off
rem ============================================================================
rem  RAG in Docker on Windows - one-click REINSTALL (clean rebuild from scratch).
rem  Double-click this file, or run from a terminal:
rem     reinstall.cmd
rem     reinstall.cmd -Cpu          (rebuild WITHOUT GPU; GPU is the default)
rem     reinstall.cmd -Data         (ALSO wipe data volumes: Qdrant index + model cache)
rem     reinstall.cmd -State        (ALSO reset settings/logs: the state folder)
rem     reinstall.cmd -NoPull       (rebuild from current files, do not pull from GitHub)
rem     reinstall.cmd -Yes          (no confirmation prompt)
rem  Rebuilds the app image with --no-cache and recreates the containers. By
rem  default DATA IS KEPT (Qdrant index, model cache, settings, docs, backups).
rem  Requires Docker Desktop.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker reinstaller ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running reinstaller...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reinstall_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
