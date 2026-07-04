@echo off
rem ============================================================================
rem  RAG in Docker on Windows - enable (or disable) Redis cache in ONE command.
rem  Double-click this file, or run from a terminal:
rem     redis.cmd            (bring up Redis container + wire the app to it)
rem     redis.cmd -Cuda      (same, with GPU stack)
rem     redis.cmd -Off       (disable the Redis cache instead)
rem  App data is preserved. Requires Docker Desktop.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker - Redis cache ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0enable_redis_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
