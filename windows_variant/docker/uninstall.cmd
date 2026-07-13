@echo off
rem ============================================================================
rem  RAG in Docker on Windows - UNINSTALL in one command.
rem  Double-click this file, or run from a terminal:
rem     uninstall.cmd            (remove containers; DATA IS KEPT)
rem     uninstall.cmd -Volumes   (also remove volumes: Qdrant index, model cache)
rem     uninstall.cmd -Images    (also remove the built app image)
rem     uninstall.cmd -State     (also remove local settings/logs: state\, .env.docker)
rem     uninstall.cmd -Docs      (also remove the documents folder - YOUR FILES!)
rem     uninstall.cmd -Purge     (full wipe: containers+volumes+image+state+backups)
rem     uninstall.cmd -Yes       (do not ask for confirmation)
rem  By default only containers/network are removed; all data is preserved so a
rem  reinstall (start.cmd) brings everything back. Requires Docker Desktop.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker - uninstall ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running uninstaller...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_windows_docker.ps1" %*

echo.
pause
