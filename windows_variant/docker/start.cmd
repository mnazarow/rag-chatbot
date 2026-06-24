@echo off
rem ============================================================================
rem  RAG in Docker on Windows - one-click launcher.
rem  Double-click this file, or run from a terminal:
rem     start.cmd
rem     start.cmd -DocsDir "C:\db" -AdminToken "your-password"
rem  It fixes .ps1 encoding (UTF-8 BOM) so Windows PowerShell 5.1 reads Cyrillic
rem  correctly, then runs the installer. Requires Docker Desktop (and Ollama).
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker launcher ===
echo Fixing script encoding (UTF-8 BOM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Recurse -Filter *.ps1 -Path '%~dp0..\..' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Starting installer...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows_docker.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
