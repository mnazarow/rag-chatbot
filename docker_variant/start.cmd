@echo off
rem ============================================================================
rem  RAG in Docker with Redis - one-click AUTOMATIC installer (Windows).
rem  Double-click this file. It installs Docker Desktop and Ollama if missing,
rem  pulls the model, builds and starts qdrant + redis + app, and opens the UI.
rem  Optional: start.cmd -DocsDir "C:\rag\BD" -AdminToken "your-password"
rem  Messages stay ASCII here; the PowerShell installer prints Russian.
rem ============================================================================
chcp 65001 >nul
cd /d "%~dp0"

echo === RAG Docker (with Redis) - automatic installer ===
echo Fixing script encoding (UTF-8 BOM for Windows PowerShell 5.1)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Filter *.ps1 -Path '%~dp0' | ForEach-Object { $t = Get-Content $_.FullName -Raw -Encoding UTF8; [IO.File]::WriteAllText($_.FullName, $t, (New-Object Text.UTF8Encoding($true))) }"

echo Running installer (this may take a while on first run)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*

echo.
echo Done. Web UI: http://localhost:8000
pause
