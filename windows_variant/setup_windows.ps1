# =============================================================================
#  Установка корпоративного RAG-чатбота на чистый Windows-сервер.
#  Ставит Python, Git, Ollama (через winget), модель, Qdrant (Docker),
#  Python-окружение и регистрирует автозапуск (Scheduled Task).
#
#  Запуск (PowerShell от администратора):
#     powershell -ExecutionPolicy Bypass -File windows_variant\setup_windows.ps1 `
#         -AdminToken "пароль" -Model "qwen2.5:14b-instruct"
#  Параметры:
#     -AdminToken  пароль админ-панели (рекомендуется)
#     -DocsDir     папка с документами (по умолчанию C:\rag\db)
#     -Model       модель Ollama (по умолчанию qwen2.5:14b-instruct)
#     -Cuda        ставить torch под CUDA (нужна видеокарта NVIDIA + драйвер)
# =============================================================================
#Requires -RunAsAdministrator
param(
  [string]$AdminToken = "",
  [string]$DocsDir    = "C:\rag\db",
  [string]$Model      = "qwen2.5:14b-instruct",
  [switch]$Cuda,
  [string]$TorchCuda  = "cu124"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
function Log($m){ Write-Host "[setup-win] $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[warn] $m" -ForegroundColor Yellow }

# ----- 1. winget -----
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  throw "winget не найден. Установите 'App Installer' из Microsoft Store (или поставьте Python/Git/Ollama вручную) и повторите."
}

Log "Устанавливаю Python 3.12, Git, Ollama (winget)..."
winget install -e --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements 2>$null
winget install -e --id Git.Git           --silent --accept-source-agreements --accept-package-agreements 2>$null
winget install -e --id Ollama.Ollama      --silent --accept-source-agreements --accept-package-agreements 2>$null
# Tesseract OCR (для CR2/фото-документов) — необязательно
winget install -e --id UB-Mannheim.TesseractOCR --silent --accept-source-agreements --accept-package-agreements 2>$null

# обновить PATH в текущей сессии
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
if (-not (Test-Path $py)) { throw "Python не найден после установки. Перезапустите PowerShell и повторите." }

# ----- 2. модель Ollama -----
Log "Скачиваю модель Ollama: $Model (надолго при первом запуске)..."
ollama pull $Model

# ----- 3. Qdrant (Docker) -----
if (Get-Command docker -ErrorAction SilentlyContinue) {
  Log "Поднимаю Qdrant в Docker..."
  docker rm -f rag_qdrant 2>$null | Out-Null
  docker run -d --restart unless-stopped -p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4
} else {
  Warn "Docker не найден — Qdrant не запущен. Установите Docker Desktop:"
  Warn "  winget install -e --id Docker.DockerDesktop  (нужны WSL2/Hyper-V и перезагрузка)"
  Warn "затем запустите: docker run -d --restart unless-stopped -p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4"
}

# ----- 4. Python-окружение -----
Set-Location $Root
Log "Создаю виртуальное окружение и ставлю зависимости..."
& $py -m venv .venv
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip wheel
$torchIndex = if ($Cuda) { "https://download.pytorch.org/whl/$TorchCuda" } else { "https://download.pytorch.org/whl/cpu" }
& $venvPy -m pip install torch --index-url $torchIndex
& $venvPy -m pip install -r (Join-Path $Root "gpu_variant\requirements-gpu.txt")
& $venvPy -m pip install ezdxf   # чертежи DWG/DXF

# ----- 5. папка документов и .env -----
New-Item -ItemType Directory -Force -Path $DocsDir | Out-Null
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) { Copy-Item (Join-Path $Root "gpu_variant\.env.gpu.example") $envFile }
function SetEnv($k,$v){
  $lines = @(Get-Content $envFile)
  if ($lines -match "^$k=") { $lines = $lines -replace "^$k=.*", "$k=$v" } else { $lines += "$k=$v" }
  Set-Content -Path $envFile -Value $lines -Encoding UTF8
}
SetEnv "LLM_BACKEND"     "ollama"
SetEnv "DEVICE"          ($(if ($Cuda) { "cuda" } else { "cpu" }))
SetEnv "WHISPER_BACKEND" "faster"
SetEnv "DOCS_DIR"        $DocsDir
SetEnv "LLM_MODEL"       $Model
SetEnv "ADMIN_TOKEN"     $AdminToken

# ----- 6. автозапуск (Scheduled Task) -----
$startCmd = Join-Path $Root "windows_variant\start_app.cmd"
@"
@echo off
cd /d "$Root"
call ".venv\Scripts\activate.bat"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 >> "%TEMP%\rag_api.log" 2>&1
"@ | Set-Content -Path $startCmd -Encoding ASCII

Log "Регистрирую автозапуск (Scheduled Task 'RagApi')..."
schtasks /Create /TN "RagApi" /TR "`"$startCmd`"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN "RagApi" | Out-Null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Готово. Сервис запущен и стартует при загрузке системы."   -ForegroundColor Green
Write-Host "  Веб-панель:  http://localhost:8000  (раздел «Администратор»)"
Write-Host "  Управление:  powershell -File windows_variant\manage_windows.ps1 status|start|stop|restart|logs"
Write-Host "============================================================" -ForegroundColor Green
