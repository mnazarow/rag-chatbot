# =============================================================================
#  Деплой RAG из GitHub на чистый Windows-сервер: ставит Git, клонирует
#  репозиторий и запускает setup_windows.ps1.
#
#  Запуск (PowerShell от администратора):
#     powershell -ExecutionPolicy Bypass -File deploy_windows.ps1 `
#         -Repo "https://github.com/USER/rag-chatbot.git" -AdminToken "пароль"
# =============================================================================
#Requires -RunAsAdministrator
param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [string]$Branch     = "main",
  [string]$TargetDir  = "C:\rag-chatbot",
  [string]$AdminToken = "",
  [string]$Model      = "qwen2.5:14b-instruct",
  [switch]$Cuda
)
$ErrorActionPreference = "Stop"
function Log($m){ Write-Host "[deploy-win] $m" -ForegroundColor Cyan }

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  throw "winget не найден. Установите 'App Installer' из Microsoft Store."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Log "Устанавливаю Git..."
  winget install -e --id Git.Git --silent --accept-source-agreements --accept-package-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
}

if (Test-Path (Join-Path $TargetDir ".git")) {
  Log "Обновляю $TargetDir ..."
  git -C $TargetDir fetch --all -q
  git -C $TargetDir reset --hard "origin/$Branch"
} else {
  Log "Клонирую $Repo в $TargetDir ..."
  git clone -q -b $Branch $Repo $TargetDir
}

Log "Запускаю установку..."
$args = @("-AdminToken", $AdminToken, "-Model", $Model)
if ($Cuda) { $args += "-Cuda" }
& powershell -ExecutionPolicy Bypass -File (Join-Path $TargetDir "windows_variant\setup_windows.ps1") @args
