# =============================================================================
#  Включение нескольких GPU (NVIDIA) и перезапуск всего проекта — Windows + Docker.
#  Генерация (LLM) идёт через Ollama на хосте: скрипт задаёт ей переменные для
#  использования 2-3 карт и перезапускает Ollama, затем перезапускает контейнеры
#  (Qdrant + приложение). По ключу -Cuda приложение тоже собирается/запускается на GPU.
#
#  Запуск (проще всего — двойной клик по gpus.cmd), либо:
#     powershell -ExecutionPolicy Bypass -File enable_gpus_windows_docker.ps1 -Gpus 2
#  Параметры:
#     -Gpus N      сколько карт задействовать (по умолчанию 0 = все обнаруженные)
#     -Parallel N  OLLAMA_NUM_PARALLEL — сколько запросов параллельно (0 = = числу карт)
#     -Cuda        запускать и контейнер приложения на GPU (docker-compose.gpu.yml)
#     -Machine     задать переменные на уровне СИСТЕМЫ (нужен запуск от администратора;
#                  требуется, если Ollama работает как служба Windows)
# =============================================================================
param(
    [int]$Gpus = 0,
    [int]$Parallel = 0,
    [switch]$Cuda,
    [switch]$Machine
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Item($status,$label,$detail=""){
    switch ($status) { "ok"{$mark="[OK]  ";$col="Green"} "fail"{$mark="[X]   ";$col="Red"} default{$mark="[~]   ";$col="Yellow"} }
    $line=" $mark $label"; if ($detail){ $line += "  - $detail" }; Write-Host $line -ForegroundColor $col
}
function HttpOk($url,$sec=5){ try{ (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec).StatusCode -eq 200 }catch{ $false } }
function ContainerUp($n){ try{ ((docker inspect -f '{{.State.Running}}' $n 2>$null) -eq 'true') }catch{ $false } }

# ----- 1. Обнаружение карт -----
$gpuNames = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $eap0 = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $gpuNames = @(nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Where-Object { $_ -and $_.Trim() -ne '' }) } catch {}
    $ErrorActionPreference = $eap0
}
$detected = $gpuNames.Count
if ($detected -lt 1) {
    Warn "NVIDIA GPU не обнаружены (nvidia-smi недоступен). Установите драйвер NVIDIA и повторите."
    exit 1
}
Log "Обнаружено видеокарт: $detected"
for ($i=0; $i -lt $detected; $i++) { Write-Host ("     GPU {0}: {1}" -f $i, $gpuNames[$i].Trim()) -ForegroundColor DarkGray }

$use = if ($Gpus -gt 0) { [Math]::Min($Gpus, $detected) } else { $detected }
if ($Gpus -gt $detected) { Warn "Запрошено $Gpus карт, но доступно $detected — использую $detected." }
$devs = (0..($use-1)) -join ','
$par  = if ($Parallel -gt 0) { $Parallel } else { $use }
Log "Задействую карты: $devs  (OLLAMA_SCHED_SPREAD=1, OLLAMA_NUM_PARALLEL=$par)"

# ----- 2. Переменные окружения для Ollama -----
$scope = if ($Machine) { 'Machine' } else { 'User' }
function SetVar($name,$val){
    try { [Environment]::SetEnvironmentVariable($name, $val, $scope) }
    catch { Warn "Не удалось задать $name на уровне '$scope' (нужны права администратора?) — задаю для пользователя."
            [Environment]::SetEnvironmentVariable($name, $val, 'User') }
    Set-Item -Path "Env:$name" -Value $val    # текущий процесс — чтобы Ollama, запущенная ниже, сразу увидела
}
SetVar 'CUDA_VISIBLE_DEVICES' $devs
SetVar 'OLLAMA_SCHED_SPREAD'  '1'
SetVar 'OLLAMA_NUM_PARALLEL'  "$par"
Item ok "Переменные Ollama заданы ($scope)" "CUDA_VISIBLE_DEVICES=$devs; SCHED_SPREAD=1; NUM_PARALLEL=$par"

# ----- 3. Перезапуск Ollama на хосте -----
Log "Перезапускаю Ollama на хосте..."
try { taskkill /IM ollama.exe /F *> $null } catch {}
try { taskkill /IM "ollama app.exe" /F *> $null } catch {}
Start-Sleep -Seconds 2
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Start-Process -WindowStyle Hidden -FilePath "ollama" -ArgumentList "serve"
    $ollamaUp = $false
    for ($i=0; $i -lt 20; $i++){ if (HttpOk "http://localhost:11434/api/tags" 3){ $ollamaUp=$true; break }; Start-Sleep 2 }
    if ($ollamaUp) { Item ok "Ollama перезапущена и отвечает" "http://localhost:11434" }
    else { Item warn "Ollama не ответила за отведённое время" "запустите вручную: ollama serve"; }
} else {
    Warn "Команда ollama не найдена в PATH — перезапустите Ollama вручную (значок в трее -> Quit, затем снова запуск)."
}

# ----- 4. Перезапуск проекта в Docker -----
$Compose = @("-f","docker-compose.windows.yml")
if ($Cuda) { $Compose += @("-f","docker-compose.gpu.yml") }
Log "Перезапускаю контейнеры проекта (данные сохраняются)..."
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
if ($Cuda) { docker compose @Compose up -d --build --force-recreate }
else       { docker compose @Compose up -d --force-recreate }
$composeOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $eap

# ----- 5. Чек-лист -----
Write-Host ""
Log "Жду готовности приложения..."
$appOk = $false
for ($i=0; $i -lt 40; $i++){ if (HttpOk "http://localhost:8000/health" 4){ $appOk=$true; break }; Start-Sleep 3 }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Мультикарта GPU + перезапуск проекта" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Item ok "Карт задействовано для Ollama" "$use из $detected (CUDA_VISIBLE_DEVICES=$devs)"
if ($composeOk) { Item ok "Контейнеры перезапущены (docker compose)" } else { Item fail "Перезапуск контейнеров не удался"; }
if (ContainerUp "rag_qdrant") { Item ok "Qdrant (rag_qdrant) работает" } else { Item warn "Qdrant не запущен"; }
if (ContainerUp "rag_app")    { Item ok "Приложение (rag_app) работает" } else { Item warn "Приложение не запущено"; }
if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" } else { Item warn "Веб-интерфейс ещё поднимается" "http://localhost:8000/health"; }

# загрузка карт после прогрева модели видна в nvidia-smi
Write-Host ""
Log "Текущая загрузка карт (nvidia-smi):"
try { nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | ForEach-Object { Write-Host "     $_" -ForegroundColor DarkGray } } catch {}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Готово. Чтобы модель реально легла на несколько карт, она должна быть" -ForegroundColor Green
Write-Host "  загружена: задайте первый вопрос в чате или выполните 'ollama run <модель>'." -ForegroundColor Green
Write-Host "  Проверка распределения: ollama ps   и   nvidia-smi" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
if ($appOk) { Start-Process "http://localhost:8000" }
