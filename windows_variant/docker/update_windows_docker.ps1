# =============================================================================
#  Обновление и перезапуск RAG в Docker на Windows — одной командой.
#  Тянет свежий код из GitHub (если это git-репозиторий), пересобирает образ
#  приложения и перезапускает контейнеры. Данные сохраняются: индекс Qdrant
#  (том qdrant_storage), кеш моделей (hf_cache), настройки/логи (папка state),
#  документы и резервные копии — всё на своих местах.
#
#  Запуск (проще всего — двойной клик по update.cmd), либо:
#     powershell -ExecutionPolicy Bypass -File update_windows_docker.ps1
#  Параметры:
#     -Cpu       пересобрать БЕЗ GPU (по умолчанию — с поддержкой GPU NVIDIA)
#     -NoPull    не тянуть код из GitHub, только пересобрать/перезапустить
#     -Branch    ветка git (по умолчанию main)
# =============================================================================
param(
    [switch]$Cpu,
    [switch]$NoPull,
    [string]$Branch = "main"
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$RepoRoot = (Resolve-Path (Join-Path $Here "..\..")).Path

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Item($status,$label,$detail=""){
    switch ($status) { "ok"{$mark="[OK]  ";$col="Green"} "fail"{$mark="[X]   ";$col="Red"} default{$mark="[~]   ";$col="Yellow"} }
    $line=" $mark $label"; if ($detail){ $line += "  - $detail" }; Write-Host $line -ForegroundColor $col
}
function HttpOk($url,$sec=5){ try{ (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec).StatusCode -eq 200 }catch{ $false } }
function ContainerUp($n){ try{ ((docker inspect -f '{{.State.Running}}' $n 2>$null) -eq 'true') }catch{ $false } }

# Файлы compose: базовый + GPU-override (по умолчанию). Ключ -Cpu отключает GPU.
$Compose = @("-f","docker-compose.windows.yml")
if (-not $Cpu) { $Compose += @("-f","docker-compose.gpu.yml") }

# ----- 0. Docker доступен? -----
docker info *> $null
if (-not $?) {
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Log "Запускаю Docker Desktop..."; Start-Process $dd }
    Log "Жду запуска движка Docker (до ~3 минут)..."
    $ready=$false; for($i=0;$i -lt 36;$i++){ docker info *> $null; if($?){ $ready=$true; break }; Start-Sleep 5 }
    if (-not $ready) { Warn "Движок Docker не запущен. Запустите Docker Desktop и повторите."; exit 1 }
}

# ----- 1. Обновление кода из GitHub -----
$oldRev = ""; $newRev = ""
if (-not $NoPull) {
    if ((Test-Path (Join-Path $RepoRoot ".git")) -and (Get-Command git -ErrorAction SilentlyContinue)) {
        Log "Обновляю код из GitHub (origin/$Branch)..."
        try {
            $oldRev = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
            git -C $RepoRoot fetch --all -q
            git -C $RepoRoot reset --hard "origin/$Branch"
            $newRev = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
            if ($oldRev -eq $newRev) { Log "Код уже актуален ($newRev)." } else { Log "Код обновлён: $oldRev -> $newRev" }
        } catch { Warn "git pull не удался: $($_.Exception.Message). Продолжаю с текущим кодом." }
    } else {
        Warn "Это не git-репозиторий (или git не установлен) — обновляю только контейнеры из текущих файлов."
        Warn "Чтобы тянуть обновления автоматически, разверните через deploy_windows.ps1 или git clone."
    }
}

# ----- 2. Пересборка образа и перезапуск контейнеров -----
if (-not $Cpu) { Log "Режим GPU (по умолчанию): пересобираю CUDA-образ и пробрасываю NVIDIA GPU. Для CPU — ключ -Cpu." }
Log "Пересобираю образ приложения и перезапускаю контейнеры (данные сохраняются)..."
# docker пишет прогресс сборки/загрузки в stderr; при ErrorActionPreference=Stop это
# прервало бы скрипт (NativeCommandError). На время команды переключаемся на Continue
# и определяем успех по коду возврата.
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
# Убрать одиночный rag_redis (из старого обходного запуска) — иначе конфликт имени с compose.
try {
    if (docker ps -aq -f "name=^rag_redis$") {
        $proj = (docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' rag_redis 2>$null)
        if (-not $proj) { Log "Удаляю одиночный контейнер rag_redis (конфликт имени с compose)..."; docker rm -f rag_redis *> $null }
    }
} catch {}
# Сборка/запуск с авто-повтором: временные сбои сети к Docker Hub (например
# «TLS handshake timeout» при получении токена/базового образа) — не редкость.
# Повторяем до 3 раз с паузой; кэш слоёв переиспользуется, повторяется только
# упавший сетевой шаг.
$composeOk = $false
for ($try = 1; $try -le 3; $try++) {
    if ($try -gt 1) {
        Log "Сетевой сбой при сборке. Повтор $try из 3 через 6 c (проверьте интернет/Docker Hub)..."
        Start-Sleep 6
    }
    docker compose @Compose up -d --build
    if ($LASTEXITCODE -eq 0) { $composeOk = $true; break }
}
$ErrorActionPreference = $eap

# ----- 3. Чеклист после обновления -----
Write-Host ""
Log "Жду готовности приложения (перезапуск + прогрев моделей — до ~2 минут)..."
$appOk = $false
for ($i=0;$i -lt 40;$i++){ if (HttpOk "http://localhost:8000/health" 4){ $appOk=$true; break }; Start-Sleep 3 }

$fails=0; $warns=0
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чек-лист обновления (Docker на Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($newRev) { Item ok "Код обновлён из GitHub" $(if($oldRev -eq $newRev){"актуален ($newRev)"}else{"$oldRev -> $newRev"}) }
if ($composeOk) { Item ok "Пересборка образа и перезапуск (docker compose)" } else { Item fail "Пересборка/перезапуск не удались"; $fails++ }
if (ContainerUp "rag_qdrant") { Item ok "Контейнер Qdrant (rag_qdrant) работает" } else { Item fail "Контейнер Qdrant не запущен"; $fails++ }
$appUp = ContainerUp "rag_app"
if ($appUp) { Item ok "Контейнер приложения (rag_app) работает" } else { Item fail "Контейнер приложения не запущен"; $fails++ }
if (HttpOk "http://localhost:6333/collections" 4) { Item ok "Qdrant отвечает (порт 6333)" } else { Item warn "Qdrant пока не отвечает" "возможно, ещё стартует"; $warns++ }
if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" } else { Item fail "Веб-интерфейс не отвечает (/health)"; $fails++ }

# CUDA / GPU внутри контейнера
if ($appUp) {
    $hostGpu=$null
    try { if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $hostGpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) } } catch {}
    if ($hostGpu) { Item ok "NVIDIA GPU на хосте (драйвер)" ($hostGpu.Trim()) } else { Item warn "NVIDIA GPU/драйвер на хосте не обнаружен" "вычисления на CPU"; $warns++ }
    try {
        $cu = (docker exec rag_app /opt/venv/bin/python -c "import torch; a=torch.cuda.is_available(); print('AVAIL', a); print('COUNT', torch.cuda.device_count()); print('NAME', torch.cuda.get_device_name(0) if a else '')" 2>&1 | Out-String)
        if ($cu -match "AVAIL\s+True") {
            $cnt = if ($cu -match "COUNT\s+(\d+)"){$Matches[1]}else{"0"}; $name = if ($cu -match "NAME\s+(.+)"){$Matches[1].Trim()}else{""}
            $d="устройств: $cnt"; if($name){$d+="; $name"}; Item ok "CUDA доступна в контейнере (эмбеддинги/реранк на GPU)" $d
        } elseif (-not $Cpu) {
            Item fail "Ожидался GPU (режим по умолчанию), но CUDA в контейнере НЕ доступна"; $fails++
            Write-Host "     Проверьте драйвер NVIDIA, Docker Desktop -> WSL2 + GPU и docker-compose.gpu.yml." -ForegroundColor Gray
            Write-Host "     Если GPU нет — обновляйтесь на CPU: update.cmd -Cpu" -ForegroundColor Gray
        } else {
            Item ok "Запущено на CPU (ключ -Cpu)" "для GPU обновитесь без -Cpu: update.cmd"
        }
    } catch { Item warn "Не удалось проверить CUDA в контейнере" "$($_.Exception.Message)"; $warns++ }
}

Write-Host "============================================================" -ForegroundColor Cyan
if ($fails -eq 0 -and $warns -eq 0) { Write-Host "  ИТОГ: обновление успешно" -ForegroundColor Green }
elseif ($fails -eq 0) { Write-Host "  ИТОГ: обновлено, есть предупреждения ($warns)." -ForegroundColor Yellow }
else { Write-Host "  ИТОГ: есть ошибки ($fails). Логи: docker compose $($Compose -join ' ') logs -f app" -ForegroundColor Red }
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Log "Веб-интерфейс: http://localhost:8000"
Log "Логи:          docker compose $($Compose -join ' ') logs -f app"
if ($appOk) { Start-Process "http://localhost:8000" }
