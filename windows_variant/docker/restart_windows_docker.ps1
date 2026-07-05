# =============================================================================
#  Перезапуск RAG в Docker на Windows — одной командой (без пересборки образа).
#  Перезапускает контейнеры проекта (Qdrant + приложение, а также Redis, если он
#  поднят). Данные сохраняются. Для обновления кода используйте update.cmd, для
#  смены/включения GPU — gpus.cmd.
#
#  Запуск (проще всего — двойной клик по restart.cmd), либо:
#     powershell -ExecutionPolicy Bypass -File restart_windows_docker.ps1
#  Параметры:
#     -Cpu    перезапуск без GPU-оверрайда (по умолчанию учитывается GPU)
# =============================================================================
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Item($status,$label,$detail=""){
    switch ($status) { "ok"{$mark="[OK]  ";$col="Green"} "fail"{$mark="[X]   ";$col="Red"} default{$mark="[~]   ";$col="Yellow"} }
    $line=" $mark $label"; if ($detail){ $line += "  - $detail" }; Write-Host $line -ForegroundColor $col
}
function HttpOk($url,$sec=5){ try{ (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec).StatusCode -eq 200 }catch{ $false } }
function ContainerExists($n){ try{ docker inspect $n *> $null; return ($LASTEXITCODE -eq 0) }catch{ return $false } }
function ContainerUp($n){ try{ ((docker inspect -f '{{.State.Running}}' $n 2>$null) -eq 'true') }catch{ $false } }

# ----- проверка Docker -----
docker info *> $null
if (-not $?) {
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Log "Запускаю Docker Desktop..."; Start-Process $dd }
    Log "Жду запуска движка Docker (до ~3 минут)..."
    $ready=$false; for($i=0;$i -lt 36;$i++){ docker info *> $null; if($?){ $ready=$true; break }; Start-Sleep 5 }
    if (-not $ready) { Item fail "Движок Docker не запущен" "запустите Docker Desktop и повторите"; exit 1 }
}

$Compose = @("-f","docker-compose.windows.yml")
if (-not $Cpu) { $Compose += @("-f","docker-compose.gpu.yml") }

Log "Перезапускаю контейнеры проекта (данные сохраняются)..."
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
docker compose @Compose restart
$ok = ($LASTEXITCODE -eq 0)
# отдельно поднятый Redis (вне compose) — перезапустим, если есть
if (ContainerExists "rag_redis") { docker restart rag_redis *> $null }
$ErrorActionPreference = $eap

# ----- краткий статус -----
Log "Жду готовности приложения..."
$appOk=$false
for ($i=0;$i -lt 40;$i++){ if (HttpOk "http://localhost:8000/health" 4){ $appOk=$true; break }; Start-Sleep 3 }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Перезапуск (Docker на Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
if ($ok) { Item ok "Команда перезапуска выполнена" } else { Item fail "docker compose restart вернул ошибку"; }
if (ContainerUp "rag_qdrant") { Item ok "Qdrant (rag_qdrant) работает" } else { Item warn "Qdrant не запущен"; }
if (ContainerUp "rag_app")    { Item ok "Приложение (rag_app) работает" } else { Item warn "Приложение не запущено"; }
if (ContainerExists "rag_redis") { if (ContainerUp "rag_redis") { Item ok "Redis (rag_redis) работает" } else { Item warn "Redis не запущен"; } }
if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" } else { Item warn "Веб-интерфейс ещё поднимается" "http://localhost:8000/health"; }
Write-Host "============================================================" -ForegroundColor Cyan
if ($appOk) { Start-Process "http://localhost:8000" }
