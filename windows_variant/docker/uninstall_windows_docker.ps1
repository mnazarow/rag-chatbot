# =============================================================================
#  Удаление RAG в Docker на Windows.
#  Проще всего — двойной клик по uninstall.cmd, либо:
#     powershell -ExecutionPolicy Bypass -File uninstall_windows_docker.ps1 [ключи]
#
#  По умолчанию (без ключей): останавливает и удаляет КОНТЕЙНЕРЫ и сеть проекта.
#  ДАННЫЕ СОХРАНЯЮТСЯ — индекс Qdrant, кеш моделей, настройки/логи (state), бэкапы,
#  а также папка документов (напр. C:\db) НЕ удаляются. Повторная установка поднимет
#  всё «как было».
#
#  Ключи (можно комбинировать):
#     -Volumes   удалить тома Docker: индекс Qdrant, кеш моделей (hf_cache), Milvus,
#                кеш превью. ДАННЫЕ ВЕКТОРНОЙ БАЗЫ И КЕШ МОДЕЛЕЙ ТЕРЯЮТСЯ (модели
#                скачаются заново при следующем старте).
#     -Images    удалить собранный образ приложения (освободить место; при следующей
#                установке образ соберётся заново).
#     -State     удалить локальные настройки/логи: папка state\ и файлы .env.docker/.env.
#                НАСТРОЙКИ АДМИНКИ И ЖУРНАЛ ЗАПРОСОВ ТЕРЯЮТСЯ (бэкапы сохраняются).
#     -Docs      удалить папку документов (DOCS_DIR из .env). ОПАСНО — это ваши файлы.
#     -Purge     ПОЛНАЯ очистка: контейнеры + тома + образ + state + бэкапы. Не трогает
#                папку документов (для неё нужен отдельный ключ -Docs).
#     -Yes       не спрашивать подтверждение (для автоматизации).
# =============================================================================
param(
    [switch]$Volumes,
    [switch]$Images,
    [switch]$State,
    [switch]$Docs,
    [switch]$Purge,
    [switch]$Yes
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Done($m){ Write-Host " [OK]  $m" -ForegroundColor Green }

# -Purge включает всё разрушительное, КРОМЕ папки документов (её нужно попросить явно -Docs)
if ($Purge) { $Volumes = $true; $Images = $true; $State = $true }

# ----- Docker доступен? -----
docker info *> $null
if (-not $?) {
    Warn "Docker Desktop не запущен. Запустите его и повторите (или удалите вручную через Docker Desktop)."
    exit 1
}

# Профиль milvus включаем, чтобы down убрал и его контейнеры (если поднимались)
$Compose = @("-f", "docker-compose.windows.yml", "--profile", "milvus")

# ----- Сводка того, что будет удалено -----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Удаление RAG (Docker на Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Будет удалено:" -ForegroundColor Gray
Write-Host "   - контейнеры и сеть проекта (rag_app, rag_qdrant, rag_redis, Milvus…)" -ForegroundColor Gray
if ($Volumes) { Write-Host "   - ТОМА: индекс Qdrant, кеш моделей, Milvus, превью  (ДАННЫЕ ТЕРЯЮТСЯ)" -ForegroundColor Yellow }
if ($Images)  { Write-Host "   - собранный ОБРАЗ приложения" -ForegroundColor Yellow }
if ($State)   { Write-Host "   - state\ (настройки, журнал) и .env.docker/.env  (НАСТРОЙКИ/ЛОГИ ТЕРЯЮТСЯ)" -ForegroundColor Yellow }
if ($Purge)   { Write-Host "   - backups\ (резервные копии)" -ForegroundColor Yellow }
$docsPath = $null
if ($Docs) {
    try { $docsPath = (Select-String -Path ".env" -Pattern '^DOCS_DIR_HOST=(.*)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }) } catch {}
    Write-Host "   - ПАПКА ДОКУМЕНТОВ: $docsPath  (ВАШИ ФАЙЛЫ — БУДУТ УДАЛЕНЫ)" -ForegroundColor Red
}
Write-Host "  Сохранится:" -ForegroundColor Gray
if (-not $Volumes) { Write-Host "   - тома (индекс/кеш моделей) — переустановка поднимет всё как было" -ForegroundColor Gray }
if (-not $State)   { Write-Host "   - настройки/логи (state\), .env.docker" -ForegroundColor Gray }
if (-not $Purge)   { Write-Host "   - резервные копии (backups\)" -ForegroundColor Gray }
if (-not $Docs)    { Write-Host "   - папка документов (ваши файлы)" -ForegroundColor Gray }
Write-Host ""

# ----- Подтверждение -----
if (-not $Yes) {
    $warnLevel = ($Volumes -or $State -or $Docs -or $Purge)
    $prompt = if ($warnLevel) { "Это удалит ДАННЫЕ. Введите 'yes' для подтверждения" } else { "Удалить контейнеры (данные сохранятся)? Введите 'yes'" }
    $ans = Read-Host $prompt
    if ($ans -ne "yes") { Warn "Отменено."; exit 0 }
}

# ----- 1. Останавливаем и удаляем контейнеры (+тома/образ по ключам) -----
$downArgs = @("down", "--remove-orphans")
if ($Volumes) { $downArgs += "--volumes" }
if ($Images)  { $downArgs += @("--rmi", "local") }
Log "Останавливаю и удаляю контейнеры проекта..."
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
docker compose @Compose @downArgs
# одиночный rag_redis (из старого обходного запуска) — если остался вне compose
try { if (docker ps -aq -f "name=^rag_redis$") { docker rm -f rag_redis *> $null } } catch {}
$ErrorActionPreference = $eap
Done "Контейнеры и сеть удалены."
if ($Volumes) { Done "Тома удалены (индекс/кеш моделей/Milvus)." }
if ($Images)  { Done "Образ приложения удалён." }

# ----- 2. Локальные файлы состояния -----
if ($State) {
    foreach ($p in @("state", ".env.docker", ".env")) {
        if (Test-Path $p) { try { Remove-Item -Recurse -Force $p; Done "Удалено: $p" } catch { Warn "Не удалось удалить $p: $($_.Exception.Message)" } }
    }
}
if ($Purge) {
    if (Test-Path "backups") { try { Remove-Item -Recurse -Force "backups"; Done "Удалено: backups" } catch { Warn "backups: $($_.Exception.Message)" } }
}

# ----- 3. Папка документов (только по явному -Docs) -----
if ($Docs -and $docsPath) {
    if (Test-Path $docsPath) {
        try { Remove-Item -Recurse -Force $docsPath; Done "Удалена папка документов: $docsPath" }
        catch { Warn "Папку документов удалить не удалось ($docsPath): $($_.Exception.Message)" }
    } else { Warn "Папка документов не найдена: $docsPath" }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Готово." -ForegroundColor Green
if (-not $Volumes -and -not $State) {
    Write-Host "  Данные сохранены. Повторная установка (start.cmd) поднимет всё как было." -ForegroundColor Gray
} elseif ($Purge) {
    Write-Host "  Выполнена полная очистка (кроме папки документов)." -ForegroundColor Gray
}
Write-Host "  Docker Desktop, Ollama и модели Ollama остаются в системе — удалите вручную при желании" -ForegroundColor Gray
Write-Host "  (winget uninstall Docker.DockerDesktop / Ollama.Ollama)." -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
