# Запуск RAG в Docker на Windows.
# Проще всего — двойной клик по start.cmd (он чинит кодировку и зовёт этот скрипт).
# Либо напрямую:
#   powershell -ExecutionPolicy Bypass -File start_windows_docker.ps1 `
#       -DocsDir "C:\path\to\BD" -AdminToken "ваш-пароль"
# Требуется: Docker Desktop и (для генерации) Ollama, установленные на Windows.
param(
    [string]$DocsDir = "C:\db",                         # папка с документами (по умолчанию C:\db)
    [string]$LlmModel = "qwen3.6:35b-a3b-q4_K_M",   # модель Ollama для генерации
    [string]$AdminToken = "",                           # пароль админ-панели (пусто = не менять)
    [switch]$Cpu                                        # -Cpu: запустить на CPU; ПО УМОЛЧАНИЮ используется GPU NVIDIA (WSL2 + драйвер)
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

# Файлы compose: базовый + GPU-override (по умолчанию). Ключ -Cpu отключает GPU.
$Compose = @("-f", "docker-compose.windows.yml")
if (-not $Cpu) { $Compose += @("-f", "docker-compose.gpu.yml") }

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path","User")
}
# Пункт чеклиста: status = ok | fail | warn
function Item($status, $label, $detail) {
    switch ($status) {
        "ok"   { $mark = "[OK]  "; $col = "Green"  }
        "fail" { $mark = "[X]   "; $col = "Red"    }
        default{ $mark = "[~]   "; $col = "Yellow" }
    }
    $line = " $mark $label"
    if ($detail) { $line += "  - $detail" }
    Write-Host $line -ForegroundColor $col
}
# HTTP-проверка: вернёт $true при коде 200
function HttpOk($url, $timeoutSec = 5) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $timeoutSec -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}
# Контейнер запущен?
function ContainerUp($name) {
    try { return ((docker inspect -f '{{.State.Running}}' $name 2>$null) -eq 'true') } catch { return $false }
}
# Наличие команды внутри контейнера приложения
function InAppHas($cmd) {
    try { docker exec rag_app sh -lc "command -v $cmd" *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}
# Подробный лог по упавшему пункту (выводит результат диагностической команды)
function ShowLog($label, [scriptblock]$action) {
    Write-Host ""
    Write-Host "     --- подробный лог: $label ---" -ForegroundColor DarkYellow
    try {
        $out = & $action 2>&1
        if ($out) { $out | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray } }
        else { Write-Host "     (пусто)" -ForegroundColor Gray }
    } catch {
        Write-Host "     (не удалось получить лог: $($_.Exception.Message))" -ForegroundColor Gray
    }
    Write-Host "     --- конец лога ---" -ForegroundColor DarkYellow
    Write-Host ""
}

# ----- 1. Docker: установка (winget) + запуск + ожидание движка -----
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Warn "Docker Desktop не установлен."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Устанавливаю Docker Desktop (winget; может потребоваться подтверждение прав)..."
        winget install -e --id Docker.DockerDesktop --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
    } else {
        Warn "winget не найден. Установите Docker Desktop вручную:"
        Warn "  https://www.docker.com/products/docker-desktop/  затем повторите start.cmd."
        exit 1
    }
}
docker info *> $null
if (-not $?) {
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Log "Запускаю Docker Desktop..."; Start-Process $dd }
    Log "Жду запуска движка Docker (до ~5 минут)..."
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) { docker info *> $null; if ($?) { $ready = $true; break }; Start-Sleep -Seconds 5 }
    if (-not $ready) {
        Warn "Docker ещё не запустился. Если это первая установка — нужна ПЕРЕЗАГРУЗКА (WSL2/Hyper-V)."
        Warn "Перезагрузите Windows и снова запустите start.cmd — всё уже установлено, он просто продолжит."
        exit 1
    }
}
Log "Docker готов."

# ----- 2. Ollama: установка (winget) + модель -----
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Устанавливаю Ollama (winget)..."
        winget install -e --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
        Start-Sleep -Seconds 5
    }
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    # интерактивный выбор модели (если -LlmModel не задан явно и есть консоль)
    if (-not $PSBoundParameters.ContainsKey('LlmModel') -and [Environment]::UserInteractive) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "  Выбор модели генерации (Ollama). Рекомендуется: $LlmModel" -ForegroundColor Cyan
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "  1) qwen2.5:7b-instruct           ~4.7 ГБ  - быстрая (CPU/слабый GPU)"
        Write-Host "  2) qwen2.5:14b-instruct          ~9 ГБ    - баланс"
        Write-Host "  3) qwen2.5:32b-instruct-q4_K_M   ~20 ГБ   - сильный RU (24 ГБ+ GPU)"
        Write-Host "  4) qwen3:8b                      ~5 ГБ    - reasoning, лёгкая"
        Write-Host "  5) qwen3.6:35b-a3b-q4_K_M        ~20 ГБ   - MoE 35B, топ (мощный сервер)"
        Write-Host "  6) glm4:9b                       ~6 ГБ    - GLM-4 (Zhipu), RU/CN"
        Write-Host "  7) Ввести свою (ollama-тег)"
        Write-Host "  0) Рекомендованную (Enter)"
        Write-Host "============================================================" -ForegroundColor Cyan
        $sel = Read-Host "Выбор [0-7]"
        switch ($sel) {
            '1' { $LlmModel = 'qwen2.5:7b-instruct' }
            '2' { $LlmModel = 'qwen2.5:14b-instruct' }
            '3' { $LlmModel = 'qwen2.5:32b-instruct-q4_K_M' }
            '4' { $LlmModel = 'qwen3:8b' }
            '5' { $LlmModel = 'qwen3.6:35b-a3b-q4_K_M' }
            '6' { $LlmModel = 'glm4:9b' }
            '7' { $ct = Read-Host "ollama-тег"; if ($ct) { $LlmModel = $ct } }
            default { }
        }
        Log "Модель: $LlmModel"
    }
    Log "Скачиваю модель Ollama: $LlmModel (при первом запуске долго)..."
    $pulled = $false
    for ($i = 0; $i -lt 6; $i++) {
        try { ollama pull $LlmModel; $pulled = $true; break } catch { Start-Sleep -Seconds 5 }
    }
    if (-not $pulled) { Warn "Модель не скачалась. Запустите Ollama (значок в трее) и выполните: ollama pull $LlmModel" }
} else {
    Warn "Ollama не установлена — контейнер поднимется, но отвечать на вопросы не сможет."
    Warn "Установите вручную: winget install -e --id Ollama.Ollama"
}

# ----- 2b. Git (нужен для обновлений через update.cmd). Остальные пакеты (Python,
#           ffmpeg, tesseract, playwright/chromium, espeak и т.п.) — внутри образа. -----
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Устанавливаю Git (winget; нужен для обновлений update.cmd)..."
        winget install -e --id Git.Git --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
    } else {
        Warn "Git не найден — обновления через update.cmd будут недоступны (winget install -e --id Git.Git)."
    }
}

# ----- 3. Конфиг .env.docker -----
if (-not (Test-Path ".env.docker")) {
    Copy-Item ".env.docker.example" ".env.docker"
    Log "Создан .env.docker (из примера). При желании отредактируйте."
}
# прописываем выбранную модель
(Get-Content ".env.docker") -replace '^LLM_MODEL=.*', "LLM_MODEL=$LlmModel" | Set-Content ".env.docker"
# пароль админ-панели (если задан параметром) — иначе оставляем как есть
if ($AdminToken -ne "") {
    (Get-Content ".env.docker") -replace '^ADMIN_TOKEN=.*', "ADMIN_TOKEN=$AdminToken" | Set-Content ".env.docker"
    Log "Пароль админ-панели задан."
}

# ----- 4. Папка с документами -----
if (-not $DocsDir) { $DocsDir = "C:\db" }            # каталог документов по умолчанию
New-Item -ItemType Directory -Force -Path $DocsDir | Out-Null   # создаём, если ещё нет
Log "Папка документов: $DocsDir (положите туда файлы и запустите переиндексацию)."
"DOCS_DIR_HOST=$DocsDir" | Set-Content ".env"   # compose читает .env для подстановки пути

# ----- 5. Файлы состояния (настройки + логи) -----
New-Item -ItemType Directory -Force -Path "state" | Out-Null
if (-not (Test-Path "state\runtime_config.json")) { "{}" | Set-Content "state\runtime_config.json" }
if (-not (Test-Path "state\ingest_stats.json"))   { "{}" | Set-Content "state\ingest_stats.json" }
if (-not (Test-Path "state\rag_logs.db"))         { New-Item -ItemType File -Force -Path "state\rag_logs.db" | Out-Null }
New-Item -ItemType Directory -Force -Path "backups" | Out-Null   # резервные копии (том)

# ----- 6. Сборка и запуск -----
if (-not $Cpu) { Log "Режим GPU (по умолчанию): собираю CUDA-образ и пробрасываю NVIDIA GPU в контейнер. Для CPU — ключ -Cpu." }
Log "Собираю и запускаю контейнеры (первый раз — долго: качаются образы и модели)..."
# docker пишет прогресс в stderr; при ErrorActionPreference=Stop это прервало бы скрипт.
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
# Убрать одиночный rag_redis (из старого обходного запуска) — иначе конфликт имени с compose.
try {
    if (docker ps -aq -f "name=^rag_redis$") {
        $proj = (docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' rag_redis 2>$null)
        if (-not $proj) { docker rm -f rag_redis *> $null }
    }
} catch {}
docker compose @Compose up -d --build
$composeOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $eap

# ----- 7. Чеклист после сборки -----
Write-Host ""
Log "Жду готовности приложения (загрузка моделей эмбеддингов может занять до ~2 минут)..."
$appOk = $false
for ($i = 0; $i -lt 40; $i++) {
    if (HttpOk "http://localhost:8000/health" 4) { $appOk = $true; break }
    Start-Sleep -Seconds 3
}

$ollamaModel = (Select-String -Path ".env.docker" -Pattern '^LLM_MODEL=(.*)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }) 2>$null
$ollamaUp = HttpOk "http://localhost:11434/api/tags" 4
$ollamaHasModel = $false
if ($ollamaUp -and $ollamaModel) {
    try { $tags = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:11434/api/tags" -TimeoutSec 5).Content
          $ollamaHasModel = ($tags -match [Regex]::Escape($ollamaModel.Split(':')[0])) } catch {}
}

$fails = 0; $warns = 0
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чеклист сборки и запуска" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. docker compose
if ($composeOk) {
    Item ok "Сборка образа и запуск docker compose"
} else {
    Item fail "Сборка/запуск docker compose"; $fails++
    ShowLog "docker compose ps + последние строки" { docker compose @Compose ps; docker compose @Compose logs --tail 40 }
}

# 2. Qdrant контейнер
if (ContainerUp "rag_qdrant") {
    Item ok "Контейнер Qdrant (rag_qdrant) работает"
} else {
    Item fail "Контейнер Qdrant не запущен"; $fails++
    ShowLog "docker logs rag_qdrant" { docker logs --tail 40 rag_qdrant }
}

# 3. App контейнер
$appUp = ContainerUp "rag_app"
if ($appUp) {
    Item ok "Контейнер приложения (rag_app) работает"
} else {
    Item fail "Контейнер приложения не запущен"; $fails++
    ShowLog "docker logs rag_app" { docker logs --tail 60 rag_app }
}

# 4. Qdrant API
if (HttpOk "http://localhost:6333/collections" 4) {
    Item ok "Qdrant отвечает (порт 6333)"
} else {
    Item warn "Qdrant пока не отвечает" "возможно, ещё стартует"; $warns++
    ShowLog "docker logs rag_qdrant" { docker logs --tail 30 rag_qdrant }
}

# 5. Веб-интерфейс приложения
if ($appOk) {
    Item ok "Веб-интерфейс отвечает" "http://localhost:8000"
} else {
    Item fail "Веб-интерфейс не отвечает (/health)"; $fails++
    if ($appUp) { ShowLog "docker logs rag_app (последние 60 строк)" { docker logs --tail 60 rag_app } }
}

# 5b. Приложение видит Qdrant (проверка реального соединения app -> qdrant)
if ($appOk) {
    $qOnline = $false
    try {
        $sys = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8000/api/system" -TimeoutSec 6).Content | ConvertFrom-Json
        $qOnline = [bool]$sys.qdrant.online
    } catch {}
    if ($qOnline) {
        Item ok "Приложение видит Qdrant" "QDRANT_URL=http://qdrant:6333"
    } else {
        Item fail "Приложение НЕ видит Qdrant (хотя контейнер БД работает)"; $fails++
        Write-Host "     Причина обычно — неверный QDRANT_URL у приложения." -ForegroundColor Gray
        Write-Host "     Этот образ фиксирует QDRANT_URL=http://qdrant:6333 в compose." -ForegroundColor Gray
        Write-Host "     Если адрес был сохранён ранее в админке — откройте Администратор -> QDRANT_URL," -ForegroundColor Gray
        Write-Host "     поставьте http://qdrant:6333, сохраните и «Перезапустить сервис»;" -ForegroundColor Gray
        Write-Host "     либо очистите state\runtime_config.json (сделайте его '{}') и пересоздайте контейнер." -ForegroundColor Gray
        ShowLog "QDRANT_URL: окружение и эффективное значение (должно быть http://qdrant:6333)" { docker exec rag_app printenv QDRANT_URL; docker exec rag_app /opt/venv/bin/python -c "import settings;print('effective:', settings.get('QDRANT_URL'))" }
    }
}

# 6. Ollama на хосте
if ($ollamaUp) {
    if ($ollamaHasModel) { Item ok "Ollama на хосте, модель загружена" $ollamaModel }
    else {
        Item warn "Ollama доступна, но модель не найдена" "выполните: ollama pull $ollamaModel"; $warns++
        ShowLog "ollama list" { ollama list }
    }
} else {
    Item fail "Ollama на хосте недоступна (http://localhost:11434)"; $fails++
    Write-Host "     Подсказка: установите и запустите Ollama: winget install -e --id Ollama.Ollama" -ForegroundColor Gray
    ShowLog "проверка ollama" { ollama --version }
}

# 7. Инструменты внутри образа
if ($appUp) {
    if (InAppHas "dwg2dxf")   { Item ok "DWG-конвертер (dwg2dxf) в образе" }
    else { Item warn "DWG-конвертер недоступен" "DWG-чертежи не индексируются"; $warns++
           ShowLog "лог сборки libredwg (последние строки)" { docker exec rag_app sh -lc "tail -n 40 /opt/libredwg-build.log 2>/dev/null || echo 'лог сборки не найден'" } }
    if (InAppHas "tesseract") { Item ok "OCR (tesseract) в образе" } else { Item warn "tesseract недоступен" "OCR картинок отключён"; $warns++ }
    if (InAppHas "ffmpeg")    { Item ok "ffmpeg (видео/аудио) в образе" } else { Item warn "ffmpeg недоступен" "кадры/транскрибация отключены"; $warns++ }
}

# 8. Python-пакеты внутри образа (ключевые импорты)
if ($appUp) {
    $pyProbe = 'import importlib' + "`n" +
      'req=["fastapi","uvicorn","qdrant_client","sentence_transformers","FlagEmbedding","torch","transformers","rank_bm25","fitz","docx","pptx","openpyxl","faster_whisper"]' + "`n" +
      'miss=[m for m in req if importlib.util.find_spec(m) is None]' + "`n" +
      'print("MISSING:"+",".join(miss) if miss else "ALLOK")'
    try {
        $res = (docker exec rag_app /opt/venv/bin/python -c $pyProbe 2>&1 | Out-String)
        if ($res -match "ALLOK") {
            Item ok "Python-пакеты приложения в образе (обязательные)"
        } elseif ($res -match "MISSING:(.*)") {
            Item fail "В образе не хватает Python-пакетов" $Matches[1].Trim(); $fails++
            ShowLog "проверка импортов внутри rag_app" { docker exec rag_app /opt/venv/bin/pip check }
        } else {
            Item warn "Не удалось проверить Python-пакеты в образе" ($res.Trim()); $warns++
        }
    } catch { Item warn "Проверка Python-пакетов в образе не выполнена" "$($_.Exception.Message)"; $warns++ }

    # 9. Проверка CUDA / GPU (хост-драйвер + доступность в контейнере)
    # 9a. драйвер NVIDIA на хосте Windows
    $hostGpu = $null
    try { if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $hostGpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) } } catch {}
    if ($hostGpu) { Item ok "NVIDIA GPU на хосте (драйвер)" ($hostGpu.Trim()) }
    else { Item warn "NVIDIA GPU/драйвер на хосте не обнаружен" "GPU-ускорение недоступно — вычисления на CPU"; $warns++ }

    # 9b. CUDA внутри контейнера приложения (PyTorch)
    try {
        $cu = (docker exec rag_app /opt/venv/bin/python -c "import torch; a=torch.cuda.is_available(); print('AVAIL', a); print('COUNT', torch.cuda.device_count()); print('NAME', torch.cuda.get_device_name(0) if a else ''); print('TVER', torch.version.cuda)" 2>&1 | Out-String)
        $cudaOk = ($cu -match "AVAIL\s+True")
        $cnt  = if ($cu -match "COUNT\s+(\d+)") { $Matches[1] } else { "0" }
        $name = if ($cu -match "NAME\s+(.+)")   { $Matches[1].Trim() } else { "" }
        $tver = if ($cu -match "TVER\s+(.+)")   { $Matches[1].Trim() } else { "" }
        if ($cudaOk) {
            $d = "устройств: $cnt"; if ($name) { $d += "; $name" }; if ($tver -and $tver -ne "None") { $d += "; CUDA $tver" }
            Item ok "CUDA доступна в контейнере (эмбеддинги/реранк на GPU)" $d
        } elseif (-not $Cpu) {
            Item fail "Ожидался GPU (режим по умолчанию), но CUDA в контейнере НЕ доступна"; $fails++
            Write-Host "     Проверьте: драйвер NVIDIA; Docker Desktop -> Settings -> Resources -> WSL2 + GPU;" -ForegroundColor Gray
            Write-Host "     образ собран с CUDA-torch (docker-compose.gpu.yml). Тест проброса GPU в Docker:" -ForegroundColor Gray
            Write-Host "     docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi" -ForegroundColor Gray
            Write-Host "     Если GPU нет — запустите на CPU: start.cmd -Cpu" -ForegroundColor Gray
            ShowLog "torch/CUDA внутри rag_app" { docker exec rag_app /opt/venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" }
        } else {
            Item ok "Запущено на CPU (ключ -Cpu)" "генерация идёт через Ollama на хосте (может использовать GPU). Уберите -Cpu для GPU-эмбеддингов"
        }
    } catch { Item warn "Не удалось проверить CUDA в контейнере" "$($_.Exception.Message)"; $warns++ }

    # 10. Стабильность контейнера: проброс GPU в WSL2 иногда даёт SIGBUS/периодические
    # перезапуски (exit 135). Наблюдаем несколько секунд и проверяем счётчик рестартов.
    Start-Sleep -Seconds 6
    $rc = 0
    try { $rc = [int](docker inspect -f '{{.RestartCount}}' rag_app 2>$null) } catch {}
    if ($rc -ge 1) {
        Item warn "Контейнер приложения уже перезапускался ($rc)" "возможна нестабильность"; $warns++
        if (-not $Cpu) {
            Write-Host "     Частая причина на Windows — проброс GPU в контейнер (WSL2): периодические" -ForegroundColor Gray
            Write-Host "     падения с SIGBUS (exit 135). Если чат/индексация «отваливаются» — перезапустите" -ForegroundColor Gray
            Write-Host "     в CPU-режиме:  update.cmd -Cpu   (генерация всё равно на GPU хоста через Ollama)." -ForegroundColor Gray
        }
        ShowLog "состояние rag_app" { docker inspect -f 'Restarts={{.RestartCount}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} Status={{.State.Status}}' rag_app }
    } else {
        Item ok "Контейнер приложения стабилен" "перезапусков нет"
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
if ($fails -eq 0 -and $warns -eq 0) {
    Write-Host "  ИТОГ: всё успешно ✓" -ForegroundColor Green
} elseif ($fails -eq 0) {
    Write-Host "  ИТОГ: запущено, есть предупреждения ($warns). Подробности — в логах выше." -ForegroundColor Yellow
} else {
    Write-Host "  ИТОГ: есть ошибки ($fails). Подробные логи — выше у соответствующих пунктов." -ForegroundColor Red
}
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Log "Веб-интерфейс: http://localhost:8000   (раздел «Администратор»)"
$cfStr = ($Compose -join ' ')
Log "Логи:          docker compose $cfStr logs -f app"
Log "Обновление:    update.cmd  (GPU по умолчанию)  либо  update.cmd -Cpu"
Log "Остановить:    docker compose $cfStr down    (или restart.cmd для перезапуска)"
Log "Удаление:      uninstall.cmd  (контейнеры; данные сохраняются)  /  uninstall.cmd -Purge  (полная очистка)"
if ($Cpu) { Log "Запущено на CPU (-Cpu). Чтобы задействовать NVIDIA GPU — перезапустите без ключа -Cpu (нужны WSL2 + драйвер NVIDIA)." }
Log "Дальше: откройте панель -> «Администратор» -> «Переиндексировать»."
