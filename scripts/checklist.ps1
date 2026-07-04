# =============================================================================
#  Полная проверка установки RAG-чатбота (Windows, нативный запуск).
#  Проверяет winget-инструменты, Python-зависимости в .venv и сервисы
#  (Qdrant / Ollama / веб-интерфейс). Печатает цветной чек-лист с итогом.
#
#  Вызывается в конце setup_windows.ps1, но можно запускать и отдельно:
#     powershell -ExecutionPolicy Bypass -File scripts\checklist.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\checklist.ps1 -Root C:\rag-chatbot
#
#  (Для Docker-варианта чек-лист встроен в start_windows_docker.ps1.)
# =============================================================================
param([string]$Root = "")

if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition) }
$EnvFile = Join-Path $Root ".env"

$script:Fails = 0; $script:Warns = 0; $script:Oks = 0
function OkI($m,$d="")   { Write-Host ("  [OK] " + $m) -ForegroundColor Green -NoNewline; if($d){Write-Host ("  - $d") -ForegroundColor DarkGray}else{Write-Host ""}; $script:Oks++ }
function WarnI($m,$d="") { Write-Host ("  [~]  " + $m) -ForegroundColor Yellow -NoNewline; if($d){Write-Host ("  - $d") -ForegroundColor DarkGray}else{Write-Host ""}; $script:Warns++ }
function FailI($m,$d="") { Write-Host ("  [X]  " + $m) -ForegroundColor Red -NoNewline; if($d){Write-Host ("  - $d") -ForegroundColor DarkGray}else{Write-Host ""}; $script:Fails++ }
function Head($m) { Write-Host ""; Write-Host $m -ForegroundColor Cyan }
function Has($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function HttpOk($url,$sec=5) { try { (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec) | Out-Null; $true } catch { $false } }
function EnvVal($k) {
  if (-not (Test-Path $EnvFile)) { return "" }
  $m = Select-String -Path $EnvFile -Pattern "^$k=(.*)$" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($m) { return $m.Matches[0].Groups[1].Value.Trim('"').Trim("'") } else { return "" }
}

# venv python
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { $VPy = (Get-Command python -ErrorAction SilentlyContinue).Source }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чек-лист установки RAG — проверка пакетов и компонентов" -ForegroundColor Cyan
Write-Host ("  Проект: " + $Root) -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ===== 1. Инструменты (winget) =====
Head "1. Системные инструменты"
if (Has "python") { OkI "Python" ((& python --version 2>&1) -join " ") } else { FailI "Python не найден" "winget install Python.Python.3.12" }
if (Has "git")    { OkI "git" } else { WarnI "git не найден" "обновление через git недоступно" }
if (Has "ollama") { OkI "Ollama (клиент)" } else { WarnI "Ollama не найдена" "winget install Ollama.Ollama" }
if (Has "docker") {
  if (HttpOk "http://localhost:6333/collections" 3) { OkI "Docker (Qdrant поднят)" }
  else { try { docker info *> $null; OkI "Docker (демон запущен)" } catch { WarnI "Docker установлен, но демон не отвечает" "запустите Docker Desktop" } }
} else { WarnI "Docker не найден" "нужен для Qdrant (Docker Desktop)" }
if (Has "tesseract") { OkI "tesseract OCR" } else { WarnI "tesseract не найден" "OCR картинок отключён (UB-Mannheim.TesseractOCR)" }
if (Has "ffmpeg") { OkI "ffmpeg (видео/аудио, TTS)" } else { WarnI "ffmpeg не найден" "кадры видео/транскрибация/голос ограничены (Gyan.FFmpeg)" }
if ((Has "7z") -or (Test-Path "C:\Program Files\7-Zip\7z.exe")) { OkI "7-Zip (архивы)" } else { WarnI "7-Zip не найден" "распаковка .7z/.rar отключена" }

# ===== 2. Python-пакеты =====
Head ("2. Python-пакеты (окружение: " + $(if($VPy){$VPy}else{"нет"}) + ")")
if (-not $VPy -or -not (Test-Path $VPy)) {
  FailI "Python-окружение (.venv) не найдено" "создайте venv и поставьте requirements"
} else {
  $pyCode = @'
import importlib, importlib.util, importlib.metadata as md
req = ["fastapi","uvicorn","qdrant_client","sentence_transformers","FlagEmbedding",
       "torch","transformers","rank_bm25","fitz","docx","pptx","openpyxl",
       "bs4","lxml","charset_normalizer","PIL","psutil","requests","numpy"]
opt = ["xlrd","ezdxf","rawpy","pytesseract","extract_msg","py7zr","rarfile",
       "paramiko","multipart","playwright","pyVoIP","redis","TTS","networkx","lightrag"]
stt = ["faster_whisper","mlx_whisper"]
def ver(m):
    for n in (m, m.replace("_","-")):
        try: return md.version(n)
        except Exception: pass
    return ""
def check(mods, kind):
    for m in mods:
        try:
            importlib.import_module(m); print(kind+"\t"+m+"\tok\t"+ver(m))
        except Exception:
            print(kind+"\t"+m+"\tmissing\t")
check(req,"req"); check(opt,"opt")
present=[m for m in stt if importlib.util.find_spec(m)]
print("stt\t"+("/".join(present) if present else "none")+"\t"+("ok" if present else "missing")+"\t")
'@
  $report = & $VPy -c $pyCode 2>$null
  if (-not $report) {
    FailI "не удалось запустить проверку импортов" "venv повреждён?"
  } else {
    $labels = @{
      "qdrant_client"="qdrant-client (векторная БД)"; "sentence_transformers"="sentence-transformers (эмбеддинги)";
      "FlagEmbedding"="FlagEmbedding (реранкер)"; "rank_bm25"="rank-bm25 (лексический поиск)";
      "fitz"="PyMuPDF (чтение PDF)"; "docx"="python-docx"; "pptx"="python-pptx";
      "bs4"="beautifulsoup4"; "PIL"="Pillow"; "multipart"="python-multipart (загрузка файлов)";
      "pyVoIP"="pyVoIP (SIP, опц.)"; "TTS"="coqui-tts / XTTS (клон голоса, опц.)";
      "redis"="redis (кэш, опц.)"; "lightrag"="LightRAG (граф-RAG, опц.)"; "networkx"="networkx (граф, опц.)"
    }
    foreach ($line in $report) {
      $p = $line -split "`t"
      if ($p.Count -lt 3) { continue }
      $kind=$p[0]; $mod=$p[1]; $st=$p[2]; $v=if($p.Count -ge 4){$p[3]}else{""}
      $label = if ($labels.ContainsKey($mod)) { $labels[$mod] } else { $mod }
      if ($kind -eq "req") {
        if ($st -eq "ok") { OkI $label $v } else { FailI ($label + " - не установлен") }
      } elseif ($kind -eq "stt") {
        if ($st -eq "ok") { OkI "Whisper STT (транскрибация)" $mod } else { WarnI "Whisper (STT) не установлен" "faster-whisper - распознавание голоса отключено" }
      } else {
        if ($st -eq "ok") { OkI $label $v } else { WarnI ($label + " - не установлен (опционально)") }
      }
    }
  }
}

# ===== 3. Сервисы =====
Head "3. Сервисы и подключения"
$qurl = EnvVal "QDRANT_URL"; if (-not $qurl) { $qurl = "http://localhost:6333" }
if (HttpOk ($qurl.TrimEnd('/') + "/collections") 5) { OkI "Qdrant отвечает" $qurl } else { WarnI "Qdrant не отвечает" "$qurl - возможно, ещё стартует" }

$ollamaModel = EnvVal "LLM_MODEL"
if (HttpOk "http://localhost:11434/api/tags" 4) {
  if ($ollamaModel) {
    $base = $ollamaModel.Split(':')[0]
    $hit = $false
    try { $tags = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:11434/api/tags" -TimeoutSec 5).Content; $hit = ($tags -match [Regex]::Escape($base)) } catch {}
    if ($hit) { OkI "Ollama, модель загружена" $ollamaModel } else { WarnI "Ollama работает, но модель не найдена" "ollama pull $ollamaModel" }
  } else { OkI "Ollama отвечает" "http://localhost:11434" }
} else { WarnI "Ollama недоступна" "http://localhost:11434 - запустите Ollama" }

$port = EnvVal "API_PORT"; if (-not $port) { $port = "8000" }
if (HttpOk ("http://localhost:$port/health") 4) { OkI "Веб-интерфейс отвечает" "http://localhost:$port" } else { WarnI "Веб-интерфейс не отвечает" "http://localhost:$port/health - сервис ещё стартует или не запущен" }

# ===== Итог =====
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($script:Fails -eq 0 -and $script:Warns -eq 0) {
  Write-Host ("  ИТОГ: всё на месте (успешно: " + $script:Oks + ")") -ForegroundColor Green
} elseif ($script:Fails -eq 0) {
  Write-Host ("  ИТОГ: работает, предупреждений: " + $script:Warns + " (опциональные). Успешно: " + $script:Oks) -ForegroundColor Yellow
} else {
  Write-Host ("  ИТОГ: есть ошибки: " + $script:Fails + ". Предупреждений: " + $script:Warns + ". Успешно: " + $script:Oks) -ForegroundColor Red
  Write-Host "  Пункты [X] выше - обязательные компоненты; устраните их." -ForegroundColor DarkGray
}
Write-Host "============================================================" -ForegroundColor Cyan
if ($script:Fails -eq 0) { exit 0 } else { exit 1 }
