# Установка на Windows-сервер

Развёртывание на чистом Windows Server (или Windows 10/11). Код приложения тот же,
что и для Linux/Mac; отличаются установка зависимостей (PowerShell + winget),
запуск Qdrant (Docker Desktop) и автозапуск (Scheduled Task).

Есть два способа установки:

1. **Нативно** (PowerShell + winget) — приложение работает прямо в Windows, Qdrant
   в Docker. Описано в этом файле ниже.
2. **Полностью в Docker** — приложение и Qdrant в контейнерах, генерация через
   Ollama на хосте. Самый изолированный вариант, не «засоряет» систему. См. раздел
   [«Установка полностью в Docker»](#установка-полностью-в-docker).

## Требования

- Windows Server 2022/2025 или Windows 10/11 с **winget** («App Installer» из
  Microsoft Store обычно уже есть; на Server core может потребоваться установить).
- Права администратора (PowerShell «Запуск от имени администратора»).
- Для Qdrant — **Docker Desktop** (нужны WSL2/Hyper-V и перезагрузка). Без него
  приложение запустится, но поиск работать не будет до запуска Qdrant.
- Для генерации — **Ollama для Windows** (ставится автоматически). GPU NVIDIA
  опционально (ключ `-Cuda`).

## Установка одной командой (локально)

PowerShell от администратора, из папки проекта:

```powershell
powershell -ExecutionPolicy Bypass -File windows_variant\setup_windows.ps1 `
    -AdminToken "придумайте-пароль" -Model "qwen3.6:35b-a3b-q4_K_M"
```

Скрипт: ставит Python/Git/Ollama (winget), качает модель, поднимает Qdrant в
Docker (если установлен), создаёт `.venv` и зависимости, прописывает `.env`
(папка документов по умолчанию `C:\db`, бэкенд Ollama, Whisper = faster) и
регистрирует автозапуск (Scheduled Task `RagApi`). Откройте `http://localhost:8000`.

С GPU NVIDIA добавьте `-Cuda` (поставит torch под CUDA `cu124`).

## Деплой из GitHub

```powershell
powershell -ExecutionPolicy Bypass -File windows_variant\deploy_windows.ps1 `
    -Repo "https://github.com/USER/rag-chatbot.git" -AdminToken "пароль"
```

Клонирует репозиторий в `C:\rag-chatbot` и запускает установку.

## Управление

```powershell
powershell -File windows_variant\manage_windows.ps1 status   # статус задачи и контейнера
powershell -File windows_variant\manage_windows.ps1 start|stop|restart|logs
```

## Особенности и ограничения

- **Qdrant.** На Windows запускается в Docker Desktop. Для автозапуска после
  перезагрузки включите в Docker Desktop «Start when you log in», а контейнер
  поднят с `--restart unless-stopped`.
- **Автозапуск приложения.** Через Scheduled Task `ONSTART` от SYSTEM. Кнопка
  «Перезапустить сервис» в админке (она завершает процесс) на Windows не поднимет
  его автоматически — используйте `manage_windows.ps1 restart`.
- **Whisper.** Используется `faster-whisper` (кросс-платформенный); `mlx-whisper`
  (Apple) на Windows не нужен.
- **Удалённые хосты, дообучение, vLLM.** Дообучение и vLLM рассчитаны на Linux+GPU;
  на Windows доступны вектор-поиск, граф-RAG (LightRAG) и Ollama-генерация.
- **PowerShell не тестировался на реальном сервере** — при первом запуске возможны
  правки под конкретную версию Windows/winget; ошибки видны в выводе и в
  `%TEMP%\rag_api.log`.

---

## Установка полностью в Docker

Приложение и Qdrant запускаются в контейнерах, а генерация (LLM) — через **Ollama,
установленный на самом Windows-хосте** (контейнер обращается к нему по адресу
`host.docker.internal:11434`). Так Ollama использует GPU/ресурсы Windows напрямую,
а контейнер с приложением остаётся лёгким и не требует CUDA.

Все файлы — в папке `windows_variant\docker\`.

### Требования

- **Docker Desktop** для Windows (WSL2-бэкенд). Установка:
  `winget install -e --id Docker.DockerDesktop`, затем перезагрузка и запуск.
- **Ollama для Windows** для генерации:
  `winget install -e --id Ollama.Ollama`. Без неё контейнеры поднимутся, но
  отвечать на вопросы приложение не сможет.

### Запуск одной командой (или двойным кликом)

Самый простой способ — файл **`windows_variant\docker\start.cmd`**: дважды кликните
по нему в проводнике, либо запустите из терминала:

```bat
start.cmd
start.cmd -DocsDir "C:\db" -AdminToken "ваш-пароль"
```

`start.cmd` сам приводит `.ps1` к UTF-8 с BOM (чтобы Windows PowerShell 5.1 не
ломался на кириллице) и запускает установку. Папка документов по умолчанию —
`C:\db` (создаётся автоматически). Дальше всё делает автоматически: проверяет Docker, качает
модель в Ollama, создаёт `.env.docker`/`.env`, готовит тома и выполняет
`docker compose up -d --build`. По завершении откройте `http://localhost:8000`.

То же напрямую через PowerShell (если предпочитаете):

```powershell
powershell -ExecutionPolicy Bypass -File start_windows_docker.ps1 `
    -DocsDir "C:\db" -AdminToken "ваш-пароль"
```

### Запуск вручную

```powershell
cd windows_variant\docker
copy .env.docker.example .env.docker         # при желании отредактируйте модель и т.п.
# путь к папке с документами (его читает compose для подстановки):
"DOCS_DIR_HOST=C:\db" | Set-Content .env
docker compose -f docker-compose.windows.yml up -d --build
```

### Что где хранится

- **Индекс Qdrant** — в именованном томе `qdrant_storage` (сохраняется между
  пересборками).
- **Кеш моделей** эмбеддингов/реранка — в томе `hf_cache` (чтобы не качать заново).
- **Документы** — папка Windows монтируется в контейнер как `/data/docs`
  **на запись**: туда же сохраняются загруженные через веб-интерфейс файлы и папки,
  спарсенные сайты и документы, восстановленные из резервной копии. Меняете
  `DOCS_DIR_HOST` в `.env` — меняется источник.
- **Настройки админки, логи и тайминги** — `windows_variant\docker\state\`
  (`runtime_config.json`, `rag_logs.db`, `ingest_stats.json`), примонтированы в контейнер.
- **Резервные копии** — папка `windows_variant\docker\backups\` (доступна прямо на
  хосте, переживает пересборку контейнера).
- **Кеш превью** (миниатюры, кадры) — в томе `previews_cache`.

### Управление

```powershell
cd windows_variant\docker
docker compose -f docker-compose.windows.yml logs -f app   # логи приложения
docker compose -f docker-compose.windows.yml restart app   # перезапуск
docker compose -f docker-compose.windows.yml down          # остановить
docker compose -f docker-compose.windows.yml up -d --build  # пересобрать после git pull
```

После старта откройте `http://localhost:8000` → вкладка **Админ** →
**Переиндексировать**, чтобы проиндексировать документы.

### Особенности Docker-варианта

- **Только CPU.** Эмбеддинги/реранк в контейнере считаются на CPU (`DEVICE=cpu`).
  Это надёжно работает в Docker Desktop на Windows; для больших объёмов первая
  индексация может быть небыстрой. Генерация идёт через Ollama на хосте и может
  использовать GPU.
- **Выбор модели.** По умолчанию `qwen3.6:35b-a3b-q4_K_M` (MoE: только ~3B активных
  параметров — быстро даже на CPU, но нужно ~19–22 ГБ ОЗУ под веса). На машинах с
  малым объёмом памяти укажите модель поменьше: `-LlmModel "qwen3:8b"` или
  `gemma3:12b` (либо смените в админ-панели).
- **Транскрибация** аудио/видео — через `faster-whisper` (модель по умолчанию
  `base`; крупнее = точнее, но медленнее на CPU).
- **Распаковка архивов, OCR и чертежи** работают «из коробки»: в образ включены
  `p7zip`/`unar`, `tesseract` с русским языком, `antiword`, `ffmpeg`, а `dwg2dxf`
  (поддержка DWG) собирается из исходников `libredwg` отдельной стадией сборки.
  Если эта стадия по какой-то причине не соберётся (нет сети/новая версия) — образ
  всё равно поднимется, просто без DWG (в логе сборки будет «libredwg build failed»);
  версия задаётся аргументом `LIBREDWG_VERSION` в Dockerfile.
- **Docker/Ollama-кнопки в админке** (управление контейнерами, `ollama pull`)
  внутри контейнера недоступны — управляйте Ollama и образами с хоста Windows.
