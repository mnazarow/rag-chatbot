# Документация по установке

Полное руководство по развёртыванию корпоративного RAG-чатбота. Система
существует в двух платформенных вариантах с общим кодом приложения:

- **Apple Silicon** (Mac Studio M3 и т.п.) — генерация через Ollama (Metal);
- **Linux + NVIDIA GPU** — генерация через vLLM (CUDA), для нагрузки на команду.

Поверх любого из них включается расширенный режим **hybrid+** (структурные
LLM-метаданные, умные фильтры, граф-RAG) и опциональная отдельная ветка
**LightRAG** для сравнения.

---

## 1. Требования

### Общие
- Папка с документами (прайс-листы, презентации, записи обучения и т.д.).
  Поддерживаются: PDF, DOCX, DOC (Word 97-2003), PPTX, XLSX/XLSM/XLS/CSV,
  TXT/MD, HTML/MHTML, XML, JSON, SVG, ярлыки .url, письма Outlook .msg,
  чертежи DXF/DWG, 3D-CAD STEP/IGES (метаданные),
  изображения (JPG/PNG/WEBP/GIF/BMP/TIFF/JFIF → OCR) и RAW-фото (CR2/NEF/ARW… → OCR),
  архивы (ZIP/RAR/7Z/TAR/GZ — распаковываются, содержимое индексируется),
  а также аудио/видео (MP3, WAV, M4A, AAC, MP4, MOV, MKV, WEBM — транскрибируются Whisper).
  Для .doc нужен `antiword` или LibreOffice; для OCR — Tesseract (+ языковой пакет rus);
  для .rar/.7z — системные `p7zip`/`unar` (или пакеты py7zr/rarfile).
- Доступ в интернет при первой установке (скачивание моделей и образов).
- ~50–100 ГБ свободного места под модели, образы и индекс.

### Apple Silicon
- macOS на чипе Apple (arm64). Рекомендуется Mac Studio M3, 64–96 ГБ ОЗУ.
- Homebrew (ставится скриптом автоматически).

### Linux + NVIDIA GPU
- Ubuntu 22.04 / 24.04 (или совместимый дистрибутив с apt).
- Установленный драйвер NVIDIA (`nvidia-smi` должен работать).
- Права sudo.
- VRAM определяет доступную модель (см. таблицу в разделе 3).

---

## 2. Установка на Apple Silicon (Mac Studio)

```bash
cd /путь/к/проекту
chmod +x setup.sh
./setup.sh
```

`setup.sh` выполняет:
1. установку Homebrew (если нет);
2. системные пакеты: `python@3.11`, `ffmpeg` (кадры/фрагменты видео и транскрибация),
   `poppler`, `tesseract` (+ `tesseract-lang` для OCR на русском), `libredwg`,
   `antiword`, `p7zip`/`unar` (архивы), Ollama, Docker; Python-зависимости (включая
   `matplotlib` для рендера чертежей, `psutil` для метрик, `pytesseract`/`rawpy`/
   `Pillow` для OCR) ставятся из `requirements.txt`;
3. запуск Ollama как сервиса и скачивание LLM (по умолчанию
   `qwen3.6:35b-a3b-q4_K_M`);
4. поднятие Qdrant в Docker (`docker-compose.yml`);
5. создание виртуального окружения `.venv` и установку зависимостей
   (`requirements.txt`);
6. создание `.env` из `.env.example`;
7. прогрев моделей эмбеддингов (`bge-m3`) и реранка (`bge-reranker-v2-m3`) на MPS.

Далее:

```bash
nano .env                       # укажите DOCS_DIR — путь к папке с документами
source .venv/bin/activate
python ingest.py                # индексация документов
uvicorn app:app --host 0.0.0.0 --port 8000
```

Веб-панель: `http://<ip-сервера>:8000`.

> На Apple Silicon можно также задать более крупную модель (`qwen2.5:72b`) —
> хватит 64–96 ГБ унифицированной памяти; меняется в `.env` или позже в админке.

---

## 3. Установка на Linux + NVIDIA GPU

### 3.1. Быстрый старт «одной командой» (рекомендуется)

На чистом сервере с рабочим `nvidia-smi`:

```bash
cd gpu_variant
ADMIN_TOKEN='придумайте-пароль' sudo -E bash run_gpu.sh
```

Если ваш sudo не разрешает `-E` (сообщение «preserving the entire environment is
not supported»), используйте форму с заданием переменных внутри root-шелла:

```bash
sudo bash -c "ADMIN_TOKEN='пароль' bash run_gpu.sh"
```

`run_gpu.sh` выполняет всё автоматически:
1. системные пакеты, Docker, NVIDIA Container Toolkit;
2. поднятие **vLLM** (OpenAI-совместимый сервер) и **Qdrant** через
   `docker-compose.gpu.yml`;
3. Python-окружение `.venv` с CUDA-сборкой `torch` и зависимостями
   (`requirements-gpu.txt`);
4. регистрацию **systemd-сервиса `rag-api`** с автозапуском и `Restart=always`.

После завершения откройте `http://<ip>:8000` → раздел **«Администратор»** и
донастройте всё в веб-панели (папку документов, модель и т.д.).

Стартовую модель можно задать заранее:

```bash
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ VLLM_TP=1 ADMIN_TOKEN='пароль' sudo -E bash run_gpu.sh
```

### 3.2. Подбор модели под VRAM

| GPU (VRAM)            | Рекомендуемая модель                | Параметры             |
|-----------------------|-------------------------------------|-----------------------|
| 24 ГБ (RTX 4090/3090) | `Qwen/Qwen2.5-14B-Instruct-AWQ`     | `VLLM_TP=1`           |
| 48 ГБ (A6000/L40S)    | `Qwen/Qwen2.5-32B-Instruct-AWQ`     | `VLLM_TP=1`           |
| 80 ГБ (A100/H100)     | `Qwen/Qwen2.5-72B-Instruct-AWQ`     | `VLLM_TP=1`           |
| 2×24–48 ГБ            | 32B/72B без квантизации             | `VLLM_TP=2`           |

`VLLM_MODEL` (модель контейнера) и `LLM_MODEL` (имя для запросов) должны совпадать.
Менять модель потом можно прямо в админке кнопкой «Применить модель LLM».

### 3.3. Управление сервисом

```bash
bash gpu_variant/manage.sh status      # статус сервиса и контейнеров
bash gpu_variant/manage.sh logs        # логи API (journalctl)
bash gpu_variant/manage.sh vllm-logs   # логи vLLM
bash gpu_variant/manage.sh restart|stop|start
```

---

## 4. Деплой из GitHub на чистый сервер

Если код в репозитории GitHub, разворачивать сервер можно без ручного копирования.

```bash
# на чистом сервере (репозиторий публичный)
curl -fsSL https://raw.githubusercontent.com/USER/rag-chatbot/main/deploy.sh -o deploy.sh
sudo bash -c "ADMIN_TOKEN='пароль' bash deploy.sh https://github.com/USER/rag-chatbot.git"
```

`deploy.sh`:
1. ставит git;
2. клонирует репозиторий в `/opt/rag` (или `TARGET_DIR`), при повторе — обновляет;
3. запускает `gpu_variant/run_gpu.sh` (полная GPU-установка).

Переменные: `REPO`, `BRANCH` (по умолчанию `main`), `TARGET_DIR` (по умолчанию
`/opt/rag`), `ADMIN_TOKEN`, `VLLM_MODEL`, `VLLM_TP`.

### Обновление развёрнутого сервера

```bash
sudo bash /opt/rag/update.sh              # git pull + зависимости + перезапуск
sudo REINDEX=1 bash /opt/rag/update.sh    # то же + переиндексация
```

`update.sh` работает только на уже установленном сервере; если установки нет, он
подскажет запустить `run_gpu.sh`.

### Деплой из GitHub на Mac (Apple Silicon)

Аналог для Mac: клонирует репозиторий, запускает `setup.sh` и регистрирует
автозапуск через launchd (без systemd).

```bash
curl -fsSL https://raw.githubusercontent.com/USER/rag-chatbot/main/mac_variant/deploy_mac.sh -o deploy_mac.sh
REPO=https://github.com/USER/rag-chatbot.git ADMIN_TOKEN='пароль' bash deploy_mac.sh
```

После установки сервис стартует автоматически (launchd, `KeepAlive`), панель —
`http://localhost:8000`. Управление и обновление:

```bash
bash mac_variant/manage_mac.sh {status|logs|restart|stop|start}
bash mac_variant/update_mac.sh
```

В отличие от GPU-варианта здесь генерация идёт через Ollama (Metal), а операция
«Применить модель LLM» (рестарт vLLM) не применяется.

---

## 5. Расширенный режим hybrid+ (граф-RAG)

Базовые улучшения (LLM-метаданные, умные фильтры) включаются прямо в админке.
Слой граф-RAG требует LightRAG и построенного графа:

```bash
sudo bash gpu_variant/setup_hybrid.sh
```

Скрипт ставит LightRAG в существующее `.venv` и строит граф из `DOCS_DIR`
(использует vLLM + bge-m3). Построение прогоняет документы через LLM и идёт долго —
для пробы укажите в админке `DOCS_DIR` на подпапку из 30–50 файлов.

Затем в админ-панели выберите режим **«Hybrid+ (граф)»** или включите тумблеры
по отдельности (см. документацию по использованию).

Перестроить граф вручную:

```bash
source .venv/bin/activate
python -m graph_rag ingest
```

---

## 4a. Установка на Windows-сервер

PowerShell от администратора (нужен winget; Qdrant — через Docker Desktop;
генерация — Ollama для Windows):

```powershell
# локально из папки проекта
powershell -ExecutionPolicy Bypass -File windows_variant\setup_windows.ps1 -AdminToken "пароль"

# или деплой из GitHub
powershell -ExecutionPolicy Bypass -File windows_variant\deploy_windows.ps1 `
    -Repo "https://github.com/USER/rag-chatbot.git" -AdminToken "пароль"
```

С GPU NVIDIA добавьте `-Cuda`. Автозапуск — через Scheduled Task; управление —
`windows_variant\manage_windows.ps1 status|start|stop|restart|logs`. Подробности и
ограничения — в `windows_variant/README_windows.md`.

## 4b. Установка на Windows в Docker

Полностью контейнерный вариант: приложение и Qdrant — в контейнерах, генерация —
через Ollama на хосте Windows. Не «засоряет» систему, всё в Docker Desktop.

```powershell
# из папки windows_variant\docker
powershell -ExecutionPolicy Bypass -File start_windows_docker.ps1 `
    -DocsDir "C:\db" -LlmModel "qwen2.5:7b-instruct-q4_K_M"
```

Скрипт проверит Docker, скачает модель в Ollama, подготовит `.env`/состояние и
выполнит `docker compose up -d --build`. Затем откройте `http://localhost:8000`
и запустите переиндексацию из вкладки «Админ». Эмбеддинги/реранк в контейнере
считаются на CPU; индекс и кеш моделей сохраняются в томах Docker. Подробности —
в `windows_variant/README_windows.md` (раздел «Установка полностью в Docker»).

## 4c. Docker с Redis (Windows / Linux / macOS)

Единый кросс-платформенный Docker-вариант, где **Redis включён и настроен**: три
контейнера — `qdrant`, `redis`, `app` — в одной сети. Папка — `docker_variant/`.

```bash
# Windows: дважды кликните start.cmd (или в терминале: start.cmd)
# Linux/macOS:
cd docker_variant
chmod +x start.sh && ./start.sh
# своя папка документов:  DOCS_DIR_HOST=/path/to/docs ./start.sh
# универсально:           docker compose up -d --build
```

Стартовый скрипт проверит Docker, подготовит `.env.docker`/состояние, соберёт и
поднимет контейнеры и покажет статус (Qdrant / Redis / приложение). Redis включается
автоматически (`REDIS_ENABLED=1`, `REDIS_HOST=redis`); его статистика видна в разделе
«Система» → «⚡ Кэш Redis». Настройки Redis — в `docker_variant/redis.conf`
(лимит памяти, персистентность, опциональный пароль). Генерация — через Ollama на
хосте. Подробности — в `docker_variant/README.md`.

## 5a. Вариант с дообучением (fine-tuning, LoRA)

Только GPU. Дообучает модель на ваших документах (QLoRA) и подаёт адаптер через
vLLM; переключение базовая/дообученная — в веб-панели.

```bash
# на развёрнутом GPU-сервере: установка зависимостей + весь пайплайн
sudo bash gpu_variant/setup_finetune.sh
# затем применить адаптер в vLLM
sudo bash gpu_variant/apply_finetuned.sh
```

Или из админ-панели: «Модель генерации» → «Запустить дообучение», затем
«Применить дообученную модель», затем выбрать карточку «Дообученная (LoRA)».

Подробности — в `finetune/README_finetune.md`. Дообучение требует видеопамяти и
времени; рекомендуется как дополнение к RAG (стиль/терминология), а не замена.

## 5b. Разнесённое (split) развёртывание на отдельных серверах

Опционально: вынести Qdrant + vLLM на отдельный GPU-сервер, а приложение
(FastAPI + эмбеддинги + LightRAG) запустить на другом. Код тот же — отличаются
только адреса бэкенда в `.env`.

```bash
# 1) на бэкенд-сервере (GPU):
sudo VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ APP_HOST=<ip-app> \
     bash remote_variant/setup_backend.sh
# 2) на сервере приложения:
sudo BACKEND_HOST=<ip-бэкенда> BACKEND_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ \
     DEVICE=cpu ADMIN_TOKEN='пароль' bash remote_variant/setup_app.sh
```

Подробности и безопасность — в `remote_variant/README_remote.md`. Qdrant и vLLM
поднимаются без авторизации, поэтому держите их в приватной сети и ограничивайте
доступ по `APP_HOST` (firewall).

## 6. Отдельная ветка LightRAG (для сравнения)

Изолированный стек для A/B-сравнения чистого граф-RAG с вектором.

```bash
cd lightrag_variant
python3.11 -m venv .venv-lr && source .venv-lr/bin/activate
pip install -r requirements-lightrag.txt
ollama pull bge-m3            # эмбеддер для LightRAG в этой ветке
python ingest_lightrag.py    # построение графа (медленно)
python query_lightrag.py "вопрос" --mode mix
```

Сравнение двух пайплайнов на одинаковых вопросах — `python compare.py`.

---

## 7. Переменные окружения (.env)

`.env` создаётся автоматически; задаётся как минимум `DOCS_DIR`. Большинство
параметров потом меняются в админке (значения из админки имеют приоритет).

| Переменная        | Назначение                                   | По умолчанию                       |
|-------------------|----------------------------------------------|------------------------------------|
| `DOCS_DIR`        | Папка с документами                          | `/opt/db`                          |
| `LLM_BACKEND`     | `ollama` (Apple) или `openai` (vLLM)         | `ollama`                           |
| `LLM_MODEL`       | Имя модели для запросов                       | `qwen3.6:35b-a3b-q4_K_M`           |
| `OLLAMA_URL`      | Адрес Ollama                                  | `http://localhost:11434`           |
| `LLM_BASE_URL`    | Адрес vLLM (OpenAI API)                       | `http://localhost:8001/v1`         |
| `EMBED_MODEL`     | Модель эмбеддингов                            | `BAAI/bge-m3`                      |
| `RERANK_MODEL`    | Модель реранка                                | `BAAI/bge-reranker-v2-m3`          |
| `DEVICE`          | `mps` / `cuda` / `cpu`                        | `mps`                              |
| `QDRANT_URL`      | Адрес Qdrant                                  | `http://localhost:6333`            |
| `QDRANT_COLLECTION`| Коллекция                                    | `company_kb`                       |
| `WHISPER_BACKEND` | `mlx` (Apple) / `faster` (GPU)               | `mlx`                              |
| `OCR_IMAGES` / `OCR_RAW` | OCR картинок / RAW-фото при индексации | `1` (вкл)                          |
| `PARSE_CAD` / `TRANSCRIBE_AV` | Чтение CAD / транскрибация медиа  | `1` (вкл)                          |
| `FILE_PARSE_TIMEOUT` | Лимит времени на файл при индексации, с   | `0` (без лимита)                   |
| `TELEGRAM_BOT_TOKEN` | Токен Телеграм-бота (@BotFather; пусто = выкл) | пусто                          |
| `TELEGRAM_AUTO_APPROVE` | Авто-подтверждение пользователей бота      | `0` (ручное)                       |
| `TELEGRAM_PROXY`  | SOCKS5/HTTP-прокси для бота (не MTProto)      | пусто                              |
| `DB_BACKEND`      | БД журнала/настроек: `sqlite`/`mysql`/`postgresql` | `sqlite`                     |
| `MYSQL_HOST` … `MYSQL_DB` | Подключение к MySQL/MariaDB (нужен PyMySQL) | пусто / `rag`                 |
| `PG_HOST` … `PG_DB` | Подключение к PostgreSQL (нужен psycopg2)   | пусто / `rag`                      |
| `REDIS_ENABLED`   | Кэш агрегатов в Redis (нужен redis)          | `0` (выкл)                         |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | Подключение к Redis           | `127.0.0.1` / `6379` / `0`         |
| `ADMIN_TOKEN`     | Пароль админ-панели (пусто = без пароля)      | пусто                              |
| `API_PORT`        | Порт веб-сервиса                              | `8000`                             |

Параметры, изменяемые в админке (TOP_K, порог, температура, промпт, режимы и т.д.),
сохраняются в `runtime_config.json` и переопределяют `.env`.

---

## 8. Проверка установки

```bash
curl http://localhost:8000/health                 # сервис жив, видит модель
curl http://localhost:6333/healthz                 # Qdrant
curl http://localhost:8001/v1/models               # vLLM (только GPU)
curl -X POST http://localhost:8000/chat \
     -H 'Content-Type: application/json' \
     -d '{"question":"тестовый вопрос"}'           # сквозной ответ RAG
```

В веб-панели раздел «Администратор» → «Состояние и операции» показывает
доступность Qdrant и LLM и число проиндексированных чанков.

---

## 9. Типичные проблемы

**`E: Невозможно найти пакет python3.11`.** В репозиториях сервера нет именно
3.11. Актуальные скрипты ставят системный `python3` (подходит 3.10–3.12). Если у вас
старая копия скрипта — пропатчите:
`sudo sed -i 's/python3\.11-venv/python3-venv/g; s/python3\.11/python3/g' run_gpu.sh`.

**`sudo: '-E' is ignored`.** Политика sudo запрещает перенос окружения. Используйте
`sudo bash -c "ADMIN_TOKEN='...' bash run_gpu.sh"` — переменные задаются внутри
root-шелла.

**`./.venv/bin/pip: Нет такого файла` / `docker: command not found` при update.**
Сервер ещё не развёрнут — это ошибка запуска `update.sh` до первичной установки.
Запустите `run_gpu.sh` (полная установка), и только потом пользуйтесь `update.sh`.

**Первый старт vLLM долгий.** Скачиваются веса модели (десятки ГБ). Следите за
`bash gpu_variant/manage.sh vllm-logs`.

**Граф-RAG не отвечает / падает.** Не установлен LightRAG или не построен граф —
выполните `gpu_variant/setup_hybrid.sh`. При недоступности граф автоматически
откатывается на вектор.

**Нет доступа к моделям Hugging Face (приватные/гейтед).** Экспортируйте
`HF_TOKEN` перед запуском — он пробрасывается в контейнер vLLM.
