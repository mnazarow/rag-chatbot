"""Централизованные настройки — единый источник правды в рантайме.

Значения по умолчанию берутся из config (.env) при первом запуске. Дальше
правятся из админ-панели и сохраняются в runtime_config.json.

scope каждого поля определяет, как применяется изменение:
  - live    : сразу (поиск/генерация читают на каждом запросе)
  - reindex : вступит в силу после переиндексации (кнопка в админке)
  - restart : вступит в силу после перезапуска сервиса (кнопка в админке)
  - vllm    : требует перезапуска контейнера vLLM (кнопка «Применить модель»)
  - secret  : пароль; в API не отдаётся открытым текстом
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import config
import prompts

_RUNTIME = Path(__file__).resolve().parent / "runtime_config.json"
_LOCK = threading.Lock()

# Полная схема: ключ, подпись, группа, тип, scope, доп. параметры.
FIELDS: list[dict] = [
    # --- Поиск и генерация (на лету) ---
    {"key": "TOP_K_RETRIEVE", "label": "Кандидатов из БД", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 5, "max": 50, "step": 1, "default": config.TOP_K_RETRIEVE,
     "desc": "Сколько фрагментов достаётся из векторной базы до реранка. Больше — шире охват "
             "(выше шанс найти нужное), но реранк медленнее. Рекомендация: 20 по умолчанию; 10–15 ради "
             "скорости на слабом железе; 30–40 для большой/разнородной базы, где ответ может быть далеко в выдаче."},
    {"key": "TOP_K_RERANK", "label": "Контекста после реранка", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 1, "max": 15, "step": 1, "default": config.TOP_K_RERANK,
     "desc": "Сколько лучших фрагментов после кросс-энкодера попадает в контекст LLM. Больше — богаче "
             "контекст, но больше шума и токенов (медленнее). Рекомендация: 6 по умолчанию; 3–4 для точечных "
             "фактических вопросов (цена, артикул); 8–10 для сводных вопросов и обзоров."},
    {"key": "MIN_SCORE", "label": "Порог релевантности", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 0, "max": 1, "step": 0.01, "default": config.MIN_SCORE,
     "desc": "Минимальная оценка релевантности фрагмента (0–1) от реранкера, чтобы попасть в ответ; если "
             "ничего не проходит порог — система честно отвечает «не знаю». Выше — строже, меньше выдумок; "
             "ниже — больше ответов, но больше шума. Рекомендация: 0.30 по умолчанию; 0.40–0.50, если бот "
             "отвечает не по теме или фантазирует; 0.20, если слишком часто говорит «нет данных», хотя они есть."},
    {"key": "TEMPERATURE", "label": "Температура генерации", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 0, "max": 1, "step": 0.05, "default": 0.1,
     "desc": "Степень «случайности» генерации. 0 — детерминированно и максимально точно (лучшее для фактов "
             "и прайсов), выше — разнообразнее, но растёт риск домыслов. Рекомендация: 0.0–0.2 для базы знаний "
             "(по умолчанию 0.1); 0.3–0.5 — только если нужны более естественные формулировки."},
    {"key": "AUTO_FILTER", "label": "Авто-фильтр по категории вопроса", "group": "Поиск и генерация",
     "type": "bool", "scope": "live", "default": True,
     "desc": "Угадывать категорию вопроса по ключевым словам (цена/обучение/презентация) и искать только "
             "внутри неё. Включено — точнее для явно «категорийных» вопросов, но может «отрезать» ответ из "
             "другой категории. Рекомендация: включено, если документы аккуратно разложены по типам; "
             "выключите, если категории размыты."},
    {"key": "AUTO_CALIBRATE", "label": "Авто-калибровка настроек по оценкам ответов",
     "group": "Поиск и генерация", "type": "bool", "scope": "live", "default": False,
     "desc": "Автоматически подстраивать параметры поиска (порог, число фрагментов) по накопленным оценкам "
             "👍/👎. Рекомендация: держите выключенной, пока оценок мало (хотя бы несколько десятков); "
             "включайте для «само-настройки». Ручная калибровка — в блоке «Калибровка по оценкам»."},
    {"key": "SYSTEM_PROMPT", "label": "Системный промпт", "group": "Поиск и генерация",
     "type": "textarea", "scope": "live", "default": prompts.SYSTEM_PROMPT,
     "desc": "Инструкция модели: отвечать строго по документам, ссылаться на источники, говорить «не знаю», "
             "если данных нет; язык и стиль. Сильно влияет на ответы. Рекомендация: сохраняйте правило "
             "«отвечай только по контексту, не выдумывай»; здесь же меняйте тон, язык и формат (кратко/подробно)."},

    # --- Генерация / бэкенд ---
    {"key": "LLM_BACKEND", "label": "Бэкенд LLM", "group": "Генерация (LLM)",
     "type": "select", "scope": "restart", "options": ["ollama", "openai"], "default": config.LLM_BACKEND},
    {"key": "LLM_MODEL", "label": "Имя модели (для запросов)", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_MODEL},
    {"key": "OLLAMA_URL", "label": "URL Ollama", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.OLLAMA_URL},
    {"key": "LLM_BASE_URL", "label": "URL vLLM (OpenAI API)", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_BASE_URL},
    {"key": "LLM_API_KEY", "label": "API-ключ vLLM", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_API_KEY},

    # --- vLLM контейнер (GPU) ---
    {"key": "VLLM_MODEL", "label": "Модель vLLM (контейнер)", "group": "vLLM (GPU)",
     "type": "text", "scope": "vllm", "default": config.VLLM_MODEL},
    {"key": "VLLM_MAX_LEN", "label": "Макс. длина контекста", "group": "vLLM (GPU)",
     "type": "int", "scope": "vllm", "default": config.VLLM_MAX_LEN},
    {"key": "VLLM_TP", "label": "Tensor-parallel (число GPU)", "group": "vLLM (GPU)",
     "type": "int", "scope": "vllm", "default": config.VLLM_TP},

    # --- Эмбеддинги / устройство ---
    {"key": "EMBED_MODEL", "label": "Модель эмбеддингов", "group": "Эмбеддинги и устройство",
     "type": "text", "scope": "restart", "default": config.EMBED_MODEL,
     "desc": "Модель, кодирующая документы и запрос в векторы для семантического поиска — "
             "это главный фактор качества RAG (важнее размера LLM). Рекомендация: "
             "<code>BAAI/bge-m3</code> (по умолчанию, многоязычная, сильный русский, 1024d); "
             "<code>intfloat/multilingual-e5-large</code> — проверенная альтернатива (1024d); "
             "<code>Qwen/Qwen3-Embedding-0.6B</code> — топ MTEB, лёгкая (1024d); "
             "<code>nomic-ai/nomic-embed-text-v2-moe</code> — лёгкая (768d, поставьте EMBED_DIM=768). "
             "⚠️ Смена модели требует «Сбросить индекс» и «Переиндексировать», и согласованной "
             "EMBED_DIM. Применяется после перезапуска сервиса."},
    {"key": "RERANK_MODEL", "label": "Модель реранка", "group": "Эмбеддинги и устройство",
     "type": "text", "scope": "restart", "default": config.RERANK_MODEL,
     "desc": "Кросс-энкодер: переоценивает кандидатов из поиска и оставляет самые релевантные — "
             "главный рычаг точности и порога «не знаю». Рекомендация: "
             "<code>BAAI/bge-reranker-v2-m3</code> (по умолчанию, многоязычный); "
             "<code>BAAI/bge-reranker-base</code> — легче и быстрее на CPU ценой качества. "
             "Чем мощнее реранкер, тем меньше «похожего, но не того» контекста. "
             "Применяется после перезапуска сервиса (переиндексация не нужна)."},
    {"key": "EMBED_DIM", "label": "Размерность эмбеддингов", "group": "Эмбеддинги и устройство",
     "type": "int", "scope": "reindex", "default": config.EMBED_DIM,
     "desc": "Размерность вектора эмбеддера — ДОЛЖНА совпадать с выбранной моделью, иначе индексация "
             "сломается. Для bge-m3 / e5-large / Qwen3-Embedding-0.6B — <b>1024</b>; для "
             "nomic-embed-text-v2 — <b>768</b>. При несоответствии: поставьте верное значение, "
             "«Сбросить индекс», затем «Переиндексировать»."},
    {"key": "DEVICE", "label": "Устройство", "group": "Эмбеддинги и устройство",
     "type": "select", "scope": "restart", "options": ["cuda", "mps", "cpu"], "default": config.DEVICE,
     "desc": "На чём считать эмбеддинги и реранк: <code>mps</code> — Apple Silicon (Metal), "
             "<code>cuda</code> — NVIDIA GPU, <code>cpu</code> — везде, но заметно медленнее "
             "(особенно реранк и индексация). Рекомендация: ставьте тот ускоритель, что есть в "
             "системе; <code>cpu</code> — только если GPU/Metal недоступны. Применяется после "
             "перезапуска сервиса."},

    # --- Хранилище ---
    {"key": "QDRANT_URL", "label": "URL Qdrant", "group": "Хранилище",
     "type": "text", "scope": "restart", "default": config.QDRANT_URL},
    {"key": "QDRANT_COLLECTION", "label": "Коллекция", "group": "Хранилище",
     "type": "text", "scope": "restart", "default": config.QDRANT_COLLECTION},
    {"key": "QDRANT_TIMEOUT", "label": "Таймаут запросов Qdrant, с", "group": "Хранилище",
     "type": "int", "scope": "restart", "default": config.QDRANT_TIMEOUT},
    {"key": "QDRANT_INGEST_TIMEOUT", "label": "Таймаут индексации Qdrant, с", "group": "Хранилище",
     "type": "int", "scope": "reindex", "default": config.QDRANT_INGEST_TIMEOUT},

    # --- Документы и индексация ---
    {"key": "DOCS_DIR", "label": "Папка с документами", "group": "Документы и индексация",
     "type": "text", "scope": "reindex", "default": str(config.DOCS_DIR)},
    {"key": "CHUNK_SIZE", "label": "Размер чанка (символов)", "group": "Документы и индексация",
     "type": "int", "scope": "reindex", "default": config.CHUNK_SIZE},
    {"key": "CHUNK_OVERLAP", "label": "Перекрытие чанков", "group": "Документы и индексация",
     "type": "int", "scope": "reindex", "default": config.CHUNK_OVERLAP},
    {"key": "OCR_IMAGES", "label": "OCR изображений (jpg/png/…)", "group": "Документы и индексация",
     "type": "bool", "scope": "reindex", "default": config.OCR_IMAGES,
     "desc": "Распознавать текст на картинках. Самый долгий этап индексации (секунды на файл). "
             "Рекомендация: выключите для быстрой первой индексации большого каталога, затем включите "
             "и переиндексируйте (инкрементально) — текстовые документы проиндексируются сразу, "
             "а картинки добавятся отдельным проходом."},
    {"key": "OCR_RAW", "label": "OCR RAW-фото (CR2/NEF/…)", "group": "Документы и индексация",
     "type": "bool", "scope": "reindex", "default": config.OCR_RAW,
     "desc": "Декодировать и распознавать RAW-снимки. Долго (декодирование + OCR). Выключите, если "
             "RAW-файлы не содержат текста или не нужны в поиске."},
    {"key": "PARSE_CAD", "label": "Чертежи CAD (DXF/DWG/STEP/IGES)", "group": "Документы и индексация",
     "type": "bool", "scope": "reindex", "default": config.PARSE_CAD,
     "desc": "Извлекать текст из чертежей. Конвертация DWG→DXF может быть долгой при тысячах файлов. "
             "Выключите для ускорения, если надписи на чертежах не нужны в поиске."},
    {"key": "TRANSCRIBE_AV", "label": "Транскрибация аудио/видео (Whisper)", "group": "Документы и индексация",
     "type": "bool", "scope": "reindex", "default": config.TRANSCRIBE_AV,
     "desc": "Расшифровывать речь в записях. Самое долгое на файл (минуты). Рекомендация: при первой "
             "индексации большого каталога выключите, затем включите для отдельного прохода по медиа."},
    {"key": "FILE_PARSE_TIMEOUT", "label": "Лимит времени на файл, с (0 = без лимита)",
     "group": "Документы и индексация", "type": "int", "scope": "reindex", "default": config.FILE_PARSE_TIMEOUT,
     "desc": "Если обработка одного файла превышает лимит — он пропускается (защита от «зависания» на "
             "тяжёлом DWG/видео). Рекомендация: 0 без лимита; 60–180 для каталога со «сложными» файлами. "
             "Работает на Linux/macOS."},

    # --- Транскрибация ---
    {"key": "WHISPER_BACKEND", "label": "Бэкенд Whisper", "group": "Транскрибация",
     "type": "select", "scope": "reindex", "options": ["faster", "mlx"], "default": config.WHISPER_BACKEND},
    {"key": "WHISPER_MODEL", "label": "Модель Whisper", "group": "Транскрибация",
     "type": "text", "scope": "reindex", "default": config.WHISPER_MODEL},

    # --- Расширенный поиск (hybrid+) ---
    {"key": "LLM_METADATA", "label": "LLM-метаданные при индексации (продукт/тема/тип)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "reindex", "default": False},
    {"key": "SMART_FILTER", "label": "Умные фильтры из вопроса (LLM)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "live", "default": False},
    {"key": "GRAPH_RAG", "label": "Граф-RAG для сводных вопросов (нужен LightRAG)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "live", "default": False},
    {"key": "GRAPH_MODE", "label": "Режим графа", "group": "Расширенный поиск (hybrid+)",
     "type": "select", "scope": "live", "options": ["mix", "hybrid", "local", "global", "naive"],
     "default": "mix"},

    # --- Движок ответов ---
    {"key": "ENGINE", "label": "Движок ответов", "group": "Движок ответов",
     "type": "select", "scope": "live", "options": ["vector", "lightrag"],
     "default": "vector"},

    # --- Дообучение (fine-tuning) ---
    {"key": "USE_FINETUNED", "label": "Использовать дообученную модель",
     "group": "Дообучение (fine-tuning)", "type": "bool", "scope": "live", "default": False,
     "desc": "Переключить генерацию на дообученную LoRA-модель (вместо базовой). Включайте "
             "только после того, как адаптер обучен и применён («Применить дообученную модель»). "
             "Дообучение настраивает стиль/терминологию ответов; факты по-прежнему берутся из "
             "документов через RAG. Доступно для бэкенда vLLM (GPU)."},
    {"key": "FINETUNED_MODEL", "label": "Имя дообученной модели (LoRA в vLLM)",
     "group": "Дообучение (fine-tuning)", "type": "text", "scope": "live", "default": "company-lora",
     "desc": "Имя LoRA-адаптера, под которым он обслуживается в vLLM (то же имя указывается при "
             "запуске vLLM с адаптером). По умолчанию <code>company-lora</code> — менять обычно не "
             "нужно. Должно совпадать с именем, поданным в vLLM кнопкой «Применить дообученную модель»."},
    {"key": "FINETUNE_BASE", "label": "Базовая модель для дообучения (пусто = из VLLM_MODEL)",
     "group": "Дообучение (fine-tuning)", "type": "text", "scope": "live", "default": config.FINETUNE_BASE,
     "desc": "Базовая fp16-модель, на которой обучается LoRA (QLoRA, 4-bit). Выберите из списка ниже "
             "(каталог проверенных баз) или впишите HF-идентификатор; пусто = берётся из "
             "<code>VLLM_MODEL</code> с отбрасыванием суффиксов квантизации (-AWQ/-GPTQ/-Int4). "
             "Рекомендации по VRAM: ~16 ГБ → Qwen2.5-7B-Instruct ✅; ~24 ГБ → Qwen2.5-14B / Qwen3-14B / "
             "gemma-3-12b; ~40 ГБ → Qwen2.5-32B. Применяется при следующем запуске обучения."},

    # --- Телеграм-бот ---
    {"key": "TELEGRAM_BOT_TOKEN", "label": "Токен бота (@BotFather)", "group": "Телеграм-бот",
     "type": "secret", "scope": "live", "default": config.TELEGRAM_BOT_TOKEN,
     "desc": "Создайте бота у @BotFather в Telegram и вставьте его токен. После изменения "
             "нажмите «Перезапустить бота» в разделе «Телеграм». Пусто = бот выключен."},
    {"key": "TELEGRAM_AUTO_APPROVE", "label": "Авто-подтверждение новых пользователей",
     "group": "Телеграм-бот", "type": "bool", "scope": "live", "default": config.TELEGRAM_AUTO_APPROVE,
     "desc": "Если включено — любой написавший боту сразу получает доступ. По умолчанию "
             "выключено: новый пользователь попадает в «Запросы на доступ», и вы подтверждаете "
             "его вручную в разделе «Телеграм»."},
    {"key": "TELEGRAM_PROXY", "label": "Прокси для бота (SOCKS5/HTTP)", "group": "Телеграм-бот",
     "type": "text", "scope": "live", "default": config.TELEGRAM_PROXY,
     "desc": "Прокси для доступа к <code>api.telegram.org</code>, если Telegram заблокирован в вашей "
             "сети. Формат: <code>socks5://host:port</code>, <code>socks5h://user:pass@host:port</code>, "
             "<code>http://host:port</code> или <code>https://…</code>. Пусто = без прокси. "
             "<b>Важно:</b> бот использует HTTP Bot API, поэтому <b>MTProto-прокси</b> (tg://proxy-ссылки) "
             "здесь НЕ подходят — они для приложений-клиентов Telegram; укажите обычный SOCKS5/HTTP-прокси. "
             "После изменения нажмите «Перезапустить бота»."},

    # --- База данных и кэш ---
    {"key": "DB_BACKEND", "label": "Активная база данных", "group": "База данных и кэш",
     "type": "select", "scope": "restart", "options": ["sqlite", "mysql", "postgresql"],
     "default": config.DB_BACKEND,
     "desc": "Где хранятся журнал запросов, история Телеграм и аналитика. "
             "<code>sqlite</code> — по умолчанию, локальный файл, без внешних сервисов. "
             "<code>mysql</code>/<code>postgresql</code> — внешняя СУБД (для нескольких "
             "инстансов или централизованного хранения). <b>Не меняйте это поле вручную</b> "
             "для переезда — используйте кнопки «Мигрировать» в блоке «База данных и кэш» "
             "раздела Администратор (они и данные перенесут, и переключат бэкенд). Смена "
             "вручную лишь переключает чтение/запись на выбранную СУБД (таблицы должны "
             "существовать). Требует драйвер: PyMySQL / psycopg2."},
    {"key": "MYSQL_HOST", "label": "MySQL: хост", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.MYSQL_HOST,
     "desc": "IP/домен сервера MySQL (или MariaDB). Пусто = не настроен."},
    {"key": "MYSQL_PORT", "label": "MySQL: порт", "group": "База данных и кэш",
     "type": "int", "scope": "live", "default": config.MYSQL_PORT},
    {"key": "MYSQL_USER", "label": "MySQL: пользователь", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.MYSQL_USER},
    {"key": "MYSQL_PASSWORD", "label": "MySQL: пароль", "group": "База данных и кэш",
     "type": "secret", "scope": "live", "default": config.MYSQL_PASSWORD},
    {"key": "MYSQL_DB", "label": "MySQL: имя БД", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.MYSQL_DB,
     "desc": "Имя существующей базы (создайте её заранее: <code>CREATE DATABASE rag "
             "CHARACTER SET utf8mb4;</code>). Таблицы приложение создаст само."},
    {"key": "PG_HOST", "label": "PostgreSQL: хост", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.PG_HOST,
     "desc": "IP/домен сервера PostgreSQL. Пусто = не настроен."},
    {"key": "PG_PORT", "label": "PostgreSQL: порт", "group": "База данных и кэш",
     "type": "int", "scope": "live", "default": config.PG_PORT},
    {"key": "PG_USER", "label": "PostgreSQL: пользователь", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.PG_USER},
    {"key": "PG_PASSWORD", "label": "PostgreSQL: пароль", "group": "База данных и кэш",
     "type": "secret", "scope": "live", "default": config.PG_PASSWORD},
    {"key": "PG_DB", "label": "PostgreSQL: имя БД", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.PG_DB,
     "desc": "Имя существующей базы (создайте заранее: <code>CREATE DATABASE rag;</code>). "
             "Таблицы приложение создаст само."},
    {"key": "REDIS_ENABLED", "label": "Включить кэш Redis", "group": "База данных и кэш",
     "type": "bool", "scope": "live", "default": config.REDIS_ENABLED,
     "desc": "Кэшировать тяжёлые агрегаты (статистика, аналитика, статистика Телеграм) "
             "в Redis. По умолчанию выключено — приложение работает и без Redis. Кэш "
             "сбрасывается автоматически при каждом новом запросе, поэтому данные не "
             "устаревают. Требует модуль <code>redis</code> и доступный сервер Redis."},
    {"key": "REDIS_HOST", "label": "Redis: хост", "group": "База данных и кэш",
     "type": "text", "scope": "live", "default": config.REDIS_HOST},
    {"key": "REDIS_PORT", "label": "Redis: порт", "group": "База данных и кэш",
     "type": "int", "scope": "live", "default": config.REDIS_PORT},
    {"key": "REDIS_DB", "label": "Redis: номер БД (0–15)", "group": "База данных и кэш",
     "type": "int", "scope": "live", "default": config.REDIS_DB},
    {"key": "REDIS_PASSWORD", "label": "Redis: пароль (если задан)",
     "group": "База данных и кэш", "type": "secret", "scope": "live",
     "default": config.REDIS_PASSWORD},
    {"key": "CATALOG_SOURCE", "label": "Источник каталога документов",
     "group": "База данных и кэш", "type": "select", "scope": "live",
     "options": ["filesystem", "postgresql"], "default": config.CATALOG_SOURCE,
     "desc": "Откуда брать список документов и их текст для просмотра в «Каталоге "
             "документов». <code>filesystem</code> — папка <code>DOCS_DIR</code> (по "
             "умолчанию). <code>postgresql</code> — таблица <code>doc_catalog</code> в "
             "активной PostgreSQL (доступно после загрузки каталога). Переключается "
             "кнопками в блоке «База данных и кэш» — менять вручную обычно не нужно."},

    # --- Доступ ---
    {"key": "ADMIN_TOKEN", "label": "Токен администратора", "group": "Доступ",
     "type": "secret", "scope": "live", "default": config.ADMIN_TOKEN},
]

DEFAULTS: dict = {f["key"]: f["default"] for f in FIELDS}
_TYPES: dict = {f["key"]: f["type"] for f in FIELDS}
_OPTIONS: dict = {f["key"]: f.get("options") for f in FIELDS if f["type"] == "select"}

# Режимы работы — пресеты, которые разом включают связку улучшений.
MODES: dict = {
    "basic": {
        "label": "Базовый",
        "desc": "Вектор + реранк. Быстро, без дополнительных вызовов LLM.",
        "flags": {"SMART_FILTER": False, "GRAPH_RAG": False, "AUTO_FILTER": True},
    },
    "meta": {
        "label": "С метаданными",
        "desc": "+ умные фильтры из вопроса (категория / продукт / период).",
        "flags": {"SMART_FILTER": True, "GRAPH_RAG": False, "AUTO_FILTER": True},
    },
    "hybrid": {
        "label": "Hybrid+ (граф)",
        "desc": "+ граф-RAG для сводных вопросов (требуется LightRAG).",
        "flags": {"SMART_FILTER": True, "GRAPH_RAG": True, "AUTO_FILTER": True},
    },
}

_state: dict = dict(DEFAULTS)
_state["MODE"] = "basic"


def _load() -> None:
    if _RUNTIME.exists():
        try:
            data = json.loads(_RUNTIME.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k in _state:
                    _state[k] = v
        except Exception:
            pass


_load()


def get(key: str):
    return _state.get(key, DEFAULTS.get(key))


def active_model() -> str:
    """Имя модели для генерации: дообученная (если включена) или базовая."""
    if _state.get("USE_FINETUNED"):
        return _state.get("FINETUNED_MODEL") or _state.get("LLM_MODEL")
    return _state.get("LLM_MODEL")


def device() -> str:
    """Фактическое устройство для эмбеддингов/реранка с проверкой доступности.

    Если задан cuda/mps, но он недоступен (нет проброса GPU в контейнер или
    установлена CPU-сборка torch) — откатываемся на cpu с предупреждением, чтобы
    не падать. Так на GPU-хосте достаточно поставить DEVICE=cuda и CUDA-сборку torch.
    """
    d = (get("DEVICE") or "cpu").lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if d == "cuda":
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        print("  ! DEVICE=cuda, но CUDA недоступна — использую cpu (нет проброса GPU "
              "в контейнер или установлена CPU-сборка torch)")
        return "cpu"
    if d == "mps":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return "cpu"


def all_settings() -> dict:
    return dict(_state)


def public_settings() -> dict:
    """Значения для UI: секреты маскируются."""
    out = dict(_state)
    for f in FIELDS:
        if f["type"] == "secret":
            out[f["key"]] = ""  # не отдаём пароль наружу
    return out


def secret_is_set(key: str) -> bool:
    return bool(_state.get(key))


def _coerce(key: str, value):
    t = _TYPES.get(key, "text")
    if t == "bool":
        return bool(value)
    if t == "int" or t == "range" and isinstance(DEFAULTS[key], int):
        return int(value)
    if t == "range" or (t in ("float",)):
        return float(value)
    return str(value)


def update(changes: dict) -> dict:
    with _LOCK:
        for k, v in changes.items():
            if k not in DEFAULTS:
                continue
            # пустой секрет = «не менять»
            if _TYPES[k] == "secret" and (v is None or v == ""):
                continue
            # select: только значения из списка опций
            if _TYPES[k] == "select" and v not in (_OPTIONS.get(k) or []):
                continue
            try:
                _state[k] = _coerce(k, v)
            except (TypeError, ValueError):
                continue
        _RUNTIME.write_text(json.dumps(_state, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return public_settings()


def reset() -> dict:
    with _LOCK:
        _state.clear()
        _state.update(DEFAULTS)
        _state["MODE"] = "basic"
        if _RUNTIME.exists():
            _RUNTIME.unlink()
    return public_settings()


# ----- режимы работы -----
def modes_catalog() -> list[dict]:
    return [{"key": k, "label": m["label"], "desc": m["desc"]} for k, m in MODES.items()]


def current_mode() -> str:
    """Текущий режим; 'custom', если флаги настроены вручную и не совпадают с пресетом."""
    cur = _state.get("MODE", "basic")
    flags = MODES.get(cur, {}).get("flags", {})
    if flags and all(_state.get(k) == v for k, v in flags.items()):
        return cur
    return "custom"


def set_mode(name: str) -> dict:
    if name not in MODES:
        return public_settings()
    with _LOCK:
        _state["MODE"] = name
        for k, v in MODES[name]["flags"].items():
            _state[k] = v
        _RUNTIME.write_text(json.dumps(_state, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return public_settings()
