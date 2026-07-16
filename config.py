"""Единая конфигурация (читается из .env)."""
import os
from pathlib import Path
from dotenv import load_dotenv

# HF-токенайзеры: отключаем внутренний параллелизм — иначе при fork (распаковка
# архивов, dwg2dxf, ffmpeg) сыплется предупреждение и возможны зависания/замедления.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))

def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") not in ("0", "false", "False", "no", "")

# Документы
DOCS_DIR = Path(os.getenv("DOCS_DIR", "/opt/db")).expanduser()

# Модели
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:35b-a3b-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "mps")          # mps (Apple) | cuda (GPU) | cpu
# Размер пачки эмбеддера при индексации: сколько чанков считать за один проход.
# Больше — выше пропускная способность на GPU, но больше расход видеопамяти.
EMBED_BATCH = _int("EMBED_BATCH", 32)

# Бэкенд генерации: ollama (Apple/CPU) | openai (vLLM, OpenAI-совместимый API)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")  # для vLLM
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
# Очередь к LLM: максимум одновременных запросов к модели (генерация, vision и т. п.).
# Остальные ждут своей очереди. 0 — без ограничения. Защищает перегруженную модель/GPU.
LLM_MAX_CONCURRENCY = _int("LLM_MAX_CONCURRENCY", 0)
LLM_QUEUE_TIMEOUT = _int("LLM_QUEUE_TIMEOUT", 600)   # макс. ожидание в очереди, с (0 — без лимита)
# Минимальная пауза между началами запросов к LLM (с). Запросы стартуют не чаще,
# чем раз в LLM_REQUEST_DELAY секунд. 0 — без паузы. Бережёт модель/GPU от «пиков».
LLM_REQUEST_DELAY = _float("LLM_REQUEST_DELAY", 3.0)
# «Размышления» гибридных моделей (Qwen3/3.6, DeepSeek-R1 и т. п.). Для таких моделей
# Ollama по умолчанию генерирует длинную фазу рассуждений (поле message.thinking или
# блок <think>…</think>), из-за чего видимый ответ появляется с большой задержкой или
# «молчит». False (по умолчанию) — просить модель отвечать сразу, без размышлений;
# True — оставить размышления (медленнее, но иногда точнее на сложных вопросах).
LLM_THINK = _bool("LLM_THINK", True)
# Скрывать блок рассуждений <think>…</think> в ответе — отдельно для веб-чата и Телеграма.
# Модель может рассуждать (LLM_THINK=да), но сами «мысли» пользователю не показываем.
HIDE_THINK_CHAT = _bool("HIDE_THINK_CHAT", True)
HIDE_THINK_TELEGRAM = _bool("HIDE_THINK_TELEGRAM", True)

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_kb")
QDRANT_TIMEOUT = _int("QDRANT_TIMEOUT", 60)            # таймаут запросов (чат), с
QDRANT_INGEST_TIMEOUT = _int("QDRANT_INGEST_TIMEOUT", 480)  # таймаут индексации, с
# Ждать применения точек Qdrant (wait=true) при записи. ВКЛ (по умолч.) — обратное давление:
# каждая партия применяется синхронно, внутренняя очередь Qdrant не переполняется на огромных
# файлах (иначе «Server disconnected»/500 «queue full»). ВЫКЛ — быстрее, но рискованно на больших.
QDRANT_UPSERT_WAIT = _bool("QDRANT_UPSERT_WAIT", True)
# Режим массовой загрузки: на время индексации отключать HNSW-индексацию Qdrant
# (indexing_threshold=0) и включать в конце — иначе на больших файлах построение индекса
# на пороге ~20000 конкурирует с записью и Qdrant отдаёт 500. По умолчанию включено.
QDRANT_BULK_INDEXING = _bool("QDRANT_BULK_INDEXING", True)

# --- Векторная база: активный бэкенд (qdrant | milvus) ---
# Весь поиск/индексация идут через vectorstore.py. По умолчанию — qdrant. Milvus
# устанавливается и наполняется из админки; переключение — после проверки миграции.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")

# --- Milvus ---
# Режим: lite — встроенный (файл-хранилище в контейнере/рядом с проектом, без etcd/minio);
#        standalone — внешний сервер Milvus (контейнеры milvus+etcd+minio), URI http://host:19530.
MILVUS_MODE = os.getenv("MILVUS_MODE", "lite")
MILVUS_URI = os.getenv("MILVUS_URI", "")              # standalone: полный URI (приоритет над host/port)
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")      # standalone: имя сервиса/хост
MILVUS_PORT = _int("MILVUS_PORT", 19530)              # standalone: порт gRPC
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")          # user:password или api-key (если включена аутентификация)
MILVUS_LITE_PATH = os.getenv("MILVUS_LITE_PATH", "")  # lite: путь к файлу БД (пусто = milvus_lite.db у приложения)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "company_kb")
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "COSINE")  # COSINE (норм. эмбеддинги) | IP | L2
# Тип индекса: HNSW (CPU, как в Qdrant) | GPU_CAGRA/GPU_IVF_FLAT (GPU) | IVF_FLAT/IVF_SQ8 | FLAT (точный перебор)
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
MILVUS_HNSW_M = _int("MILVUS_HNSW_M", 16)                     # HNSW: связей на узел (качество↔память)
MILVUS_HNSW_EF_CONSTRUCTION = _int("MILVUS_HNSW_EF_CONSTRUCTION", 200)  # HNSW: ширина при построении
MILVUS_SEARCH_EF = _int("MILVUS_SEARCH_EF", 128)             # HNSW ef / CAGRA itopk при поиске
MILVUS_NLIST = _int("MILVUS_NLIST", 1024)                    # IVF: число кластеров
MILVUS_NPROBE = _int("MILVUS_NPROBE", 16)                    # IVF: сколько кластеров смотреть при поиске

# RAG-параметры
CHUNK_SIZE = _int("CHUNK_SIZE", 1800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 360)
# Предохранитель от «разбухания»: максимум чанков на один файл/архив. Если файл
# (например огромный архив с тысячами моделей) даёт больше — лишнее отбрасывается
# с предупреждением. 0 — без ограничения.
INGEST_MAX_CHUNKS = _int("INGEST_MAX_CHUNKS", 2000000)
# Путь к ODA File Converter (конвертер DWG→DXF, запасной к dwg2dxf). Пусто —
# искать автоматически (PATH и типовые места установки, вкл. .app на macOS).
ODA_CONVERTER_PATH = os.getenv("ODA_CONVERTER_PATH", "")
TOP_K_RETRIEVE = _int("TOP_K_RETRIEVE", 60)
TOP_K_RERANK = _int("TOP_K_RERANK", 6)
MIN_SCORE = _float("MIN_SCORE", 0.15)
# Авто-фильтр по категории вопроса — теперь управляется и через .env (по умолч. включён).
AUTO_FILTER = _bool("AUTO_FILTER", True)

# Телефония: голосовой мост к АТС через Asterisk AudioSocket (STT→RAG→TTS).
SIP_ENABLED = os.getenv("SIP_ENABLED", "0") not in ("0", "false", "")
SIP_BRIDGE_HOST = os.getenv("SIP_BRIDGE_HOST", "0.0.0.0")
SIP_BRIDGE_PORT = _int("SIP_BRIDGE_PORT", 8090)
SIP_GREETING = os.getenv("SIP_GREETING",
                         "Здравствуйте! Это голосовой ассистент компании. "
                         "Задайте вопрос после сигнала.")
SIP_SILENCE_MS = _int("SIP_SILENCE_MS", 700)     # пауза-тишина = конец реплики
SIP_SILENCE_RMS = _int("SIP_SILENCE_RMS", 500)   # порог громкости (тишина ниже)
SIP_MAX_UTTER_SEC = _int("SIP_MAX_UTTER_SEC", 15)

# Нативная SIP-регистрация (без AudioSocket): бот регистрируется как SIP-аккаунт
# на АТС/провайдере и принимает звонки напрямую (RTP-аудио через pyVoIP).
SIP_REGISTER_ENABLED = os.getenv("SIP_REGISTER_ENABLED", "0") not in ("0", "false", "")
SIP_SERVER = os.getenv("SIP_SERVER", "")          # хост АТС/провайдера (домен SIP)
SIP_PORT = _int("SIP_PORT", 5060)                 # порт SIP-сервера
SIP_USERNAME = os.getenv("SIP_USERNAME", "")      # логин (внутренний номер/аккаунт)
SIP_PASSWORD = os.getenv("SIP_PASSWORD", "")      # пароль SIP-аккаунта (секрет)
SIP_LOCAL_IP = os.getenv("SIP_LOCAL_IP", "")      # наш IP для SDP (пусто = автоопредел.)
SIP_LOCAL_PORT = _int("SIP_LOCAL_PORT", 5060)     # локальный SIP-порт (bind)
SIP_RTP_PORT_LOW = _int("SIP_RTP_PORT_LOW", 10000)
SIP_RTP_PORT_HIGH = _int("SIP_RTP_PORT_HIGH", 20000)
SIP_DEBUG = os.getenv("SIP_DEBUG", "0") not in ("0", "false", "")   # подробный лог pyVoIP
# Озвучивать в звонке ответ модели (LLM/RAG). Выкл — бот произносит короткую отметку
# (SIP_ACK_PHRASE), но запрос всё равно обрабатывается и попадает в журнал.
SIP_SPEAK_ANSWER = os.getenv("SIP_SPEAK_ANSWER", "1") not in ("0", "false", "")
SIP_ACK_PHRASE = os.getenv("SIP_ACK_PHRASE", "Ваш запрос принят и записан. Спасибо.")
# Ускорение отправки RTP в pyVoIP (TRANSMIT_DELAY_REDUCTION): если звук в звонке
# «медленный»/с провалами (pyVoIP не успевает слать пакеты в реальном времени),
# увеличьте (0.5, 1, 2). 0 — штатный темп pyVoIP.
SIP_TX_SPEEDUP = _float("SIP_TX_SPEEDUP", 0.0)

# Прайс-папка: на «ценовых» вопросах брать контекст напрямую из указанной папки
# (без индексации — файлы читаются по требованию и реранкуются под вопрос).
PRICE_FOLDER = os.getenv("PRICE_FOLDER", "0") not in ("0", "false", "")  # включение
PRICE_DIR = os.getenv("PRICE_DIR", "")          # путь к папке с прайс-листами
PRICE_TOP_K = _int("PRICE_TOP_K", 6)            # сколько фрагментов прайса в контекст

# Транскрибация
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "mlx")  # mlx (Apple) | faster (GPU/CPU)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# vLLM (GPU-вариант): параметры контейнера генерации
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
VLLM_MAX_LEN = _int("VLLM_MAX_LEN", 16384)
VLLM_TP = _int("VLLM_TP", 2)
# Доля видеопамяти под vLLM (0.5–0.98). Ниже (напр. 0.80) — оставить запас под vision-энкодер
# мультимодальных моделей, иначе на картинках движок может падать (OOM). По умолчанию 0.9.
VLLM_GPU_UTIL = _float("VLLM_GPU_UTIL", 0.9)

# Дообучение (QLoRA): базовая fp16-модель. Пусто = берётся из VLLM_MODEL
# (с отбрасыванием суффиксов квантизации -AWQ/-GPTQ/-Int4).
FINETUNE_BASE = os.getenv("FINETUNE_BASE", "")


# Индексация: какие тяжёлые экстракторы включать (отключение ускоряет индексацию).
OCR_IMAGES = _bool("OCR_IMAGES", True)        # OCR изображений (jpg/png/…) — самый долгий
OCR_RAW = _bool("OCR_RAW", True)              # OCR RAW-фото (CR2/NEF/…)
PARSE_CAD = _bool("PARSE_CAD", True)          # чертежи DXF/DWG и 3D-CAD (конвертация DWG долгая)
# Дополнительно отрисовать чертёж DWG/DXF в изображение и описать vision-моделью
# (как «глазами»): помогает по чертежам без текстовых надписей. Требуется мультимодальная
# модель (VISION_MODEL или основная LLM) + matplotlib. По умолчанию выкл. (тяжело/медленно).
CAD_LLM_DESCRIBE = _bool("CAD_LLM_DESCRIBE", False)
TRANSCRIBE_AV = _bool("TRANSCRIBE_AV", True)  # транскрибация аудио/видео (Whisper, минуты на файл)
FILE_PARSE_TIMEOUT = _int("FILE_PARSE_TIMEOUT", 0)  # лимит времени на файл, c (0 = без лимита)
# Параллельное извлечение файлов при индексации (парсинг/OCR/конвертация в несколько
# потоков; эмбеддинг и запись в Qdrant — в основном потоке). 0 = авто (по числу ядер).
# При заданном FILE_PARSE_TIMEOUT принудительно 1 (таймаут работает только однопоточно).
INGEST_WORKERS = _int("INGEST_WORKERS", 0)

# --- Расширенные параметры OCR (tesseract) ---
# Языки распознавания (через '+', напр. "rus+eng"). Пусто = автоопределение по
# установленным языковым пакетам tesseract (rus+eng, если есть).
OCR_LANGS = os.getenv("OCR_LANGS", "")
# Масштаб рендера страниц PDF в картинку перед OCR: 2.5 ≈ 180 DPI. Больше — точнее
# на мелком шрифте, но медленнее и больше памяти.
OCR_SCALE = _float("OCR_SCALE", 2.5)
# Максимальная сторона изображения (пиксели): крупнее — даунскейл (ускоряет OCR).
OCR_MAX_DIM = _int("OCR_MAX_DIM", 3500)
# Сколько символов на странице PDF считать «достаточным» текстовым слоем; ниже —
# страница считается «картиночной» и прогоняется через OCR.
OCR_MIN_CHARS = _int("OCR_MIN_CHARS", 25)
# Если OCR дал мало текста (≤ OCR_LLM_MAX_CHUNKS чанков), передать изображение
# vision-модели за описанием и тоже добавить его в базу знаний.
OCR_LLM_DESCRIBE = _bool("OCR_LLM_DESCRIBE", False)
OCR_LLM_MAX_CHUNKS = _int("OCR_LLM_MAX_CHUNKS", 1)
# Описывать КАЖДЫЙ файл при индексации: LLM делает краткое описание/summary документа,
# оно тоже добавляется в базу знаний (улучшает поиск). Медленно (вызов LLM на файл).
INDEX_LLM_DESCRIBE = _bool("INDEX_LLM_DESCRIBE", False)
INDEX_LLM_DESCRIBE_MAXCHARS = _int("INDEX_LLM_DESCRIBE_MAXCHARS", 6000)  # сколько текста слать LLM
# Дедуп документов по содержимому при индексации: файлы с одинаковым извлечённым
# текстом (SHA-256) индексируются один раз (последующие копии пропускаются). По умолч. выкл.
INDEX_DEDUP = _bool("INDEX_DEDUP", True)
# Автоматическая инкрементальная переиндексация папки по расписанию (off|hourly|daily|
# weekly|Nh, напр. 6h). Индексируются только новые/изменённые файлы. Крутит monitor.py.
REINDEX_SCHEDULE = os.getenv("REINDEX_SCHEDULE", "off")

# --- Качество поиска: обогащение чанков ---
# Контекстные чанки: перед эмбеддингом добавлять к чанку заголовок документа и тему
# (contextual retrieval) — повышает и полноту, и точность. В базе хранится исходный
# текст для показа; обогащается только то, что кодируется в вектор.
INDEX_CONTEXTUAL = _bool("INDEX_CONTEXTUAL", True)
# Small-to-big: для каждого чанка хранить «родительский» фрагмент (окно соседних чанков).
# Поиск/реранк — по маленькому точному чанку, а в LLM подаётся более крупный контекст.
INDEX_PARENT_CONTEXT = _bool("INDEX_PARENT_CONTEXT", True)
# Сколько соседних чанков с каждой стороны включать в «родительский» фрагмент.
PARENT_WINDOW = _int("PARENT_WINDOW", 1)

# --- Проверка обоснованности ответа (антигаллюцинации) ---
# После генерации LLM сверяет, следует ли ответ из найденного контекста. Если нет —
# ответ помечается и (при verify=strict) перегенерируется со строгой инструкцией.
# Значения: off | warn (пометить) | strict (перегенерировать при необеспеченности).
ANSWER_VERIFY = os.getenv("ANSWER_VERIFY", "off")

# Структурное чанкование: резать документ по заголовкам/абзацам, таблицы и списки
# держать целиком; крупные блоки — по предложениям. Осмысленнее фиксированного размера.
STRUCTURE_CHUNK = _bool("STRUCTURE_CHUNK", True)
# Инлайн-цитаты: просить модель ставить [Фрагмент N] у каждого утверждения в ответе.
INLINE_CITATIONS = _bool("INLINE_CITATIONS", True)
# Защита от промпт-инъекций: добавлять в промпт указание считать контекст ДАННЫМИ, а не
# инструкциями (важно, т.к. в контекст попадает содержимое спарсенных сайтов/документов).
PROMPT_INJECTION_GUARD = _bool("PROMPT_INJECTION_GUARD", True)
# Разрешение контекста диалога: переписывать follow-up-вопрос в самостоятельный
# (с учётом истории) перед поиском. Генерация ответа — по оригиналу и истории.
DIALOG_REWRITE = _bool("DIALOG_REWRITE", True)
VISION_MODEL = os.getenv("VISION_MODEL", "")   # vision-модель (пусто = основная LLM)
# Уменьшать изображение перед отправкой vision-модели (тумблер) и до какого размера.
# Большие фото/сканы раздувают число image-токенов и память vLLM (частая причина
# «Server disconnected»/краша движка). VISION_DOWNSCALE=выкл — слать в исходном размере.
VISION_DOWNSCALE = _bool("VISION_DOWNSCALE", True)
VISION_MAX_SIDE = _int("VISION_MAX_SIDE", 1536)   # макс. сторона (px), если уменьшение включено
VISION_TIMEOUT = _int("VISION_TIMEOUT", 180)   # таймаут запроса к vision-модели, сек
VISION_RETRIES = _int("VISION_RETRIES", 2)     # число попыток описать изображение
# Tesseract PSM (page segmentation mode): 3 — авто; 4 — колонками; 6 — единый блок;
# 11 — разрозненный текст. OEM: 1 — нейросеть LSTM; 3 — авто (LSTM+legacy).
OCR_PSM = _int("OCR_PSM", 3)
OCR_OEM = _int("OCR_OEM", 3)
# Предобработка изображения перед OCR (оттенки серого + автоконтраст; для сканов
# с шумом/неравномерным фоном повышает качество).
OCR_PREPROCESS = _bool("OCR_PREPROCESS", False)

# Телеграм-бот: токен от @BotFather (пусто = бот выключен) и авто-подтверждение
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_AUTO_APPROVE = _bool("TELEGRAM_AUTO_APPROVE", False)
# Прокси для доступа к api.telegram.org (где Telegram заблокирован). Поддерживаются
# socks5://, socks5h://, http://, https:// (можно с user:pass@). ВНИМАНИЕ: MTProto-прокси
# (tg://proxy-ссылки) — для клиентов Telegram и НЕ работают с Bot API; нужен SOCKS5/HTTP.
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "socks5h://10.0.0.2:1080")
# Голосовые сообщения бота: распознавание входящих (Whisper) и ответ голосом (TTS).
TELEGRAM_VOICE_IN = _bool("TELEGRAM_VOICE_IN", True)     # распознавать голосовые запросы
TELEGRAM_VOICE_OUT = _bool("TELEGRAM_VOICE_OUT", False)  # отвечать голосом на голосовые
# Распознавать приложенные к сообщению файлы (документы/фото): извлечь текст и ответить
# на подпись-вопрос по содержимому файла (без добавления в базу).
TELEGRAM_FILES = _bool("TELEGRAM_FILES", True)
# Показывать в ответе бота структуру формирования ответа (этапы конвейера, как в чате).
TELEGRAM_PIPELINE = _bool("TELEGRAM_PIPELINE", True)
# Выводить в ответе бота сам текст ответа LLM (можно отключить, оставив только
# источники и/или структуру формирования ответа).
TELEGRAM_SHOW_ANSWER = _bool("TELEGRAM_SHOW_ANSWER", True)
# Кнопки оценки ответа (👍/👎) и комментария под ответом бота.
TELEGRAM_FEEDBACK = _bool("TELEGRAM_FEEDBACK", True)
# Отправлять визуальные превью источников (картинки/чертежи/кадры видео/аудио),
# как карточки-превью в веб-чате. Кол-во превью на ответ — TELEGRAM_PREVIEW_MAX.
TELEGRAM_PREVIEWS = _bool("TELEGRAM_PREVIEWS", True)
TELEGRAM_PREVIEW_MAX = _int("TELEGRAM_PREVIEW_MAX", 4)
# Движок синтеза речи: auto (пробует доступные) | xtts | piper | say (macOS) | espeak | off.
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")
# Голос/модель: для macOS `say` — имя голоса (напр. Milena/Yuri); для piper — путь к .onnx;
# для espeak — код языка (напр. ru). Пусто = по умолчанию для движка.
TTS_VOICE = os.getenv("TTS_VOICE", "")

# --- Клонирование голоса (Coqui XTTS-v2): «обучение» голосового вывода по образцу ---
# Zero-shot: загружается короткий (6–30 c) чистый образец речи, и синтез идёт этим
# голосом (в т.ч. русский). Пакет `coqui-tts` ставится по кнопке в админке (тяжёлый,
# тянет torch). Модель скачивается автоматически при первом запуске.
XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_SAMPLE = os.getenv("XTTS_SAMPLE", "")        # путь к WAV-образцу голоса (16 кГц моно)
XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "ru")  # язык синтеза (ru/en/…)
XTTS_USE_GPU = _bool("XTTS_USE_GPU", False)       # использовать GPU (CUDA) при синтезе
# Адрес отдельного микросервиса XTTS (свой venv с coqui-tts + transformers>=4.57, чтобы не
# конфликтовать с ядром RAG на transformers==4.44.2). Если задан — синтез идёт по HTTP на этот
# сервис; если пусто — пытается синтезировать in-process (coqui-tts в этом же venv, не рекоменд.).
XTTS_URL = os.getenv("XTTS_URL", "")              # напр. http://127.0.0.1:8020

# --- Парсинг сайтов в базу знаний ---
# Глубина обхода ссылок (0 = только указанная страница), лимит страниц на сайт,
# ходить ли только по тому же домену.
# Фолбэк, когда обычный поиск не нашёл ответа: лексический (полнотекст/имена файлов),
# затем «глубокий» (LLM выбирает файлы по списку имён). По умолчанию выключен.
NO_ANSWER_FALLBACK = _bool("NO_ANSWER_FALLBACK", True)
# Не показывать источники/«дополнительные документы», если ответ — честное
# «В доступных документах нет точного ответа на этот вопрос».
HIDE_SOURCES_IF_NO_ANSWER = _bool("HIDE_SOURCES_IF_NO_ANSWER", True)

WEB_CRAWL_DEPTH = _int("WEB_CRAWL_DEPTH", 14)
WEB_MAX_PAGES = _int("WEB_MAX_PAGES", 200000)
WEB_MAX_FILES = _int("WEB_MAX_FILES", 500000)       # лимит скачиваемых файлов на сайт
WEB_SAME_DOMAIN = _bool("WEB_SAME_DOMAIN", True)
# Рендерить страницы headless-браузером (Playwright Chromium) — для сайтов на
# JavaScript. Если Playwright/браузер не установлены — мягкий откат на обычную загрузку.
WEB_JS_RENDER = _bool("WEB_JS_RENDER", True)
# «Умный» режим браузера: сначала быстрая обычная загрузка (httpx), а headless-браузер
# запускается ТОЛЬКО для страниц, где обычной загрузкой получилось мало текста (реальные
# JS-сайты). Резко ускоряет обход статических/серверных сайтов. Выключите, чтобы
# рендерить браузером КАЖДУЮ страницу (медленно, старое поведение).
WEB_JS_AUTO = _bool("WEB_JS_AUTO", True)
# Условие готовности страницы в браузере: domcontentloaded (быстро, по умолчанию),
# load (дождаться ресурсов) или networkidle (дождаться простоя сети — самый медленный,
# многие сайты с аналитикой/поллингом не «затихают» и упираются в таймаут).
WEB_JS_WAIT = os.getenv("WEB_JS_WAIT", "load")
# Доп. пауза после загрузки JS-страницы (мс) — чтобы догрузился динамический контент.
WEB_JS_WAIT_MS = _int("WEB_JS_WAIT_MS", 0)
# Не загружать в браузере картинки/стили/шрифты/медиа — на текст не влияет, но сильно
# ускоряет рендер JS-страниц (меньше сетевых запросов).
WEB_JS_BLOCK_ASSETS = _bool("WEB_JS_BLOCK_ASSETS", True)
# Таймаут загрузки одной страницы (сек) — и для httpx, и для перехода в браузере.
WEB_PAGE_TIMEOUT = _int("WEB_PAGE_TIMEOUT", 25)
# Пересоздавать контекст браузера каждые N отрендеренных страниц. На длинных обходах
# (тысячи страниц) Chromium накапливает память даже при закрытых вкладках → замедление
# и «зависание». Периодическое пересоздание освобождает память. 0 — не пересоздавать.
WEB_JS_RECYCLE = _int("WEB_JS_RECYCLE", 150)
# Параллелизм парсинга: сколько страниц/файлов качать одновременно (ускоряет обход).
# Обычные загрузки (httpx) идут параллельно всегда; в браузере рендерятся только
# «тяжёлые» JS-страницы — они обрабатываются последовательно (Playwright не потокобезопасен).
WEB_CONCURRENCY = _int("WEB_CONCURRENCY", 6)
# Ежедневный автоматический повторный парсинг сохранённых сайтов в 00:05 (обновляет
# их содержимое в базе знаний). Расписание крутит monitor.py.
WEB_AUTO_REPARSE = _bool("WEB_AUTO_REPARSE", False)
# Брать список URL из sitemap.xml (быстрее и полнее слепого обхода; обходит ловушки
# поиска/пагинации). При отсутствии карты — обычный обход по ссылкам.
WEB_USE_SITEMAP = _bool("WEB_USE_SITEMAP", True)
# Учитывать robots.txt (Disallow) и его Crawl-delay при обходе.
WEB_RESPECT_ROBOTS = _bool("WEB_RESPECT_ROBOTS", True)
# Соблюдать паузу Crawl-delay из robots.txt вообще. False — игнорировать задержку
# (Disallow всё равно учитывается, если включён robots). Быстрый тумблер «задержки».
WEB_RESPECT_CRAWL_DELAY = _bool("WEB_RESPECT_CRAWL_DELAY", False)
# Верхний предел паузы Crawl-delay из robots.txt (сек). Некоторые сайты задают 20+ с,
# из-за чего обход идёт ~1 стр./20 с и кажется зависшим. robots соблюдается, но не
# медленнее этого предела. 0 — не ограничивать (полностью доверять robots).
WEB_CRAWL_DELAY_MAX = _float("WEB_CRAWL_DELAY_MAX", 5.0)
# Сколько САЙТОВ парсить одновременно (каждый — в своём потоке, со своим браузером).
# 1 — последовательно (как раньше). Больше — быстрее на нескольких сайтах, но каждый
# параллельный сайт поднимает свой Chromium (расход памяти).
WEB_SITE_CONCURRENCY = _int("WEB_SITE_CONCURRENCY", 3)
# Инкрементальный парсинг: условными запросами (ETag/Last-Modified) пропускать
# неизменённые страницы и файлы — быстрый ежедневный автопарсинг.
WEB_INCREMENTAL = _bool("WEB_INCREMENTAL", True)

# --- База данных приложения (журнал/история/настройки) ---
# sqlite (по умолчанию, без внешних сервисов) | mysql | postgresql
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = _int("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "rag")
PG_HOST = os.getenv("PG_HOST", "")
PG_PORT = _int("PG_PORT", 5432)
PG_USER = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DB = os.getenv("PG_DB", "rag")

# Источник каталога документов: filesystem (папка DOCS_DIR) | postgresql (таблица
# doc_catalog в активной PostgreSQL). Переключается кнопками в админке.
CATALOG_SOURCE = os.getenv("CATALOG_SOURCE", "filesystem")

# --- Кэш Redis (по умолчанию выключен) ---
REDIS_ENABLED = _bool("REDIS_ENABLED", False)
# Меж-процессный обмен состоянием LLM (список вызовов + очередь) через общую rag_logs.db,
# когда Redis выключен. Позволяет видеть и учитывать vision-вызовы из процесса индексации
# без Redis. По умолчанию включено; выключить: PROC_SHARE_SQLITE=0 (тогда без Redis видны
# только вызовы самого веб-процесса — прежнее поведение).
PROC_SHARE_SQLITE = _bool("PROC_SHARE_SQLITE", True)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
# TTL кэша результатов поиска, сек. Пространство 'index' и так сбрасывается по версии
# при переиндексации, а ключ включает влияющие настройки — поэтому TTL можно держать
# большим (по умолчанию 6 ч) для высокого hit-rate. 0 — практически без кэша поиска.
CACHE_SEARCH_TTL = _int("CACHE_SEARCH_TTL", 21600)
# Кэшировать готовые ответы LLM в Redis (для одинаковых вопросов). По умолчанию выкл.:
# экономит время/нагрузку, но один и тот же вопрос будет получать один ответ до
# переиндексации/смены модели. Требует REDIS_ENABLED.
ANSWER_CACHE = _bool("ANSWER_CACHE", False)
# Семантический кэш ответов: отдавать готовый ответ на ПОХОЖИЙ по смыслу вопрос
# (эмбеддинг + косинус), а не только на точное совпадение. Требует ANSWER_CACHE и Redis.
ANSWER_CACHE_SEMANTIC = _bool("ANSWER_CACHE_SEMANTIC", False)
# Порог косинусной близости для семантического кэша (0..1). Выше — строже (реже попадания,
# меньше риск отдать ответ не на тот вопрос). Рекомендация: 0.92–0.96.
ANSWER_CACHE_SIM = _float("ANSWER_CACHE_SIM", 0.94)

# --- Улучшение запроса перед поиском ---
# off — как есть; rewrite — LLM переформулирует вопрос в чистый поисковый запрос;
# hyde — LLM пишет гипотетический абзац-ответ, его эмбеддинг используется для поиска
# (часто повышает полноту на «размытых» вопросах). Стоит один доп. вызов LLM.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "off")

# --- KAG (Knowledge Augmented Generation) ---
# Движок ответов «знание-усиленной генерации»: сложный вопрос раскладывается на
# под-вопросы (логические шаги), по каждому идёт поиск, результаты объединяются и
# (опц.) дополняются знаниями из графа; финальный ответ генерируется по собранному
# знанию со ссылками. Включается выбором ENGINE=kag.
KAG_DECOMPOSE = _bool("KAG_DECOMPOSE", True)        # раскладывать вопрос на под-вопросы
KAG_MAX_HOPS = _int("KAG_MAX_HOPS", 3)              # макс. число под-вопросов/шагов
KAG_CHUNKS_PER_HOP = _int("KAG_CHUNKS_PER_HOP", 4)  # фрагментов на под-вопрос
KAG_CONTEXT_CHUNKS = _int("KAG_CONTEXT_CHUNKS", 8)  # итоговых фрагментов в контексте
KAG_GRAPH = _bool("KAG_GRAPH", False)              # дополнять знаниями из графа (LightRAG)
KAG_GRAPH_MODE = os.getenv("KAG_GRAPH_MODE", "local")   # режим извлечения знаний из графа
KAG_MUTUAL_INDEX = _bool("KAG_MUTUAL_INDEX", True)  # взаимное индексирование текст⇄знания
KAG_REQUIRE_CITATIONS = _bool("KAG_REQUIRE_CITATIONS", True)  # требовать ссылки на источники
KAG_TEMPERATURE = _float("KAG_TEMPERATURE", 0.1)    # температура финальной генерации

# --- Надёжность: алерты о сбоях ---
# Уведомления в Телеграм и/или на e-mail при падении критичных компонентов
# (векторная база, LLM и т.п.) и при падении фоновых задач (индексация, парсинг).
# Есть троттлинг: повторный алерт по той же проблеме — не чаще ALERT_COOLDOWN секунд;
# при восстановлении приходит отдельное сообщение «снова доступен».
ALERTS_ENABLED = _bool("ALERTS_ENABLED", False)
ALERT_TG_CHAT = os.getenv("ALERT_TG_CHAT", "")        # chat_id для алертов (пусто = не слать в ТГ)
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")            # получатель(и) e-mail через запятую
ALERT_COOLDOWN = _int("ALERT_COOLDOWN", 900)          # пауза между повторами по одной проблеме, с
ALERT_ON_JOB_FAIL = _bool("ALERT_ON_JOB_FAIL", True)  # слать при падении фоновых задач
# SMTP для e-mail-алертов (если ALERT_EMAIL задан)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = _int("SMTP_PORT", 587)
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")                # адрес отправителя (пусто = SMTP_USER)
SMTP_TLS = os.getenv("SMTP_TLS", "starttls")          # starttls | ssl | none

# Доступ
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # пусто = админка без пароля (только LAN!)

EMBED_DIM = 1024  # размерность bge-m3
