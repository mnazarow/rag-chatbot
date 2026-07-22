# RAG-проект — changelog исправлений

**Дата:** 22 июля 2026
**Изменено/добавлено:** 54 файла (51 правка + 3 новых: `admin/__init__.py`, `admin/common.py`, `admin/jobs.py`, `tests/test_imports_smoke.py`).
**Проверка:** весь проект компилируется (`py_compile`); 46 модулей импортируются под заглушками без ошибок; тесты — **21 passed, 0 failed** (5 прежних + 4 новых смок-теста + расширенные).

Правки применены **на месте** в папке проекта. Ваш `.git` не тронут — просмотреть всё можно через `git diff`. Архив с копиями файлов также доставлен в чат. Технический артефакт `_rag_fixes.tar.gz` и устаревший `.git/index.lock` перемещены в папку `_to_delete/` (её можно удалить).

Изменения помечены в коде так: рискованные/требующие ручной проверки места содержат комментарий `# FIXME(review): ...`.

---

## CRITICAL

**C1 — мёртвый API `retriever._client`/`_COLLECTION` (calibrate, tg_train, org_index).**
Все три модуля переведены на фасад `vectorstore`: `upsert([{id,vector,payload}])`, `delete(flt)`, `search(vec,limit,flt)`. Точки получают детерминированные `uuid5`-id; tg-документы — payload `tg=True`/`tg_chat_id`. В `org_structure.sync()` ошибка индексации теперь отражается в статусе синхронизации, а не только в тексте лога. `calibrate._candidates` дополнительно приведён к боевому конвейеру (эмбеддинг после синонимов/`QUERY_REWRITE`, отбрасывание точек без text — **M43/B3**). Добавлен смок-тест, который ловит этот класс дефектов на будущее.

**C2 — закоммиченный токен + открытая по умолчанию админка.**
`runtime_config.json`: `ADMIN_TOKEN` `"secret123"` → `""`. `_check_admin` (app.py) стал fail-closed: при незаданном токене админ-запросы отклоняются (лазейка `ADMIN_ALLOW_NO_TOKEN`, по умолчанию off), сравнение через `hmac.compare_digest`. CORS больше не `"*"` — origin берутся из `CORS_ORIGINS` (по умолчанию пусто → доступ закрыт; `CORS_ALLOW_LOCALHOST` для отладки). Qdrant в `docker-compose.yml` привязан к `127.0.0.1` + healthcheck (**H12**).

**C3 — стриминг LLM не проверял статус ответа.**
`chat_stream`: при HTTP ≥400 читается тело и поднимается исключение; в ollama-ветке проверяется `obj.get("error")`. Ошибка теперь уходит в учёт как `ok=False`, а не как «успешный пустой ответ». Ключи маскируются в логах (`_redact`).

---

## HIGH

**H1** — синхронный горячий путь убран из event loop: `_embed_query`, `standalone_question`, `search`, `no_answer_fallback`, парсинг+чанкинг в `/chat-doc`, `load_file` в `/api/transcribe` обёрнуты в `asyncio.to_thread`; `get_event_loop()` → `get_running_loop()`.
**H2** — синглтоны моделей теперь кэшируются по параметрам (`_embedder_for(model,device)`), добавлен `retriever.reset_models()`; `settings.update` вызывает его при смене `EMBED_MODEL`/`RERANK_MODEL`/`DEVICE`.
**H3** — в последовательном пути `ingest` удаление старых версий перенесено на **после** успешного парсинга (документ больше не исчезает при сбое). `# FIXME(review)` про детерминированные id.
**H4** — кодировки: общий `_read_text_any()` (charset_normalizer → utf-8-sig → cp1251 → latin-1) применён во всех текстовых загрузчиках (.txt/.md/.html/.svg/.json/.url/STEP/IGES/XML). cp1251-файлы больше не теряются.
**H5** — утечка слота LLM-очереди при отмене: `release` гарантируется через `future.add_done_callback`.
**H6** — не-стриминговый `chat()` использует отдельный увеличенный таймаут (`LLM_NONSTREAM_READ_TIMEOUT`, 900с), не жёсткий read-timeout стрима.
**H7** — при сбое Redis/SQLite очередь больше не проваливается в пустой локальный счётчик (обход лимита) — трактует как «слот не выдан», для SQLite ретраит `BEGIN IMMEDIATE`.
**H8** — SIP AudioSocket по умолчанию слушает `127.0.0.1`, проверяет UUID/секрет из настроек, лимит одновременных сессий, allowlist IP.
**H9** — Telegram: `restart()` дожидается завершения прежнего поллера (join с таймаутом) — бот больше не «молча мёртв»; введён пул воркеров (`TELEGRAM_WORKERS`) с сериализацией по `chat_id` и семафором на тяжёлые стадии.
**H10** — SSRF/инъекция в API-хуках: URL-подстановки через `quote()`, JSON — через `json.dumps`, `follow_redirects=False`, allowlist схем http/https.
**H11** — зависимости: `torch>=2.6.0` (CVE-2025-32434), `Pillow>=11.3.0`, `transformers==4.44.2` (пин ядра), `mlx-whisper ; sys_platform=="darwin"`.
**H13** — `reinstall_server.sh`: авто-бэкап перед `rm -rf`, второе подтверждение (`DELETE`), лог.
**H14** — LLM-автокалибровка и `retrieval_autotune` снимают снапшот настроек до цикла и восстанавливают в `finally` (прод больше не остаётся на экспериментальных значениях).

---

## MEDIUM (48 находок — сводно)

**Ядро RAG:** M1 реальный гибрид BM25 (веса `HYBRID_BM25_WEIGHT`=0.0 по умолчанию → поведение не меняется); M2 семантический кэш учитывает фильтры; M3 `answered` считается во всех потоках; M4 сбойный поиск не выполняется дважды; M5 вынесены единые хелперы стриминга (`_visible_sources`, `_answer_chunks`) — устранены ~6 копий; M6 удалён обходной `QdrantClient` → `vectorstore.count()`; M7 экранирование и allow-list полей в фильтре Milvus; M9 verify режет контекст по границам фрагментов + CAVEAT в strict; M10 валидация ролей в истории диалога.

**Данные:** M11 `lo_unlink` больших объектов PG в `catalog_clear`; M12 контроль объёма распаковки для 7z/rar/tar/gz; M13 валидация членов tar (path traversal); M14 уникальный tmp для DWG-конверсии; M15 закрытие `fitz`-документов; M16 SQLite WAL+busy_timeout; M17 транзакции+`executemany` в миграциях; M18 гистограмма латентности через SQL, аналитика ограничена окном; M19 индексы под фактические запросы; M20 threadlocal-кэш соединений с ping; M21 `load_file` пробрасывает ошибки парсеров; M22 согласованный снимок SQLite (`VACUUM INTO`) + white-list путей restore. M31 — `ingest` исключает подпапку `telegram/` из общего обхода (сохранены tg-метки).

**LLM-слой:** M23 пул httpx-клиентов (keep-alive); M24 `dns_override` c uninstall/restore и опциональным allowlist; M25 lock на счётчик API-ключей; M26 пейсинг не удерживает слот во время sleep; M27 LRU-лимит кэша хуков; M28 SSH `RejectPolicy`+known_hosts вместо `AutoAddPolicy`.

**Боты/голос:** M29 callback Telegram проверяет `approved` и владение записью; M30 эхо-дрейн SIP ограничен по времени + выход при hangup; M32 lock вокруг загрузки/синтеза XTTS; M33 общий семафор тяжёлого конвейера (`HEAVY_PIPELINE_LIMIT`); M34 рабочий fallback движков TTS; M35 `xtts_service` ограничивает `sample_path`, токен-заголовок, лимит длины.

**Admin/settings:** M36 `Popen` не попадает в JSON `/api/admin/status`; M37 lock на `stats_map`; M38 обязательный кэш каталога файлов; M39 кэш `component_analytics`; M40 закрытие FD в reinstall; M41 блок-лист приватных адресов (SSRF `ingest_web`); M42 `LLM_API_KEY` помечен `secret` + маскирование секретов по имени. B6 честный обход больших коллекций (без «магических» лимитов); B7 частичный успех не помечается провалом.

**Инфраструктура:** M44 `update.sh` падает при ошибке `pip` (не тихо); M45 precedence-баг `&` в setup.sh + ожидание Ollama через curl; M46 `exit 1` на не-macOS; M47 `reinstall.sh` собирает `.venv.new` атомарно; M48 `kb_eval` через фасад + флаг «данные неполные»; C7 lock в `kb_eval.evaluate`; C9 проверка `git status` перед `git reset --hard`.

**Декомпозиция admin_ops.py:** создан пакет `admin/` (`common.py` — общие хелперы и дедуплицированный `_jlen`; `jobs.py` — централизованный `jobview`/чтение логов, где и живёт фикс M36). `admin_ops.py` ре-экспортирует перенесённые имена — все 82 публичных символа, которые использует `app.py`, на месте (проверено).

---

## LOW (сводно)
Маскировка токена бота в логах; `is_no_answer` по началу ответа; temp-файлы через `uuid`/`mkstemp` (вместо `time.time()`/`mktemp`); валидация месяца в regex дат; `ESCAPE` в LIKE; атомарная запись progress/stats; `_pending_comment`/кэш хуков с TTL; учёт `429 retry_after`; espeak/say через `--`; monitor не спамит без psutil и пишет `reindex_last` при ошибке; alerts проверяет `ALERTS_ENABLED` до сброса состояния; lock на счётчики звонков; `gen_pmi` под `if __name__=="__main__"`; корневой `docker-compose.yml` в CI-проверке; `REDIS_ENABLED` синхронизирован между `.env.example` и конфигом; ретраи httpx только на 5xx/timeout; параллельные под-поиски KAG.

---

## Осознанно НЕ сделано / помечено `# FIXME(review)`

- **M5 (полное слияние `/chat`)** — унифицированы повторы и вынесены хелперы, но 5 веток стриминга не слиты в единый объект-результат: разные наборы полей debug/meta и сигнатуры `log_request`, а требование «сохранить wire-формат байт-в-байт» делает полное слияние опасным без интеграционных тестов. Wire-формат сохранён.
- **M18 (ключевые слова/источники)** — оставлены в Python (не мапятся чисто на портируемый SQL для 3 диалектов), ограничены окном 100k строк.
- **M20** — threadlocal-кэш соединения с ping/reconnect, а не полноценный пул (перестройка вокруг `_LOCK` рискованна).
- **H3 детерминированные id ingest** — оставлен перенос delete + uuid4; переход на uuid5 требует отдельного прохода очистки «хвоста» при сокращении числа чанков.
- **M26 дефолт `LLM_REQUEST_DELAY`** — численно не менялся (смена дефолта затронула бы существующие установки).
- **M39** — добавлен кэш, но ~26 точных `count` не заменены одним `facet` (разная семантика `must_not`/`is_empty` — нужна сверка на живом стенде).
- **M25 счётчик calls** — добавлен lock; атомарный `UPDATE` в БД оставлен как рекомендация.
- **torch>=2.6** — совместимость с sentence-transformers/FlagEmbedding на Apple MPS не проверялась установкой (нет окружения) — проверьте при обновлении.
- **docker healthcheck** — образ qdrant может быть без curl; проверить на реальном образе.
- **Декомпозиция admin_ops** — вынесены инфраструктура и дубли; целые домены (web-парсинг, milvus, catalog) НЕ вынесены (риск разрыва связности выше пользы) — приоритет отдан работоспособности.
- **Дедуп sip_bridge/sip_phone** (общая `DialogSession`) — оставлен как рекомендация: крупный рефактор двух рабочих трактов без стенда рискован.

---

## Новые настройки (читаются с безопасными дефолтами, зарегистрированы в settings)
`CORS_ORIGINS`, `CORS_ALLOW_LOCALHOST`, `ADMIN_ALLOW_NO_TOKEN`, `HYBRID_BM25_WEIGHT`, `HYBRID_CE_WEIGHT`, `LLM_NONSTREAM_READ_TIMEOUT`, `TELEGRAM_WORKERS`, `HEAVY_PIPELINE_LIMIT`, `SIP_AUDIOSOCKET_HOST/UUID/SECRET/ALLOW`, `SIP_MAX_CONCURRENT`, `SIP_ECHO_DRAIN_SEC`, `XTTS_TOKEN/MAX_TEXT`, `INGEST_CONTENT_HASH(_MAX)`, `ARCHIVE_*`. Все имеют дефолт, сохраняющий прежнее поведение (гибрид, семафор и лимиты выключены по умолчанию).

---

## Рекомендации после применения
1. Прогнать `pytest tests/` на своём окружении (здесь pytest недоступен — тесты гонялись через ручной раннер).
2. Задать `ADMIN_TOKEN` (обязателен — админка теперь fail-closed) и при внешнем доступе `CORS_ORIGINS`.
3. Проверить обновление `torch>=2.6` на целевом железе.
4. Просмотреть изменения через `git diff`, обратив внимание на места с `# FIXME(review)`.
5. Удалить папку `_to_delete/` (туда убраны технический архив и устаревший git-lock).
