# Тесты

Быстрые, детерминированные юнит-тесты чистых функций RAG-чатбота. Сеть и тяжёлые
модели (torch, sentence-transformers, qdrant/milvus, FlagEmbedding, ...) **не
требуются**: `tests/conftest.py` подкладывает лёгкие заглушки в `sys.modules`
**до** импорта тестируемых модулей.

## Запуск

С установленным pytest:

```bash
python -m pip install "pytest>=8,<9" PyYAML
python -m pytest -q
```

Без установленного pytest (офлайн) — ручной раннер (conftest применяется первым):

```bash
python3 - <<'PY'
import importlib.util, os, sys, traceback
ROOT = os.path.dirname(os.path.abspath("tests")); sys.path.insert(0, ".")
s = importlib.util.spec_from_file_location("conftest", "tests/conftest.py")
c = importlib.util.module_from_spec(s); s.loader.exec_module(c)
p = f = 0
for fn in sorted(os.listdir("tests")):
    if fn.startswith("test_") and fn.endswith(".py"):
        sp = importlib.util.spec_from_file_location(fn[:-3], "tests/" + fn)
        m = importlib.util.module_from_spec(sp)
        try:
            sp.loader.exec_module(m)
        except Exception as e:
            f += 1; print("IMPORT FAIL", fn, repr(e)); traceback.print_exc(); continue
        for n in dir(m):
            if n.startswith("test_") and callable(getattr(m, n)):
                try:
                    getattr(m, n)(); p += 1
                except Exception as e:
                    f += 1; print("FAIL", fn, n, repr(e))
print(p, "passed", f, "failed")
PY
```

## Что покрыто

| Файл | Модуль | Что проверяется |
|------|--------|-----------------|
| `test_chunking.py` | `ingest` | `chunk_text` / `_chunk_fixed` / `_chunk_structured`: границы, overlap, пустой вход, юникод, срез по заголовкам |
| `test_loaders.py` | `loaders` | `_read_text_any`: utf-8 / utf-8-sig / cp1251-фолбэк, устойчивость к мусорным байтам |
| `test_metadata.py` | `metadata` | регэкспы дат (валидные/невалидные месяцы, версии `v2023.2`), категории |
| `test_retriever.py` | `retriever` | `_tokenize`, `infer_category`, `_build_filter`, `_apply_hybrid` (min-max нормировка) |
| `test_calibrate.py` | `calibrate` | `_select`, `_to_bool`, `_coerce_opt` |
| `test_vectorstore.py` | `vectorstore` | `_m_expr`: экранирование строк и allow-list имён полей |
| `test_query_filters.py` | `query_filters`, `synonyms` | разбор фильтров из LLM-ответа, расширение синонимами |
| `test_db.py` | `db` | SQLite: `kv_get`/`kv_set`, `init()`, `log_request`→`recent`, идемпотентность миграций |
| `test_chat_integration.py` | `app` | wire-формат NDJSON-стрима `/chat` (кадры `stage`/`answer`/`sources`/`meta`, вызов `db.log_request`) |
| `test_prompts.py`, `test_settings.py`, `test_think_filter.py`, `test_imports_smoke.py` | — | ранее существовавшие |

### Замечания по подходу

- **`test_chat_integration.py`** не импортирует `app.py` целиком (FastAPI и десятки
  тяжёлых зависимостей). Реальные сериализаторы NDJSON (`_stg`, `_answer_chunks`,
  `_visible_sources`) извлекаются из исходника через AST — тест ломается, если
  рефактор `/chat` изменит формат кадров. **TODO:** при наличии `fastapi` расширить
  до полного ASGI-прогона через `httpx.ASGITransport(app=app)`.
- **`synonyms`** в `conftest.py` подменён облегчённой заглушкой; настоящий модуль
  тесты загружают отдельной копией из файла с фейковым `db`.

## Lock-файл зависимостей

Полный резолв офлайн невозможен (нет доступа к PyPI и к torch/CUDA-колёсам),
поэтому в репозитории лежит только `constraints.txt` — пиннинг критичных
транзитивных пакетов (совместимых с ядром `transformers==4.44.2`). Использование:

```bash
pip install -r requirements.txt -c constraints.txt
```

Полный воспроизводимый lock генерируйте на машине **с доступом к индексам**:

```bash
# вариант 1: uv (быстро)
uv pip compile requirements.txt -o requirements.lock

# вариант 2: pip-tools
pip install pip-tools
pip-compile --generate-hashes -o requirements.lock requirements.txt
```

Для GPU-окружения соберите отдельный lock из `requirements-gpu.txt` (torch ставится
под конкретную CUDA-сборку).
