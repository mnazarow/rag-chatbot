"""Общие настройки тестов.

Чтобы юнит-тесты гоняли ЧИСТЫЕ функции без установки тяжёлых зависимостей (torch,
sentence-transformers, redis, playwright и т.п.), подкладываем лёгкие заглушки в
sys.modules ДО импорта тестируемых модулей. Реальные пакеты, если установлены,
используются как есть (setdefault не перезатирает).
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# dotenv нужен config.py только для load_dotenv — заглушки достаточно
_stub("dotenv", load_dotenv=lambda *a, **k: None)
# httpx импортируется на уровне модулей, но в тестах чистых функций не вызывается.
# Некоторые модули создают пулы соединений на уровне импорта (httpx.Limits/Client),
# поэтому заглушка должна отдавать безобидные объекты для этих имён.
_stub(
    "httpx",
    Limits=lambda *a, **k: object(),
    Timeout=lambda *a, **k: object(),
    Client=lambda *a, **k: object(),
    AsyncClient=lambda *a, **k: object(),
    HTTPError=type("HTTPError", (Exception,), {}),
    TimeoutException=type("TimeoutException", (Exception,), {}),
    HTTPStatusError=type("HTTPStatusError", (Exception,), {}),
    TransportError=type("TransportError", (Exception,), {}),
    RequestError=type("RequestError", (Exception,), {}),
)
# synonyms тянет БД (sqlite) — для тестов сборки промпта подсказки синонимов не нужны
_stub("synonyms", hint=lambda q: "")

# --- Лёгкие заглушки тяжёлых ML/utility-зависимостей ---------------------------
# Нужны, чтобы импортировать чистые функции из retriever/ingest/calibrate без
# установки моделей. setdefault-семантика (_stub не перезатирает уже импортированные
# реальные пакеты), поэтому если numpy/пр. установлены — используется настоящий.
class _DummyModel:  # общий безобидный синглтон для моделей
    def __init__(self, *a, **k):
        pass

    def encode(self, *a, **k):
        return []

    def compute_score(self, *a, **k):
        return []


_stub("sentence_transformers", SentenceTransformer=_DummyModel)
_stub("FlagEmbedding", FlagReranker=_DummyModel)


class _DummyBM25:
    def __init__(self, *a, **k):
        pass

    def get_scores(self, *a, **k):
        return []


_stub("rank_bm25", BM25Okapi=_DummyBM25)
# tqdm: from tqdm import tqdm — прогресс-бар как прозрачная обёртка над итерируемым
_stub("tqdm", tqdm=lambda it=None, *a, **k: (it if it is not None else []))
