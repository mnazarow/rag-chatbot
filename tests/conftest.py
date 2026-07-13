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
# httpx импортируется на уровне модулей, но в тестах чистых функций не вызывается
_stub("httpx")
# synonyms тянет БД (sqlite) — для тестов сборки промпта подсказки синонимов не нужны
_stub("synonyms", hint=lambda q: "")
