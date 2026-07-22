"""Смок-тесты межмодульных контрактов.

Ловит класс дефектов «рефакторинг переехал, потребители остались»: именно так
три модуля (calibrate/tg_train/org_index) продолжали звать несуществующие
retriever._client/_COLLECTION после переноса работы с БД в фасад vectorstore.
Эти тесты дешёвые и не требуют тяжёлых зависимостей.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Модули, которые обязаны обращаться к векторной БД только через фасад vectorstore.
_FACADE_CONSUMERS = [
    "calibrate.py", "tg_train.py", "org_index.py",
    "retriever.py", "ingest.py", "kb_eval.py", "admin_ops.py",
]


def _attr_chain(node):
    """`retriever._client` → ('retriever', '_client'); иначе None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return (node.value.id, node.attr)
    return None


def test_no_dead_retriever_client_refs():
    """Ни один модуль не должен трогать retriever._client / retriever._COLLECTION —
    этих атрибутов не существует (работа с БД вынесена в фасад vectorstore)."""
    offenders = []
    for fn in _FACADE_CONSUMERS:
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=fn)
        for node in ast.walk(tree):
            chain = _attr_chain(node)
            if chain and chain[0] == "retriever" and chain[1] in ("_client", "_COLLECTION"):
                offenders.append(f"{fn}:{node.lineno} retriever.{chain[1]}")
    assert not offenders, "Обращения к несуществующему API retriever: " + "; ".join(offenders)


def test_vectorstore_facade_surface():
    """Публичный контракт фасада, на который переведены потребители, на месте."""
    import vectorstore
    for name in ("search", "upsert", "delete", "scroll", "count", "facet", "list_values"):
        assert hasattr(vectorstore, name), f"vectorstore.{name} отсутствует"


def test_retriever_reset_models_exists():
    """settings.update вызывает retriever.reset_models() при смене модели — он должен быть."""
    src = open(os.path.join(ROOT, "retriever.py"), encoding="utf-8").read()
    assert "def reset_models(" in src


def test_all_modules_parse():
    """Все .py в корне синтаксически корректны (быстрый предохранитель до импорта)."""
    bad = []
    for fn in os.listdir(ROOT):
        if fn.endswith(".py"):
            try:
                ast.parse(open(os.path.join(ROOT, fn), encoding="utf-8").read(), filename=fn)
            except SyntaxError as e:
                bad.append(f"{fn}: {e}")
    assert not bad, "Синтаксические ошибки: " + "; ".join(bad)
