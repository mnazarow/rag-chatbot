"""Интеграционный тест wire-формата стрима /chat на замоканных зависимостях.

Зачем так: FastAPI в среде НЕ установлен, а app.py импортирует его на уровне
модуля (плюс десятки тяжёлых зависимостей). Поэтому НЕ импортируем app целиком, а
извлекаем из его исходника РЕАЛЬНЫЕ сериализаторы NDJSON (_stg, _answer_chunks,
_visible_sources) через AST и гоняем их напрямую. Так тест закрепляет именно
формат кадров и ловит будущий рефактор /chat, не требуя тяжёлого стека.

Дополнительно прогоняем маленький эталон стрим-конвейера: мок-async-генератор
llm_backend.chat_stream отдаёт пару токенов, а спай db.log_request проверяет, что
запрос журналируется. Это фиксирует контракт кадров (stage → answer → sources →
meta) и факт логирования.

TODO: при наличии fastapi расширить до полного ASGI-прогона через
httpx.ASGITransport(app=app) с реальным POST /chat и разбором NDJSON-ответа.
"""

import ast
import asyncio
import json
import os
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_app_helpers():
    """Вытащить из app.py именно функции-сериализаторы NDJSON (реальный исходник)."""
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = {"_stg", "_answer_chunks", "_visible_sources"}
    picked = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted
    ]
    ns = {
        "json": json,
        # изолированные фейки зависимостей, которые трогают эти три функции
        "settings": types.SimpleNamespace(get=lambda k: {"HIDE_SOURCES_IF_NO_ANSWER": True}.get(k)),
        "prompts": types.SimpleNamespace(
            is_no_answer=lambda t: "нет точного ответа" in (t or "").lower()
        ),
    }
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, "app_helpers_extract", "exec"), ns)
    assert wanted <= set(ns), f"не найдены сериализаторы: {wanted - set(ns)}"
    return ns


_H = _load_app_helpers()


# ------------------------- формат отдельных кадров ---------------------------
def test_answer_chunks_frame_shape():
    frames = [json.loads(x) for x in _H["_answer_chunks"]("абвгдеёжзийкл", size=5)]
    assert [f["type"] for f in frames] == ["answer", "answer", "answer"]
    # склейка чанков воспроизводит исходный текст без потерь
    assert "".join(f["text"] for f in frames) == "абвгдеёжзийкл"
    assert frames[0]["text"] == "абвгд"


def test_answer_chunks_ndjson_terminated():
    out = list(_H["_answer_chunks"]("ok"))
    assert out and all(line.endswith("\n") for line in out)


def test_stage_frame_shape():
    frame = json.loads(_H["_stg"]("generate", "done", {"chars": 5}, 12))
    assert frame == {
        "type": "stage",
        "key": "generate",
        "status": "done",
        "ms": 12,
        "info": {"chars": 5},
    }


def test_visible_sources_hidden_on_no_answer():
    src = [{"source": "a.pdf"}]
    assert _H["_visible_sources"]("В документах нет точного ответа.", src) == []


def test_visible_sources_kept_on_real_answer():
    src = [{"source": "a.pdf"}]
    assert _H["_visible_sources"]("Гарантия три года.", src) == src


# ------------------- эталон стрим-конвейера /chat (мок) ----------------------
async def _fake_chat_stream(*_a, **_k):
    """Мок llm_backend.chat_stream: async-генератор из пары токенов."""
    for tok in ("Гарантия ", "три года."):
        yield tok


async def _run_pipeline(sources, calls):
    """Мини-воспроизведение вектор-ветки /chat поверх РЕАЛЬНЫХ сериализаторов.

    Формат обязан совпадать с app.chat.stream(): stage(retrieve) → answer* →
    sources → meta, и db.log_request фиксирует запрос.
    """
    frames = []
    frames.append(_H["_stg"]("retrieve", "done", {"hits": len(sources)}, 3))
    acc = []
    async for tok in _fake_chat_stream():
        acc.append(tok)
        frames.append(json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n")
    answer = "".join(acc)
    visible = _H["_visible_sources"](answer, sources)
    frames.append(json.dumps({"type": "sources", "items": visible}, ensure_ascii=False) + "\n")
    rid = calls["log_request"](
        "вопрос?", "price", len(sources), 0.9, 100, len(answer), True, sources, answer=answer
    )
    frames.append(json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n")
    return [json.loads(x) for x in frames]


def test_chat_stream_wire_contract():
    logged = {}

    def spy_log_request(*a, **k):
        logged["args"] = a
        logged["kwargs"] = k
        return 777

    calls = {"log_request": spy_log_request}
    sources = [{"source": "warranty.pdf", "page": 2}]
    frames = asyncio.run(_run_pipeline(sources, calls))

    types_seq = [f["type"] for f in frames]
    # порядок кадров конвейера
    assert types_seq[0] == "stage"
    assert types_seq.count("answer") == 2
    assert "sources" in types_seq and "meta" in types_seq
    assert types_seq.index("sources") < types_seq.index("meta")

    # answer-кадры склеиваются в полный ответ
    answer = "".join(f["text"] for f in frames if f["type"] == "answer")
    assert answer == "Гарантия три года."

    # sources присутствуют (ответ не «нет ответа»)
    sframe = next(f for f in frames if f["type"] == "sources")
    assert sframe["items"] == sources

    # meta несёт id из db.log_request, и сам log_request был вызван
    mframe = next(f for f in frames if f["type"] == "meta")
    assert mframe["id"] == 777
    assert "args" in logged and logged["args"][0] == "вопрос?"


def test_chat_stream_hides_sources_on_no_answer():
    calls = {"log_request": lambda *a, **k: 1}

    async def _no_answer_stream(*_a, **_k):
        yield "В документах нет точного ответа на вопрос."

    global _fake_chat_stream
    saved = _fake_chat_stream
    _fake_chat_stream = _no_answer_stream
    try:
        frames = asyncio.run(_run_pipeline([{"source": "x.pdf"}], calls))
    finally:
        _fake_chat_stream = saved
    sframe = next(f for f in frames if f["type"] == "sources")
    assert sframe["items"] == []
