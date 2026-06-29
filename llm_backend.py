"""Единый интерфейс к LLM поверх двух бэкендов:

  - ollama  : нативный Ollama (Apple Metal / CPU), эндпоинт /api/chat
  - openai  : OpenAI-совместимый сервер (vLLM на GPU), эндпоинт /v1/chat/completions

Выбор бэкенда и адреса — из рантайм-настроек (settings), правятся из админки.
Остальной код (app.py, compare.py) просто зовёт chat()/chat_stream().
"""
from __future__ import annotations
import json
from typing import AsyncIterator

import httpx

import settings


async def chat_stream(messages: list[dict], temperature: float = 0.1,
                      model: str | None = None) -> AsyncIterator[str]:
    """Асинхронно отдаёт токены ответа по мере генерации."""
    model = model or settings.get("LLM_MODEL")
    if settings.get("LLM_BACKEND") == "openai":
        url = f"{settings.get('LLM_BASE_URL')}/chat/completions"
        payload = {"model": model, "messages": messages,
                   "stream": True, "temperature": temperature}
        headers = {"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"}
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, json=payload, headers=headers) as r:
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
    else:  # ollama
        url = f"{settings.get('OLLAMA_URL')}/api/chat"
        payload = {"model": model, "messages": messages,
                   "stream": True, "options": {"temperature": temperature}}
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, json=payload) as r:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    tok = json.loads(line).get("message", {}).get("content", "")
                    if tok:
                        yield tok


def chat(messages: list[dict], temperature: float = 0.1,
         model: str | None = None) -> str:
    """Синхронный полный ответ (для скриптов/сравнения)."""
    model = model or settings.get("LLM_MODEL")
    if settings.get("LLM_BACKEND") == "openai":
        r = httpx.post(
            f"{settings.get('LLM_BASE_URL')}/chat/completions", timeout=None,
            headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
            json={"model": model, "messages": messages,
                  "stream": False, "temperature": temperature},
        )
        return r.json()["choices"][0]["message"]["content"]
    r = httpx.post(
        f"{settings.get('OLLAMA_URL')}/api/chat", timeout=None,
        json={"model": model, "messages": messages,
              "stream": False, "options": {"temperature": temperature}},
    )
    return r.json()["message"]["content"]


_DEFAULT_VISION_PROMPT = (
    "Опиши, что изображено, подробно и по-деловому: текст, таблицы, схемы, графики, "
    "объекты, назначение. Если это документ, прайс-лист, чертёж или диаграмма — передай "
    "ключевую информацию и числа. Ответь по-русски, без вступлений.")


def describe_image(image, prompt: str | None = None, model: str | None = None) -> str:
    """Описать изображение визуальной (vision) моделью. `image` — путь/Path,
    bytes или PIL.Image. Возвращает текст описания ('' при недоступности).
    Бэкенд openai (vLLM) — content с image_url; ollama — поле images:[base64]."""
    import base64
    import io
    try:
        if hasattr(image, "save"):                    # PIL.Image
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
        elif isinstance(image, (bytes, bytearray)):
            data = bytes(image)
        else:
            with open(image, "rb") as f:
                data = f.read()
    except Exception as e:
        print(f"[vision] чтение изображения: {e}")
        return ""
    b64 = base64.b64encode(data).decode("ascii")
    model = (model or settings.get("VISION_MODEL") or settings.get("LLM_MODEL"))
    prompt = prompt or _DEFAULT_VISION_PROMPT
    try:
        if settings.get("LLM_BACKEND") == "openai":
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}}]
            r = httpx.post(
                f"{settings.get('LLM_BASE_URL')}/chat/completions", timeout=180,
                headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                json={"model": model, "stream": False, "temperature": 0.2,
                      "messages": [{"role": "user", "content": content}]})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        r = httpx.post(
            f"{settings.get('OLLAMA_URL')}/api/chat", timeout=180,
            json={"model": model, "stream": False, "options": {"temperature": 0.2},
                  "messages": [{"role": "user", "content": prompt, "images": [b64]}]})
        r.raise_for_status()
        return (r.json().get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        print(f"[vision] описание изображения не удалось (model={model}): {e}")
        return ""
