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


def _label_from_messages(messages: list[dict]) -> str:
    """Короткая подпись запроса — последнее сообщение пользователя (без контекста)."""
    try:
        for m in reversed(messages or []):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):       # мультимодальное содержимое
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                c = (c or "").strip().replace("\n", " ")
                return c[:160]
    except Exception:
        pass
    return ""


def _full_request(messages: list[dict]) -> str:
    """Полный текст запроса к LLM (роли + содержимое) — для раскрытия строки на дашборде."""
    parts = []
    try:
        for m in messages or []:
            role = m.get("role", "")
            c = m.get("content")
            if isinstance(c, list):          # мультимодальное содержимое
                c = " ".join(p.get("text", "[изображение]") for p in c
                             if isinstance(p, dict))
            parts.append(f"[{role}]\n{c}")
    except Exception:
        return ""
    return "\n\n".join(parts)


def _act_begin(kind: str, model: str, label: str = "", prompt: str = ""):
    try:
        import llm_activity
        return llm_activity.begin(kind, model, settings.get("LLM_BACKEND"), label, prompt)
    except Exception:
        return None


def _act_tokens(cid, chars: int):
    if cid is None:
        return
    try:
        import llm_activity
        llm_activity.tokens(cid, chars)
    except Exception:
        pass


def _act_end(cid, ok: bool, chars: int = 0, error: str | None = None,
             ptok: int = 0, ctok: int = 0, gen_ms: int = 0):
    if cid is None:
        return
    try:
        import llm_activity
        llm_activity.end(cid, ok=ok, chars=chars, error=error,
                         ptok=ptok, ctok=ctok, gen_ms=gen_ms)
    except Exception:
        pass


async def chat_stream(messages: list[dict], temperature: float = 0.1,
                      model: str | None = None, kind: str = "chat",
                      label: str = "") -> AsyncIterator[str]:
    """Асинхронно отдаёт токены ответа по мере генерации."""
    model = model or settings.get("LLM_MODEL")
    # очередь к LLM: ждём свободный слот (не блокируя event loop)
    import asyncio
    import llm_queue
    _qtok = await asyncio.get_event_loop().run_in_executor(None, llm_queue.acquire)
    cid = _act_begin(kind, model, label or _label_from_messages(messages),
                     _full_request(messages))
    nchars = 0
    ptok = ctok = gen_ms = 0
    ok = True
    err = None
    try:
        if settings.get("LLM_BACKEND") == "openai":
            url = f"{settings.get('LLM_BASE_URL')}/chat/completions"
            # include_usage — чтобы сервер вернул счётчики токенов в финальном чанке
            payload = {"model": model, "messages": messages,
                       "stream": True, "temperature": temperature,
                       "stream_options": {"include_usage": True}}
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
                        obj = json.loads(data)
                        u = obj.get("usage") or {}
                        if u:
                            ptok = int(u.get("prompt_tokens") or ptok)
                            ctok = int(u.get("completion_tokens") or ctok)
                        choices = obj.get("choices") or []
                        delta = (choices[0].get("delta", {}).get("content", "")
                                 if choices else "")
                        if delta:
                            nchars += len(delta)
                            if nchars % 64 < len(delta):
                                _act_tokens(cid, nchars)
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
                        obj = json.loads(line)
                        tok = obj.get("message", {}).get("content", "")
                        if obj.get("done"):     # финальный ответ Ollama несёт счётчики
                            ptok = int(obj.get("prompt_eval_count") or ptok)
                            ctok = int(obj.get("eval_count") or ctok)
                            gen_ms = int((obj.get("eval_duration") or 0) / 1e6)  # нс→мс
                        if tok:
                            nchars += len(tok)
                            if nchars % 64 < len(tok):
                                _act_tokens(cid, nchars)
                            yield tok
    except Exception as e:
        ok = False
        err = str(e)
        raise
    finally:
        _act_end(cid, ok=ok, chars=nchars, error=err, ptok=ptok, ctok=ctok, gen_ms=gen_ms)
        try:
            llm_queue.release(_qtok)
        except Exception:
            pass


def chat(messages: list[dict], temperature: float = 0.1,
         model: str | None = None, kind: str = "llm", label: str = "") -> str:
    """Синхронный полный ответ (для скриптов/сравнения)."""
    model = model or settings.get("LLM_MODEL")
    import llm_queue
    _qtok = llm_queue.acquire()
    cid = _act_begin(kind, model, label or _label_from_messages(messages),
                     _full_request(messages))
    try:
        ptok = ctok = gen_ms = 0
        if settings.get("LLM_BACKEND") == "openai":
            r = httpx.post(
                f"{settings.get('LLM_BASE_URL')}/chat/completions", timeout=None,
                headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                json={"model": model, "messages": messages,
                      "stream": False, "temperature": temperature},
            )
            j = r.json()
            out = j["choices"][0]["message"]["content"]
            u = j.get("usage") or {}
            ptok, ctok = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        else:
            r = httpx.post(
                f"{settings.get('OLLAMA_URL')}/api/chat", timeout=None,
                json={"model": model, "messages": messages,
                      "stream": False, "options": {"temperature": temperature}},
            )
            j = r.json()
            out = j["message"]["content"]
            ptok, ctok = int(j.get("prompt_eval_count") or 0), int(j.get("eval_count") or 0)
            gen_ms = int((j.get("eval_duration") or 0) / 1e6)
        _act_end(cid, ok=True, chars=len(out or ""), ptok=ptok, ctok=ctok, gen_ms=gen_ms)
        return out
    except Exception as e:
        _act_end(cid, ok=False, error=str(e))
        raise
    finally:
        try:
            llm_queue.release(_qtok)
        except Exception:
            pass


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
    # таймаут и число попыток — из настроек (большие vision-модели бывают медленными)
    try:
        timeout = float(settings.get("VISION_TIMEOUT") or 180)
    except Exception:
        timeout = 180.0
    try:
        attempts = max(1, int(settings.get("VISION_RETRIES") or 1))
    except Exception:
        attempts = 1

    import llm_queue
    last_err = None
    for attempt in range(1, attempts + 1):
        _qtok = llm_queue.acquire()
        cid = _act_begin("vision", model,
                         "описание изображения" + (f" (попытка {attempt})" if attempt > 1 else ""),
                         prompt=prompt + "\n\n[изображение прикреплено]")
        try:
            ptok = ctok = gen_ms = 0
            if settings.get("LLM_BACKEND") == "openai":
                content = [{"type": "text", "text": prompt},
                           {"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + b64}}]
                r = httpx.post(
                    f"{settings.get('LLM_BASE_URL')}/chat/completions", timeout=timeout,
                    headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                    json={"model": model, "stream": False, "temperature": 0.2,
                          "messages": [{"role": "user", "content": content}]})
                r.raise_for_status()
                j = r.json()
                out = (j["choices"][0]["message"]["content"] or "").strip()
                u = j.get("usage") or {}
                ptok, ctok = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            else:
                r = httpx.post(
                    f"{settings.get('OLLAMA_URL')}/api/chat", timeout=timeout,
                    json={"model": model, "stream": False, "options": {"temperature": 0.2},
                          "messages": [{"role": "user", "content": prompt, "images": [b64]}]})
                r.raise_for_status()
                j = r.json()
                out = (j.get("message", {}).get("content", "") or "").strip()
                ptok, ctok = int(j.get("prompt_eval_count") or 0), int(j.get("eval_count") or 0)
                gen_ms = int((j.get("eval_duration") or 0) / 1e6)
            _act_end(cid, ok=True, chars=len(out), ptok=ptok, ctok=ctok, gen_ms=gen_ms)
            return out
        except Exception as e:
            last_err = e
            _act_end(cid, ok=False, error=str(e))
            if attempt < attempts:
                print(f"[vision] попытка {attempt}/{attempts} не удалась (model={model}): "
                      f"{e} — повтор")
                continue
        finally:
            try:
                llm_queue.release(_qtok)
            except Exception:
                pass
    print(f"[vision] описание изображения не удалось (model={model}, "
          f"попыток {attempts}, таймаут {timeout:.0f}с): {last_err}")
    return ""
