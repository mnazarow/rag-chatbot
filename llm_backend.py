"""Единый интерфейс к LLM поверх двух бэкендов:

  - ollama  : нативный Ollama (Apple Metal / CPU), эндпоинт /api/chat
  - openai  : OpenAI-совместимый сервер (vLLM на GPU), эндпоинт /v1/chat/completions

Выбор бэкенда и адреса — из рантайм-настроек (settings), правятся из админки.
Остальной код (app.py, compare.py) просто зовёт chat()/chat_stream().
"""
from __future__ import annotations
import atexit
import json
import threading
from typing import AsyncIterator

import httpx

import settings


# --- Переиспользуемые httpx-клиенты (M23: connection pooling / keep-alive) --- #
# Раньше на КАЖДЫЙ вызов LLM создавался новый httpx.Client/AsyncClient (новое TCP+TLS-
# соединение). Теперь держим общий модульный клиент с пулом keep-alive соединений и
# закрываем его при завершении процесса (atexit). Таймаут задаётся ПО-ЗАПРОСНО (в .post/
# .stream), т.к. он разный для стрима/не-стрима/vision.
_HTTP_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=50,
                            keepalive_expiry=30.0)
_sync_client: httpx.Client | None = None
_sync_lock = threading.Lock()
# Async-клиент привязан к event loop, поэтому создаём его ЛЕНИВО в текущем loop и кэшируем
# по id(loop) — иначе клиент, созданный в одном loop, нельзя безопасно использовать в другом.
_async_clients: dict[int, httpx.AsyncClient] = {}
_async_lock = threading.Lock()


def _client() -> httpx.Client:
    """Общий синхронный httpx-клиент (потокобезопасен для запросов), с пулом keep-alive."""
    global _sync_client
    c = _sync_client
    if c is None or c.is_closed:
        with _sync_lock:
            if _sync_client is None or _sync_client.is_closed:
                _sync_client = httpx.Client(limits=_HTTP_LIMITS)
            c = _sync_client
    return c


def _aclient() -> httpx.AsyncClient:
    """Async httpx-клиент для текущего event loop (создаётся лениво, кэшируется по loop)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    c = _async_clients.get(key)
    if c is None or c.is_closed:
        with _async_lock:
            c = _async_clients.get(key)
            if c is None or c.is_closed:
                c = httpx.AsyncClient(limits=_HTTP_LIMITS)
                _async_clients[key] = c
    return c


@atexit.register
def _close_clients() -> None:
    """Закрыть модульные httpx-клиенты при завершении процесса (best-effort)."""
    global _sync_client
    try:
        if _sync_client is not None and not _sync_client.is_closed:
            _sync_client.close()
    except Exception:
        pass
    if _async_clients:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            for c in list(_async_clients.values()):
                try:
                    if not c.is_closed:
                        loop.run_until_complete(c.aclose())
                except Exception:
                    pass
            loop.close()
        except Exception:
            pass
        _async_clients.clear()


def _redact(s: str) -> str:
    """Убрать секреты (API-ключ LLM) из текста ошибки/лога, чтобы не утёк в журнал/дашборд."""
    try:
        s = str(s)
        key = settings.get("LLM_API_KEY")
        if key and str(key) not in ("", "EMPTY") and len(str(key)) >= 6 and str(key) in s:
            s = s.replace(str(key), "***")
    except Exception:
        pass
    return s


def _think() -> bool:
    """Разрешены ли «размышления» гибридных моделей (Qwen3/3.6, DeepSeek-R1 …)."""
    return bool(settings.get("LLM_THINK"))


def _llm_timeout(stream: bool = True):
    """Таймаут httpx для запросов к LLM: без общего лимита (генерация бывает долгой), но с
    таймаутом на ЧТЕНИЕ очередного куска (LLM_READ_TIMEOUT). Зависший движок прервётся и
    освободит слот очереди, а активная генерация таймаут не трогает (сброс на каждом токене).

    stream=False (H6): не-стриминговый ответ приходит одним куском в КОНЦЕ генерации, поэтому
    read-таймаут должен покрывать всю генерацию, а не один токен — берём отдельный увеличенный
    LLM_NONSTREAM_READ_TIMEOUT (из config), иначе длинные ответы рвутся по LLM_READ_TIMEOUT."""
    try:
        read = float(settings.get("LLM_READ_TIMEOUT") or 0) or None
    except Exception:
        read = 180.0
    if not stream:
        try:
            import config
            read = float(getattr(config, "LLM_NONSTREAM_READ_TIMEOUT", 900) or 0) or None
        except Exception:
            read = 900.0
    return httpx.Timeout(read, connect=15.0, write=60.0, pool=15.0)


# Кэш: какие модели Ollama поддерживают «thinking» (чтобы не слать параметр `think`
# несовместимым моделям — иначе Ollama вернёт HTTP 400). TTL небольшой.
_THINK_CAP: dict[str, bool] = {}
_THINK_CAP_TS = 0.0


def _ollama_supports_think(model: str) -> bool:
    """True, если у модели Ollama есть возможность рассуждений (capabilities:thinking).
    Результат кэшируется на 5 минут; при ошибке — консервативно False (не слать `think`)."""
    global _THINK_CAP_TS
    import time as _t
    now = _t.time()
    if now - _THINK_CAP_TS > 300:
        _THINK_CAP.clear()
        _THINK_CAP_TS = now
    if model in _THINK_CAP:
        return _THINK_CAP[model]
    ok = False
    try:
        r = _client().post(f"{settings.get('OLLAMA_URL')}/api/show",
                           json={"model": model}, timeout=4)
        if r.status_code == 200:
            caps = r.json().get("capabilities") or []
            ok = "thinking" in caps
    except Exception:
        ok = False
    _THINK_CAP[model] = ok
    return ok


def _ollama_think_payload(model: str) -> dict:
    """Ключ `think` для payload Ollama — только для моделей, что его поддерживают."""
    return {"think": _think()} if _ollama_supports_think(model) else {}


def _strip_think(text: str) -> str:
    """Убрать блок рассуждений <think>…</think> из готового ответа (если модель
    всё же вставила его в content, несмотря на think=false)."""
    if not text or "<think>" not in text:
        return text
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _emit_len(buf: str, tag: str) -> int:
    """Сколько символов из начала buf можно отдать/отбросить, не разрезав возможный
    неполный тег в конце: len(buf) минус самый длинный суффикс buf, являющийся
    префиксом tag."""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if tag.startswith(buf[-k:]):
            return len(buf) - k
    return len(buf)


class _ThinkFilter:
    """Потоковый фильтр: не пропускает наружу содержимое между <think> и </think>,
    даже если теги пришли в разных чанках. Нужен на случай, когда модель игнорирует
    think=false и всё равно печатает рассуждения прямо в content."""
    def __init__(self):
        self._buf = ""
        self._in = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while True:
            if not self._in:
                i = self._buf.find("<think>")
                if i < 0:
                    cut = _emit_len(self._buf, "<think>")
                    out.append(self._buf[:cut]); self._buf = self._buf[cut:]
                    break
                out.append(self._buf[:i]); self._buf = self._buf[i + len("<think>"):]
                self._in = True
            else:
                j = self._buf.find("</think>")
                if j < 0:
                    cut = _emit_len(self._buf, "</think>")
                    self._buf = self._buf[cut:]          # отбрасываем «мысли»
                    break
                self._buf = self._buf[j + len("</think>"):]
                self._in = False
        return "".join(out)

    def flush(self) -> str:
        """Отдать остаток (только если мы не внутри блока размышлений)."""
        r = "" if self._in else self._buf
        self._buf = ""
        return r


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


# Троттлинг обновлений счётчика токенов на дашборде — ПО ВРЕМЕНИ (а не по модулю длины
# чанка): раньше `nchars % 64 < len(piece)` пропускало обновления неравномерно (зависело от
# размера чанка) и на крупных чанках почти всегда срабатывало. Обновляем не чаще раза в _TOK_UPDATE_S.
_TOK_UPDATE_S = 0.25


class _TokThrottle:
    def __init__(self):
        self._last = 0.0

    def due(self) -> bool:
        import time as _t
        now = _t.time()
        if now - self._last >= _TOK_UPDATE_S:
            self._last = now
            return True
        return False


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
                      label: str = "", hide_think: bool | None = None) -> AsyncIterator[str]:
    """Асинхронно отдаёт токены ответа по мере генерации.
    hide_think: True/None — вырезать <think>…</think> (по умолчанию); False — показать."""
    model = model or settings.get("LLM_MODEL")
    _hide_think = True if hide_think is None else bool(hide_think)
    # очередь к LLM: ждём свободный слот (не блокируя event loop)
    import asyncio
    import llm_queue
    # H5: acquire() выполняется в потоке пула. Если корутину отменят (клиент отвалился)
    # ПОКА мы ждём слот, поток всё равно доведёт acquire() до конца и займёт слот — а release
    # уже не вызовется (finally ниже не выполнится, т.к. _qtok не присвоен) → утечка слота.
    # Поэтому при CancelledError навешиваем callback на future: как только acquire вернёт
    # токен, слот гарантированно освобождается.
    _fut = asyncio.get_event_loop().run_in_executor(None, llm_queue.acquire)
    try:
        _qtok = await _fut
    except asyncio.CancelledError:
        def _release_leaked(f):
            try:
                _tok = f.result()
            except Exception:
                return
            if _tok:
                try:
                    llm_queue.release(_tok)
                except Exception:
                    pass
        _fut.add_done_callback(_release_leaked)
        raise
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
            # vLLM с reasoning-моделью может печатать <think>…</think> прямо в content —
            # фильтруем, если размышления выключены (как в ollama-ветке)
            tf = _ThinkFilter() if _hide_think else None
            thr = _TokThrottle()
            async with _aclient().stream("POST", url, json=payload, headers=headers,
                                         timeout=_llm_timeout()) as r:
                # C3: без проверки статуса стрим с HTTP 4xx/5xx давал тихий ПУСТОЙ ответ
                # с ok=True. Читаем тело ошибки и поднимаем исключение (уйдёт в ok=False).
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "ignore")
                    raise RuntimeError(f"LLM HTTP {r.status_code}: {_redact(body)[:500]}")
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    # некоторые OpenAI-совместимые серверы шлют объект ошибки в потоке
                    if isinstance(obj, dict) and obj.get("error"):
                        raise RuntimeError(f"LLM error: {_redact(str(obj.get('error')))[:500]}")
                    u = obj.get("usage") or {}
                    if u:
                        ptok = int(u.get("prompt_tokens") or ptok)
                        ctok = int(u.get("completion_tokens") or ctok)
                    choices = obj.get("choices") or []
                    delta = (choices[0].get("delta", {}).get("content", "")
                             if choices else "")
                    if delta:
                        piece = tf.feed(delta) if tf else delta
                        if piece:
                            nchars += len(piece)
                            if thr.due():
                                _act_tokens(cid, nchars)
                            yield piece
                if tf:
                    tail = tf.flush()
                    if tail:
                        nchars += len(tail); _act_tokens(cid, nchars); yield tail
        else:  # ollama
            url = f"{settings.get('OLLAMA_URL')}/api/chat"
            # проба поддержки thinking делает синхронный httpx-запрос — уводим её с event loop
            _think_kw = await asyncio.get_event_loop().run_in_executor(
                None, _ollama_think_payload, model)
            payload = {"model": model, "messages": messages,
                       "stream": True, "options": {"temperature": temperature},
                       **_think_kw}
            tf = _ThinkFilter() if _hide_think else None
            thr = _TokThrottle()
            async with _aclient().stream("POST", url, json=payload,
                                         timeout=_llm_timeout()) as r:
                # C3: проверяем HTTP-статус — иначе HTTP 400/500 давал тихий пустой ответ ok=True
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "ignore")
                    raise RuntimeError(f"Ollama HTTP {r.status_code}: {body[:500]}")
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # C3: Ollama сообщает об ошибке полем "error" в NDJSON — поднимаем исключение
                    if isinstance(obj, dict) and obj.get("error"):
                        raise RuntimeError(f"Ollama error: {str(obj.get('error'))[:500]}")
                    msg = obj.get("message", {}) or {}
                    tok = msg.get("content", "")   # «мысли» приходят в message.thinking — их не отдаём
                    if obj.get("done"):     # финальный ответ Ollama несёт счётчики
                        ptok = int(obj.get("prompt_eval_count") or ptok)
                        ctok = int(obj.get("eval_count") or ctok)
                        gen_ms = int((obj.get("eval_duration") or 0) / 1e6)  # нс→мс
                    if tok:
                        piece = tf.feed(tok) if tf else tok   # на случай <think>…</think> в content
                        if piece:
                            nchars += len(piece)
                            if thr.due():
                                _act_tokens(cid, nchars)
                            yield piece
                tail = tf.flush() if tf else ""
                if tail:
                    nchars += len(tail); _act_tokens(cid, nchars); yield tail
    except Exception as e:
        ok = False
        err = _redact(str(e))
        raise
    finally:
        _act_end(cid, ok=ok, chars=nchars, error=err, ptok=ptok, ctok=ctok, gen_ms=gen_ms)
        try:
            llm_queue.release(_qtok)
        except Exception:
            pass


def chat(messages: list[dict], temperature: float = 0.1,
         model: str | None = None, kind: str = "llm", label: str = "",
         hide_think: bool | None = None) -> str:
    """Синхронный полный ответ (для скриптов/сравнения).
    hide_think: True/None — вырезать <think>…</think> (по умолчанию); False — показать."""
    model = model or settings.get("LLM_MODEL")
    _hide_think = True if hide_think is None else bool(hide_think)
    import llm_queue
    _qtok = llm_queue.acquire()
    cid = _act_begin(kind, model, label or _label_from_messages(messages),
                     _full_request(messages))
    try:
        ptok = ctok = gen_ms = 0
        # H6: stream=False — весь ответ приходит одним куском в конце генерации, поэтому
        # используем отдельный увеличенный read-таймаут (не рвём длинный ответ по LLM_READ_TIMEOUT).
        if settings.get("LLM_BACKEND") == "openai":
            r = _client().post(
                f"{settings.get('LLM_BASE_URL')}/chat/completions",
                timeout=_llm_timeout(stream=False),
                headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                json={"model": model, "messages": messages,
                      "stream": False, "temperature": temperature},
            )
            r.raise_for_status()
            j = r.json()
            out = j["choices"][0]["message"]["content"]
            if _hide_think:
                out = _strip_think(out)
            u = j.get("usage") or {}
            ptok, ctok = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        else:
            r = _client().post(
                f"{settings.get('OLLAMA_URL')}/api/chat", timeout=_llm_timeout(stream=False),
                json={"model": model, "messages": messages,
                      "stream": False, "options": {"temperature": temperature},
                      **_ollama_think_payload(model)},
            )
            r.raise_for_status()
            j = r.json()
            _raw = j["message"]["content"]
            out = _strip_think(_raw) if _hide_think else _raw
            ptok, ctok = int(j.get("prompt_eval_count") or 0), int(j.get("eval_count") or 0)
            gen_ms = int((j.get("eval_duration") or 0) / 1e6)
        _act_end(cid, ok=True, chars=len(out or ""), ptok=ptok, ctok=ctok, gen_ms=gen_ms)
        return out
    except Exception as e:
        _act_end(cid, ok=False, error=_redact(str(e)))
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
        from PIL import Image
        if hasattr(image, "save"):                    # PIL.Image
            im = image
        elif isinstance(image, (bytes, bytearray)):
            im = Image.open(io.BytesIO(bytes(image)))
        else:
            im = Image.open(image)
        im = im.convert("RGB")
        # Уменьшаем большие изображения (опция VISION_DOWNSCALE): гигантские фото/сканы дают
        # очень много image-токенов и раздувают память vLLM — частая причина «Server
        # disconnected» (краш воркера). Отключить — VISION_DOWNSCALE=выкл.
        _ds = settings.get("VISION_DOWNSCALE")
        _ds = True if _ds is None else bool(_ds)
        try:
            max_side = int(settings.get("VISION_MAX_SIDE") or 1536)
        except Exception:
            max_side = 1536
        if _ds and max_side > 0 and max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
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
                r = _client().post(
                    f"{settings.get('LLM_BASE_URL')}/chat/completions", timeout=timeout,
                    headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                    json={"model": model, "stream": False, "temperature": 0.2,
                          "messages": [{"role": "user", "content": content}]})
                r.raise_for_status()
                j = r.json()
                out = _strip_think((j["choices"][0]["message"]["content"] or "")).strip()
                u = j.get("usage") or {}
                ptok, ctok = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            else:
                r = _client().post(
                    f"{settings.get('OLLAMA_URL')}/api/chat", timeout=timeout,
                    json={"model": model, "stream": False, "options": {"temperature": 0.2},
                          "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                          **_ollama_think_payload(model)})
                r.raise_for_status()
                j = r.json()
                out = _strip_think((j.get("message", {}).get("content", "") or "")).strip()
                ptok, ctok = int(j.get("prompt_eval_count") or 0), int(j.get("eval_count") or 0)
                gen_ms = int((j.get("eval_duration") or 0) / 1e6)
            _act_end(cid, ok=True, chars=len(out), ptok=ptok, ctok=ctok, gen_ms=gen_ms)
            return out
        except Exception as e:
            last_err = e
            _act_end(cid, ok=False, error=_redact(str(e)))
            # low: не повторять на 4xx (кроме 429) — это ошибка запроса (плохая модель/картинка/
            # роль), повтор не поможет и только жжёт время/слот очереди. 429 (rate limit) и 5xx
            # (временный сбой движка) — повторяем.
            _status = getattr(getattr(e, "response", None), "status_code", None)
            _no_retry = isinstance(_status, int) and 400 <= _status < 500 and _status != 429
            if attempt < attempts and not _no_retry:
                print(f"[vision] попытка {attempt}/{attempts} не удалась (model={model}): "
                      f"{_redact(str(e))} — повтор")
                continue
            if _no_retry:
                print(f"[vision] {_status} — запрос отклонён моделью, без повторов (model={model})")
                break
        finally:
            try:
                llm_queue.release(_qtok)
            except Exception:
                pass
    print(f"[vision] описание изображения не удалось (model={model}, "
          f"попыток {attempts}, таймаут {timeout:.0f}с): {_redact(str(last_err))}")
    return ""
