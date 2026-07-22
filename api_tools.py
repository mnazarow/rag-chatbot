"""Внешние API-хуки: для определённых типов вопросов дёргать сторонний REST-сервис
и подмешивать его ответ в контекст LLM.

Каждый хук (настраивается в админке) описывает: когда срабатывать (триггер —
ключевые слова / регэксп / ИИ-интент), как извлечь параметры из вопроса, как
вызвать API (метод, URL-шаблон с {param}, заголовки, тело) и как взять текст из
ответа (resp_path по JSON). Результат добавляется в контекст ответа как фрагмент с
источником-меткой (модель формулирует ответ по данным и правилам, со ссылкой).

Безопасность: хуки задаёт только администратор; исходящие запросы идут с учётом
статических DNS-записей; есть таймаут и короткий кэш по (хук, параметры).
"""
from __future__ import annotations
import atexit
import hashlib
import json
import re
import threading
import time
import urllib.parse

import db

_cache: dict = {}
_lock = threading.Lock()
_TTL = 20.0          # сек кэш ответа API
_MAX_TEXT = 4000     # обрезка ответа API
_CACHE_MAX = 500     # M27: верхняя граница числа записей кэша (иначе неограниченный рост)

# Разрешённые схемы URL для исходящих API-хуков (H10): file://, gopher:// и пр. запрещены,
# чтобы админский шаблон/подстановка параметров не привели к чтению локальных файлов/SSRF.
_ALLOWED_SCHEMES = {"http", "https"}

# M23: переиспользуемый httpx-клиент с пулом keep-alive вместо нового клиента на каждый хук.
# follow_redirects=False (H10) — редирект мог увести запрос на другой (внутренний) хост в обход
# намерения; ответы 3xx возвращаются как есть.
_http_lock = threading.Lock()
_http_client = None


def _get_client():
    global _http_client
    import httpx
    if _http_client is None or _http_client.is_closed:
        with _http_lock:
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.Client(
                    follow_redirects=False,
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20,
                                        keepalive_expiry=30.0))
    return _http_client


@atexit.register
def _close_client():
    global _http_client
    try:
        if _http_client is not None and not _http_client.is_closed:
            _http_client.close()
    except Exception:
        pass


def _cache_put(key: str, now: float, text: str) -> None:
    """Положить ответ в кэш, выселив истёкшие и ограничив размер (M27)."""
    with _lock:
        # снять истёкшие
        for k in [k for k, v in _cache.items() if now - v[0] >= _TTL]:
            _cache.pop(k, None)
        _cache[key] = (now, text)
        # если всё ещё сверх лимита — выселяем самые старые по времени записи
        if len(_cache) > _CACHE_MAX:
            for k in sorted(_cache, key=lambda kk: _cache[kk][0])[: len(_cache) - _CACHE_MAX]:
                _cache.pop(k, None)


def list_hooks() -> list[dict]:
    return db.api_hooks_list()


def save_hook(d: dict) -> int:
    return db.api_hook_save(d)


def delete_hook(hook_id: int) -> bool:
    return db.api_hook_delete(hook_id)


def _enabled() -> list[dict]:
    return [h for h in db.api_hooks_list() if h.get("enabled")]


def _extract_json(s: str) -> str:
    """Выделить первый сбалансированный JSON-объект из текста LLM.

    Раньше брали срез от первой '{' до последней '}' — это ломалось, если после объекта шёл
    поясняющий текст с ещё одной '}' (или несколько объектов). Теперь честно считаем баланс
    скобок с учётом строк и экранирования и возвращаем ровно первый завершённый объект."""
    s = s or ""
    start = s.find("{")
    if start < 0:
        return "{}"
    depth = 0
    in_str = False
    esc = False
    for k in range(start, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:k + 1]
    return "{}"


def _intent_match(hook: dict, q: str):
    """ИИ решает, нужен ли вызов, и извлекает параметры. Возвращает dict или None."""
    try:
        import llm_backend
        pnames = [p.strip() for p in re.split(r"[,\n;]+", hook.get("param_spec") or "")
                  if p.strip()]
        prompt = (
            f"Вопрос пользователя: «{q}»\n"
            f"Сервис: «{hook.get('name')}» — {hook.get('trigger_val') or ''}.\n"
            f"Нужно ли для ответа вызвать этот сервис? Если да, извлеки параметры: "
            f"{', '.join(pnames) if pnames else 'нет'}.\n"
            'Ответь СТРОГО в JSON без пояснений: {"match": true|false, "params": {…}}')
        out = llm_backend.chat([{"role": "user", "content": prompt}], temperature=0,
                               kind="api-intent")
        j = json.loads(_extract_json(out))
        if j.get("match"):
            p = j.get("params") or {}
            return {k: v for k, v in p.items() if v not in (None, "")}
    except Exception as e:
        print(f"[api] intent-матч «{hook.get('name')}»: {e}")
    return None


def _match(hook: dict, q: str):
    """Сработал ли хук. Возвращает dict параметров (возможно пустой) или None."""
    tt = (hook.get("trigger_type") or "keywords").lower()
    trig = hook.get("trigger_val") or ""
    ql = q.lower()
    if tt == "keywords":
        kws = [k.strip().lower() for k in re.split(r"[,\n;]+", trig) if k.strip()]
        if not kws or not any(k in ql for k in kws):
            return None
        pr = (hook.get("param_spec") or "").strip()
        if pr:
            try:
                m = re.search(pr, q, re.IGNORECASE)
                return (m.groupdict() if m else {}) or {}
            except re.error:
                return {}
        return {}
    if tt == "regex":
        try:
            m = re.search(trig, q, re.IGNORECASE)
        except re.error:
            return None
        if not m:
            return None
        return m.groupdict() or {}
    if tt == "intent":
        return _intent_match(hook, q)
    return None


def _fill(tpl: str, params: dict, q: str) -> str:
    """Сырая подстановка (без экранирования). Оставлена для совместимости; для URL/JSON-тела
    используйте _fill_url/_fill_json, которые экранируют значения (H10)."""
    s = tpl or ""
    s = s.replace("{q}", q)
    for k, v in (params or {}).items():
        s = s.replace("{" + str(k) + "}", str(v))
    return s


def _fill_url(tpl: str, params: dict, q: str) -> str:
    """Подстановка в URL с percent-экранированием значений (H10): {q}/{param} не должны
    ломать структуру URL или уводить запрос (инъекция пути/квери, обход хоста)."""
    def esc(v):
        return urllib.parse.quote(str(v), safe="")
    s = (tpl or "").replace("{q}", esc(q))
    for k, v in (params or {}).items():
        s = s.replace("{" + str(k) + "}", esc(v))
    return s


def _fill_json(tpl: str, params: dict, q: str) -> str:
    """Подстановка в JSON-тело с экранированием как строковое JSON-значение (H10): значение
    вставляется внутрь кавычек шаблона (напр. \"key\": \"{param}\"), поэтому берём внутренность
    json.dumps (без обрамляющих кавычек) — спецсимволы/кавычки не ломают JSON и не инъектят поля."""
    def esc(v):
        return json.dumps(str(v), ensure_ascii=False)[1:-1]
    s = (tpl or "").replace("{q}", esc(q))
    for k, v in (params or {}).items():
        s = s.replace("{" + str(k) + "}", esc(v))
    return s


def _dot(data, path: str):
    cur = data
    for part in (path or "").split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _call(hook: dict, params: dict, q: str) -> str:
    # H10: экранируем подстановки в URL (percent) и в JSON-тело (как строковое значение),
    # чтобы {q}/{param} не ломали структуру и не приводили к инъекции/SSRF.
    url = _fill_url(hook.get("url"), params, q)
    # H10: допускаем только http/https (никаких file://, gopher:// и пр.)
    scheme = (urllib.parse.urlparse(url).scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"недопустимая схема URL: {scheme or '(пусто)'}")
    method = (hook.get("method") or "GET").upper()
    try:
        headers = json.loads(hook.get("headers") or "{}") if hook.get("headers") else {}
        if not isinstance(headers, dict):
            headers = {}
    except Exception:
        headers = {}
    timeout = int(hook.get("timeout") or 15)
    body = _fill_json(hook.get("body") or "", params, q)
    # M23: общий клиент с пулом; H10: follow_redirects=False (см. _get_client)
    c = _get_client()
    if method == "POST":
        payload = None
        if body.strip():
            try:
                payload = json.loads(body)
            except Exception:
                payload = None
        if payload is not None:
            r = c.post(url, json=payload, headers=headers, timeout=timeout)
        else:
            r = c.post(url, content=body or None, headers=headers, timeout=timeout)
    else:
        r = c.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    rp = (hook.get("resp_path") or "").strip()
    if rp:
        try:
            val = _dot(r.json(), rp)
            if val is not None:
                return (json.dumps(val, ensure_ascii=False)
                        if isinstance(val, (dict, list)) else str(val))[:_MAX_TEXT]
        except Exception:
            pass
    return (r.text or "")[:_MAX_TEXT]


def run_for(question: str) -> dict | None:
    """Первый сработавший хук → {source, text, hook, params}. Иначе None."""
    for h in _enabled():
        try:
            params = _match(h, question)
        except Exception as e:
            print(f"[api] матч «{h.get('name')}»: {e}")
            params = None
        if params is None:
            continue
        key = hashlib.sha1((str(h.get("id")) + "|" +
                            json.dumps(params, sort_keys=True, ensure_ascii=False)
                            ).encode("utf-8")).hexdigest()
        now = time.time()
        with _lock:
            c = _cache.get(key)
        if c and now - c[0] < _TTL:
            text = c[1]
        else:
            try:
                text = _call(h, params, question)
            except Exception as e:
                print(f"[api] вызов «{h.get('name')}»: {e}")
                continue
            _cache_put(key, now, text)   # M27: с выселением истёкших и лимитом размера
        label = (h.get("source_label") or h.get("name") or "Внешний API").strip()
        return {"source": label, "params": params, "hook": h.get("name"),
                "text": f"[Данные из внешнего сервиса «{label}»]\n{text}"}
    return None


def augment_hit(question: str):
    """Фрагмент для подмешивания в контекст ответа (или None)."""
    r = run_for(question)
    if not r:
        return None
    return {"source": r["source"], "text": r["text"], "page": None, "score": 1.0}


def test(question: str) -> dict:
    """Прогнать хуки по вопросу для проверки в админке."""
    r = run_for(question)
    if not r:
        return {"matched": False}
    return {"matched": True, "source": r["source"], "hook": r["hook"],
            "params": r["params"], "text": r["text"]}
