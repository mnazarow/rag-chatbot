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
import hashlib
import json
import re
import threading
import time

import db

_cache: dict = {}
_lock = threading.Lock()
_TTL = 20.0          # сек кэш ответа API
_MAX_TEXT = 4000     # обрезка ответа API


def list_hooks() -> list[dict]:
    return db.api_hooks_list()


def save_hook(d: dict) -> int:
    return db.api_hook_save(d)


def delete_hook(hook_id: int) -> bool:
    return db.api_hook_delete(hook_id)


def _enabled() -> list[dict]:
    return [h for h in db.api_hooks_list() if h.get("enabled")]


def _extract_json(s: str) -> str:
    s = s or ""
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if (i >= 0 and j > i) else "{}"


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
        out = llm_backend.chat([{"role": "user", "content": prompt}], temperature=0)
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
    s = tpl or ""
    s = s.replace("{q}", q)
    for k, v in (params or {}).items():
        s = s.replace("{" + str(k) + "}", str(v))
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
    import httpx
    url = _fill(hook.get("url"), params, q)
    method = (hook.get("method") or "GET").upper()
    try:
        headers = json.loads(hook.get("headers") or "{}") if hook.get("headers") else {}
        if not isinstance(headers, dict):
            headers = {}
    except Exception:
        headers = {}
    timeout = int(hook.get("timeout") or 15)
    body = _fill(hook.get("body") or "", params, q)
    with httpx.Client(timeout=timeout, follow_redirects=True) as c:
        if method == "POST":
            payload = None
            if body.strip():
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = None
            if payload is not None:
                r = c.post(url, json=payload, headers=headers)
            else:
                r = c.post(url, content=body or None, headers=headers)
        else:
            r = c.get(url, headers=headers)
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
            with _lock:
                _cache[key] = (now, text)
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
