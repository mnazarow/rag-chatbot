"""Проверка обоснованности ответа (антигаллюцинации).

После генерации LLM сверяет, следует ли ответ строго из найденного контекста. Режимы
(настройка ANSWER_VERIFY):
  off    — не проверять;
  warn   — если ответ не обеспечен источниками, добавить пометку;
  strict — при необеспеченности перегенерировать ответ со строгой инструкцией.
"""
from __future__ import annotations
import json
import re

import settings
import llm_backend
import prompts

CAVEAT = ("\n\n⚠️ Возможно, этот ответ не полностью подтверждён найденными документами — "
          "сверьтесь с источниками ниже.")

_CHECK_SYS = (
    "Ты — контролёр достоверности. Даны КОНТЕКСТ (фрагменты корпоративных документов), "
    "ВОПРОС и ОТВЕТ. Определи, следует ли ОТВЕТ СТРОГО из КОНТЕКСТА, без добавленных "
    "фактов и домыслов. Ответ «в документах нет данных» считается обоснованным. "
    "Верни СТРОГО JSON: {\"grounded\": true|false}. Без пояснений."
)


_FRAG_RE = re.compile(r"\n\n(?=\[Фрагмент \d)")


def _trim_context(context: str, answer: str, budget: int = 8000) -> str:
    """Урезать контекст под бюджет символов, НЕ разрывая фрагменты: режем по границам
    фрагментов (см. prompts.build_context) и в первую очередь оставляем те, чей источник
    упомянут в ответе (использованные). Грубое context[:8000] могло оборвать фрагмент
    на полуслове и «спрятать» именно тот кусок, на который опирался ответ."""
    if not context or len(context) <= budget:
        return context
    frags = _FRAG_RE.split(context)
    ans = (answer or "").lower()

    def _cited(frag: str) -> bool:
        head = frag.split("\n", 1)[0]
        m = re.search(r"—\s*(.+?)\]", head)          # "[Фрагмент i — источник, стр. N]"
        name = (m.group(1) if m else "").split(",")[0].strip().lower()
        return bool(name) and name in ans

    ordered = [f for f in frags if _cited(f)] + [f for f in frags if not _cited(f)]
    out, used = [], 0
    for f in ordered:
        add = len(f) + (2 if out else 0)             # +разделитель "\n\n"
        if used + add > budget:
            continue
        out.append(f)
        used += add
    if not out:
        return frags[0][:budget]                     # ни один фрагмент не влез — крайний случай
    return "\n\n".join(out)


def is_grounded(question: str, answer: str, context: str) -> bool:
    """True — ответ обеспечен контекстом (или проверка недоступна/не нужна)."""
    if not answer or not context:
        return True
    ctx = _trim_context(context, answer, 8000)
    msg = (f"КОНТЕКСТ:\n{ctx}\n\nВОПРОС:\n{question}\n\n"
           f"ОТВЕТ:\n{answer[:4000]}")
    try:
        out = llm_backend.chat(
            [{"role": "system", "content": _CHECK_SYS},
             {"role": "user", "content": msg}],
            temperature=0, model=settings.active_model())
        m = re.search(r"\{.*\}", out or "", re.S)
        d = json.loads(m.group(0)) if m else {}
        return bool(d.get("grounded", True))
    except Exception:
        return True   # при сбое проверки не мешаем ответу


def regenerate_strict(question: str, context: str) -> str | None:
    """Перегенерировать ответ с усиленной инструкцией «только по контексту»."""
    sys = (settings.get("SYSTEM_PROMPT") or "") + (
        "\n\nОСОБО ВАЖНО: используй ТОЛЬКО факты из КОНТЕКСТА. Ничего не добавляй от себя, "
        "не обобщай сверх текста. Если точного ответа в контексте нет — честно ответь, что "
        "в доступных документах нет ответа на этот вопрос.")
    try:
        return llm_backend.chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": prompts.build_user_message(question, context)}],
            temperature=0, model=settings.active_model())
    except Exception:
        return None


def apply(question: str, answer: str, context: str, mode: str | None = None) -> dict:
    """Проверить и (в strict) поправить ответ. Возвращает {answer, grounded, changed}."""
    mode = (mode or settings.get("ANSWER_VERIFY") or "off").strip().lower()
    if mode not in ("warn", "strict") or not answer:
        return {"answer": answer, "grounded": True, "changed": False}
    if prompts.is_no_answer(answer):
        return {"answer": answer, "grounded": True, "changed": False}
    grounded = is_grounded(question, answer, context)
    if grounded:
        return {"answer": answer, "grounded": True, "changed": False}
    if mode == "strict":
        fixed = regenerate_strict(question, context)
        if fixed and fixed.strip():
            g2 = is_grounded(question, fixed, context)
            if g2 or prompts.is_no_answer(fixed):
                return {"answer": fixed, "grounded": g2, "changed": True}
            # повторная необоснованность после перегенерации — отдаём с пометкой CAVEAT
            return {"answer": fixed + CAVEAT, "grounded": False, "changed": True}
    return {"answer": answer + CAVEAT, "grounded": False, "changed": True}
