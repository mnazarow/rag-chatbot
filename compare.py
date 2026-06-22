"""Сравнение двух пайплайнов на одинаковых вопросах.

Слева — основной вектор-вариант (retriever + Ollama),
справа — LightRAG (если установлен и проиндексирован).

Запуск:
  python compare.py questions.txt            # вопросы построчно из файла
  python compare.py "вопрос1" "вопрос2"      # вопросы аргументами
  python compare.py                          # встроенный демо-набор
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

import config
import prompts
import llm_backend
from retriever import search

DEMO_QUESTIONS = [
    "Сколько стоит базовый тариф?",
    "Что входит в программу онбординга новых сотрудников?",
    "Какие основные темы затрагиваются в материалах обучения?",
]


def load_questions(args: list[str]) -> list[str]:
    if not args:
        return DEMO_QUESTIONS
    if len(args) == 1 and Path(args[0]).is_file():
        return [ln.strip() for ln in Path(args[0]).read_text().splitlines() if ln.strip()]
    return args


def vector_answer(question: str) -> str:
    hits = search(question)
    if not hits:
        return "В доступных документах нет точного ответа на этот вопрос."
    context = prompts.build_context(hits)
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_user_message(question, context)},
    ]
    answer = llm_backend.chat(messages, temperature=0.1)
    srcs = "; ".join(f"{h['source']}" + (f" стр.{h['page']}" if h.get('page') else "")
                     for h in hits)
    return answer + f"\n  [источники: {srcs}]"


async def lightrag_answer(question: str, mode: str = "mix") -> str:
    try:
        sys.path.insert(0, str(Path(__file__).parent / "lightrag_variant"))
        from lightrag import QueryParam
        from rag_lightrag import build_rag
    except Exception as e:
        return f"(LightRAG не установлен/не настроен: {e})"
    rag = await build_rag()
    from lightrag import QueryParam  # noqa
    return await rag.aquery(question, param=QueryParam(mode=mode))


async def main():
    questions = load_questions(sys.argv[1:])
    for q in questions:
        print("=" * 80)
        print("ВОПРОС:", q)
        print("-" * 80)
        print("ВЕКТОР:\n", vector_answer(q))
        print("-" * 80)
        print("LIGHTRAG (mix):\n", await lightrag_answer(q))
        print()


if __name__ == "__main__":
    asyncio.run(main())
