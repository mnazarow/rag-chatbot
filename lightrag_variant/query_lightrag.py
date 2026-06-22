"""Запрос к LightRAG из консоли.

Запуск:
  python query_lightrag.py "какие условия по продукту X" --mode hybrid

Режимы:
  naive  — обычный вектор-поиск (≈ базовый вариант)
  local  — вокруг конкретных сущностей (факты, детали)
  global — по сообществам графа (сводные/тематические вопросы)
  hybrid — local + global
  mix    — граф + вектор (обычно лучший баланс)
"""
from __future__ import annotations
import argparse
import asyncio

from lightrag import QueryParam
from rag_lightrag import build_rag


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="mix",
                    choices=["naive", "local", "global", "hybrid", "mix"])
    args = ap.parse_args()

    rag = await build_rag()
    answer = await rag.aquery(args.question, param=QueryParam(mode=args.mode))
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
