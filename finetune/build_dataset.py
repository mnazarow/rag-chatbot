"""Сборка обучающего датасета из проиндексированных документов.

Берёт чанки из Qdrant и для каждого (или выборки) просит текущую LLM
сгенерировать пару «вопрос — ответ», основанную строго на фрагменте.
Результат — JSONL в чат-формате для SFT/LoRA (finetune/data/train.jsonl).

Запуск:  python finetune/build_dataset.py [--limit N] [--per-chunk K]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# доступ к модулям проекта из корня
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402
import settings  # noqa: E402
import prompts  # noqa: E402
import llm_backend  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "train.jsonl"

_GEN_PROMPT = (
    "На основе ТОЛЬКО этого фрагмента корпоративного документа придумай {k} "
    "пар(ы) «вопрос-ответ» на русском, как мог бы спросить сотрудник. Ответ — "
    "строго по фрагменту, кратко и точно. Верни СТРОГО JSON-массив объектов "
    '[{{"q":"...","a":"..."}}]. Только JSON.\n\nФРАГМЕНТ:\n'
)


def _parse(s: str) -> list[dict]:
    import re
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [d for d in data if isinstance(d, dict) and d.get("q") and d.get("a")]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="макс. чанков (0 = все)")
    ap.add_argument("--per-chunk", type=int, default=1, help="пар Q&A на чанк")
    args = ap.parse_args()

    client = QdrantClient(url=settings.get("QDRANT_URL"))
    coll = settings.get("QDRANT_COLLECTION")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = n_pairs = 0
    offset = None
    with OUT.open("w", encoding="utf-8") as f:
        while True:
            points, offset = client.scroll(
                coll, with_payload=True, limit=128, offset=offset)
            if not points:
                break
            for p in points:
                text = (p.payload or {}).get("text", "")
                if len(text.strip()) < 80:
                    continue
                try:
                    out = llm_backend.chat(
                        [{"role": "user",
                          "content": _GEN_PROMPT.format(k=args.per_chunk) + text[:2000]}],
                        temperature=0.2)
                except Exception:
                    continue
                for qa in _parse(out):
                    record = {"messages": [
                        {"role": "system", "content": prompts.SYSTEM_PROMPT},
                        {"role": "user", "content": qa["q"]},
                        {"role": "assistant", "content": qa["a"]},
                    ]}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_pairs += 1
                n_chunks += 1
                if args.limit and n_chunks >= args.limit:
                    offset = None
                    break
            if offset is None:
                break

    print(f"Готово: чанков обработано {n_chunks}, пар Q&A {n_pairs} -> {OUT}")
    if n_pairs < 50:
        print("ВНИМАНИЕ: пар мало — дообучение может быть нестабильным. "
              "Добавьте документы или увеличьте --per-chunk.")


if __name__ == "__main__":
    main()
