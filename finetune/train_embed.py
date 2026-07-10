"""Дообучение модели эмбеддингов на оценках 👍 из журнала.

Идея: положительно оценённые ответы связывают ВОПРОС сотрудника с процитированными
ФРАГМЕНТАМИ документов. Из этих пар (вопрос → релевантный фрагмент) дообучаем
sentence-transformers (MultipleNegativesRankingLoss, negatives — в батче), чтобы
эмбеддинги лучше находили нужное именно на вашей терминологии.

Запуск:  python finetune/train_embed.py [--epochs 1] [--batch 16] [--out DIR]
После обучения: укажите EMBED_MODEL = путь к DIR, «Сбросить индекс» и «Переиндексировать».
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import settings  # noqa: E402
import db  # noqa: E402

DEFAULT_OUT = ROOT / "models" / "embed-finetuned"


def build_pairs(min_len: int = 20) -> list[tuple[str, str]]:
    """Пары (вопрос, релевантный фрагмент) из 👍-оценённых ответов журнала."""
    pairs, seen = [], set()
    try:
        rows = db._all(
            "SELECT question, sources FROM requests WHERE rating=1 AND answered=1")
    except Exception as e:
        print(f"Не удалось прочитать журнал: {e}")
        return pairs
    for r in rows:
        q = (r.get("question") or "").strip()
        if len(q) < 5:
            continue
        try:
            srcs = json.loads(r.get("sources") or "[]")
        except Exception:
            srcs = []
        for s in srcs:
            snip = (s.get("snippet") or "").strip()
            if len(snip) < min_len:
                continue
            key = (q, snip[:80])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((q, snip))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pairs = build_pairs()
    print(f"Обучающих пар (вопрос→фрагмент из 👍-ответов): {len(pairs)}")
    if len(pairs) < 20:
        print("FATAL: слишком мало данных для дообучения (нужно ≥20 пар с оценкой 👍 и "
              "процитированными источниками). Накопите оценки в чате и повторите.")
        raise SystemExit(2)

    base = settings.get("EMBED_MODEL")
    device = settings.device()
    print(f"Базовая модель: {base} · устройство: {device}")
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except Exception as e:
        print(f"FATAL: нет sentence-transformers/torch: {e}")
        raise SystemExit(3)

    model = SentenceTransformer(base, device=device)
    examples = [InputExample(texts=[q, p]) for (q, p) in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=max(2, args.batch))
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(1, int(len(loader) * args.epochs * 0.1))
    print(f"Старт обучения: эпох {args.epochs}, батч {args.batch}, шагов/эпоху {len(loader)}")
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=warmup, show_progress_bar=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Модель сохранена: {out}")
    print(f"SUMMARY embed_ft pairs={len(pairs)} out={out}")
    print("Дальше: в настройках задайте EMBED_MODEL = " + str(out) +
          ", затем «Сбросить индекс» и «Переиндексировать».")


if __name__ == "__main__":
    main()
