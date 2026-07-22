"""Юнит-тесты чистых функций retriever: токенизация, интент-роутер, сборка
нейтрального фильтра и нормировка гибридного скоринга.

Тяжёлые модели (sentence_transformers, FlagEmbedding, rank_bm25) подменены
заглушками в conftest — эти функции их не используют.
"""

import retriever
import settings


def test_tokenize_lowercases_and_splits():
    assert retriever._tokenize("Привет  Мир 123") == ["привет", "мир", "123"]


def test_tokenize_empty():
    assert retriever._tokenize("   ") == []
    assert retriever._tokenize("") == []


def test_infer_category_price():
    assert retriever.infer_category("сколько стоит подписка") == "price"
    assert retriever.infer_category("какой прайс на услуги") == "price"


def test_infer_category_training_and_presentation():
    assert retriever.infer_category("есть ли вебинар по продукту") == "training"
    assert retriever.infer_category("покажи слайд из презентации") == "presentation"


def test_infer_category_none_for_neutral_question():
    assert retriever.infer_category("привет, как дела") is None


def test_build_filter_drops_empty_values():
    assert retriever._build_filter({"a": 1, "b": None, "c": ""}) == {"a": 1}


def test_build_filter_none_and_empty():
    assert retriever._build_filter(None) is None
    assert retriever._build_filter({}) is None
    # все значения «пустые» → фильтр не строится
    assert retriever._build_filter({"x": "", "y": None}) is None


def test_apply_hybrid_disabled_by_default_no_mutation():
    # HYBRID_BM25_WEIGHT по умолчанию 0.0 → скоринг не трогаем (обратная совместимость)
    cands = [{"ce": 0.5, "bm25": 2.0}, {"ce": 0.9, "bm25": 10.0}]
    retriever._apply_hybrid(cands)
    assert "score" not in cands[0] and "bm25_norm" not in cands[0]


def test_apply_hybrid_minmax_normalization():
    orig = settings.get

    def fake(k):
        return {"HYBRID_BM25_WEIGHT": 1.0, "HYBRID_CE_WEIGHT": 1.0}.get(k, orig(k))

    settings.get = fake
    try:
        cands = [{"ce": 0.0, "bm25": 2.0}, {"ce": 0.0, "bm25": 10.0}]
        retriever._apply_hybrid(cands)
    finally:
        settings.get = orig
    # min-max нормировка bm25: минимум → 0.0, максимум → 1.0
    assert cands[0]["bm25_norm"] == 0.0
    assert cands[1]["bm25_norm"] == 1.0
    # score = w_ce*ce + w_bm*bm25_norm
    assert cands[0]["score"] == 0.0
    assert cands[1]["score"] == 1.0


def test_apply_hybrid_equal_bm25_no_div_by_zero():
    orig = settings.get

    def fake(k):
        return {"HYBRID_BM25_WEIGHT": 0.5, "HYBRID_CE_WEIGHT": 0.5}.get(k, orig(k))

    settings.get = fake
    try:
        cands = [{"ce": 1.0, "bm25": 5.0}, {"ce": 0.0, "bm25": 5.0}]
        retriever._apply_hybrid(cands)  # rng == 0 → защита от деления на ноль
    finally:
        settings.get = orig
    assert all("score" in c for c in cands)
