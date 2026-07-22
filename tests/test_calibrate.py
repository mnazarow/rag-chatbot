"""Юнит-тесты чистых функций автокалибровки (calibrate): отбор фрагментов и
приведение параметров к каталогу.

calibrate тянет retriever/vectorstore/db — все под заглушками conftest; сами
проверяемые функции внешних сервисов не вызывают.
"""

import calibrate


def _c(dense_rank, score):
    return {"dense_rank": dense_rank, "score": score}


def test_select_respects_retrieve_window():
    cands = [_c(0, 0.9), _c(1, 0.8), _c(5, 0.95)]
    # k_retrieve=3 отсекает кандидата с dense_rank=5, даже если его score выше
    out = calibrate._select(cands, min_score=0.5, k_rerank=10, k_retrieve=3)
    assert all(c["dense_rank"] < 3 for c in out)
    assert _c(5, 0.95) not in out


def test_select_min_score_threshold():
    cands = [_c(0, 0.9), _c(1, 0.3), _c(2, 0.6)]
    out = calibrate._select(cands, min_score=0.5, k_rerank=10, k_retrieve=10)
    assert [c["score"] for c in out] == [0.9, 0.6]  # 0.3 отброшен, сортировка по убыванию


def test_select_k_rerank_limit():
    cands = [_c(i, 0.9 - i * 0.01) for i in range(10)]
    out = calibrate._select(cands, min_score=0.0, k_rerank=3, k_retrieve=10)
    assert len(out) == 3


def test_to_bool_truthy_ru_en():
    for v in ("true", "1", "yes", "да", "вкл", "on", 1, True):
        assert calibrate._to_bool(v) is True


def test_to_bool_falsy_ru_en():
    for v in ("false", "0", "no", "нет", "выкл", "off", 0, False):
        assert calibrate._to_bool(v) is False


def test_to_bool_unknown_is_none():
    assert calibrate._to_bool("может быть") is None


def test_coerce_opt_float_clamped():
    spec = calibrate._OPT_CATALOG["MIN_SCORE"]
    assert calibrate._coerce_opt("MIN_SCORE", 999) == spec["hi"]
    assert calibrate._coerce_opt("MIN_SCORE", -999) == spec["lo"]


def test_coerce_opt_int_rounds_and_clamps():
    assert calibrate._coerce_opt("TOP_K_RERANK", "5") == 5
    assert isinstance(calibrate._coerce_opt("TOP_K_RERANK", 3.6), int)


def test_coerce_opt_unknown_key_none():
    assert calibrate._coerce_opt("__NOPE__", 1) is None


def test_coerce_opt_none_value():
    assert calibrate._coerce_opt("MIN_SCORE", None) is None
