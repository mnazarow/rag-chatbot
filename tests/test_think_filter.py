"""Юнит-тесты потокового фильтра <think>…</think> в llm_backend."""
import llm_backend as lb


def _run(chunks):
    f = lb._ThinkFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_plain_passthrough():
    assert _run(["привет, мир"]) == "привет, мир"


def test_think_block_removed():
    assert _run(["<think>рассуждаю</think>Ответ"]) == "Ответ"


def test_tags_split_across_chunks():
    assert _run(["<th", "ink>ага", "</thi", "nk>Готово"]) == "Готово"


def test_multiple_think_blocks():
    assert _run(["A<think>x</think>B<think>y</think>C"]) == "ABC"


def test_lt_sign_not_a_tag():
    # «5<10» не должен восприниматься как открытие тега
    assert _run(["итог: 5<10 верно"]) == "итог: 5<10 верно"


def test_unterminated_think_yields_nothing():
    assert _run(["<think>только размышления без ответа"]) == ""


def test_strip_think_helper():
    assert lb._strip_think("<think>abc</think>  Ответ") == "Ответ"
    assert lb._strip_think("нет тегов") == "нет тегов"


def test_emit_len_partial_tag_suffix():
    # хвост, который может оказаться началом тега, не отдаём
    assert lb._emit_len("hi <thi", "<think>") == 3
    assert lb._emit_len("no partial tag", "<think>") == len("no partial tag")
