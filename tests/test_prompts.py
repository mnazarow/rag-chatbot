"""Юнит-тесты сборки промпта и защиты от инъекций."""
import prompts


def test_context_and_question_present():
    msg = prompts.build_user_message("Сколько стоит гарантия?", "КОНТЕКСТ-ТЕКСТ")
    assert "КОНТЕКСТ-ТЕКСТ" in msg
    assert "Сколько стоит гарантия?" in msg


def test_injection_guard_present_by_default():
    # по умолчанию PROMPT_INJECTION_GUARD включён — в промпте есть указание про «данные»
    msg = prompts.build_user_message("вопрос", "контекст")
    low = msg.lower()
    assert "данные" in low and "инструкции" in low


def test_build_context_numbers_fragments():
    ctx = prompts.build_context([
        {"text": "первый", "source": "a.pdf", "page": 1, "score": 0.9},
        {"text": "второй", "source": "b.pdf", "score": 0.8},
    ])
    assert "Фрагмент 1" in ctx and "Фрагмент 2" in ctx
    assert "a.pdf" in ctx and "первый" in ctx


def test_is_no_answer():
    assert prompts.is_no_answer("В доступных документах нет точного ответа на этот вопрос.")
    assert not prompts.is_no_answer("Гарантия составляет 3 года.")
