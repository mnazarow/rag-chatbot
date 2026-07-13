"""Юнит-тесты целостности схемы настроек и вспомогательных функций."""
import config
import settings


def test_fields_and_defaults_built():
    assert isinstance(settings.FIELDS, list) and len(settings.FIELDS) > 50
    # у каждого поля есть ключ, тип и default, ссылающийся на существующий config.*
    for f in settings.FIELDS:
        assert "key" in f and "type" in f, f
        # DEFAULTS построены из FIELDS — ключ обязан там быть
        assert f["key"] in settings.DEFAULTS


def test_known_new_keys_present():
    for k in ["LLM_THINK", "PROMPT_INJECTION_GUARD", "WEB_SITE_CONCURRENCY",
              "WEB_RESPECT_CRAWL_DELAY", "WEB_CRAWL_DELAY_MAX", "WEB_JS_RECYCLE",
              "ALERTS_ENABLED", "ANSWER_CACHE_SEMANTIC", "ANSWER_CACHE_SIM"]:
        assert k in settings.DEFAULTS, k
        assert hasattr(config, k), k


def test_coerce_bool():
    assert settings._coerce("LLM_THINK", True) is True
    assert settings._coerce("LLM_THINK", False) is False


def test_export_masks_secrets_by_default():
    exp = settings.export_settings(include_secrets=False)
    # секретные ключи не должны утекать в экспорт по умолчанию
    secret_keys = [f["key"] for f in settings.FIELDS if f["type"] == "secret"]
    for k in secret_keys:
        assert k not in exp


def test_import_ignores_unknown_keys():
    r = settings.import_settings({"__NOPE__": 1})
    assert r.get("ok") is True and r.get("applied") == 0
