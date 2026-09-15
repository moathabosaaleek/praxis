from zoneinfo import ZoneInfo

import pytest

from core.config import load_settings

REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "123:abc",
    "ADMIN_TELEGRAM_ID": "42",
    "GEMINI_API_KEY": "test-key",
}


@pytest.fixture
def env(monkeypatch):
    for name in [*REQUIRED_ENV, "LLM_MODEL", "TIMEZONE", "ENV"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_loads_defaults(env):
    settings = load_settings()

    assert settings.admin_telegram_id == 42
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.timezone == ZoneInfo("UTC")
    assert settings.env == "development"


def test_reads_optional_overrides(env):
    env.setenv("LLM_MODEL", "gemini-2.5-pro")
    env.setenv("TIMEZONE", "Asia/Amman")

    settings = load_settings()

    assert settings.llm_model == "gemini-2.5-pro"
    assert settings.timezone == ZoneInfo("Asia/Amman")


@pytest.mark.parametrize("name", list(REQUIRED_ENV))
def test_missing_required_variable_fails(env, name):
    env.delenv(name)

    with pytest.raises(RuntimeError, match=name):
        load_settings()


def test_non_numeric_admin_id_fails(env):
    env.setenv("ADMIN_TELEGRAM_ID", "not-a-number")

    with pytest.raises(RuntimeError, match="ADMIN_TELEGRAM_ID"):
        load_settings()


def test_placeholder_token_fails(env):
    env.setenv("TELEGRAM_BOT_TOKEN", "paste_your_botfather_token_here")

    with pytest.raises(RuntimeError, match="placeholder"):
        load_settings()


@pytest.mark.parametrize("name", ["Mars/Olympus", "../etc/passwd"])
def test_invalid_timezone_fails(env, name):
    env.setenv("TIMEZONE", name)

    with pytest.raises(RuntimeError, match="TIMEZONE"):
        load_settings()
