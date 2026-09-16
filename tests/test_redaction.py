import pytest

from core.redaction import REDACTED, redact

# Built by concatenation, not written as literals, so these fake-but-real-shaped
# fixtures don't trip GitHub secret scanning even though no such secret exists.
_FAKE_TELEGRAM_TOKEN = "8123456789" + ":" + "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw2"
_FAKE_GOOGLE_KEY = "AIza" + "SyD-1234567890abcdefghijklmnopqrstu"
_FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
_FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdefghijklmnopqrstuvwxyz"
_FAKE_OPENAI_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz1234"


@pytest.mark.parametrize(
    "secret",
    [
        _FAKE_TELEGRAM_TOKEN,
        _FAKE_GOOGLE_KEY,
        _FAKE_AWS_KEY,
        _FAKE_GITHUB_TOKEN,
        _FAKE_OPENAI_KEY,
        "4111 1111 1111 1111",
    ],
)
def test_removes_secret_shaped_values(secret):
    result = redact(f"here it is: {secret} ok")

    assert secret not in result
    assert REDACTED in result


def test_removes_labelled_secrets():
    assert redact("api_key = hunter2trustno1") == f"api_key = {REDACTED}"
    assert redact("password: swordfish") == f"password: {REDACTED}"


def test_removes_private_key_blocks():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----"

    assert redact(text) == REDACTED


def test_leaves_ordinary_text_untouched():
    text = "What is TLS 1.3 and why is it faster than 1.2?"

    assert redact(text) == text


def test_leaves_short_numbers_untouched():
    text = "I have 2 meetings and 15 tabs open in 2026"

    assert redact(text) == text
