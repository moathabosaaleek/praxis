from types import SimpleNamespace

from core.config import Settings
from interfaces.telegram.bot import require_admin

SETTINGS = Settings(telegram_bot_token="123:abc", admin_telegram_id=42, gemini_api_key="key")


def make_update(user_id):
    user = SimpleNamespace(id=user_id) if user_id is not None else None
    return SimpleNamespace(effective_user=user)


async def call_guarded(user_id):
    calls = []

    async def handler(update, context):
        calls.append(update)
        return "handled"

    result = await require_admin(SETTINGS)(handler)(make_update(user_id), None)
    return result, calls


async def test_admin_is_allowed():
    result, calls = await call_guarded(42)

    assert result == "handled"
    assert len(calls) == 1


async def test_other_user_is_blocked():
    result, calls = await call_guarded(7)

    assert result is None
    assert calls == []


async def test_update_without_user_is_blocked():
    result, calls = await call_guarded(None)

    assert result is None
    assert calls == []
