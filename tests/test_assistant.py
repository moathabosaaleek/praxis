from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from core.assistant import (
    ASK_USAGE,
    EMPTY_MESSAGE,
    GREETING,
    LLM_FAILURE,
    PONG,
    UNKNOWN_COMMAND,
    Assistant,
    build_system_prompt,
)
from core.config import Settings
from core.llm_router import LLMError
from core.messages import ConversationTurn, IncomingMessage
from core.redaction import REDACTED

ADMIN_ID = 42
SETTINGS = Settings(
    telegram_bot_token="123:abc",
    admin_telegram_id=ADMIN_ID,
    gemini_api_key="key",
    timezone=ZoneInfo("Asia/Amman"),
)


class FakeLLM:
    def __init__(self, answer="an answer", error=None):
        self.answer = answer
        self.error = error
        self.prompts = []
        self.histories = []

    async def generate_response(self, prompt, system_instruction=None, history=()):
        self.prompts.append(prompt)
        self.histories.append(tuple(history))
        if self.error:
            raise self.error
        return self.answer


class FakeMessages:
    def __init__(self, history=()):
        self.history = tuple(history)
        self.added = []

    async def add(self, *, chat_id, role, content, user_id=None, sensitivity="low"):
        self.added.append((chat_id, role, content, user_id))

    async def recent(self, chat_id, limit):
        return self.history[-limit:]


def make_message(text, user_id=ADMIN_ID, chat_id=7):
    return IncomingMessage(
        user_id=user_id, chat_id=chat_id, text=text, received_at=datetime.now(UTC)
    )


def make_assistant(llm=None, messages=None):
    return Assistant(SETTINGS, llm or FakeLLM(), messages)


async def test_unauthorized_user_is_ignored():
    llm = FakeLLM()
    store = FakeMessages()

    assert await make_assistant(llm, store).handle(make_message("hi", user_id=999)) is None
    assert llm.prompts == []
    assert store.added == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/start", GREETING),
        ("/ping", PONG),
        ("/ping@praxis_sys_bot", PONG),
        ("/ask", ASK_USAGE),
        ("/unknown", UNKNOWN_COMMAND),
        ("   ", EMPTY_MESSAGE),
    ],
)
async def test_instant_replies_do_not_call_the_llm(text, expected):
    llm = FakeLLM()

    response = await make_assistant(llm).handle(make_message(text))

    assert response.text == expected
    assert llm.prompts == []


async def test_plain_text_is_sent_to_the_llm():
    llm = FakeLLM(answer="42")

    response = await make_assistant(llm).handle(make_message("what is 6 times 7?"))

    assert response.text == "42"
    assert llm.prompts == ["what is 6 times 7?"]


async def test_ask_command_strips_the_command_prefix():
    llm = FakeLLM()

    await make_assistant(llm).handle(make_message("/ask what is TLS?"))

    assert llm.prompts == ["what is TLS?"]


async def test_llm_failure_returns_a_friendly_message():
    llm = FakeLLM(error=LLMError("boom"))

    response = await make_assistant(llm).handle(make_message("hello"))

    assert response.text == LLM_FAILURE


async def test_previous_turns_are_sent_as_history():
    history = (ConversationTurn("user", "my name is Moath"), ConversationTurn("assistant", "noted"))
    llm = FakeLLM()

    await make_assistant(llm, FakeMessages(history)).handle(make_message("what is my name?"))

    assert llm.histories[0] == history


async def test_both_sides_of_the_exchange_are_stored():
    store = FakeMessages()

    await make_assistant(FakeLLM(answer="hi there"), store).handle(make_message("hello"))

    assert store.added == [
        (7, "user", "hello", ADMIN_ID),
        (7, "assistant", "hi there", None),
    ]


async def test_secrets_are_redacted_before_storage_and_the_llm():
    store = FakeMessages()
    llm = FakeLLM()
    token = "8123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw2"

    await make_assistant(llm, store).handle(make_message(f"my token is {token}"))

    assert token not in llm.prompts[0]
    assert REDACTED in llm.prompts[0]
    assert token not in store.added[0][2]


def test_system_prompt_uses_configured_timezone():
    prompt = build_system_prompt(SETTINGS, now=datetime(2026, 9, 16, 14, 30))

    assert "Wednesday, September 16, 2026 14:30" in prompt
    assert "Asia/Amman" in prompt
