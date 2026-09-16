from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from core.assistant import (
    ASK_USAGE,
    CANCELLED,
    EMPTY_MESSAGE,
    GREETING,
    NOTHING_TO_CANCEL,
    PONG,
    UNKNOWN_COMMAND,
    Assistant,
)
from core.config import Settings
from core.messages import AssistantResponse, IncomingMessage
from core.redaction import REDACTED
from core.version import get_version
from storage.repositories import Session

ADMIN_ID = 42
SETTINGS = Settings(
    telegram_bot_token="123:abc",
    admin_telegram_id=ADMIN_ID,
    gemini_api_key="key",
    timezone=ZoneInfo("Asia/Amman"),
)


class FakePlugin:
    def __init__(self, name="chat", reply="a reply"):
        self.name = name
        self.description = f"fake plugin {name}"
        self.reply = reply
        self.received = []

    async def handle(self, message, ctx):
        self.received.append(message)
        return AssistantResponse(self.reply)


class FakeLLM:
    def __init__(self, answer="chat"):
        self.answer = answer

    async def generate_response(self, prompt, system_instruction=None, history=()):
        return self.answer


class FakeMessages:
    def __init__(self):
        self.added = []

    async def add(self, *, chat_id, role, content, user_id=None, sensitivity="low"):
        self.added.append((chat_id, role, content, user_id))

    async def recent(self, chat_id, limit):
        return ()


class FakeSessions:
    def __init__(self, sessions: dict[int, Session] | None = None):
        self._sessions = dict(sessions or {})
        self.cleared = []

    async def get(self, chat_id):
        return self._sessions.get(chat_id)

    async def set(self, chat_id, plugin, state):
        self._sessions[chat_id] = Session(plugin=plugin, state=state)

    async def clear(self, chat_id):
        self.cleared.append(chat_id)
        self._sessions.pop(chat_id, None)


def make_message(text, user_id=ADMIN_ID, chat_id=7):
    return IncomingMessage(
        user_id=user_id, chat_id=chat_id, text=text, received_at=datetime.now(UTC)
    )


def make_assistant(plugin=None, messages=None, plugins=None, sessions=None, llm=None):
    plugin = plugin or FakePlugin()
    return Assistant(
        SETTINGS,
        llm=llm or FakeLLM(),
        plugins=plugins or [plugin],
        default_plugin_name=(plugins[0].name if plugins else plugin.name),
        messages=messages,
        sessions=sessions,
    )


async def test_unauthorized_user_is_ignored():
    plugin = FakePlugin()
    store = FakeMessages()

    result = await make_assistant(plugin, store).handle(make_message("hi", user_id=999))

    assert result is None
    assert plugin.received == []
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
async def test_meta_commands_do_not_reach_a_plugin(text, expected):
    plugin = FakePlugin()

    response = await make_assistant(plugin).handle(make_message(text))

    assert response.text == expected
    assert plugin.received == []


async def test_version_command_reports_the_running_version():
    response = await make_assistant().handle(make_message("/version"))

    assert response.text == f"Praxis v{get_version()}"


async def test_plain_text_is_dispatched_to_the_default_plugin():
    plugin = FakePlugin(reply="42")

    response = await make_assistant(plugin).handle(make_message("what is 6 times 7?"))

    assert response.text == "42"
    assert [m.text for m in plugin.received] == ["what is 6 times 7?"]


async def test_ask_command_strips_the_command_prefix_before_dispatch():
    plugin = FakePlugin()

    await make_assistant(plugin).handle(make_message("/ask what is TLS?"))

    assert [m.text for m in plugin.received] == ["what is TLS?"]


async def test_both_sides_of_the_exchange_are_stored():
    store = FakeMessages()

    await make_assistant(FakePlugin(reply="hi there"), store).handle(make_message("hello"))

    assert store.added == [
        (7, "user", "hello", ADMIN_ID),
        (7, "assistant", "hi there", None),
    ]


async def test_secrets_are_redacted_before_dispatch_and_storage():
    store = FakeMessages()
    plugin = FakePlugin()
    token = "8123456789" + ":" + "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw2"

    await make_assistant(plugin, store).handle(make_message(f"my token is {token}"))

    assert token not in plugin.received[0].text
    assert REDACTED in plugin.received[0].text
    assert token not in store.added[0][2]


async def test_active_session_routes_directly_to_its_owner_bypassing_the_router():
    chat = FakePlugin(name="chat", reply="chat reply")
    writer = FakePlugin(name="writer", reply="writer reply")
    sessions = FakeSessions({7: Session(plugin="writer", state={"step": "await_stance"})})

    response = await make_assistant(plugins=[chat, writer], sessions=sessions).handle(
        make_message("agree")
    )

    assert response.text == "writer reply"
    assert chat.received == []
    assert [m.text for m in writer.received] == ["agree"]


async def test_no_active_session_falls_back_to_the_default_plugin():
    chat = FakePlugin(name="chat", reply="chat reply")
    writer = FakePlugin(name="writer", reply="writer reply")

    response = await make_assistant(plugins=[chat, writer], sessions=FakeSessions()).handle(
        make_message("hello")
    )

    assert response.text == "chat reply"
    assert writer.received == []


async def test_generic_plugin_command_starts_that_plugin_fresh():
    chat = FakePlugin(name="chat")
    writer = FakePlugin(name="writer", reply="writer reply")
    sessions = FakeSessions({7: Session(plugin="chat", state={"leftover": True})})

    response = await make_assistant(plugins=[chat, writer], sessions=sessions).handle(
        make_message("/writer")
    )

    assert response.text == "writer reply"
    assert sessions.cleared == [7]
    assert [m.text for m in writer.received] == [""]


async def test_generic_plugin_command_passes_its_argument_as_text():
    chat = FakePlugin(name="chat")
    writer = FakePlugin(name="writer")

    await make_assistant(plugins=[chat, writer], sessions=FakeSessions()).handle(
        make_message("/writer about ransomware trends")
    )

    assert [m.text for m in writer.received] == ["about ransomware trends"]


async def test_cancel_clears_an_active_session():
    sessions = FakeSessions({7: Session(plugin="chat", state={})})

    response = await make_assistant(sessions=sessions).handle(make_message("/cancel"))

    assert response.text == CANCELLED
    assert sessions.cleared == [7]


async def test_cancel_with_no_active_session():
    response = await make_assistant(sessions=FakeSessions()).handle(make_message("/cancel"))

    assert response.text == NOTHING_TO_CANCEL


async def test_cancel_without_a_session_store_configured():
    response = await make_assistant(sessions=None).handle(make_message("/cancel"))

    assert response.text == NOTHING_TO_CANCEL
