from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from core.assistant import ASK_USAGE, EMPTY_MESSAGE, GREETING, PONG, UNKNOWN_COMMAND, Assistant
from core.config import Settings
from core.messages import AssistantResponse, IncomingMessage
from core.redaction import REDACTED
from core.version import get_version

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


class FakeMessages:
    def __init__(self):
        self.added = []

    async def add(self, *, chat_id, role, content, user_id=None, sensitivity="low"):
        self.added.append((chat_id, role, content, user_id))

    async def recent(self, chat_id, limit):
        return ()


def make_message(text, user_id=ADMIN_ID, chat_id=7):
    return IncomingMessage(
        user_id=user_id, chat_id=chat_id, text=text, received_at=datetime.now(UTC)
    )


def make_assistant(plugin=None, messages=None):
    plugin = plugin or FakePlugin()
    return Assistant(
        SETTINGS, llm=None, plugins=[plugin], default_plugin_name=plugin.name, messages=messages
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
