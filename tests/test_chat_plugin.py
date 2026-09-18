from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from core.config import Settings
from core.llm_router import LLMError
from core.messages import ConversationTurn, IncomingMessage
from core.plugin import PluginContext
from plugins.chat import LLM_FAILURE, ChatPlugin, build_system_prompt

SETTINGS = Settings(
    telegram_bot_token="123:abc",
    admin_telegram_id=42,
    gemini_api_key="key",
    timezone=ZoneInfo("Asia/Amman"),
)


class FakeLLM:
    def __init__(self, answer="an answer", error=None):
        self.answer = answer
        self.error = error
        self.prompts = []
        self.histories = []
        self.searches = []

    async def generate_response(self, prompt, system_instruction=None, history=(), search=False):
        self.prompts.append(prompt)
        self.histories.append(tuple(history))
        self.searches.append(search)
        if self.error:
            raise self.error
        return self.answer


class FakeMessages:
    def __init__(self, history=()):
        self.history = tuple(history)

    async def recent(self, chat_id, limit):
        return self.history[-limit:]


def make_message(text):
    return IncomingMessage(user_id=42, chat_id=7, text=text, received_at=datetime.now(UTC))


def make_ctx(llm=None, messages=None):
    return PluginContext(settings=SETTINGS, llm=llm or FakeLLM(), messages=messages)


async def test_chat_asks_for_web_grounding():
    llm = FakeLLM()

    await ChatPlugin().handle(make_message("what happened today?"), make_ctx(llm))

    assert llm.searches == [True]


async def test_plain_prompt_is_sent_to_the_llm():
    llm = FakeLLM(answer="42")

    response = await ChatPlugin().handle(make_message("what is 6 times 7?"), make_ctx(llm))

    assert response.text == "42"
    assert llm.prompts == ["what is 6 times 7?"]


async def test_llm_failure_returns_a_friendly_message():
    llm = FakeLLM(error=LLMError("boom"))

    response = await ChatPlugin().handle(make_message("hello"), make_ctx(llm))

    assert response.text == LLM_FAILURE


async def test_previous_turns_are_sent_as_history():
    history = (
        ConversationTurn("user", "my name is Moath"),
        ConversationTurn("assistant", "noted"),
    )
    llm = FakeLLM()

    await ChatPlugin().handle(
        make_message("what is my name?"), make_ctx(llm, FakeMessages(history))
    )

    assert llm.histories[0] == history


async def test_no_history_without_a_message_store():
    llm = FakeLLM()

    await ChatPlugin().handle(make_message("hello"), make_ctx(llm, messages=None))

    assert llm.histories[0] == ()


def test_system_prompt_uses_configured_timezone():
    prompt = build_system_prompt(SETTINGS, now=datetime(2026, 9, 16, 14, 30))

    assert "Wednesday, September 16, 2026 14:30" in prompt
    assert "Asia/Amman" in prompt
