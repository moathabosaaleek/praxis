from datetime import UTC, datetime

import pytest

from core.llm_router import LLMError
from core.messages import AssistantResponse, IncomingMessage
from core.plugin import PluginContext
from core.router import Router


class FakePlugin:
    def __init__(self, name, description="a fake plugin"):
        self.name = name
        self.description = description

    async def handle(self, message, ctx):
        return AssistantResponse(f"handled by {self.name}")


class FakeLLM:
    def __init__(self, answer="chat", error=None):
        self.answer = answer
        self.error = error
        self.prompts = []

    async def generate_response(self, prompt, system_instruction=None, history=()):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.answer


def make_message(text="hello"):
    return IncomingMessage(user_id=1, chat_id=1, text=text, received_at=datetime.now(UTC))


def make_ctx(llm):
    return PluginContext(settings=None, llm=llm, messages=None)


async def test_single_plugin_is_returned_without_calling_the_llm():
    chat = FakePlugin("chat")
    llm = FakeLLM()
    router = Router([chat], default_name="chat")

    chosen = await router.choose(make_message(), make_ctx(llm))

    assert chosen is chat
    assert llm.prompts == []


async def test_llm_choice_selects_the_matching_plugin():
    chat = FakePlugin("chat")
    writer = FakePlugin("writer")
    router = Router([chat, writer], default_name="chat")

    chosen = await router.choose(make_message(), make_ctx(FakeLLM(answer="writer")))

    assert chosen is writer


async def test_unrecognized_llm_answer_falls_back_to_default():
    chat = FakePlugin("chat")
    writer = FakePlugin("writer")
    router = Router([chat, writer], default_name="chat")

    chosen = await router.choose(make_message(), make_ctx(FakeLLM(answer="not-a-real-plugin")))

    assert chosen is chat


async def test_llm_failure_falls_back_to_default():
    chat = FakePlugin("chat")
    writer = FakePlugin("writer")
    router = Router([chat, writer], default_name="chat")

    chosen = await router.choose(make_message(), make_ctx(FakeLLM(error=LLMError("boom"))))

    assert chosen is chat


def test_unknown_default_name_is_rejected():
    with pytest.raises(ValueError, match="default_name"):
        Router([FakePlugin("chat")], default_name="writer")
