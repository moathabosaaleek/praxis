import logging
from datetime import datetime

from core.config import Settings
from core.llm_router import LLMError, PraxisLLM
from core.messages import AssistantResponse, IncomingMessage

logger = logging.getLogger(__name__)

GREETING = "Praxis is online."
PONG = "Pong! The core engine is responsive."
ASK_USAGE = "Ask me anything. Example: /ask What is a Git submodule?"
EMPTY_MESSAGE = "I didn't get any text to work with."
LLM_FAILURE = "Sorry, I couldn't get an answer right now. Please try again."
UNKNOWN_COMMAND = "I don't know that command. Just type a normal message instead."


def build_system_prompt(settings: Settings, now: datetime | None = None) -> str:
    now = now or datetime.now(settings.timezone)
    return (
        "You are Praxis, a concise technical assistant for a single user. "
        f"The user's current local time is {now:%A, %B %d, %Y %H:%M} ({settings.timezone.key}). "
        "Use it for any question about today, tomorrow, or other dates. "
        "Keep answers concise and accurate, and avoid corporate jargon. "
        "Format only with **bold**, `inline code`, and fenced code blocks. Do not use tables."
    )


class Assistant:
    """Platform-neutral core. It must not import any messaging-platform code."""

    def __init__(self, settings: Settings, llm: PraxisLLM):
        self._settings = settings
        self._llm = llm

    def is_authorized(self, user_id: int) -> bool:
        return user_id == self._settings.admin_telegram_id

    async def handle(self, message: IncomingMessage) -> AssistantResponse | None:
        if not self.is_authorized(message.user_id):
            logger.warning("Ignoring message from unauthorized user ID: %s", message.user_id)
            return None

        text = message.text.strip()
        if not text:
            return AssistantResponse(EMPTY_MESSAGE)

        if text.startswith("/"):
            return await self._handle_command(text)

        return await self._answer(text)

    async def _handle_command(self, text: str) -> AssistantResponse:
        command, _, argument = text.partition(" ")
        name = command[1:].split("@")[0].lower()

        if name == "start":
            return AssistantResponse(GREETING)
        if name == "ping":
            return AssistantResponse(PONG)
        if name != "ask":
            return AssistantResponse(UNKNOWN_COMMAND)

        question = argument.strip()
        if not question:
            return AssistantResponse(ASK_USAGE)
        return await self._answer(question)

    async def _answer(self, prompt: str) -> AssistantResponse:
        try:
            answer = await self._llm.generate_response(
                prompt, system_instruction=build_system_prompt(self._settings)
            )
        except LLMError:
            return AssistantResponse(LLM_FAILURE)
        return AssistantResponse(answer)
