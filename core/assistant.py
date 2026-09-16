import logging
from datetime import datetime

from core.config import Settings
from core.llm_router import LLMError, PraxisLLM
from core.messages import AssistantResponse, ConversationTurn, IncomingMessage
from core.redaction import redact
from storage.repositories import MessageRepository

logger = logging.getLogger(__name__)

GREETING = "Praxis is online."
PONG = "Pong! The core engine is responsive."
ASK_USAGE = "Ask me anything. Example: /ask What is a Git submodule?"
EMPTY_MESSAGE = "I didn't get any text to work with."
LLM_FAILURE = "Sorry, I couldn't get an answer right now. Please try again."
UNKNOWN_COMMAND = "I don't know that command. Just type a normal message instead."

HISTORY_LIMIT = 10


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

    def __init__(
        self,
        settings: Settings,
        llm: PraxisLLM,
        messages: MessageRepository | None = None,
    ):
        self._settings = settings
        self._llm = llm
        self._messages = messages

    def is_authorized(self, user_id: int) -> bool:
        return user_id == self._settings.admin_telegram_id

    async def handle(self, message: IncomingMessage) -> AssistantResponse | None:
        if not self.is_authorized(message.user_id):
            logger.warning("Ignoring message from unauthorized user ID: %s", message.user_id)
            return None

        text = redact(message.text.strip())
        if not text:
            return AssistantResponse(EMPTY_MESSAGE)

        response = await self._route(text, message.chat_id)
        await self._remember(message, text, response)
        return response

    async def _route(self, text: str, chat_id: int) -> AssistantResponse:
        if not text.startswith("/"):
            return await self._answer(text, chat_id)

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
        return await self._answer(question, chat_id)

    async def _answer(self, prompt: str, chat_id: int) -> AssistantResponse:
        history = await self._history(chat_id)
        try:
            answer = await self._llm.generate_response(
                prompt,
                system_instruction=build_system_prompt(self._settings),
                history=history,
            )
        except LLMError:
            return AssistantResponse(LLM_FAILURE)
        return AssistantResponse(answer)

    async def _history(self, chat_id: int) -> tuple[ConversationTurn, ...]:
        if self._messages is None:
            return ()
        return await self._messages.recent(chat_id, HISTORY_LIMIT)

    async def _remember(
        self, message: IncomingMessage, text: str, response: AssistantResponse
    ) -> None:
        if self._messages is None:
            return
        await self._messages.add(
            chat_id=message.chat_id, user_id=message.user_id, role="user", content=text
        )
        await self._messages.add(
            chat_id=message.chat_id, role="assistant", content=redact(response.text)
        )
