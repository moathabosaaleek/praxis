import logging
from collections.abc import Sequence
from dataclasses import replace

from core.config import Settings
from core.llm_router import PraxisLLM
from core.messages import AssistantResponse, IncomingMessage
from core.plugin import Plugin, PluginContext
from core.redaction import redact
from core.router import Router
from core.version import get_version
from storage.repositories import MessageRepository, SessionRepository

logger = logging.getLogger(__name__)

GREETING = "Praxis is online."
PONG = "Pong! The core engine is responsive."
ASK_USAGE = "Ask me anything. Example: /ask What is a Git submodule?"
EMPTY_MESSAGE = "I didn't get any text to work with."
UNKNOWN_COMMAND = "I don't know that command. Just type a normal message instead."
CANCELLED = "Cancelled."
NOTHING_TO_CANCEL = "Nothing to cancel."


class Assistant:
    """Platform-neutral core. It must not import any messaging-platform code."""

    def __init__(
        self,
        settings: Settings,
        llm: PraxisLLM,
        plugins: Sequence[Plugin],
        default_plugin_name: str,
        messages: MessageRepository | None = None,
        sessions: SessionRepository | None = None,
    ):
        self._settings = settings
        self._messages = messages
        self._sessions = sessions
        self._router = Router(plugins, default_plugin_name)
        self._ctx = PluginContext(settings=settings, llm=llm, messages=messages, sessions=sessions)

    def is_authorized(self, user_id: int) -> bool:
        return user_id == self._settings.admin_telegram_id

    async def handle(self, message: IncomingMessage) -> AssistantResponse | None:
        if not self.is_authorized(message.user_id):
            logger.warning("Ignoring message from unauthorized user ID: %s", message.user_id)
            return None

        text = redact(message.text.strip())
        if not text:
            return AssistantResponse(EMPTY_MESSAGE)

        response = await self._route(text, message)
        await self._remember(message, text, response)
        return response

    async def _route(self, text: str, message: IncomingMessage) -> AssistantResponse:
        if not text.startswith("/"):
            return await self._dispatch(text, message)

        command, _, argument = text.partition(" ")
        name = command[1:].split("@")[0].lower()

        if name == "start":
            return AssistantResponse(GREETING)
        if name == "ping":
            return AssistantResponse(PONG)
        if name == "version":
            return AssistantResponse(f"Praxis v{get_version()}")
        if name == "cancel":
            return await self._cancel(message.chat_id)
        if name == "ask":
            question = argument.strip()
            if not question:
                return AssistantResponse(ASK_USAGE)
            return await self._dispatch(question, message)

        plugin = self._router.get(name)
        if plugin is not None:
            return await self._start_plugin(plugin, argument.strip(), message)

        return AssistantResponse(UNKNOWN_COMMAND)

    async def _start_plugin(
        self, plugin: Plugin, text: str, message: IncomingMessage
    ) -> AssistantResponse:
        """Explicitly invoking a plugin by its /<name> command always starts it fresh."""
        if self._sessions is not None:
            await self._sessions.clear(message.chat_id)
        plugin_message = replace(message, text=text)
        return await plugin.handle(plugin_message, self._ctx)

    async def _dispatch(self, text: str, message: IncomingMessage) -> AssistantResponse:
        plugin_message = replace(message, text=text)
        plugin = await self._select_plugin(plugin_message)
        return await plugin.handle(plugin_message, self._ctx)

    async def _select_plugin(self, message: IncomingMessage) -> Plugin:
        if self._sessions is not None:
            session = await self._sessions.get(message.chat_id)
            if session is not None:
                owner = self._router.get(session.plugin)
                if owner is not None:
                    return owner
        return await self._router.choose(message, self._ctx)

    async def _cancel(self, chat_id: int) -> AssistantResponse:
        if self._sessions is None:
            return AssistantResponse(NOTHING_TO_CANCEL)
        had_session = await self._sessions.get(chat_id) is not None
        await self._sessions.clear(chat_id)
        return AssistantResponse(CANCELLED if had_session else NOTHING_TO_CANCEL)

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
