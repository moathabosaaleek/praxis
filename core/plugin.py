from dataclasses import dataclass
from typing import Protocol

from core.config import Settings
from core.llm_router import PraxisLLM
from core.messages import AssistantResponse, IncomingMessage
from storage.repositories import MessageRepository, SessionRepository


@dataclass(frozen=True)
class PluginContext:
    settings: Settings
    llm: PraxisLLM
    messages: MessageRepository | None
    sessions: SessionRepository | None = None


class Plugin(Protocol):
    name: str
    description: str

    async def handle(self, message: IncomingMessage, ctx: PluginContext) -> AssistantResponse: ...
