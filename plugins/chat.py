from datetime import datetime

from core.llm_router import LLMError
from core.messages import AssistantResponse, ConversationTurn, IncomingMessage
from core.plugin import PluginContext

HISTORY_LIMIT = 10
LLM_FAILURE = "Sorry, I couldn't get an answer right now. Please try again."


def build_system_prompt(settings, now: datetime | None = None) -> str:
    now = now or datetime.now(settings.timezone)
    return (
        "You are Praxis, a concise technical assistant for a single user. "
        f"The user's current local time is {now:%A, %B %d, %Y %H:%M} ({settings.timezone.key}). "
        "Use it for any question about today, tomorrow, or other dates. "
        "Keep answers concise and accurate, and avoid corporate jargon. "
        "Format only with **bold**, `inline code`, and fenced code blocks. Do not use tables."
    )


class ChatPlugin:
    name = "chat"
    description = "General conversation and questions with no other specific plugin."

    async def handle(self, message: IncomingMessage, ctx: PluginContext) -> AssistantResponse:
        history = await self._history(message.chat_id, ctx)
        try:
            answer = await ctx.llm.generate_response(
                message.text,
                system_instruction=build_system_prompt(ctx.settings),
                history=history,
            )
        except LLMError:
            return AssistantResponse(LLM_FAILURE)
        return AssistantResponse(answer)

    async def _history(self, chat_id: int, ctx: PluginContext) -> tuple[ConversationTurn, ...]:
        if ctx.messages is None:
            return ()
        return await ctx.messages.recent(chat_id, HISTORY_LIMIT)
