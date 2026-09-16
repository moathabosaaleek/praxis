import logging
from collections.abc import Sequence

from google import genai
from google.genai import types

from core.config import Settings
from core.messages import ConversationTurn

logger = logging.getLogger(__name__)

_GEMINI_ROLES = {"user": "user", "assistant": "model"}


class LLMError(Exception):
    pass


class PraxisLLM:
    def __init__(self, settings: Settings):
        self._model = settings.llm_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents = [
            types.Content(role=_GEMINI_ROLES[turn.role], parts=[types.Part(text=turn.text)])
            for turn in history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            logger.error("LLM request failed: %s: %s", type(exc).__name__, exc)
            raise LLMError("LLM request failed") from exc

        if not response.text or not response.text.strip():
            logger.error("LLM returned an empty response")
            raise LLMError("LLM returned an empty response")

        return response.text
