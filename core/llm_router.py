import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence

from google import genai
from google.genai import types

from core.config import Settings
from core.messages import ConversationTurn

logger = logging.getLogger(__name__)

_GEMINI_ROLES = {"user": "user", "assistant": "model"}


class LLMError(Exception):
    pass


class LLMQuotaError(LLMError):
    """The daily allowance is gone. Waiting a few seconds will not help."""


class LLMRateLimitError(LLMError):
    """The provider refused for now. retry_after is None when it gave no hint."""

    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


_RETRY_DELAY = re.compile(r"'retryDelay': '(\d+(?:\.\d+)?)s'")
_QUOTA_ID = re.compile(r"'quotaId': '([^']+)'")
MAX_RETRY_AFTER = 10.0


def _walk_details(details: object) -> list[dict]:
    """Collect the dicts Google nests under error.details, whatever the shape."""
    if isinstance(details, dict):
        found = [details]
        for value in details.values():
            found.extend(_walk_details(value))
        return found
    if isinstance(details, list):
        return [d for item in details for d in _walk_details(item)]
    return []


def _quota_facts(exc: Exception) -> tuple[str | None, float | None]:
    """Read the quota id and retry hint from the SDK's parsed error, not from text."""
    quota_id: str | None = None
    retry_after: float | None = None

    for entry in _walk_details(getattr(exc, "details", None)):
        quota_id = quota_id or entry.get("quotaId")
        delay = entry.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                retry_after = float(delay[:-1])
            except ValueError:
                pass

    # Older payloads and plain strings still need the text fallback.
    text = str(exc)
    if quota_id is None:
        found = _QUOTA_ID.search(text)
        quota_id = found.group(1) if found else None
    if retry_after is None:
        found = _RETRY_DELAY.search(text)
        retry_after = float(found.group(1)) if found else None

    return quota_id, retry_after


def _as_llm_error(exc: Exception, what: str) -> LLMError:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        logger.error("%s failed: %s: %s", what, type(exc).__name__, exc)
        return LLMError(text)

    quota_id, retry_after = _quota_facts(exc)

    # Per-day limits do not clear by waiting a moment, so never retry those.
    if quota_id and "PerDay" in quota_id:
        logger.error("%s hit the DAILY quota (%s)", what, quota_id)
        return LLMQuotaError(text)

    if retry_after is None:
        # No hint means we cannot know when it clears. Retrying would just spend
        # another request from the same allowance, so report instead of guessing.
        # Log the provider's own text: a refusal with no quota id is usually a
        # separately metered add-on, such as search grounding, rather than the model.
        logger.error(
            "%s was refused with no retry hint (quotaId=%s). Provider said: %s",
            what,
            quota_id,
            text[:500],
        )
        return LLMRateLimitError(text, retry_after=None)

    delay = min(retry_after, MAX_RETRY_AFTER)
    logger.warning("%s was rate limited (quotaId=%s), retry in %.1fs", what, quota_id, delay)
    return LLMRateLimitError(text, retry_after=delay)


async def _call_with_retry(send: Callable[[], Awaitable], what: str):
    """Retry once, but only when the provider told us how long to wait."""
    try:
        return await send()
    except Exception as exc:
        error = _as_llm_error(exc, what)
        if not isinstance(error, LLMRateLimitError) or error.retry_after is None:
            raise error from exc
        await asyncio.sleep(error.retry_after)

    try:
        return await send()
    except Exception as exc:
        raise _as_llm_error(exc, f"{what} (retry)") from exc


def _build_schema(spec: dict[str, list[str] | None]) -> types.Schema:
    """Turn a plain field spec into a provider schema, so plugins stay provider-agnostic."""
    properties = {
        name: types.Schema(type=types.Type.STRING, enum=list(allowed))
        if allowed
        else types.Schema(type=types.Type.STRING)
        for name, allowed in spec.items()
    }
    return types.Schema(type=types.Type.OBJECT, properties=properties, required=list(spec))


class PraxisLLM:
    def __init__(self, settings: Settings):
        self._model = settings.llm_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        history: Sequence[ConversationTurn] = (),
        search: bool = False,
    ) -> str:
        """Set search=True only when the answer needs live web grounding.

        Grounding is metered separately from the model, so attaching it to calls
        that do not need it burns a quota that has nothing to do with the request.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())] if search else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents = [
            types.Content(role=_GEMINI_ROLES[turn.role], parts=[types.Part(text=turn.text)])
            for turn in history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        response = await _call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            ),
            "LLM request",
        )

        if not response.text or not response.text.strip():
            logger.error("LLM returned an empty response")
            raise LLMError("LLM returned an empty response")

        return response.text

    async def generate_json(
        self,
        prompt: str,
        spec: dict[str, list[str] | None],
        system_instruction: str | None = None,
    ) -> dict[str, str]:
        """Ask for a structured decision instead of prose.

        No search tool is attached: grounding is not wanted for a routing decision,
        and it conflicts with structured output.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=_build_schema(spec),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = await _call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=config,
            ),
            "Structured LLM request",
        )

        try:
            decision = json.loads(response.text or "")
        except json.JSONDecodeError as exc:
            logger.error("LLM returned malformed JSON")
            raise LLMError("LLM returned malformed JSON") from exc

        if not isinstance(decision, dict):
            raise LLMError("LLM returned JSON that is not an object")
        return decision
