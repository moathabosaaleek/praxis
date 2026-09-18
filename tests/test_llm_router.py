import pytest

from core.llm_router import (
    MAX_RETRY_AFTER,
    LLMError,
    LLMQuotaError,
    LLMRateLimitError,
    _as_llm_error,
    _call_with_retry,
)

# Shortened versions of what Google actually returns.
DAILY = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '27s'}]}}"
)
BURST = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, {'retryDelay': '3.74s'}]}}"
)


def test_a_daily_limit_is_not_retryable():
    error = _as_llm_error(Exception(DAILY), "test")

    assert isinstance(error, LLMQuotaError)
    assert not isinstance(error, LLMRateLimitError)


def test_a_burst_limit_is_retryable_and_keeps_the_suggested_delay():
    error = _as_llm_error(Exception(BURST), "test")

    assert isinstance(error, LLMRateLimitError)
    assert error.retry_after == 3.74


def test_a_refusal_without_a_retry_hint_is_not_retried():
    """Guessing a delay would spend a second request from the same allowance."""
    error = _as_llm_error(Exception("429 RESOURCE_EXHAUSTED, no hint here"), "test")

    assert isinstance(error, LLMRateLimitError)
    assert error.retry_after is None


class FakeAPIError(Exception):
    """Mirrors google.genai APIError, which exposes parsed details."""

    def __init__(self, details):
        super().__init__(f"429 RESOURCE_EXHAUSTED. {details}")
        self.details = details


def test_structured_details_are_preferred_over_text_parsing():
    exc = FakeAPIError(
        {
            "error": {
                "details": [
                    {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProject-FreeTier"}]},
                    {"retryDelay": "7s"},
                ]
            }
        }
    )

    error = _as_llm_error(exc, "test")

    assert isinstance(error, LLMRateLimitError)
    assert error.retry_after == 7.0


def test_a_structured_daily_quota_is_never_retryable():
    exc = FakeAPIError(
        {
            "error": {
                "details": [
                    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]},
                    {"retryDelay": "3s"},
                ]
            }
        }
    )

    error = _as_llm_error(exc, "test")

    assert isinstance(error, LLMQuotaError)


async def test_a_refusal_without_a_hint_costs_only_one_request():
    calls = []

    async def send():
        calls.append(1)
        raise Exception("429 RESOURCE_EXHAUSTED with no hint")

    with pytest.raises(LLMRateLimitError):
        await _call_with_retry(send, "test")
    assert len(calls) == 1


def test_a_very_long_suggested_delay_is_capped():
    error = _as_llm_error(Exception("429 RESOURCE_EXHAUSTED 'retryDelay': '600s'"), "test")

    assert error.retry_after == MAX_RETRY_AFTER


def test_an_ordinary_failure_is_not_treated_as_a_limit():
    error = _as_llm_error(Exception("500 INTERNAL"), "test")

    assert type(error) is LLMError


async def test_a_burst_limit_is_retried_once_and_can_succeed():
    calls = []

    async def send():
        calls.append(1)
        if len(calls) == 1:
            raise Exception("429 RESOURCE_EXHAUSTED 'retryDelay': '0.01s'")
        return "worked on the retry"

    assert await _call_with_retry(send, "test") == "worked on the retry"
    assert len(calls) == 2


async def test_a_daily_limit_is_not_retried():
    calls = []

    async def send():
        calls.append(1)
        raise Exception(DAILY)

    with pytest.raises(LLMQuotaError):
        await _call_with_retry(send, "test")
    assert len(calls) == 1


async def test_a_burst_limit_that_never_clears_still_raises():
    async def send():
        raise Exception("429 RESOURCE_EXHAUSTED 'retryDelay': '0.01s'")

    with pytest.raises(LLMRateLimitError):
        await _call_with_retry(send, "test")


async def test_a_normal_call_is_not_retried():
    calls = []

    async def send():
        calls.append(1)
        return "fine"

    assert await _call_with_retry(send, "test") == "fine"
    assert len(calls) == 1
