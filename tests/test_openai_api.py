"""Error handling around the OpenAI SDK, with a fake network (no real requests)."""

import httpx
import openai
import pytest

import openai_api
from exceptions import OpenAIAccountError, TranscriptionError, VisualContextError

_NO_CREDIT = {
    "error": {
        "message": "You exceeded your current quota.",
        "type": "insufficient_quota",
        "code": "insufficient_quota",
    }
}
_BALANCE_EXHAUSTED = {
    "error": {
        "message": "Your credit balance is too low.",
        "type": "insufficient_quota",
        "code": "credit_balance_exhausted",
    }
}
_RATE_LIMITED = {
    "error": {
        "message": "Rate limit reached for requests",
        "type": "requests",
        "code": "rate_limit_exceeded",
    }
}


def _client(status: int, body: dict, calls: list[int]) -> openai.OpenAI:
    """A real SDK client whose network answers every request with one response."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        # Retry-After in ms keeps the SDK's own backoff out of the test's time.
        return httpx.Response(status, json=body, headers={"retry-after-ms": "1"})

    return openai_api.make_client("test-key", transport=httpx.MockTransport(handler))


def _chat(client: openai.OpenAI) -> None:
    client.chat.completions.create(
        model="gpt-5.4-nano", messages=[{"role": "user", "content": "hi"}]
    )


@pytest.mark.parametrize("body", [_NO_CREDIT, _BALANCE_EXHAUSTED])
def test_no_credit_is_sent_once_not_retried(body):
    calls: list[int] = []
    with pytest.raises(openai.RateLimitError) as caught:
        _chat(_client(429, body, calls))

    assert len(calls) == 1
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert isinstance(translated, OpenAIAccountError)
    assert "credit exhausted" in str(translated)
    assert "{" not in str(translated)  # never the raw JSON


def test_a_real_rate_limit_is_still_retried():
    calls: list[int] = []
    with pytest.raises(openai.RateLimitError) as caught:
        _chat(_client(429, _RATE_LIMITED, calls))

    assert len(calls) == 1 + openai_api.OPENAI_MAX_RETRIES
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert type(translated) is TranscriptionError
    assert "rate limit" in str(translated)


def test_a_bad_key_is_sent_once_and_named():
    calls: list[int] = []
    body = {"error": {"message": "Incorrect API key provided: sk-...", "code": None}}
    with pytest.raises(openai.AuthenticationError) as caught:
        _chat(_client(401, body, calls))

    assert len(calls) == 1
    translated = openai_api.translate_error(caught.value, VisualContextError)
    assert isinstance(translated, OpenAIAccountError)
    assert "API key" in str(translated)


def test_server_errors_are_retried_then_reported_briefly():
    calls: list[int] = []
    with pytest.raises(openai.InternalServerError) as caught:
        _chat(_client(500, {"error": {"message": "boom"}}, calls))

    assert len(calls) == 1 + openai_api.OPENAI_MAX_RETRIES
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert type(translated) is TranscriptionError
    assert "server error" in str(translated)


def test_a_bad_request_shows_the_api_message_not_the_json():
    calls: list[int] = []
    body = {"error": {"message": "Audio file is too short.", "code": "audio_too_short"}}
    with pytest.raises(openai.BadRequestError) as caught:
        _chat(_client(400, body, calls))

    assert len(calls) == 1
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert str(translated) == "OpenAI refused the request: Audio file is too short."


def test_no_access_concerns_one_model_not_the_account():
    calls: list[int] = []
    body = {"error": {"message": "Project does not have access to model x"}}
    with pytest.raises(openai.PermissionDeniedError) as caught:
        _chat(_client(403, body, calls))

    assert len(calls) == 1
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert type(translated) is TranscriptionError
    assert "no access" in str(translated)


def test_an_odd_429_body_is_retried_as_a_rate_limit_not_a_lost_connection():
    # Some gateways send {"error": "Too Many Requests"}; the hook must not crash
    # on it, or the SDK reports the crash as a connection error.
    calls: list[int] = []
    with pytest.raises(openai.RateLimitError):
        _chat(_client(429, {"error": "Too Many Requests"}, calls))
    assert len(calls) == 1 + openai_api.OPENAI_MAX_RETRIES


def test_inactive_billing_is_an_account_error_sent_once():
    calls: list[int] = []
    body = {"error": {"message": "not active", "code": "billing_not_active"}}
    with pytest.raises(openai.RateLimitError) as caught:
        _chat(_client(429, body, calls))

    assert len(calls) == 1
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert isinstance(translated, OpenAIAccountError)
    assert "billing is not active" in str(translated)


def test_a_server_side_timeout_reads_as_a_timeout():
    calls: list[int] = []
    with pytest.raises(openai.APIStatusError) as caught:
        _chat(_client(408, {"error": {"message": "Request timed out"}}, calls))
    translated = openai_api.translate_error(caught.value, TranscriptionError)
    assert "did not answer in time" in str(translated)


def test_the_client_carries_the_retry_and_timeout_policy():
    client = openai_api.make_client("test-key")
    assert client.max_retries == openai_api.OPENAI_MAX_RETRIES
    assert client.timeout.read == openai_api.OPENAI_TIMEOUT_SECONDS
    assert client.timeout.connect == openai_api.OPENAI_CONNECT_TIMEOUT_SECONDS
    client.close()


def test_connection_failures_say_so():
    request = httpx.Request("POST", "https://api.openai.com/v1/x")
    error = openai.APIConnectionError(request=request)
    translated = openai_api.translate_error(error, TranscriptionError)
    assert "Cannot reach OpenAI" in str(translated)

    timeout = openai.APITimeoutError(request=request)
    translated = openai_api.translate_error(timeout, TranscriptionError)
    assert "did not answer in time" in str(translated)
