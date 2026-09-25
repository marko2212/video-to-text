"""The OpenAI client and the translation of its errors into app errors.

One place decides how requests are retried and how failures read in the UI.
The SDK already retries what can recover — connection errors, timeouts,
408/409/429 and 5xx, honouring ``Retry-After`` — so the app adds no retry loop
of its own. What cannot recover (a bad key, an empty balance) is never retried
and is reported in one short sentence instead of the raw JSON body. This module
is UI-agnostic — it never imports Streamlit.
"""

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    ConflictError,
    DefaultHttpxClient,
    InternalServerError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from config import (
    OPENAI_CONNECT_TIMEOUT_SECONDS,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)
from exceptions import AppError, OpenAIAccountError

# Error codes OpenAI uses for an account that cannot be billed. They arrive as
# HTTP 429 like an ordinary rate limit, but waiting does not help.
_NO_CREDIT_CODES = {"insufficient_quota", "credit_balance_exhausted"}
_BILLING_INACTIVE_CODES = {"billing_not_active"}
_ACCOUNT_CODES = _NO_CREDIT_CODES | _BILLING_INACTIVE_CODES
_BILLING_URL = "platform.openai.com/settings/organization/billing"


def _no_retry_without_credit(response: httpx.Response) -> None:
    """Tell the SDK not to retry a 429 that means "no credit".

    The SDK retries every 429, because it cannot tell an exhausted balance from
    a momentary rate limit; it does obey the ``x-should-retry`` header. Without
    this, each chunk or frame of an account with no credit was sent 4 times.

    Args:
        response: A response, before the SDK decides whether to retry it.
    """
    if response.status_code != 429:
        return
    # An exception here would reach the SDK as a connection error, so this hook
    # must never raise: any body it does not understand is left alone.
    try:
        response.read()
        body = response.json()
    except (ValueError, httpx.HTTPError):
        return
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return
    if error.get("code") in _ACCOUNT_CODES or error.get("type") in _ACCOUNT_CODES:
        response.headers["x-should-retry"] = "false"


def make_client(api_key: str, transport: httpx.BaseTransport | None = None) -> OpenAI:
    """Return an OpenAI client with the app's retry and timeout policy.

    Args:
        api_key: OpenAI API key.
        transport: Replaces the network, for tests only.

    Returns:
        A configured client.
    """
    return OpenAI(
        api_key=api_key,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=httpx.Timeout(
            OPENAI_TIMEOUT_SECONDS, connect=OPENAI_CONNECT_TIMEOUT_SECONDS
        ),
        http_client=DefaultHttpxClient(
            transport=transport,
            event_hooks={"response": [_no_retry_without_credit]},
        ),
    )


def _api_message(exc: APIStatusError) -> str:
    """Return the human-readable part of an API error, without the JSON dump.

    Args:
        exc: An error carrying the decoded response body.

    Returns:
        The API's own message when present, otherwise the HTTP status.
    """
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"].strip()
    return f"HTTP {exc.status_code}"


def is_account_error(exc: OpenAIError) -> bool:
    """Return True for errors that every further request would repeat.

    A 403 is not one of them: it concerns one model or endpoint (a restricted
    key may transcribe but not use the vision model), so it must not cancel
    work that would succeed.

    Args:
        exc: An error raised by the OpenAI SDK.

    Returns:
        True for a rejected key or an account that cannot be billed.
    """
    if isinstance(exc, AuthenticationError):
        return True
    return isinstance(exc, RateLimitError) and (
        exc.code in _ACCOUNT_CODES or exc.type in _ACCOUNT_CODES
    )


def translate_error(exc: OpenAIError, fallback: type[AppError]) -> AppError:
    """Turn an SDK error into an app error with a short, actionable message.

    Args:
        exc: The error raised by the OpenAI SDK (after its own retries).
        fallback: Error class for failures that concern only this request,
            e.g. ``TranscriptionError`` or ``VisualContextError``.

    Returns:
        :class:`OpenAIAccountError` when the account itself is refused,
        otherwise an instance of ``fallback``. Never the raw response body.
    """
    if isinstance(exc, AuthenticationError):
        return OpenAIAccountError(
            "OpenAI rejected the API key — check OPENAI_API_KEY in .env or the "
            "key in the sidebar."
        )
    if isinstance(exc, PermissionDeniedError):
        return fallback(f"This API key has no access to it: {_api_message(exc)}")
    if isinstance(exc, RateLimitError):
        if exc.code in _BILLING_INACTIVE_CODES:
            return OpenAIAccountError(
                f"OpenAI billing is not active for this account — see {_BILLING_URL}."
            )
        if is_account_error(exc):
            return OpenAIAccountError(
                f"OpenAI credit exhausted — add credit at {_BILLING_URL}."
            )
        return fallback("OpenAI rate limit reached — wait a minute and try again.")
    if isinstance(exc, APITimeoutError) or (
        isinstance(exc, APIStatusError) and exc.status_code == 408
    ):
        return fallback("OpenAI did not answer in time — try again.")
    if isinstance(exc, ConflictError):
        return fallback("OpenAI was busy with a conflicting request — try again.")
    if isinstance(exc, APIConnectionError):
        return fallback("Cannot reach OpenAI — check the internet connection.")
    if isinstance(exc, InternalServerError):
        return fallback(
            f"OpenAI had a server error (HTTP {exc.status_code}) — try again later."
        )
    if isinstance(exc, APIStatusError):
        return fallback(f"OpenAI refused the request: {_api_message(exc)}")
    return fallback(f"OpenAI request failed: {exc}")
