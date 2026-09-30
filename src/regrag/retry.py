"""Bounded retry with exponential backoff for transient external-API failures only.
Deterministic client errors (400/401/403/404/422) are never retried."""

from __future__ import annotations

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status is not None:
        return status in _TRANSIENT_STATUS
    name = type(exc).__name__
    return name in {"APITimeoutError", "APIConnectionError", "TimeoutException", "ConnectError", "ReadTimeout"}


def api_retry(fn, attempts: int = 3):
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        reraise=True,
    )(fn)
