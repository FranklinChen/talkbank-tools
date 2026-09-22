"""Provider answers are reported by Google Translate and waited out by pyannoteAI.

``googletrans`` answers any non-200 response with the INPUT text as the
translation unless it is constructed with ``raise_exception=True``: a blocked
or rate-limited call then yields the source text, stamped as a translation.
The library raises a bare ``Exception`` that carries no status, so the status
and ``Retry-After`` header are taken from the ``httpx`` response hook and
reported to the Rust control plane as a typed refusal, which decides whether
to wait. The in-process policy below is pyannoteAI's, whose client runs a
whole job inside one worker request.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from batchalign.inference.provider_retry import (
    CooldownTooLong,
    Decision,
    Exhausted,
    Final,
    GiveUp,
    LastResponse,
    ProviderRefusal,
    ProviderResponse,
    Retry,
    RetryPolicy,
    RetryState,
)

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single?client=gtx"


def _response(
    status_code: int, headers: dict[str, str] | None = None, url: str = TRANSLATE_URL
) -> Any:
    return types.SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        request=types.SimpleNamespace(url=url),
    )


def _decide(
    status: int, *, retry_after: float | None = None, state: RetryState | None = None
) -> Decision:
    policy = RetryPolicy.pyannote_ai()
    return policy.decide(
        ProviderResponse(status=status, retry_after_s=retry_after),
        state or policy.start(),
    )


def test_a_rate_limit_waits_the_scheduled_delay_first() -> None:
    decision = _decide(429)
    assert isinstance(decision, Retry)
    assert decision.delay_s == 1.0


def test_a_longer_retry_after_replaces_the_schedule_and_a_shorter_one_does_not() -> (
    None
):
    longer = _decide(429, retry_after=12.0)
    assert isinstance(longer, Retry) and longer.delay_s == 12.0
    shorter = _decide(429, retry_after=0.2)
    assert isinstance(shorter, Retry) and shorter.delay_s == 1.0


def test_the_schedule_is_spent_one_cooldown_at_a_time_then_gives_up() -> None:
    state = RetryPolicy.pyannote_ai().start()
    for _ in range(3):
        decision = _decide(429, state=state)
        assert isinstance(decision, Retry)
        state = decision.next_state
    assert _decide(429, state=state) == GiveUp(429, Exhausted(retries_made=3))


def test_a_cooldown_beyond_the_ceiling_fails_visibly_instead_of_sleeping() -> None:
    decision = _decide(429, retry_after=61.0)
    assert decision == GiveUp(429, CooldownTooLong(requested_s=61.0, ceiling_s=60.0))
    assert "61" in str(decision) and "60" in str(decision)


def test_a_non_retryable_status_gives_up_at_once() -> None:
    for status in (400, 403, 404, 500, 503):
        assert _decide(status) == GiveUp(status, Final()), status


def test_the_hook_keeps_only_the_last_response_and_hands_it_over_once() -> None:
    seen = LastResponse(endpoint="/translate_a/")
    asyncio.run(seen.record(_response(200)))
    asyncio.run(seen.record(_response(429, {"retry-after": "7"})))
    assert seen.take() == ProviderResponse(status=429, retry_after_s=7.0)
    assert seen.take() is None


def test_the_hook_ignores_answers_to_other_requests_on_the_same_client() -> None:
    """The token page the client fetches once an hour is not an answer to
    any utterance; its 200 must not be taken for the translate request's."""
    seen = LastResponse(endpoint="/translate_a/")
    asyncio.run(seen.record(_response(200, url="https://translate.google.com/")))
    assert seen.take() is None


def test_a_non_numeric_retry_after_is_treated_as_absent() -> None:
    seen = LastResponse(endpoint="/translate_a/")
    asyncio.run(
        seen.record(_response(503, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    )
    assert seen.take() == ProviderResponse(status=503, retry_after_s=None)


class _FakeClient:
    def __init__(self) -> None:
        self.event_hooks: dict[str, list[Any]] = {"request": [], "response": []}


def _install_translator(monkeypatch: pytest.MonkeyPatch, translator: type) -> Any:
    monkeypatch.setitem(
        sys.modules, "googletrans", types.SimpleNamespace(Translator=translator)
    )
    from batchalign.worker._model_loading import translation as loader
    from batchalign.worker._types import _state

    loader._load_google_translate()
    assert _state.translation is not None
    return _state.translation.translate


def test_the_session_reports_a_provider_status_instead_of_returning_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the loader: a 403 raises a typed refusal carrying
    the status and cooldown the hook saw, and nothing is retried here."""
    calls: list[str] = []

    class _FakeTranslator:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs.get("raise_exception") is True, kwargs
            assert kwargs.get("timeout") is not None
            self.client = _FakeClient()

        async def translate(self, text: str) -> Any:
            calls.append(text)
            for hook in self.client.event_hooks["response"]:
                await hook(_response(403, {"retry-after": "30"}))
            raise Exception(
                'Unexpected status code "403" from translate.googleapis.com'
            )

    translate = _install_translator(monkeypatch, _FakeTranslator)
    with pytest.raises(ProviderRefusal) as excinfo:
        translate("hello", "eng")
    assert excinfo.value.response == ProviderResponse(status=403, retry_after_s=30.0)
    assert "403" in str(excinfo.value)
    assert calls == ["hello"], "one attempt: the control plane owns retries"


def test_a_success_does_not_stand_in_for_a_later_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook records every response, the successful 200 included. A
    transport failure on the next item must be reported as no response, not
    judged on the previous item's 200."""
    answers: list[Any] = ["hola", ConnectionResetError("peer reset")]

    class _FakeTranslator:
        def __init__(self, **_kwargs: Any) -> None:
            self.client = _FakeClient()

        async def translate(self, text: str) -> Any:
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            for hook in self.client.event_hooks["response"]:
                await hook(_response(200))
            return types.SimpleNamespace(text=answer)

    translate = _install_translator(monkeypatch, _FakeTranslator)
    assert translate("hello", "eng") == "hola"
    with pytest.raises(ProviderRefusal) as excinfo:
        translate("adios", "eng")
    assert excinfo.value.response is None
    assert "no HTTP response recorded" in str(excinfo.value)


def test_a_success_status_with_an_exception_behind_it_is_an_engine_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 the library could not parse is not a provider refusal: nothing
    is waited out, and the item fails as the engine's own error."""

    class _FakeTranslator:
        def __init__(self, **_kwargs: Any) -> None:
            self.client = _FakeClient()

        async def translate(self, text: str) -> Any:
            for hook in self.client.event_hooks["response"]:
                await hook(_response(200))
            raise Exception("Expecting value: line 1 column 1 (char 0)")

    translate = _install_translator(monkeypatch, _FakeTranslator)
    with pytest.raises(RuntimeError, match="HTTP 200") as excinfo:
        translate("hello", "eng")
    assert not isinstance(excinfo.value, ProviderRefusal)
