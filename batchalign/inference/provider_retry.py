"""What an HTTP provider answered, and (for one in-process client) whether to
wait it out.

Two provider clients in this package meet rate limits. They are handled at
different altitudes on purpose:

- Google Translate's answers are REPORTED, never retried here. The worker
  returns each item's [`ProviderResponse`] to the Rust control plane, which
  owns the pacing and the cooldowns (``batchalign::translate::provider``),
  where a wait is visible to the job's deadline and cancellation. The types
  that describe an answer live in this module so the reporting is typed.
- pyannoteAI's client runs a whole diarization job inside one worker request
  (upload, submit, poll), so its per-call 429 handling has to stay in-process:
  [`RetryPolicy.decide`] maps one observed answer and one [`RetryState`] to
  exactly one of [`Retry`] or [`GiveUp`], the latter saying why ([`Final`],
  [`Exhausted`] or [`CooldownTooLong`]). The caller matches them
  exhaustively; there is no boolean and no default arm.

``googletrans`` raises a bare ``Exception`` that carries no status, so the
status and header have to come from the underlying ``httpx`` response:
[`LastResponse`] is the response hook that keeps them, and [`ProviderRefusal`]
is the exception the session raises with what it kept.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderResponse:
    """What the provider answered: the status, and its cooldown if it named one."""

    status: int
    retry_after_s: float | None


class ProviderRefusal(Exception):
    """The provider did not translate: it answered ``response``, or nothing.

    ``response`` is ``None`` when no HTTP response was recorded at all (a
    transport failure before any reply). ``cause`` is the library's own
    exception, kept for the operator-facing message.
    """

    def __init__(self, response: ProviderResponse | None, cause: BaseException) -> None:
        self.response = response
        self.cause = cause
        detail = (
            f"HTTP {response.status}"
            if response is not None
            else "no HTTP response recorded"
        )
        super().__init__(f"{detail}: {cause}")


@dataclass(frozen=True)
class RetryState:
    """The cooldowns a call still has; minted by [`RetryPolicy.start`] only.

    Holding the remaining schedule rather than a count means "exhausted" is
    an empty tuple, not a comparison against a length kept elsewhere.
    """

    remaining: tuple[float, ...]


@dataclass(frozen=True)
class Retry:
    """Wait ``delay_s``, then try again from ``next_state``."""

    delay_s: float
    next_state: RetryState


@dataclass(frozen=True)
class Final:
    """Not an answer this provider recovers from."""

    def __str__(self) -> str:
        return ""


@dataclass(frozen=True)
class Exhausted:
    """Transient, but every scheduled cooldown has been spent."""

    retries_made: int

    def __str__(self) -> str:
        return f" again after {self.retries_made} retries"


@dataclass(frozen=True)
class CooldownTooLong:
    """The provider asked for a longer cooldown than the ceiling allows."""

    requested_s: float
    ceiling_s: float

    def __str__(self) -> str:
        return (
            f" and asked for a {self.requested_s:g} s cooldown (Retry-After), "
            f"above the {self.ceiling_s:g} s ceiling"
        )


@dataclass(frozen=True)
class GiveUp:
    """Fail the call: the status the policy stopped on, and why."""

    status: int
    why: Final | Exhausted | CooldownTooLong

    def __str__(self) -> str:
        return f"returned HTTP {self.status}{self.why}"


Decision = Retry | GiveUp


@dataclass(frozen=True)
class RetryPolicy:
    """Which statuses are worth a retry, how long to wait, and where to stop.

    ``schedule_s`` is the floor for each wait: a ``Retry-After`` longer than
    the scheduled delay replaces it, a shorter one does not.
    """

    retryable: frozenset[int]
    schedule_s: tuple[float, ...]
    ceiling_s: float

    @classmethod
    def pyannote_ai(cls) -> RetryPolicy:
        return cls(
            retryable=frozenset({429}),
            schedule_s=(1.0, 1.0, 1.0),
            ceiling_s=60.0,
        )

    def start(self) -> RetryState:
        """A state with the whole schedule still available."""
        return RetryState(remaining=self.schedule_s)

    def decide(self, response: ProviderResponse, state: RetryState) -> Decision:
        if response.status not in self.retryable:
            return GiveUp(response.status, Final())
        if not state.remaining:
            return GiveUp(response.status, Exhausted(retries_made=len(self.schedule_s)))
        scheduled, *rest = state.remaining
        delay_s = max(scheduled, response.retry_after_s or 0.0)
        if delay_s > self.ceiling_s:
            return GiveUp(response.status, CooldownTooLong(delay_s, self.ceiling_s))
        return Retry(delay_s=delay_s, next_state=RetryState(remaining=tuple(rest)))


def parse_retry_after(raw: str | None) -> float | None:
    """Only the delta-seconds form; an HTTP-date is treated as absent."""
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0.0 else None


class _RequestLike(Protocol):
    url: object


class _ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]
    request: _RequestLike


class LastResponse:
    """The ``httpx`` response hook: remembers the most recent response to
    ``endpoint`` only.

    ``endpoint`` is a substring of the request URL; a response to any other
    request on the same client is ignored. ``googletrans`` fetches a token
    page from a different host once an hour through the same client, and its
    200 must not stand in for the translate request's answer. ``take`` hands
    the response over once and clears it, and ``clear`` empties it before an
    attempt, so an attempt can never be judged on the attempt before (a
    success's 200 must not stand in for a later transport failure's missing
    reply).
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._seen: ProviderResponse | None = None

    async def record(self, response: _ResponseLike) -> None:
        """The async form ``httpx.AsyncClient`` requires of a response hook."""
        if self._endpoint not in str(response.request.url):
            return
        self._seen = ProviderResponse(
            status=int(response.status_code),
            retry_after_s=parse_retry_after(response.headers.get("retry-after")),
        )

    def clear(self) -> None:
        self._seen = None

    def take(self) -> ProviderResponse | None:
        seen, self._seen = self._seen, None
        return seen
