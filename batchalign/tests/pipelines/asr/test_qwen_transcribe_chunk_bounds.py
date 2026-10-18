"""`ReadyQwenRecognizer._transcribe_chunk` classifies its own decode outcome.

No real model: `loaded.processor` and `loaded.model` are fakes, so
`generate` never runs actual inference. What is under test is the
classification, not the model: does a decode that used its whole token
budget come back as `Bounded(TokenCap)`, and does a decode whose
`_RecordingMaxTimeCriteria` recorded firing come back as `Bounded(Deadline)`,
in each case with the partial text intact. Classification reads the
criterion's OWN recorded fact (`.fired`), not a wall-clock measurement taken
outside `generate`, so the fake model below must actually invoke the
`stopping_criteria` it was handed, the way a real `generate` loop does.

Needs torch (for the tensor shapes `generate` and `decode` pass around), not
a downloaded model.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from batchalign.inference.languages.cantonese._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
    Bounded,
    BoundReason,
    ChunkDecodeBudget,
    Completed,
    DecodeBudget,
)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_torch(), reason="torch not installed")


def _recognizer(loaded):
    from batchalign.inference.languages.cantonese._qwen_common import (
        QwenRecognizerConfig,
        ReadyQwenRecognizer,
    )

    return ReadyQwenRecognizer(
        QwenRecognizerConfig(
            lang="yue", model_id="Qwen/Qwen3-ASR-0.6B-hf", device="cpu"
        ),
        loaded,
    )


def _budget_for(chunk: AudioChunk) -> ChunkDecodeBudget:
    """An ample file-level budget, so these tests are purely about the
    per-chunk classification, not about the file-level cap (that invariant
    is covered separately in test_qwen_decode_budget.py)."""
    return ChunkDecodeBudget.for_chunk(chunk, DecodeBudget(seconds=1_000_000.0))


class _FakeInputs(dict):
    """Duck-types the `BatchFeature` the real processor would return: a
    mapping `generate` unpacks with `**`, plus a no-op device/dtype move."""

    def to(self, device: object, dtype: object) -> _FakeInputs:
        return self


class _FakeProcessor:
    def __init__(self, prompt_len: int) -> None:
        self._prompt_len = prompt_len

    def apply_transcription_request(
        self, *, audio: object, language: str
    ) -> _FakeInputs:
        import torch

        return _FakeInputs(
            input_ids=torch.zeros((1, self._prompt_len), dtype=torch.long)
        )

    def decode(self, generated: object, *, return_format: str) -> list[dict[str, str]]:
        # One "token" per character keeps the test's expected text legible.
        n_tokens = generated.shape[1]  # type: ignore[attr-defined]
        return [{"transcription": "x" * n_tokens}]


class _FakeModel:
    """A `generate` that sleeps, then checks the stopping criteria it was
    handed exactly once, the way a real `generate` loop checks after every
    generated token: real inference checks every step, but the fact under
    test here (`_RecordingMaxTimeCriteria.fired`) only depends on whether
    the check ever happened AFTER the deadline passed, so one check after
    the configured sleep is sufficient to exercise it."""

    def __init__(
        self, prompt_len: int, produced_tokens: int, sleep_seconds: float
    ) -> None:
        self.device = "cpu"
        self.dtype = None
        self._prompt_len = prompt_len
        self._produced_tokens = produced_tokens
        self._sleep_seconds = sleep_seconds

    def generate(self, **kwargs: object) -> object:
        import torch

        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        output_ids = torch.zeros(
            (1, self._prompt_len + self._produced_tokens), dtype=torch.long
        )
        stopping_criteria = kwargs["stopping_criteria"]
        stopping_criteria(output_ids, None)  # type: ignore[operator]
        return output_ids


class _FakeLoaded:
    def __init__(self, model: _FakeModel, processor: _FakeProcessor) -> None:
        self.model = model
        self.processor = processor
        # Unused by `_transcribe_chunk`, present so attribute access on a
        # `LoadedQwen`-shaped object never fails if extended later.
        self.aligner = None
        self.aligner_processor = None


def _chunk(seconds: float) -> AudioChunk:
    return AudioChunk(
        samples=np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), offset=0.0
    )


def test_a_generate_that_fills_its_token_budget_is_bounded_by_the_token_cap() -> None:
    """The reference-length chunk's budget is exactly `MAX_NEW_TOKENS_CEILING`
    (see test_qwen_decode_budget), so a stub that always emits the full budget
    reproduces the reported runaway shape: it hit the cap well under the
    wall-clock deadline, so the deadline criterion never fires."""
    chunk = _chunk(185.0)
    budget = _budget_for(chunk)
    prompt_len = 3
    loaded = _FakeLoaded(
        model=_FakeModel(
            prompt_len, produced_tokens=budget.max_new_tokens, sleep_seconds=0.0
        ),
        processor=_FakeProcessor(prompt_len),
    )

    outcome = _recognizer(loaded)._transcribe_chunk(chunk, budget)

    assert isinstance(outcome, Bounded)
    assert outcome.reason is BoundReason.TOKEN_CAP
    assert outcome.text == "x" * budget.max_new_tokens


def test_a_generate_whose_deadline_criterion_fires_is_bounded_by_the_deadline() -> None:
    """A short chunk gets a short wall-clock deadline; a stub that sleeps
    past it, however few tokens it returns, must be classified Deadline, not
    TokenCap: the two are independent bounds and the shorter one can fire
    first. Classification reads `_RecordingMaxTimeCriteria.fired`, so the
    fake must actually invoke the criteria it was handed after sleeping."""
    chunk = _chunk(0.01)
    budget = _budget_for(chunk)
    assert budget.deadline_seconds < 0.2, (
        "test assumes a short chunk gets a sub-200ms deadline; if the "
        "calibration constants changed, adjust the sleep below to match"
    )
    prompt_len = 3
    loaded = _FakeLoaded(
        model=_FakeModel(
            prompt_len,
            produced_tokens=1,
            sleep_seconds=budget.deadline_seconds + 0.2,
        ),
        processor=_FakeProcessor(prompt_len),
    )

    outcome = _recognizer(loaded)._transcribe_chunk(chunk, budget)

    assert isinstance(outcome, Bounded)
    assert outcome.reason is BoundReason.DEADLINE
    assert outcome.text == "x"


def test_a_generate_that_stops_early_within_budget_is_completed() -> None:
    chunk = _chunk(180.0)
    budget = _budget_for(chunk)
    prompt_len = 3
    loaded = _FakeLoaded(
        model=_FakeModel(
            prompt_len,
            produced_tokens=budget.max_new_tokens - 1,
            sleep_seconds=0.0,
        ),
        processor=_FakeProcessor(prompt_len),
    )

    outcome = _recognizer(loaded)._transcribe_chunk(chunk, budget)

    assert isinstance(outcome, Completed)
    assert outcome.text == "x" * (budget.max_new_tokens - 1)


# --- the wrapper itself, against a REAL StoppingCriteriaList ----------------


def test_an_expired_max_time_criterion_signals_stop_via_a_real_list() -> None:
    """No fakes below `StoppingCriteriaList` here: a real `MaxTimeCriteria`
    (via `_RecordingMaxTimeCriteria`) whose clock is already past its
    deadline, run through a real `transformers.generation.StoppingCriteriaList`
    against a two-row `input_ids` batch, must report every row done -- and
    the wrapper must have recorded that it fired."""
    import torch
    from transformers.generation import StoppingCriteriaList

    from batchalign.inference.languages.cantonese._qwen_common import (
        _RecordingMaxTimeCriteria,
    )

    criterion = _RecordingMaxTimeCriteria(
        max_time=1.0, initial_timestamp=time.time() - 1_000.0
    )
    criteria = StoppingCriteriaList([criterion])
    input_ids = torch.zeros((2, 3), dtype=torch.long)  # two-row batch

    is_done = criteria(input_ids, None)

    assert bool(is_done.all())
    assert criterion.fired is True
