"""Qwen3-ASR worker provider for Cantonese.

New-style provider: ``load`` is called once at worker startup,
``infer_*`` is called per request. Mirrors the
``_funaudio_asr`` / ``_tencent_asr`` module shape so the worker
bootstrap dispatch in ``batchalign/worker/_model_loading/asr.py``
can swap engines uniformly.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import NoReturn

from pydantic import ValidationError

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse
from batchalign.worker._progress import emit_download_event
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
)

from ._asr_types import admit_provider_monologues
from ._common import EngineOverrides
from ._qwen_common import QwenRecognizer

L = logging.getLogger("batchalign.hk.qwen_asr")


# Wall-clock timeout for ``QwenRecognizer.warm()``, the call that
# resolves into ``Qwen3ASRModel.from_pretrained(...)`` in the
# ``qwen-asr`` package. That call is a single blocking import-and-load
# with no upstream progress callbacks; on 2026-05-27 it was observed to
# hang for 70+ minutes at 0% CPU with no Qwen model ever appearing in
# the HF cache (the worker had loaded the Whisper-large-v2 FA model and
# then stalled before reaching the qwen-asr code path). The timeout
# below converts that silent hang into a typed TimeoutError so the
# worker exits, the daemon marks the job failed, and the operator sees
# a useful error in the daemon log instead of staring at a 0% CPU
# process for hours. Default is generous (20 min) to accommodate first-
# time model download on slow networks; configurable via env var for
# debugging or for tighter CI budgets.
_QWEN_LOAD_TIMEOUT_SECONDS: int = int(
    os.environ.get("BATCHALIGN_QWEN_LOAD_TIMEOUT_SECONDS", "1200")
)


def _raise_qwen_load_timeout(_signum: int, _frame: object) -> NoReturn:
    """SIGALRM handler installed during ``QwenRecognizer.warm()``.

    Raises a typed ``TimeoutError`` that propagates out of ``warm()``
    naturally: Python signal handlers run between bytecodes, so
    ``Qwen3ASRModel.from_pretrained``'s pure-Python sections will
    surface the exception promptly. C-extension blocks (libtorch,
    HF Hub native) only surface it when they return to Python; if a
    hang IS deep inside C code, the worker may not interrupt cleanly
    - see [[task #9]] for the watchdog-process follow-up.
    """
    raise TimeoutError(
        f"Qwen3-ASR model load timed out after "
        f"{_QWEN_LOAD_TIMEOUT_SECONDS} second(s). The qwen-asr package's "
        f"``Qwen3ASRModel.from_pretrained`` call did not complete in the "
        f"allotted window: this matches the silent-hang failure mode "
        f"observed during the 2026-05-27 v2 Cantonese ASR benchmark "
        f"sweep. The worker is exiting; the daemon will mark the job "
        f"failed and the operator can investigate the qwen-asr package "
        f"integration. Override via BATCHALIGN_QWEN_LOAD_TIMEOUT_SECONDS."
    )


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_recognizer: QwenRecognizer | None = None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_qwen_asr(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    model_path: str | None = None,
    aligner_path: str | None = None,
) -> None:
    """Initialize the Qwen3-ASR recognizer (called once at worker startup).

    Recognized ``engine_overrides`` keys:

    - ``qwen_model``: HuggingFace model id of the Qwen3-ASR
      checkpoint, in the ``-hf`` spelling the native ``transformers``
      module targets. Default ``"Qwen/Qwen3-ASR-1.7B-hf"``;
      ``"Qwen/Qwen3-ASR-0.6B-hf"`` is the smaller faster alternative.
    - ``qwen_device``: torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
      Default ``"cpu"`` (Apple Silicon fleet has no CUDA, and MPS is
      unusable for this model: measured 2026-08-20 to be SIGKILLed on
      long audio and to segfault at load in float16).

    Eager-warms the model at the end of bootstrap so the first
    inference request sees a warm cache. The worker bootstrap
    already selected this engine; requests will come.
    """
    global _recognizer

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    device = "cpu"
    if engine_overrides:
        if "qwen_model" in engine_overrides:
            model_id = str(engine_overrides["qwen_model"])
        if "qwen_device" in engine_overrides:
            device = str(engine_overrides["qwen_device"])

    # ``model_path`` / ``aligner_path`` are snapshots the worker already
    # materialized at the revisions the control plane pinned. The ids stay for
    # the checkpoint check and the log line; the paths are what actually load,
    # so a hub repository that moved cannot change what this worker runs.
    recognizer = QwenRecognizer(
        lang=lang,
        model_id=model_id,
        device=device,
        model_path=model_path,
        aligner_path=aligner_path,
    )

    # Surface the model-load event to every UI channel per the
    # time-transparency rule in talkbank-tools/CLAUDE.md §11. The
    # ``from_pretrained`` call can take minutes (first-time HF Hub
    # download) and an operator watching the daemon log or dashboard
    # needs to see "loading Qwen3-ASR…" rather than silent dead air.
    emit_download_event(
        stage="downloading_qwen_asr",
        user_message=(
            f"Loading Qwen3-ASR model {model_id} ({device}); "
            f"first-run download may take several minutes…"
        ),
    )

    # Bracket ``warm()`` with a SIGALRM-based timeout. See
    # ``_raise_qwen_load_timeout`` for the rationale; the short version
    # is that the qwen-asr package's first-use code path is known to
    # hang silently, and the worker MUST exit loudly on that hang so
    # the daemon can mark the job failed.
    old_handler = signal.signal(signal.SIGALRM, _raise_qwen_load_timeout)
    signal.alarm(_QWEN_LOAD_TIMEOUT_SECONDS)
    try:
        recognizer.warm()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    emit_download_event(
        stage="downloading_qwen_asr_complete",
        user_message=f"Qwen3-ASR model loaded ({model_id}).",
    )

    _recognizer = recognizer
    L.info(
        "Qwen3-ASR recognizer initialized: lang=%s, model=%s, device=%s",
        lang,
        model_id,
        device,
    )


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------


def infer_qwen_asr(req: BatchInferRequest) -> BatchInferResponse:
    """Batch-mode handler: kept for symmetry with the other HK ASR
    providers even though the V2 worker path uses ``infer_qwen_asr_v2``."""
    if _recognizer is None:
        return BatchInferResponse(
            results=[
                InferResponse(
                    error="Qwen3-ASR recognizer not loaded",
                    elapsed_s=0.0,
                )
                for _ in req.items
            ],
        )

    t0 = time.monotonic()
    results: list[InferResponse] = []
    for item_idx, raw_item in enumerate(req.items):
        try:
            item = AsrBatchItem.model_validate(raw_item)
        except ValidationError:
            results.append(InferResponse(error="Invalid AsrBatchItem", elapsed_s=0.0))
            continue
        try:
            response = _transcribe_to_monologues(item)
            results.append(InferResponse(result=response.model_dump(), elapsed_s=0.0))
        except Exception as exc:
            L.warning(
                "Qwen3-ASR failed for item %d: %s",
                item_idx,
                exc,
                exc_info=True,
            )
            results.append(InferResponse(error=str(exc), elapsed_s=0.0))

    elapsed = time.monotonic() - t0
    if results:
        first = results[0]
        results[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed
        )
    return BatchInferResponse(results=results)


def _transcribe_to_monologues(item: AsrBatchItem) -> MonologueAsrResponse:
    """Run Qwen3-ASR for one batch item; return the shared monologue payload."""
    if _recognizer is None:
        raise RuntimeError("Qwen3-ASR recognizer not initialized")

    payload, _timed_words = _recognizer.transcribe(
        item.audio_path, decode_budget_seconds=item.decode_budget_seconds
    )

    # Qwen3-ASR performs no speaker separation, so its payload says so and the
    # admission carries that through; the downstream diarization stage
    # attributes speakers when one is requested.
    return admit_provider_monologues(payload, item.lang)


def infer_qwen_asr_v2(item: AsrBatchItem) -> MonologueAsrResponse:
    """V2 worker entry point, one item, returns typed payload directly."""
    return _transcribe_to_monologues(item)
