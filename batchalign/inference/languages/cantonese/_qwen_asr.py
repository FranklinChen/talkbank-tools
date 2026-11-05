"""Qwen3-ASR worker provider for Cantonese.

The ASR load builds a :class:`QwenAsr` around the ready recognizer and records
it in the worker's load registry; a request transcribes through that object.
Mirrors the ``_funaudio_asr`` / ``_tencent_asr`` module shape so the ASR load
in ``batchalign/worker/_model_loading/asr.py`` treats every provider alike.
Nothing is held at module level.
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from typing import NoReturn

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse
from batchalign.worker._progress import Phase, emit_phase_event
from batchalign.worker._types_v2 import PhaseEdgeV2, QwenRecognitionV2

from ._asr_types import admit_provider_monologues
from ._common import EngineOverrides
from ._qwen_common import QwenRecognizerConfig, QwenSnapshots, ReadyQwenRecognizer

L = logging.getLogger("batchalign.hk.qwen_asr")


# Wall-clock timeout for ``QwenRecognizerConfig.load()``, a single
# blocking import-and-load (then through the ``qwen-asr`` package's
# ``Qwen3ASRModel.from_pretrained``, now ``transformers``' native one)
# with no upstream progress callbacks; on 2026-05-27 it was observed to
# hang for 70+ minutes at 0% CPU with no Qwen model ever appearing in
# the HF cache (the worker had loaded the Whisper-large-v2 FA model and
# then stalled before reaching the qwen-asr code path). The timeout
# below converts that silent hang into a typed TimeoutError so the
# worker exits, the daemon marks the job failed, and the operator sees
# a useful error in the daemon log instead of staring at a 0% CPU
# process for hours. The 20-minute default was sized for a first-time
# download, which no longer happens inside this call: the snapshots are
# resolved before it, under ``download_liveness``. It remains the bound on
# the load from disk; configurable via env var for debugging or for
# tighter CI budgets.
_QWEN_LOAD_TIMEOUT_SECONDS: int = int(
    os.environ.get("BATCHALIGN_QWEN_LOAD_TIMEOUT_SECONDS", "1200")
)


def _raise_qwen_load_timeout(_signum: int, _frame: object) -> NoReturn:
    """SIGALRM handler installed during ``QwenRecognizerConfig.load()``.

    Raises a typed ``TimeoutError`` that propagates out of ``load()``
    naturally: Python signal handlers run between bytecodes, so
    ``from_pretrained``'s pure-Python sections will
    surface the exception promptly. C-extension blocks (libtorch,
    HF Hub native) only surface it when they return to Python; if a
    hang IS deep inside C code, the worker may not interrupt cleanly
    - see [[task #9]] for the watchdog-process follow-up.
    """
    raise TimeoutError(
        f"Qwen3-ASR model load timed out after "
        f"{_QWEN_LOAD_TIMEOUT_SECONDS} second(s): loading the resolved "
        f"snapshots did not complete in the allotted window, the "
        f"silent-hang failure mode first observed during the 2026-05-27 "
        f"v2 Cantonese ASR benchmark sweep. The worker is exiting; the "
        f"daemon will mark the job failed. Override via "
        f"BATCHALIGN_QWEN_LOAD_TIMEOUT_SECONDS."
    )


@dataclass(frozen=True, slots=True)
class QwenAsr:
    """The Qwen3-ASR recognizer one ASR load built."""

    recognizer: ReadyQwenRecognizer

    def transcribe(self, item: AsrBatchItem) -> MonologueAsrResponse:
        """Transcribe one provider media item into the shared monologue payload."""
        payload, _timed_words = self.recognizer.transcribe(
            item.audio_path, decode_budget_seconds=item.decode_budget_seconds
        )
        # Qwen3-ASR performs no speaker separation, so its payload says so and
        # the admission carries that through; the downstream diarization stage
        # attributes speakers when one is requested.
        return admit_provider_monologues(payload, item.lang)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_qwen_asr(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    recognition: QwenRecognitionV2,
    snapshots: QwenSnapshots,
) -> QwenAsr:
    """Initialize the Qwen3-ASR recognizer for one ASR load.

    ``snapshots`` are the two models the worker bootstrap already resolved
    (and, if needed, downloaded under ``download_liveness``) at the revisions
    the control plane pinned. The ``qwen_model`` override is consumed
    upstream: the control plane resolves it into the ASR member, so this
    function never reads it, and the id it logs is the snapshot's own.

    Recognized ``engine_overrides`` keys:

    - ``qwen_device``: torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
      Default ``"cpu"`` (Apple Silicon fleet has no CUDA, and MPS is
      unusable for this model: measured 2026-08-20 to be SIGKILLed on
      long audio and to segfault at load in float16).

    - ``qwen_context``: optional recognizer context/hotwords. An empty string
      explicitly supplies no context, matching the default. The exact
      value remains in the submitted job options; it does not relabel language.

    Loads the model inside the ASR load, so the first inference request
    sees a warm model.
    """
    device = "cpu"
    if engine_overrides and "qwen_device" in engine_overrides:
        device = str(engine_overrides["qwen_device"])

    model_id = snapshots.asr.model_id
    config = QwenRecognizerConfig(
        lang=lang,
        device=device,
        context_override=(engine_overrides or {}).get("qwen_context"),
        # Decided by the control plane for this language and parsed at the
        # worker boundary (`_model_loading.asr.qwen_recognition`).
        recognition=recognition,
    )

    # Name the model being loaded on every UI channel, per the
    # time-transparency rule in talkbank-tools/AGENTS.md §11. This is a load
    # from disk, never a download (see ``QwenSnapshots``), so it says so.
    emit_phase_event(
        Phase.LOADING_QWEN_ASR,
        PhaseEdgeV2.BEGIN,
        f"Loading Qwen3-ASR model {model_id} ({device})…",
    )

    # Bracket the loading transition with a SIGALRM-based timeout. See
    # ``_raise_qwen_load_timeout`` for the rationale; the short version
    # is that the qwen-asr package's first-use code path is known to
    # hang silently, and the worker MUST exit loudly on that hang so
    # the daemon can mark the job failed.
    old_handler = signal.signal(signal.SIGALRM, _raise_qwen_load_timeout)
    signal.alarm(_QWEN_LOAD_TIMEOUT_SECONDS)
    try:
        recognizer = config.load(snapshots)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    emit_phase_event(
        Phase.LOADING_QWEN_ASR,
        PhaseEdgeV2.COMPLETE,
        f"Qwen3-ASR model loaded ({model_id}).",
    )

    L.info(
        "Qwen3-ASR recognizer initialized: lang=%s, model=%s, device=%s",
        lang,
        model_id,
        device,
    )
    return QwenAsr(recognizer)


__all__ = ["QwenAsr", "load_qwen_asr"]
