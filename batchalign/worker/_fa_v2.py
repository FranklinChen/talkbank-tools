"""Thin worker-protocol V2 forced-alignment wrapper.

Rust now owns the prepared-artifact reads, request validation, backend dispatch,
and typed V2 response shaping for the worker FA boundary. Python stays only at
the model-host callback edge.

**See also:** `../../INTERFACE_MAP.md` section "3. Forced Alignment V2" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_fa_exec.rs::execute_forced_alignment_request_v2()`
- Full Rust/Python responsibility split and input/output contracts.

Concurrency note
----------------
The GPU worker serves V2 requests via ``_serve_stdio_concurrent(max_threads=4)``,
which dispatches up to 4 requests simultaneously through a ``ThreadPoolExecutor``.
PyTorch releases the GIL during compute, so threads truly run in parallel on the
CPU side.  However, the torchaudio MMS_FA / ``forced_align`` kernel is **not
thread-safe** for concurrent CPU execution from multiple threads, concurrent
calls cause SIGSEGV/SIGABRT in LibTorch, crashing the entire worker process and
failing every pending request in the job.

``_fa_inference_lock`` serializes all calls to ``execute_forced_alignment_request_v2``
so that only one FA inference runs at a time.  The lock is module-level (one per
worker process), which is the correct granularity, each GPU worker process is a
single Python interpreter serving one language's models.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
    FaPrepareV2,
    ForcedAlignmentRequestV2,
    UnitProgressV2,
)

if TYPE_CHECKING:
    import torch

    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost
    from batchalign.inference.qwen_forced_alignment import (
        IndexedWordTimingRow,
        QwenFaHost,
    )
    from batchalign.inference.types import (
        Wave2VecFAHandle,
        Wave2VecFAResult,
        WhisperFAHandle,
    )
    from batchalign.worker._loads import FaHostView, WhisperFaRunner, WordFaRunner

# Serialize all FA inference calls within a single worker process.
#
# The torchaudio MMS_FA forced_align kernel is not thread-safe for concurrent
# CPU execution. _serve_stdio_concurrent runs a ThreadPoolExecutor(max_threads=4)
# and PyTorch releases the GIL during compute, so without this lock up to 4
# threads can race inside the native C++ kernel simultaneously, causing
# SIGSEGV/SIGABRT. One lock per process is the right granularity because each
# worker process owns exactly one loaded FA model for one language.
_fa_inference_lock = threading.Lock()


def _as_tensor(audio: np.ndarray) -> torch.Tensor:
    """The prepared mono samples as the float32 tensor the aligners take."""
    import torch

    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def whisper_fa_runner(model: WhisperFAHandle) -> WhisperFaRunner:
    """The Whisper FA runner over the model its load built."""
    from batchalign.inference.fa import infer_whisper_fa

    def _run_whisper(
        audio: np.ndarray, text: str, capture_json: str | None = None
    ) -> list[tuple[str, float]]:
        # `text` arrives shaped: Rust applied the request's `FaTextModeV2`,
        # including the character-spacing that `--pauses` selects.
        if capture_json is None:
            return infer_whisper_fa(model, _as_tensor(audio), text)
        from batchalign.inference.whisper_attention import WhisperAttentionCapture

        capture = WhisperAttentionCapture(capture_json, audio, text, model.sample_rate)
        result = infer_whisper_fa(model, _as_tensor(audio), text, capture=capture)
        capture.require_recorded()
        return result

    return _run_whisper


def wave2vec_fa_runner(model: Wave2VecFAHandle) -> WordFaRunner:
    """The Wave2Vec FA runner over the model its load built."""
    from batchalign.inference.fa import infer_wave2vec_fa

    def _run_wave2vec(audio: np.ndarray, words: list[str]) -> Wave2VecFAResult:
        return infer_wave2vec_fa(model, _as_tensor(audio), words)

    return _run_wave2vec


def cantonese_fa_runner(host: CantoneseFaHost) -> WordFaRunner:
    """The Cantonese FA runner over the host its load built.

    The same call shape as Wave2Vec's: the host romanizes the words itself, so
    nothing about the request beyond the words reaches it. It used to receive
    the prepared payload and the whole request as JSON, and read only the
    words.
    """

    def _run_canto(audio: np.ndarray, words: list[str]) -> Wave2VecFAResult:
        return host.align_words(_as_tensor(audio), words)

    return _run_canto


def qwen_fa_runner(host: QwenFaHost) -> WordFaRunner:
    """The Qwen3 FA runner over the host its load built."""

    def _run_qwen3(audio: np.ndarray, words: list[str]) -> list[IndexedWordTimingRow]:
        # The host owns the whole verb (segment, align, fold onto our words),
        # so nothing is reassembled here. Times come back WINDOW-relative on
        # purpose: Rust adds the group offset.
        #
        # `to_indexed_timings` is the one lossy step, and it is called here
        # and nowhere else: the fold knows WHY a word is untimed and the
        # worker protocol's row has no slot for it, so the reason stops at
        # this line rather than at an unmarked boundary.
        return host.align_words(audio, words).to_indexed_timings()

    return _run_qwen3


@contextmanager
def _preparation_progress(request: ExecuteRequestV2) -> Iterator[None]:
    """The metadata-preparation phase owns its request's progress edges."""
    from batchalign.worker._protocol import write_request_progress

    match request.payload:
        case ForcedAlignmentRequestV2(phase=FaPrepareV2()):
            write_request_progress(
                request.request_id,
                "preparing_alignment",
                UnitProgressV2(completed=0, total=1),
            )
            yield
            write_request_progress(
                request.request_id,
                "preparing_alignment",
                UnitProgressV2(completed=1, total=1),
            )
        case _:
            yield


def execute_forced_alignment_request_v2(
    request: ExecuteRequestV2,
    host: FaHostView,
) -> ExecuteResponseV2:
    """Execute one live V2 forced-alignment request on the loaded FA host.

    The Rust executor reads the host's ``backend`` and ``runner`` and refuses a
    request for a backend this worker did not load. The
    ``_fa_inference_lock`` is held for the duration of this call to prevent
    concurrent FA inference within the same worker process. See the
    module-level docstring for the full rationale.
    """
    import batchalign_core

    with _fa_inference_lock, _preparation_progress(request):
        return ExecuteResponseV2.model_validate_json(
            batchalign_core.execute_forced_alignment_request_v2(request, host)
        )


__all__ = [
    "cantonese_fa_runner",
    "execute_forced_alignment_request_v2",
    "qwen_fa_runner",
    "wave2vec_fa_runner",
    "whisper_fa_runner",
]
