"""Type stubs for the batchalign_core Rust worker runtime extension.

This stub exposes type information for mypy to check Python code
that calls into the Rust worker runtime. It is hand-written, so
`make batchalign-stubtest` checks it against the BUILT extension: a function
or class the extension registers and this file lacks, a parameter that
differs, or a name only this file has, fails that check. Its `__all__` is
the extension's own export list, and stubtest holds the two equal.
"""

from collections.abc import Callable
from typing import ClassVar, Literal, final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ISO_639_1_PAIRS",
    "STANZA_PROCESSORS",
    "STANZA_REQUIREMENTS",
    "WORKER_ERROR_KINDS",
    "ArtifactInputError",
    "BatchalignError",
    "CHATValidationException",
    "ConfigError",
    "ConfigNotFoundError",
    "DocumentValidationException",
    "ImmediateProtocolReply",
    "PayloadTooLargeError",
    "PendingProtocolRequest",
    "ProtocolStdin",
    "RealignLanguage",
    "SkipFileWarning",
    "align_tokens",
    "aliyun_sentences_to_asr",
    "dispatch_protocol_message",
    "error_envelope",
    "error_kind_ends_worker",
    "execute_asr_request_v2",
    "execute_avqi_request_v2",
    "execute_coref_request_v2",
    "execute_forced_alignment_request_v2",
    "execute_morphosyntax_request_v2",
    "execute_opensmile_request_v2",
    "execute_speaker_embedding_request_v2",
    "execute_speaker_request_v2",
    "execute_translate_request_v2",
    "execute_utseg_request_v2",
    "execute_word_segmentation_request_v2",
    "funaudio_segments_to_asr",
    "open_protocol_stdin",
    "prepare_protocol_message",
    "repair_feats",
    "tencent_result_detail_to_asr",
    "whisper_lexical_frame_indices",
]

def whisper_lexical_frame_indices(costs: NDArray[np.float32]) -> list[int]: ...

# ---------------------------------------------------------------------------
# Worker protocol dispatch
# ---------------------------------------------------------------------------

@final
class ProtocolStdin:
    def read_line(self) -> str | None: ...
    def stop(self) -> None: ...
    @property
    def stopped(self) -> bool: ...

def open_protocol_stdin() -> ProtocolStdin: ...

@final
class PendingProtocolRequest:
    @property
    def message(self) -> dict[str, object]: ...

@final
class ImmediateProtocolReply:
    @property
    def payload(self) -> dict[str, object]: ...
    @property
    def should_shutdown(self) -> bool: ...

def prepare_protocol_message(
    message: object,
) -> PendingProtocolRequest | ImmediateProtocolReply: ...
def dispatch_protocol_message(
    request: PendingProtocolRequest,
    *,
    health_fn: Callable[..., object],
    capabilities_fn: Callable[..., object],
    infer_fn: Callable[..., object],
    batch_infer_fn: Callable[..., object],
    execute_v2_fn: Callable[..., object],
    ensure_task_fn: Callable[..., object],
    infer_request_model: object,
    batch_infer_request_model: object,
    execute_v2_request_model: object,
    validation_error_type: object,
) -> dict[str, object]: ...

WorkerErrorKind = Literal[
    "runtime", "bootstrap", "download", "out_of_memory", "invalid_request"
]
"""What an ``{"op": "error"}`` line reports. The one Python spelling of
Rust's ``WorkerErrorKind``; ``WORKER_ERROR_KINDS`` is the Rust list itself,
and ``test_worker_error_kinds_match_rust`` holds the two equal."""

WORKER_ERROR_KINDS: tuple[WorkerErrorKind, ...]

ISO_639_1_PAIRS: tuple[tuple[str, str], ...]
"""Every (ISO 639-3, ISO 639-1) pair, from Rust's one table
(``batchalign_types::iso639_part1``), sorted by the three-letter code."""

def repair_feats(feats: str) -> tuple[str | None, list[tuple[str, str]]]:
    """FEATS without model padding (``<PAD>``, ``<UNK>``) and each removed
    ``(feature, value)``: Rust's one repair (``repair_feats``), so the worker
    never classifies a value itself."""

StanzaProcessorName = Literal[
    "tokenize", "mwt", "pos", "lemma", "depparse", "constituency", "coref"
]
"""A processor a Stanza catalog entry can ship. The one Python spelling of
Rust's ``StanzaProcessor``; ``STANZA_PROCESSORS`` is the Rust list itself,
and ``test_stanza_names_match_rust`` holds the two equal."""

StanzaRequirementName = Literal["morphosyntax", "utterance_segmentation", "coreference"]
"""A Stanza-backed command's requirement, by ``StanzaRequirement::wire_name``.
Held to ``STANZA_REQUIREMENTS`` by ``test_stanza_names_match_rust``."""

STANZA_PROCESSORS: tuple[StanzaProcessorName, ...]
"""Every processor a catalog entry can ship, from Rust's closed
``StanzaProcessor`` (``batchalign_types::stanza_catalog``), in its order."""

STANZA_REQUIREMENTS: tuple[tuple[StanzaRequirementName, list[StanzaProcessorName]], ...]
"""Each Stanza-backed command's requirement name and the processors it needs,
from Rust's closed ``StanzaRequirement`` (``batchalign_types::stanza_catalog``),
the table the server admits languages by."""

def error_envelope(
    message: str,
    kind: WorkerErrorKind,
    request_id: str | None = None,
) -> dict[str, object]: ...
def error_kind_ends_worker(kind: WorkerErrorKind) -> bool:
    """Whether a worker stops serving after reporting ``kind``: Rust's one
    decision (``WorkerErrorKind::worker_after``), which its readers act on."""

# ---------------------------------------------------------------------------
# Worker V2 execution
# ---------------------------------------------------------------------------

def execute_asr_request_v2(
    request: object,
    local_whisper_runner: Callable[..., object] | None = None,
    hk_tencent_runner: Callable[..., object] | None = None,
    hk_aliyun_runner: Callable[..., object] | None = None,
    hk_funaudio_runner: Callable[..., object] | None = None,
    hk_qwen_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_forced_alignment_request_v2(
    request: object,
    whisper_runner: Callable[..., object] | None = None,
    wave2vec_runner: Callable[..., object] | None = None,
    canto_runner: Callable[..., object] | None = None,
    qwen3_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_opensmile_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_avqi_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_speaker_request_v2(
    request: object,
    pyannote_ai_prepared_audio_runner: Callable[..., object] | None = None,
    pyannote_prepared_audio_runner: Callable[..., object] | None = None,
    nemo_prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_speaker_embedding_request_v2(
    request: object,
    pyannote_span_runner: Callable[..., object] | None = None,
) -> str: ...
def execute_morphosyntax_request_v2(
    request: object,
    runner: Callable[..., object] | None = None,
) -> str: ...
def execute_utseg_request_v2(
    request: object,
    runner: Callable[..., object] | None = None,
) -> str: ...
def execute_translate_request_v2(
    request: object,
    runner: Callable[..., object] | None = None,
) -> str: ...
def execute_coref_request_v2(
    request: object,
    runner: Callable[..., object] | None = None,
) -> str: ...
def execute_word_segmentation_request_v2(
    request: object,
    runner: Callable[[str, str], object] | None = None,
) -> str: ...

@final
class RealignLanguage:
    """A pipeline's realignment language, read from its Stanza code."""

    English: ClassVar[RealignLanguage]
    French: ClassVar[RealignLanguage]
    Italian: ClassVar[RealignLanguage]
    Other: ClassVar[RealignLanguage]
    @staticmethod
    def of_stanza_code(code: str) -> RealignLanguage: ...

def align_tokens(
    original_words: list[str],
    stanza_tokens: list[str],
    language: RealignLanguage,
) -> list[str | tuple[str, bool | list[str]]]: ...

# ---------------------------------------------------------------------------
# Cantonese ASR bridges
# ---------------------------------------------------------------------------

def funaudio_segments_to_asr(segments: object) -> str: ...
def tencent_result_detail_to_asr(result_detail: object) -> str: ...
def aliyun_sentences_to_asr(sentences: object, lang: str) -> str: ...

# ---------------------------------------------------------------------------
# Typed Python exception hierarchy crossing the PyO3 boundary.
#
# Defined in ``crates/batchalign-pyo3/src/error.rs`` via
# ``pyo3::create_exception!``. ``BatchalignError`` is the common ancestor
# (inherits from ``Exception``); the subclasses below all inherit from
# ``BatchalignError``, except ``SkipFileWarning`` which inherits from
# ``Exception`` directly (it is raised and caught as an exception, not a
# warning: see error.rs comment).
# ---------------------------------------------------------------------------

class BatchalignError(Exception): ...
class CHATValidationException(BatchalignError): ...
class DocumentValidationException(BatchalignError): ...
class ConfigNotFoundError(BatchalignError): ...
class ConfigError(BatchalignError): ...
class PayloadTooLargeError(BatchalignError): ...
class SkipFileWarning(Exception): ...

class ArtifactInputError(BatchalignError):
    """A prepared worker artifact was missing or unreadable."""

    code: str
    """The protocol error code's wire spelling: ``"missing_attachment"`` or
    ``"attachment_unreadable"``. Set on the raised instance."""
