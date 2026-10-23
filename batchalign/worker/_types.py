"""Request/Response models and worker state (V1 protocol).

These mirror Rust batchalign-types::worker and are the wire format
for the stdio JSON-lines IPC protocol between the Rust server and
stateless Python inference workers.

V1 Protocol Status: FROZEN
~~~~~~~~~~~~~~~~~~~~~~~~~
V1 is used for morphosyntax, utseg, translate, and coref batch inference.
All new task families (FA, ASR, speaker, opensmile, avqi) use V2
(see ``_types_v2.py``). V1 types are not part of the Rust→Python schema
generation pipeline and are not covered by conformance tests.

Do not add new types to V1. New engines and commands should use V2.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, assert_never

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
)

import batchalign_core
from batchalign.device import DevicePolicy
from batchalign.inference._domain_types import (
    CommandName,
    LanguageCode,
    NumSpeakers,
    RevAiApiKey,
    TimestampMs,
)

if TYPE_CHECKING:
    from batchalign.inference.qwen_forced_alignment import QwenFaHost
    from batchalign.inference.translate import LoadedTranslation
    from batchalign.inference.types import (
        Wave2VecFAHandle,
        WhisperASRHandle,
        WhisperFAHandle,
    )
    from batchalign.models.utterance.infer import BertUtteranceModel
    from batchalign.worker._pipeline_cache import StanzaPipelineCache
    from batchalign.worker._types_v2 import (
        AsrModelIdentityV2,
        AsrRequestedModelsV2,
    )

JSONPrimitive = str | int | float | bool | None

if TYPE_CHECKING:
    WorkerJSONValue = (
        JSONPrimitive | Sequence["WorkerJSONValue"] | Mapping[str, "WorkerJSONValue"]
    )
else:
    type WorkerJSONValue = (
        JSONPrimitive | Sequence["WorkerJSONValue"] | Mapping[str, "WorkerJSONValue"]
    )


class HealthResponse(BaseModel):
    """Response body for health operation."""

    status: str = "ok"
    command: CommandName = ""
    lang: LanguageCode = ""
    pid: int = 0
    uptime_s: float = 0.0


class InferTask(str, Enum):
    """Supported inference task identifiers (snake_case on the wire)."""

    MORPHOSYNTAX = "morphosyntax"
    UTSEG = "utseg"
    TRANSLATE = "translate"
    COREF = "coref"
    FA = "fa"
    ASR = "asr"
    OPENSMILE = "opensmile"
    AVQI = "avqi"
    SPEAKER = "speaker"
    WORD_SEGMENTATION = "word_segmentation"


class WorkerServing(str, Enum):
    """How one worker process serves requests.

    Decided by the Rust pool (``batchalign::worker::serving::WorkerServing``),
    which routes its requests by the same value and passes it as
    ``--serving``. The worker obeys it and never derives its own: two owners of
    this decision disagreed on 2026-09-30.
    """

    CONCURRENT = "concurrent"
    SEQUENTIAL = "sequential"


class WorkerProfile(str, Enum):
    """Worker profile grouping related InferTasks into fewer processes.

    Instead of spawning one worker per InferTask, profiles group related tasks
    to share loaded models within a single process:

    - GPU: ASR, FA, Speaker, GPU-bound models
    - STANZA: Morphosyntax, Utseg, Coref, Stanza NLP processors
    - IO: Translate, OpenSMILE, AVQI, word segmentation, lightweight API/library calls
    """

    GPU = "gpu"
    STANZA = "stanza"
    IO = "io"


PROFILE_TASKS: dict[WorkerProfile, set[str]] = {
    WorkerProfile.GPU: {"asr", "fa", "speaker"},
    WorkerProfile.STANZA: {"morphosyntax", "utseg", "coref"},
    WorkerProfile.IO: {"translate", "opensmile", "avqi", "word_segmentation"},
}


@dataclass(frozen=True, slots=True)
class WorkerBootstrapRuntime:
    """Typed worker bootstrap inputs resolved once at process startup."""

    task: InferTask | None
    lang: LanguageCode
    num_speakers: NumSpeakers
    profile: WorkerProfile | None = None
    engine_overrides: dict[str, str] = field(default_factory=dict)
    test_echo: bool = False
    verbose: int = 0
    device_policy: DevicePolicy = field(default_factory=DevicePolicy)
    revai_api_key: RevAiApiKey | None = None


class AsrEngine(str, Enum):
    """Which ASR backend is loaded for this worker."""

    WHISPER = "whisper"
    WHISPER_HUB = "whisper_hub"
    REV = "rev"
    TENCENT = "tencent"
    ALIYUN = "aliyun"
    FUNAUDIO = "funaudio"
    QWEN = "qwen"


class FaEngine(str, Enum):
    """Which forced-alignment backend is loaded for this worker."""

    WHISPER = "whisper"
    WAVE2VEC = "wave2vec"
    WAV2VEC_CANTO = "wav2vec_canto"
    QWEN3 = "qwen3_fa"


_ENGINE_NAME_FORBIDDEN_CHARACTERS = frozenset("|;]\n\r")
"""Characters a reported engine name may never contain.

An engine name is written into a provenance stamp
(``[fc-ba3 <command> | key=value ; key=value | timestamp]``), where ``|``,
``;`` and ``]`` are the grammar's own separators and a line break ends the
header. Rust's ``StampSafeText::STAMP_STRUCTURE`` refuses the same set; this is
the Python producer's copy, so a bad name fails where it is born instead of as
a refused item on the far side of the bridge.
"""

_STAMP_WHITESPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0                　")
"""What counts as whitespace in an engine name: the Unicode ``White_Space`` set.

Written out rather than taken from ``str.isspace`` or ``str.strip``, which also
treat the information separators U+001C to U+001F as space, so Python and Rust
(``StampSafeText::WHITESPACE``) refuse exactly the same names. The shared cases
in ``tests/fixtures/stamp_safe_text_cases.json`` hold both sides to it.
"""


class InvalidReportedEngineName(ValueError):
    """A would-be engine identity that cannot be reported.

    A ``ValueError`` so pydantic reports it as a validation failure when it
    fires inside a model, and so the Rust executor taxonomy classifies it the
    way it classifies every other invalid value.
    """


def reported_engine_name(value: str) -> str:
    """Admit ``value`` as a reported engine identity, unchanged, or raise.

    Refuses a blank name, surrounding whitespace (``_STAMP_WHITESPACE``), and
    any character in ``_ENGINE_NAME_FORBIDDEN_CHARACTERS``. Nothing is trimmed
    or replaced: a name that needs repair is a defect at whatever produced it,
    and a silently repaired name would stop matching the cache rows and
    provenance earlier runs wrote under the original.
    """
    if all(character in _STAMP_WHITESPACE for character in value):
        raise InvalidReportedEngineName("engine name is blank")
    if value[0] in _STAMP_WHITESPACE or value[-1] in _STAMP_WHITESPACE:
        raise InvalidReportedEngineName(
            f"engine name {value!r} has surrounding whitespace"
        )
    forbidden = sorted(set(value) & _ENGINE_NAME_FORBIDDEN_CHARACTERS)
    if forbidden:
        raise InvalidReportedEngineName(
            f"engine name {value!r} contains provenance separator characters "
            f"{forbidden!r}"
        )
    return value


ReportedEngineName: TypeAlias = Annotated[str, AfterValidator(reported_engine_name)]
"""A string admitted by ``reported_engine_name``, checked when a model is built."""


if TYPE_CHECKING:
    from batchalign_core.batchalign_core import (
        StanzaProcessorName as StanzaProcessorName,
    )
else:
    # Read from Rust's closed set at runtime, so the catalog model below
    # validates exactly the names the server's registry parses; mypy reads
    # the extension stub's spelling, held to the same list by
    # ``test_stanza_names_match_rust``.
    StanzaProcessorName = Literal[tuple(batchalign_core.STANZA_PROCESSORS)]
"""A processor a Stanza catalog entry can ship: Rust's ``StanzaProcessor``."""


class StanzaCatalog(BaseModel):
    """The installed Stanza catalog as this worker read it.

    ``languages`` maps each ISO 639-3 code the catalog can serve to the
    processors of the entry serving it. The Rust server parses the keys and
    processor names when it deserializes the report
    (``StanzaCatalogReport`` in ``batchalign-types``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["catalog"] = "catalog"
    languages: dict[str, list[StanzaProcessorName]]


StanzaReportReason: TypeAlias = Annotated[
    str, StringConstraints(min_length=1, pattern=r"\S", strict=True)
]
"""A worker's statement of why it reports no catalog: never blank, as
``StanzaReportReason`` refuses in ``batchalign-types``."""


class StanzaNotInstalledReport(BaseModel):
    """The ``stanza`` package does not import in this worker.

    ``import_error`` says why, as the worker saw it, so an operator reading
    the server's refusal knows what to fix.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["not_installed"] = "not_installed"
    import_error: StanzaReportReason


class StanzaCatalogFailed(BaseModel):
    """Stanza imported, but this worker could not build its catalog report.

    ``reason`` is the failure as the worker saw it, so the server can say why
    it has no registry instead of reading an empty catalog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["failed"] = "failed"
    reason: StanzaReportReason


StanzaCatalogReport: TypeAlias = Annotated[
    StanzaCatalog | StanzaNotInstalledReport | StanzaCatalogFailed,
    Field(discriminator="kind"),
]
"""A worker's account of its Stanza catalog: one of three outcomes by type."""


class CapabilitiesResponse(BaseModel):
    """Response body for capabilities operation."""

    commands: list[str]
    free_threaded: bool
    infer_tasks: list[InferTask]
    engine_versions: dict[InferTask, ReportedEngineName | None]
    """Engine identity per advertised task, one entry each.

    ``None`` means the worker supports the task but cannot name the engine
    behind it. It is never a guessed name or ``"unknown"``: the Rust capability
    gate admits ``None`` as unreported, and no cache namespace or provenance
    field can be built from it. A name is checked by ``reported_engine_name``
    when this response is built, so a bad one fails the capabilities operation
    here rather than being refused by the Rust gate.
    """
    stanza_catalog: StanzaCatalogReport
    """Required: a worker that cannot report a catalog says why."""


class InferRequest(BaseModel):
    """Request body for infer operation -- pure inference, no CHAT."""

    task: InferTask
    lang: LanguageCode
    payload: WorkerJSONValue = Field(default_factory=dict)


FiniteNonNegativeFloat: TypeAlias = Annotated[FiniteFloat, Field(ge=0)]
"""Finite floating-point value constrained to be non-negative: the one spelling
of an elapsed time or a position in seconds, shared with ``_types_v2``."""


def _item_clock() -> float:
    """The clock an item's own work is timed by: ``time.monotonic``. Named so
    a test can replace it for item timing alone, never for the process."""
    return time.monotonic()


class ItemProduced(BaseModel):
    """One inference item's own successful result payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["produced"] = "produced"
    result: WorkerJSONValue

    @field_validator("result")
    @classmethod
    def _result_is_not_null(cls, value: WorkerJSONValue) -> WorkerJSONValue:
        # A produced item with a null result is an item with no result, which
        # the outcome union rules out; Rust refuses it too (`ItemPayload`).
        if value is None:
            raise ValueError("a produced item's result must not be null")
        return value


class ItemFailed(BaseModel):
    """One inference item's own failure, reported against that item alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["failed"] = "failed"
    error: str


ItemOutcome: TypeAlias = Annotated[
    ItemProduced | ItemFailed, Field(discriminator="kind")
]
"""What one item's work produced: a result or a failure, never both or neither.

On the wire ``{"kind": "produced", "result": ...}`` or ``{"kind": "failed",
"error": "..."}``, mirroring Rust ``batchalign_types::worker::ItemOutcome``.
"""


class InferResponse(BaseModel):
    """Response body for infer operation, one per item of a batch.

    ``outcome`` is the item's result or its failure, one of the two by type.
    ``elapsed_s`` is REQUIRED and has no default: every construction says
    whether this item's own work was timed (a non-negative number of seconds)
    or never ran (``None``, written as ``null``, which Rust reads as
    ``ItemElapsed::NotExecuted``). Build one through :meth:`timed` or
    :meth:`unexecuted`, which make that choice by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: ItemOutcome
    elapsed_s: FiniteNonNegativeFloat | None

    @property
    def result(self) -> WorkerJSONValue | None:
        """The result payload, when the item produced one."""
        match self.outcome:
            case ItemProduced(result=result):
                return result
            case ItemFailed():
                return None
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def error(self) -> str | None:
        """The failure, when the item failed."""
        match self.outcome:
            case ItemFailed(error=error):
                return error
            case ItemProduced():
                return None
            case _ as unreachable:
                assert_never(unreachable)

    @classmethod
    def timed(cls, work: Callable[[], ItemOutcome]) -> InferResponse:
        """Run ONE item's own work and report exactly that item's time.

        Takes the work rather than a duration or a start instant, so a batch
        total, or a clock read before some other item's work, has no
        signature to travel through.
        """
        started_at = _item_clock()
        outcome = work()
        return cls(outcome=outcome, elapsed_s=_item_clock() - started_at)

    @classmethod
    def unexecuted(cls, outcome: ItemOutcome) -> InferResponse:
        """An item no work ran for, so its elapsed time is absent, not zero."""
        return cls(outcome=outcome, elapsed_s=None)


class BatchInferRequest(BaseModel):
    """Request body for batch_infer operation -- multiple inference items."""

    task: InferTask
    lang: LanguageCode
    items: list[WorkerJSONValue] = Field(default_factory=list)
    mwt: dict[str, list[str]] = Field(default_factory=dict)
    retokenize: bool = False


class BatchInferResponse(BaseModel):
    """Response body for batch_infer operation."""

    results: list[InferResponse]


BatchInferHandler = Callable[[BatchInferRequest], BatchInferResponse]
"""Callable signature for one fully wired batch-infer task handler."""


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class _WorkerState:
    """Mutable state for the worker process.

    Model objects are loaded directly at worker startup. The stanza_*
    fields hold loaded Stanza pipelines for morphosyntax/utseg inference.
    The fa_* fields hold loaded forced alignment models.
    """

    def __init__(self) -> None:
        self.command: CommandName = ""
        self.lang: LanguageCode = ""
        self.started_at: float = time.monotonic()
        self.test_echo: bool = False
        self.test_delay_ms: TimestampMs = 0
        self.ready: bool = False
        self.bootstrap: WorkerBootstrapRuntime | None = None

        # Stanza models for morphosyntax. A BOUNDED cache rather than a dict:
        # pipelines are hundreds of megabytes each and this process is
        # long-lived, so residency needs a ceiling. `None` still means "no
        # Stanza bootstrap ran in this worker", which is a different fact from
        # "the cache is empty" and is what the infer host refuses on.
        # The tokenizer contexts live INSIDE the cache, beside the pipelines
        # they belong to; they used to be a second dict keyed alike, which
        # nothing kept in step.
        self.stanza_pipelines: StanzaPipelineCache | None = None
        # There is no process-wide Stanza lock: every pipeline in the cache
        # carries its own guard (`LoadedPipeline.calling`), so one pipeline is
        # never called by two threads at once, in any interpreter mode.

        self.utterance_boundary_model: BertUtteranceModel | None = None
        self.utterance_model_name: str = ""

        # Translation: ONE record naming the backend, its reported engine
        # identity and the callable that runs it, set together by the loader
        # that loaded them. It used to be three parallel fields, and nothing
        # stopped a caller from reading an engine name that belonged to a
        # backend loaded before the current one.
        self.translation: LoadedTranslation | None = None

        # FA models (typed handles from load_whisper_fa / load_wave2vec_fa)
        self.whisper_fa_model: WhisperFAHandle | None = None
        self.wave2vec_fa_model: Wave2VecFAHandle | None = None
        # The Qwen3 aligner arrives as a whole host rather than a bare handle,
        # because the language it was loaded for is part of what it is: the
        # label is fixed at load time and never travels on the FA wire.
        self.qwen_fa_host: QwenFaHost | None = None
        # The identity of the loaded forced-alignment model, set by the loader
        # that loaded it; None until an FA engine has loaded. It is the FA
        # cache namespace on the Rust side, so it is never guessed: the old
        # empty-string default fell back to the default engine's enum value,
        # naming a model this process had not loaded.
        self.fa_model_name: str | None = None

        # ASR model
        self.whisper_asr_model: WhisperASRHandle | None = None
        self.rev_api_key: RevAiApiKey | None = None
        self.asr_engine: AsrEngine = AsrEngine.WHISPER

        # The models Rust told this worker to load, when it named an ASR
        # engine at all. `None` is a REAL case, not a missing value: the
        # control plane fills in only forced alignment for a worker's
        # preloaded tasks, so an align job's GPU worker preloads ASR without
        # naming an engine (see `EngineSelection::for_target`, which explains
        # why injecting an ASR default there would overrule a better-informed
        # per-language choice). Such a worker loads its own default and
        # reports an unpinned identity.
        self.asr_pinned_models: AsrRequestedModelsV2 | None = None

        # What the ASR engine ACTUALLY loaded, recorded by the loader that
        # loaded it and attached to every ASR result this worker returns. The
        # bridge checks it against the pin above, so a model that moved is
        # refused by name instead of silently changing transcripts.
        self.asr_model_identity: AsrModelIdentityV2 | None = None

        # Tracks which tasks have been loaded (for LazyProfile mode).
        # Empty at startup in lazy mode; populated as ensure_task_loaded() is
        # called. In eager (Profile/Task) mode, populated during bootstrap.
        self.loaded_tasks: set[str] = set()

        # Serializes on-demand model loading in LazyProfile mode. Prevents
        # concurrent ensure_task calls from loading the same model twice.
        self.loading_lock: threading.Lock = threading.Lock()

        # Request-time batch inference handlers registered during bootstrap.
        # Dynamic tasks such as translation, FA, and ASR should resolve their
        # engine-specific routing here instead of branching on raw state in the
        # hot request path.
        self.batch_infer_handlers: dict[InferTask, BatchInferHandler] = {}

        # Transient progress callback set by the V2 execution path during a
        # long-running morphosyntax batch.  Read by the handler to emit
        # ProgressEventV2 lines.  None when no V2 request is in flight.
        self.active_progress_callback: Callable[[int, int], None] | None = None

    def stanza_version(self) -> str | None:
        """The installed Stanza version, or ``None`` when none can be named.

        The ONE accessor for this fact. It used to be read four ways: a
        field written after a pipeline loaded, a second field written by the
        utseg loader, a resolver that preferred the first field and fell back
        to the package, and a capability-table builder that defaulted to
        ``"unknown"``. All four read ``stanza.__version__`` in the end, and a
        process cannot change its installed Stanza, so the copies could only
        disagree by being stale.

        ``None`` means no importable Stanza or no version string; a
        placeholder such as ``"unknown"`` is never returned, because it would
        travel on as a version into provenance. Not cached: importing an
        already-imported module is a dictionary lookup, and caching would hide
        a test's substituted module.
        """
        try:
            import stanza
        except (ImportError, ModuleNotFoundError):
            return None
        version = getattr(stanza, "__version__", None)
        if not isinstance(version, str) or not version:
            return None
        # Admitted, not trusted: it is reported as a model identity, so a
        # version string with a separator in it raises here, typed.
        return reported_engine_name(version)

    def stanza_engine(self) -> str | None:
        """The Stanza engine identity, ``stanza-<version>``, or ``None``.

        Shared by the capability report and the coreference producer, so the
        two spell one identity one way.
        """
        version = self.stanza_version()
        return None if version is None else f"stanza-{version}"

    def register_batch_infer_handler(
        self,
        task: InferTask,
        handler: BatchInferHandler,
    ) -> None:
        """Install the concrete batch-infer handler for one loaded task."""
        self.batch_infer_handlers[task] = handler

    def batch_infer_handler(
        self,
        task: InferTask,
    ) -> BatchInferHandler | None:
        """Return the bootstrap-installed batch handler for one task."""
        return self.batch_infer_handlers.get(task)

    def clear_batch_infer_handlers(self) -> None:
        """Clear any previously registered task handlers.

        Worker startup is a one-command bootstrap boundary. Clearing the
        registry before reconfiguration keeps test setup and future worker
        reinitialization paths from accidentally reusing stale engine wiring.
        """
        self.batch_infer_handlers.clear()


_state = _WorkerState()


def sleep_test_delay() -> None:
    """Sleep the test-echo delay (``--test-delay-ms``), standing in for model
    work; the one place every test-echo path waits."""
    if _state.test_delay_ms > 0:
        time.sleep(_state.test_delay_ms / 1000.0)
