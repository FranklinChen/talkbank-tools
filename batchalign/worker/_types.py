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
from typing import (
    TYPE_CHECKING,
    Annotated,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    assert_never,
    cast,
)

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
from batchalign.worker._pipeline_cache import StanzaPipelineCache

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


class TaskLoad(Enum):
    """Every load one worker can complete: the closed set ``ensure_task`` names.

    The Python half of Rust's ``TaskLoad`` (``worker/pool/task_load.rs``). A
    task's engine is fixed by the worker key, so it is not part of the load,
    with one exception: the speaker model is not worker identity (one GPU
    worker serves every speaker model), so each speaker model is its own load,
    and pyannoteAI, which is hosted and builds nothing, is a load of its own
    rather than an absent model.
    """

    MORPHOSYNTAX = "morphosyntax"
    UTSEG = "utseg"
    TRANSLATE = "translate"
    COREF = "coref"
    FA = "fa"
    ASR = "asr"
    OPENSMILE = "opensmile"
    AVQI = "avqi"
    WORD_SEGMENTATION = "word_segmentation"
    SPEAKER_HOSTED = "speaker:pyannote_ai"
    SPEAKER_LOCAL_PYANNOTE = "speaker:pyannote"
    SPEAKER_EMBEDDING = "speaker:embedding:pyannote"


def task_load(task: InferTask, engine_overrides: Mapping[str, str]) -> TaskLoad:
    """The load ``ensure_task`` is asked for, from its task and its overrides.

    The one parse from the wire to a :class:`TaskLoad`. Only the speaker task
    reads its overrides: ``speaker`` names the local diarization engine and
    ``speaker_embedding`` the embedding model, at most one of them (the plan
    has one speaker slot), and neither for a hosted engine. Any other
    combination is a control plane defect and is refused here, before
    anything loads, so a refused load is never recorded.
    """
    match task:
        case InferTask.MORPHOSYNTAX:
            return TaskLoad.MORPHOSYNTAX
        case InferTask.UTSEG:
            return TaskLoad.UTSEG
        case InferTask.TRANSLATE:
            return TaskLoad.TRANSLATE
        case InferTask.COREF:
            return TaskLoad.COREF
        case InferTask.FA:
            return TaskLoad.FA
        case InferTask.ASR:
            return TaskLoad.ASR
        case InferTask.OPENSMILE:
            return TaskLoad.OPENSMILE
        case InferTask.AVQI:
            return TaskLoad.AVQI
        case InferTask.WORD_SEGMENTATION:
            return TaskLoad.WORD_SEGMENTATION
        case InferTask.SPEAKER:
            pass
        case _:
            assert_never(task)
    diarization = engine_overrides.get("speaker")
    embedding = engine_overrides.get("speaker_embedding")
    match (diarization, embedding):
        case (None, None):
            return TaskLoad.SPEAKER_HOSTED
        case ("pyannote", None):
            return TaskLoad.SPEAKER_LOCAL_PYANNOTE
        case (None, "pyannote"):
            return TaskLoad.SPEAKER_EMBEDDING
        case _:
            raise ValueError(
                "the speaker task load takes at most one speaker model: "
                "'speaker' = 'pyannote' or 'speaker_embedding' = 'pyannote'; got "
                f"speaker={diarization!r} speaker_embedding={embedding!r}"
            )


Host = TypeVar("Host")
"""The loaded host one :class:`LoadKey` stands for. Invariant on purpose."""


# A classic invariant TypeVar, not PEP 695 syntax: the parameter is a phantom
# (no field uses it), and mypy infers an unused PEP 695 parameter as
# covariant, under which `record(ASR_LOAD, <an FA host>)` type-checks with the
# host solved as their join. Invariance is what makes the pairing a check.
@dataclass(frozen=True, slots=True)
class LoadKey(Generic[Host]):  # noqa: UP046
    """One :class:`TaskLoad`, typed by the host its load builds.

    The type parameter is the pairing: :meth:`TaskLoadRegistry.record` takes a
    ``LoadKey[H]`` with an ``H``, and :meth:`TaskLoadRegistry.get` returns an
    ``H``, so recording the wrong host under a load, or reading a load as the
    wrong host, is a type error. The keys are the module constants in
    ``batchalign.worker._loads``, one per load.
    """

    load: TaskLoad


def not_loaded_message(load: TaskLoad) -> str:
    """The one wording of "this load has not completed here", for every
    route that refuses a request whose load is absent."""
    return (
        f"the {load.value} load has not completed on this worker: the control "
        "plane prepares a worker (ensure_task) before its first request for a load"
    )


class LoadAlreadyRecorded(RuntimeError):
    """A second record of one load: a load inserts exactly one entry."""


class TaskLoadRegistry:
    """What this worker's loads built, by load: the Python mirror of Rust's
    ``TaskLoads``.

    Empty at startup in every mode. ``ensure_task_loaded`` is the only writer
    (:meth:`record`, the transition from not loaded to loaded), and each entry
    is the host the load built, holding the model itself, never a copy. A
    request resolves its load once, at its boundary (:meth:`get`): absent is
    the named ``model_unavailable`` refusal, present is the host its handler
    receives.
    """

    __slots__ = ("_hosts",)

    def __init__(self) -> None:
        # Values are the host each key's type parameter names; `record` is the
        # only writer and takes the key and host together, which is what makes
        # the cast in `get` sound.
        self._hosts: dict[TaskLoad, object] = {}

    def record(self, key: LoadKey[Host], host: Host) -> None:
        """Record one completed load. A second record of a load is refused."""
        if key.load in self._hosts:
            raise LoadAlreadyRecorded(f"the {key.load.value} load is already recorded")
        self._hosts[key.load] = host

    def get(self, key: LoadKey[Host]) -> Host | None:
        """The host ``key``'s load built, or ``None`` when it has not loaded."""
        return cast("Host | None", self._hosts.get(key.load))

    def has(self, load: TaskLoad) -> bool:
        """Whether ``load`` completed on this worker."""
        return load in self._hosts

    def completed(self) -> frozenset[TaskLoad]:
        """Every load this worker completed."""
        return frozenset(self._hosts)


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
    """Which worker-hosted ASR engine an ASR load builds, as the control plane
    names it in the ``asr`` override.

    Rev.AI and whisper.cpp are not here: they run in the Rust control plane,
    which never names them to a worker (``EngineSelection`` drops an engine
    with no worker loader), so a worker cannot be asked to load one.
    """

    WHISPER = "whisper"
    WHISPER_HUB = "whisper_hub"
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


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class _WorkerState:
    """Mutable state for the worker process.

    Built at import time, before any load. Every load's model is an entry
    of ``loads``, recorded by the load that built it. The one model store
    here besides it is ``stanza_pipelines``: a bounded cache of per-language
    Stanza pipelines, loaded on a request's first use of a language rather
    than by a load, which the morphosyntax and utseg runners read.
    """

    def __init__(self) -> None:
        self.command: CommandName = ""
        self.lang: LanguageCode = ""
        self.started_at: float = time.monotonic()
        self.test_echo: bool = False
        self.test_delay_ms: TimestampMs = 0
        self.ready: bool = False
        self.bootstrap: WorkerBootstrapRuntime | None = None

        # Stanza models for morphosyntax and utseg. A BOUNDED cache rather than
        # a dict: pipelines are hundreds of megabytes each and this process is
        # long-lived, so residency needs a ceiling. Present from the start
        # (empty), because it holds pipelines loaded per language on first
        # use, not one load's model; the morphosyntax load's host reads it.
        # The tokenizer contexts live INSIDE the cache, beside the pipelines
        # they belong to; they used to be a second dict keyed alike, which
        # nothing kept in step.
        self.stanza_pipelines: StanzaPipelineCache = StanzaPipelineCache()
        # There is no process-wide Stanza lock: every pipeline in the cache
        # carries its own guard (`LoadedPipeline.calling`), so one pipeline is
        # never called by two threads at once, in any interpreter mode.

        # What this worker's loads built, by load. Empty at startup in every
        # mode; `ensure_task_loaded` is the only writer, and every request
        # resolves its load here once, at its boundary.
        self.loads: TaskLoadRegistry = TaskLoadRegistry()

        # Serializes on-demand model loading. Prevents concurrent ensure_task
        # calls from loading the same model twice.
        self.loading_lock: threading.Lock = threading.Lock()

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
        except ImportError, ModuleNotFoundError:
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


_state = _WorkerState()


def sleep_test_delay() -> None:
    """Sleep the test-echo delay (``--test-delay-ms``), standing in for model
    work; the one place every test-echo path waits."""
    if _state.test_delay_ms > 0:
        time.sleep(_state.test_delay_ms / 1000.0)
