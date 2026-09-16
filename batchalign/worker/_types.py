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
from typing import TYPE_CHECKING, Annotated, TypeAlias

from pydantic import AfterValidator, BaseModel, Field

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


class WorkerProfile(str, Enum):
    """Worker profile grouping related InferTasks into fewer processes.

    Instead of spawning one worker per InferTask, profiles group related tasks
    to share loaded models within a single process:

    - GPU: ASR, FA, Speaker, GPU-bound models, concurrent via ThreadPoolExecutor
    - STANZA: Morphosyntax, Utseg, Coref, Stanza NLP processors
    - IO: Translate, OpenSMILE, AVQI, lightweight API/library calls
    """

    GPU = "gpu"
    STANZA = "stanza"
    IO = "io"


PROFILE_TASKS: dict[WorkerProfile, set[str]] = {
    WorkerProfile.GPU: {"asr", "fa", "speaker"},
    WorkerProfile.STANZA: {"morphosyntax", "utseg", "coref"},
    WorkerProfile.IO: {"translate", "opensmile", "avqi"},
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


class StanzaLanguageProcessors(BaseModel):
    """Per-language Stanza processor availability."""

    alpha2: str
    processors: list[str]


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
    stanza_capabilities: dict[str, StanzaLanguageProcessors] = Field(
        default_factory=dict
    )


class InferRequest(BaseModel):
    """Request body for infer operation -- pure inference, no CHAT."""

    task: InferTask
    lang: LanguageCode
    payload: WorkerJSONValue = Field(default_factory=dict)


class InferResponse(BaseModel):
    """Response body for infer operation."""

    result: WorkerJSONValue | None = None
    error: str | None = None
    elapsed_s: float = 0.0


class BatchInferRequest(BaseModel):
    """Request body for batch_infer operation -- multiple inference items."""

    task: InferTask
    lang: LanguageCode
    items: list[WorkerJSONValue] = Field(default_factory=list)
    mwt: dict[str, list[str]] = Field(default_factory=dict)
    retokenize: bool = False
    # Operator opt-in to the legacy Stanza constituency-parser fallback
    # for utseg when no language-specific TalkBank BERT model is
    # configured. Surfaced as `--utseg-fallback-stanza` on every
    # utseg-invoking CLI subcommand (transcribe, transcribe_s, utseg).
    # Default-refuse mirrors the WhisperHubModelNotFoundError pattern:
    # silent substitution between models is the foot-gun this flag
    # exists to prevent.
    allow_stanza_fallback: bool = False


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
        # ONE lock for the life of the process, created here and never
        # replaced: a handler captures it before inference, and a mid-batch
        # reload (the bounded cache can evict a language and bring it back)
        # must not hand a second handler a different lock, or two threads enter
        # Stanza at once.
        self.stanza_nlp_lock: threading.Lock = threading.Lock()

        # Utseg config builder (callable from StanzaUtteranceEngine). Its
        # presence is also the "utseg has loaded" fact the capability report
        # reads; there is no separate version field to keep in step with it.
        self.utseg_config_builder: (
            Callable[[list[str]], tuple[list[str], dict[str, dict[str, str | bool]]]]
            | None
        ) = None
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
