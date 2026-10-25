"""Operation handlers for worker metadata and capability reporting."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from batchalign.worker._model_loading.bootstrap import EnsureTaskResponse

from batchalign.worker._types import (
    CapabilitiesResponse,
    HealthResponse,
    InferTask,
    StanzaCatalog,
    StanzaCatalogFailed,
    StanzaCatalogReport,
    StanzaNotInstalledReport,
    _state,
)

L = logging.getLogger("batchalign.worker")


def _health() -> HealthResponse:
    """Health check with worker metadata."""
    is_ready = _state.ready or _state.test_echo
    return HealthResponse(
        status="ok" if is_ready else "loading",
        command=_state.command,
        lang=_state.lang,
        pid=os.getpid(),
        uptime_s=time.monotonic() - _state.started_at,
    )


def _reported_engine(task: InferTask) -> str | None:
    """The engine name this worker's capability report gives for ``task``.

    Only forced alignment's engine is reported. Every FA cache row is read
    under that name before any worker runs, so Rust has to learn it from the
    report. Every other stage names its engine on each result it returns, and
    provenance is built from those results, so its entry is ``None``
    ("supported, no name here"); the Rust capability gate refuses a report that
    names one. For FA, ``None`` means no FA engine has loaded yet, never a
    guessed name. The match is exhaustive, so a new task must decide.
    """
    match task:
        case InferTask.FA:
            # Unchanged byte for byte once loaded: FA cache rows and evidence
            # envelopes are namespaced by exactly this string.
            return _state.fa_model_name
        case (
            InferTask.MORPHOSYNTAX
            | InferTask.COREF
            | InferTask.UTSEG
            | InferTask.TRANSLATE
            | InferTask.ASR
            | InferTask.OPENSMILE
            | InferTask.AVQI
            | InferTask.SPEAKER
            | InferTask.WORD_SEGMENTATION
        ):
            return None
        case _:
            assert_never(task)


def _stanza_catalog_report() -> StanzaCatalogReport:
    """This worker's account of its installed Stanza catalog.

    Every outcome is reported as itself: the catalog, Stanza not installed,
    or the failure that stopped the table build, with its type and message.
    A failure is never reported as an empty catalog, which the server would
    read as "no Stanza" and refuse morphotag without saying why.
    """
    from batchalign.worker._stanza_capabilities import (
        StanzaCapabilityTable,
        StanzaNotInstalled,
        get_cached_capability_table,
    )

    try:
        outcome = get_cached_capability_table()
    except Exception as exc:  # reported, never swallowed: the server names it
        L.error("Stanza capability table build failed: %s", exc)
        return StanzaCatalogFailed(reason=f"{type(exc).__name__}: {exc}")
    match outcome:
        case StanzaNotInstalled():
            return StanzaNotInstalledReport(import_error=outcome.import_error)
        case StanzaCapabilityTable(languages=languages):
            return StanzaCatalog(
                languages={
                    iso3: sorted(cap.processors) for iso3, cap in languages.items()
                }
            )
        case _:
            assert_never(outcome)


def _capabilities() -> CapabilitiesResponse:
    """Report available commands and runtime info.

    Command advertisement is intentionally narrower than infer-task
    advertisement. Server-owned compositions such as ``transcribe`` are not
    exposed as Python commands; Rust synthesizes them from lower-level
    capability signals.
    """
    if _state.test_echo:
        from batchalign.runtime import Cmd2Task

        commands = sorted(set(Cmd2Task.keys()) | {"test-echo"})
        # Advertise all infer tasks so the server's capability gate passes.
        # Echo workers handle any task by echoing back the payload. Like a real
        # worker, only forced alignment names an engine; every other entry is
        # None, which is the only report the gate admits for those tasks.
        all_infer_tasks = list(InferTask)
        echo_versions: dict[InferTask, str | None] = {
            task: "test-echo" if task is InferTask.FA else None
            for task in all_infer_tasks
        }
        return CapabilitiesResponse(
            commands=commands,
            free_threaded=False,
            infer_tasks=all_infer_tasks,
            engine_versions=echo_versions,
            # An echo worker loads no Stanza: it serves no Stanza catalog.
            stanza_catalog=StanzaNotInstalledReport(
                import_error="not attempted: a test-echo worker imports no Stanza"
            ),
        )

    from batchalign.runtime import is_free_threaded

    infer_tasks: list[InferTask] = []
    engine_versions: dict[InferTask, str | None] = {}

    # Infer task probes: map each InferTask to the imports required to prove
    # the system *can* run it.  The probe worker only loads morphotag models,
    # so we must NOT gate on loaded model state, otherwise FA, translate,
    # utseg, ASR, etc. are silently excluded from server capabilities.
    #
    # Advertising a task and naming its engine are separate questions; the
    # engine entry comes from `_reported_engine`, which names only what this
    # process can vouch for and otherwise reports None.
    _INFER_TASK_PROBES: dict[InferTask, tuple[str, ...]] = {
        InferTask.MORPHOSYNTAX: ("stanza",),
        InferTask.UTSEG: ("stanza",),
        InferTask.COREF: ("stanza",),
        InferTask.TRANSLATE: ("googletrans",),
        InferTask.FA: ("torch", "torchaudio"),
        InferTask.OPENSMILE: ("opensmile",),
        InferTask.AVQI: ("parselmouth", "torchaudio"),
        # The segmenter is named per request; taibun is the only one today.
        InferTask.WORD_SEGMENTATION: ("taibun",),
    }

    import importlib

    def _module_importable(module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
        except ImportError, ModuleNotFoundError:
            return False
        return True

    for task, deps in _INFER_TASK_PROBES.items():
        if all(_module_importable(dep) for dep in deps):
            infer_tasks.append(task)
            engine_versions[task] = _reported_engine(task)

    # Speaker is advertised when any diarization package imports. Which one a
    # job runs is chosen per request, so this probe cannot name it.
    if _module_importable("pyannote.audio") or _module_importable(
        "nemo.collections.asr"
    ):
        infer_tasks.append(InferTask.SPEAKER)
        engine_versions[InferTask.SPEAKER] = _reported_engine(InferTask.SPEAKER)

    # ASR is special now: the server can satisfy ASR through either
    # Python-hosted local engines (for example Whisper) or the Rust-owned
    # Rev.AI path when the control plane has already injected credentials.
    from batchalign.worker._model_loading.asr import resolve_injected_revai_api_key

    has_revai_key = bool(
        (_state.bootstrap and _state.bootstrap.revai_api_key)
        or resolve_injected_revai_api_key()
    )

    has_whisper = _module_importable("whisper")

    if has_whisper or has_revai_key:
        infer_tasks.append(InferTask.ASR)
        engine_versions[InferTask.ASR] = _reported_engine(InferTask.ASR)

    stanza_catalog = _stanza_catalog_report()

    return CapabilitiesResponse(
        commands=[],
        free_threaded=is_free_threaded(),
        infer_tasks=infer_tasks,
        engine_versions=engine_versions,
        stanza_catalog=stanza_catalog,
    )


# ---------------------------------------------------------------------------
# ensure_task: on-demand model loading for LazyProfile workers
# ---------------------------------------------------------------------------


def _ensure_task(
    task: str, engine_overrides: dict[str, str] | None
) -> EnsureTaskResponse:
    """Load one task's models on demand.

    Called by the Rust control plane before dispatching work to a LazyProfile
    worker. Idempotent: if the task is already loaded, returns immediately.
    Returns the Pydantic ``EnsureTaskResponse`` directly for JSON serialization.
    """
    from batchalign.worker._model_loading import ensure_task_loaded

    return ensure_task_loaded(task, engine_overrides)
