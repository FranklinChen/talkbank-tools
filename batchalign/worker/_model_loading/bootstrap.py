"""Top-level worker bootstrap orchestration.

This module is the narrow control-plane entry for worker startup. Workers are
bootstrapped for one infer task at a time; Rust owns any higher-level command
composition.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Literal, assert_never

from pydantic import BaseModel

from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._avqi_v2 import avqi_task
from batchalign.worker._infer_hosts import (
    build_morphosyntax_runner,
    build_translate_runner,
    build_utseg_runner,
)
from batchalign.worker._loads import (
    ASR_LOAD,
    AVQI_LOAD,
    COREF_LOAD,
    FA_LOAD,
    MORPHOSYNTAX_LOAD,
    OPENSMILE_LOAD,
    SPEAKER_EMBEDDING_LOAD,
    SPEAKER_HOSTED_LOAD,
    SPEAKER_LOCAL_PYANNOTE_LOAD,
    TRANSLATE_LOAD,
    UTSEG_LOAD,
    WORD_SEGMENTATION_LOAD,
    LoadedMorphosyntax,
    LoadedTranslate,
    LoadedUtseg,
    LoadedWordSegmentation,
)
from batchalign.worker._model_loading.asr import load_asr_engine
from batchalign.worker._model_loading.coreference import load_coref_task
from batchalign.worker._model_loading.forced_alignment import load_fa_engine
from batchalign.worker._model_loading.speaker import (
    hosted_speaker_task,
    load_local_pyannote_task,
    load_speaker_embedding_task,
)
from batchalign.worker._model_loading.translation import load_translation_engine
from batchalign.worker._model_loading.utterance import load_utterance_model
from batchalign.worker._opensmile_v2 import opensmile_task
from batchalign.worker._progress import LoadingTask, emit_phase_event
from batchalign.worker._stanza_loading import load_stanza_models, prepare_utseg_stanza
from batchalign.worker._types import (
    FiniteNonNegativeFloat,
    InferTask,
    TaskLoad,
    TaskLoadRegistry,
    WorkerBootstrapRuntime,
    WorkerProfile,
    _state,
    task_load,
)
from batchalign.worker._types_v2 import PhaseEdgeV2

L = logging.getLogger("batchalign.worker")


def start_worker(bootstrap: WorkerBootstrapRuntime) -> None:
    """Start a worker with no models loaded.

    No task loads a model at startup, in any bootstrap mode: each model loads
    in its task's ``ensure_task``, which the Rust pool sends before the
    worker's first request for that task (the prepare step), under the
    loading stall window that download growth re-arms.
    """
    match (bootstrap.profile, bootstrap.task):
        case (WorkerProfile() as profile, None):
            label = f"profile:{profile.value}"
        case (None, InferTask() as task):
            label = f"infer:{task.value}"
        case _:
            raise ValueError(
                "a worker starts for one profile or one task, got "
                f"profile={bootstrap.profile!r} task={bootstrap.task!r}"
            )
    _state.bootstrap = bootstrap
    _state.command = label
    _state.lang = bootstrap.lang
    _state.ready = True
    L.info(
        "Worker ready, no models loaded: target=%s lang=%s pid=%d",
        label,
        bootstrap.lang,
        os.getpid(),
    )


class EnsureTaskResponse(BaseModel):
    """Response for the ``ensure_task`` IPC operation.

    Returned by :func:`ensure_task_loaded` and serialized directly to JSON
    over the worker protocol.
    """

    status: Literal["loaded", "already_loaded"]
    task: str
    elapsed_s: FiniteNonNegativeFloat
    """Seconds the worker spent answering, as it measured them.

    ``loaded``: the model load. ``already_loaded``: the check, including any
    wait for another thread's load of the same task. Never a constant.
    """


def ensure_task_loaded(
    task: str,
    engine_overrides: dict[str, str] | None = None,
) -> EnsureTaskResponse:
    """Load one task's model on demand and record its host. Idempotent and
    thread-safe.

    The prepare step: the Rust pool sends ``ensure_task`` before a worker's
    first request for a load, in every bootstrap mode, and waits under the
    ensure-task loading stall window that download growth re-arms. This is
    the transition from not loaded to loaded, and the only writer of
    ``_state.loads``: a completed load records exactly one host.

    ``engine_overrides`` are the ones Rust's ``TaskLoad`` carries, merged over
    the worker's bootstrap overrides; the load and its key are both derived
    from the merged set, so they cannot disagree.

    Raises ``ValueError`` if *task* is not a recognized ``InferTask`` value,
    or if the speaker overrides name no load this worker can build.
    """
    started_at = time.monotonic()

    try:
        infer_task = InferTask(task)
    except ValueError as error:
        raise ValueError(
            f"Unknown ensure_task name {task!r}; "
            f"valid names: {sorted(t.value for t in InferTask)}"
        ) from error

    if _state.test_echo:
        # A test-echo worker hosts no model for any task, and it is prepared
        # like any other worker, so it answers at once without loading.
        return EnsureTaskResponse(
            status="already_loaded",
            task=task,
            elapsed_s=time.monotonic() - started_at,
        )

    bootstrap = _state.bootstrap
    if bootstrap is None:
        raise RuntimeError("ensure_task_loaded called before worker bootstrap")
    merged_overrides = dict(bootstrap.engine_overrides)
    if engine_overrides:
        merged_overrides.update(engine_overrides)
    load = task_load(infer_task, merged_overrides)

    if _state.loads.has(load):
        return EnsureTaskResponse(
            status="already_loaded",
            task=task,
            elapsed_s=time.monotonic() - started_at,
        )

    with _state.loading_lock:
        # Double-check after acquiring lock (another thread may have loaded it).
        if _state.loads.has(load):
            return EnsureTaskResponse(
                status="already_loaded",
                task=task,
                elapsed_s=time.monotonic() - started_at,
            )

        task_bootstrap = WorkerBootstrapRuntime(
            task=bootstrap.task,
            profile=bootstrap.profile,
            lang=bootstrap.lang,
            num_speakers=bootstrap.num_speakers,
            engine_overrides=merged_overrides,
            device_policy=bootstrap.device_policy,
            revai_api_key=bootstrap.revai_api_key,
        )

        t0 = time.monotonic()
        L.info(
            "Loading on demand: load=%s engine_overrides=%s pid=%d",
            load.value,
            merged_overrides,
            os.getpid(),
        )
        phase = LoadingTask(task=task, lang=bootstrap.lang)
        emit_phase_event(
            phase, PhaseEdgeV2.BEGIN, f"Loading {task} ({bootstrap.lang})…"
        )
        record_load(load, task_bootstrap, _state.loads)
        if not _state.loads.has(load):
            # An arm recorded its host under another load's key, which the
            # key's type parameter cannot see (each arm picks its key by
            # hand). Refused here, before a request could miss it.
            raise RuntimeError(f"the {load.value} load recorded no {load.value} entry")
        elapsed = time.monotonic() - t0
        emit_phase_event(
            phase,
            PhaseEdgeV2.COMPLETE,
            f"Loaded {task} ({bootstrap.lang}) in {int(elapsed * 1000)} ms",
        )
        L.info("Loaded: load=%s elapsed=%.1fs pid=%d", load.value, elapsed, os.getpid())
        return EnsureTaskResponse(status="loaded", task=task, elapsed_s=elapsed)


def record_load(
    load: TaskLoad, bootstrap: WorkerBootstrapRuntime, loads: TaskLoadRegistry
) -> None:
    """Build ``load``'s host and record it: one load, one entry.

    Exhaustive over :class:`TaskLoad`, so a new load has to say what it
    builds. Each arm records under the key typed by the host it builds, so a
    host recorded under the wrong load is a type error.

    ``coref`` prepares the coreference model (``load_coref_task``), whose host
    holds no pipeline. ``opensmile``, ``avqi`` and ``word_segmentation`` have
    no model; their hosts are the analysis functions. pyannoteAI is hosted:
    its host calls the API and loads nothing.
    """
    match load:
        case TaskLoad.MORPHOSYNTAX:
            # The IPC wire format is `--lang STRING`, so Python sees a plain
            # string. A capability-probe or per-file worker is spawned with
            # the non-ISO `auto` or `per-file`, for which no Stanza model can
            # load; its pipelines load per language on first use, through the
            # cache the host reads.
            if bootstrap.lang in ("auto", "per-file"):
                L.info(
                    "Skipping eager Stanza load for non-ISO lang '%s'; "
                    "language-specific pipelines will be loaded per-file",
                    bootstrap.lang,
                )
            else:
                load_stanza_models(bootstrap.lang)
            loads.record(
                MORPHOSYNTAX_LOAD,
                LoadedMorphosyntax(
                    runner=build_morphosyntax_runner(_state.stanza_pipelines)
                ),
            )
        case TaskLoad.UTSEG:
            loads.record(UTSEG_LOAD, _load_utseg(bootstrap))
        case TaskLoad.TRANSLATE:
            loads.record(
                TRANSLATE_LOAD,
                LoadedTranslate(
                    runner=build_translate_runner(load_translation_engine(bootstrap))
                ),
            )
        case TaskLoad.COREF:
            loads.record(COREF_LOAD, load_coref_task())
        case TaskLoad.FA:
            loads.record(FA_LOAD, load_fa_engine(bootstrap))
        case TaskLoad.ASR:
            loads.record(ASR_LOAD, load_asr_engine(bootstrap))
        case TaskLoad.OPENSMILE:
            loads.record(OPENSMILE_LOAD, opensmile_task())
        case TaskLoad.AVQI:
            loads.record(AVQI_LOAD, avqi_task())
        case TaskLoad.WORD_SEGMENTATION:
            from batchalign.inference.word_segmentation import segment_batch

            loads.record(
                WORD_SEGMENTATION_LOAD, LoadedWordSegmentation(runner=segment_batch)
            )
        case TaskLoad.SPEAKER_HOSTED:
            loads.record(SPEAKER_HOSTED_LOAD, hosted_speaker_task())
        case TaskLoad.SPEAKER_LOCAL_PYANNOTE:
            loads.record(SPEAKER_LOCAL_PYANNOTE_LOAD, load_local_pyannote_task())
        case TaskLoad.SPEAKER_EMBEDDING:
            loads.record(SPEAKER_EMBEDDING_LOAD, load_speaker_embedding_task())
        case _:
            assert_never(load)


def _load_utseg(bootstrap: WorkerBootstrapRuntime) -> LoadedUtseg:
    """The utseg load: the language's boundary model, when it has a pin.

    A worker with no resolved language has no boundary-model pin (Rust
    injects the pin for a resolved language only), so it loads none; it loads
    a request language's Stanza pipeline on first use. Rust dispatches utseg
    on the route's resolved language, so a boundary-model request reaches a
    worker that loaded its pin.
    """
    from batchalign.inference.utseg import BoundaryModel, NoPinnedBoundaryModel

    boundary_model: BoundaryModel = NoPinnedBoundaryModel()
    if bootstrap.lang in ("auto", "per-file"):
        L.info(
            "No utseg models for non-ISO lang '%s'; a request language's "
            "Stanza pipeline loads on its first use",
            bootstrap.lang,
        )
    else:
        prepare_utseg_stanza(bootstrap.lang)
        boundary_model = load_utterance_model(
            bootstrap.lang, bootstrap.engine_overrides
        )
    return LoadedUtseg(runner=build_utseg_runner(boundary_model))


def enable_test_echo(target_label: str, lang: LanguageCode) -> None:
    """Configure the worker to echo requests without loading models."""
    _state.test_echo = True
    _state.command = target_label or "test-echo"
    _state.lang = lang
    _state.ready = True


def parse_engine_overrides(raw: str) -> dict[str, str] | None:
    """Parse the engine-override JSON payload from CLI args.

    The Rust caller serializes a ``BTreeMap<String, String>``; we validate
    the shape here at the IPC boundary.
    """
    if not raw:
        return None
    parsed: object = json.loads(raw)
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(
            f"engine overrides must be a flat {{str: str}} object, got: {raw!r}"
        )
    return parsed
