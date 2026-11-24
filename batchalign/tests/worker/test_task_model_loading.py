"""No worker loads a model at startup; the speaker task loads on first use.

A worker starts with nothing loaded, in every bootstrap mode. The speaker
task's ``ensure_task`` carries the speaker model to build (it is not worker
identity), builds exactly that model once, and records the host whose runner
closes over it in the worker's load registry; a second speaker model on the
same worker is a second load. A request for a model no load built finds no
entry and is refused, never loaded.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from batchalign.inference.pyannote_local import LoadedPyannotePipeline
from batchalign.tests._loaded_hosts import TEST_LOCAL_SPEAKER_GRAPH
from batchalign.worker._loads import (
    FA_LOAD,
    SPEAKER_EMBEDDING_LOAD,
    SPEAKER_HOSTED_LOAD,
    SPEAKER_LOCAL_PYANNOTE_LOAD,
    LoadedWordFa,
)
from batchalign.worker._model_loading.bootstrap import (
    ensure_task_loaded,
    start_worker,
)
from batchalign.worker._types import (
    InferTask,
    LoadAlreadyRecorded,
    ModelStartup,
    ModelTarget,
    ProfileTarget,
    TaskLoad,
    TaskLoadRegistry,
    TaskTarget,
    WorkerBootstrapRuntime,
    WorkerProfile,
    _state,
    task_load,
)
from batchalign.worker._types_v2 import SpeakerBackendV2


@pytest.fixture(autouse=True)
def _fresh_worker() -> Iterator[None]:
    _state.__init__()  # type: ignore[misc]
    yield
    _state.__init__()  # type: ignore[misc]


@pytest.mark.parametrize(
    "target",
    [
        ProfileTarget(WorkerProfile.GPU),
        ProfileTarget(WorkerProfile.STANZA),
        ProfileTarget(WorkerProfile.IO),
        TaskTarget(InferTask.FA),
        TaskTarget(InferTask.MORPHOSYNTAX),
    ],
)
def test_a_worker_starts_with_no_models(
    monkeypatch: pytest.MonkeyPatch, target: ModelTarget
) -> None:
    """Guards: Publishing any model target keeps loads empty and its inputs intact.
    Kind: contract
    Not a type: Startup implementation can still invoke a real loader before
        publishing this target, despite carrying the same typed settings.
    State: An isolated worker state and a refusing loader spy expose early loads.
    Concurrency: The fixture restores process state before the next case.
    """

    def refuse(*_args, **_kwargs) -> None:
        raise AssertionError("no model may load at startup")

    monkeypatch.setattr(
        "batchalign.worker._model_loading.bootstrap.record_load", refuse
    )
    bootstrap = WorkerBootstrapRuntime(lang="eng", num_speakers=1)
    start_worker(ModelStartup(target, bootstrap))

    assert _state.ready is True
    assert _state.loads.completed() == frozenset()
    assert _state.bootstrap is bootstrap
    assert _state.command == target.label


def test_two_speaker_models_are_two_loads() -> None:
    local = task_load(InferTask.SPEAKER, {"speaker": "pyannote"})
    embedding = task_load(InferTask.SPEAKER, {"speaker_embedding": "pyannote"})
    hosted = task_load(InferTask.SPEAKER, {})
    assert (local, embedding, hosted) == (
        TaskLoad.SPEAKER_LOCAL_PYANNOTE,
        TaskLoad.SPEAKER_EMBEDDING,
        TaskLoad.SPEAKER_HOSTED,
    )
    # Every other task's engine is fixed by the worker key: one load.
    assert task_load(InferTask.FA, {"fa": "wave2vec"}) == task_load(InferTask.FA, {})


def test_a_load_records_one_entry_and_refuses_a_second() -> None:
    registry = TaskLoadRegistry()
    from batchalign.tests.test_fa_resource_projection import MetadataEncoder, mms_handle

    host = LoadedWordFa.from_wave2vec(mms_handle(MetadataEncoder()))

    registry.record(FA_LOAD, host)

    with pytest.raises(LoadAlreadyRecorded, match="fa"):
        registry.record(FA_LOAD, host)
    assert registry.get(FA_LOAD) is host
    assert registry.completed() == frozenset({TaskLoad.FA})


def test_the_utterance_alias_is_not_a_task() -> None:
    _gpu_worker()

    with pytest.raises(ValueError, match="Unknown ensure_task name"):
        ensure_task_loaded("utterance")
    assert _state.loads.completed() == frozenset()


def _gpu_worker() -> None:
    start_worker(
        ModelStartup(
            ProfileTarget(WorkerProfile.GPU),
            WorkerBootstrapRuntime(lang="eng", num_speakers=1),
        )
    )


def _stub_speaker_pipeline() -> LoadedPyannotePipeline:
    return LoadedPyannotePipeline(
        object(),
        tempfile.TemporaryDirectory(),
        TEST_LOCAL_SPEAKER_GRAPH,
    )


def test_each_speaker_model_is_built_once_by_its_own_load(monkeypatch) -> None:
    """Guards: Each requested speaker model is constructed once by its own worker load.
    Kind: contract
    Not a type: Actual calls and registry reuse depend on the load sequence and observed constructor invocations.
    State: Typed fake loaded pipeline and patched constructors count the independent worker loads.
    Concurrency: Monkeypatch restores constructors after serial calls.
    """
    builds: list[str] = []

    def build_pipeline() -> LoadedPyannotePipeline:
        builds.append("pyannote")
        return _stub_speaker_pipeline()

    def build_embedding() -> object:
        builds.append("embedding")
        return SimpleNamespace(revision=lambda: "0" * 40)

    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.load_local_pyannote_pipeline",
        build_pipeline,
    )
    monkeypatch.setattr(
        "batchalign.inference.speaker_embedding.load_speaker_embedding_model",
        build_embedding,
    )
    _gpu_worker()
    # Nothing recorded: a request resolves no host, and the execute boundary
    # refuses it as model_unavailable.
    assert _state.loads.get(SPEAKER_LOCAL_PYANNOTE_LOAD) is None
    assert _state.loads.get(SPEAKER_EMBEDDING_LOAD) is None

    assert ensure_task_loaded("speaker", {"speaker": "pyannote"}).status == "loaded"
    assert (
        ensure_task_loaded("speaker", {"speaker": "pyannote"}).status
        == "already_loaded"
    )
    # A second speaker model on the same worker is its own load.
    assert (
        ensure_task_loaded("speaker", {"speaker_embedding": "pyannote"}).status
        == "loaded"
    )

    assert builds == ["pyannote", "embedding"]
    assert _state.loads.completed() == frozenset(
        {TaskLoad.SPEAKER_LOCAL_PYANNOTE, TaskLoad.SPEAKER_EMBEDDING}
    )


def test_the_local_runner_runs_on_the_pipeline_its_load_built(monkeypatch) -> None:
    """Guards: The local runner consumes the actual loaded pipeline handle selected by the task loader.
    Kind: contract
    Not a type: The Python callable boundary can substitute a different handle despite typed signatures.
    State: A typed fake loaded handle and patched inference function observe the dispatched object.
    Concurrency: Monkeypatch isolates the loader and worker functions.
    """
    handle = _stub_speaker_pipeline()
    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.load_local_pyannote_pipeline",
        lambda: handle,
    )
    seen: list[object] = []

    def fake_local(audio, sample_rate_hz, num_speakers, *, pipeline):
        seen.append(pipeline)
        return SimpleNamespace()

    monkeypatch.setattr(
        "batchalign.inference.speaker.infer_local_pyannote_prepared_audio", fake_local
    )
    _gpu_worker()
    ensure_task_loaded("speaker", {"speaker": "pyannote"})

    host = _state.loads.get(SPEAKER_LOCAL_PYANNOTE_LOAD)
    assert host is not None
    assert host.backend is SpeakerBackendV2.PYANNOTE
    host.runner(None, 16000, None)
    assert seen == [handle]


def test_a_hosted_engine_loads_nothing(monkeypatch) -> None:
    def refuse() -> None:
        raise AssertionError("pyannoteAI is hosted: nothing may be loaded")

    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.load_local_pyannote_pipeline", refuse
    )
    monkeypatch.setattr(
        "batchalign.inference.speaker_embedding.load_speaker_embedding_model", refuse
    )
    _gpu_worker()

    # A hosted engine adds no override: the request names no model to load.
    assert ensure_task_loaded("speaker").status == "loaded"
    host = _state.loads.get(SPEAKER_HOSTED_LOAD)
    assert host is not None
    assert host.backend is SpeakerBackendV2.PYANNOTE_AI
    assert _state.loads.get(SPEAKER_LOCAL_PYANNOTE_LOAD) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"speaker": "nemo"},
        {"speaker": "pyannote_ai"},
        {"speaker_embedding": "other"},
        {"speaker": "pyannote", "speaker_embedding": "pyannote"},
    ],
)
def test_a_speaker_load_naming_anything_else_is_refused(overrides) -> None:
    _gpu_worker()

    with pytest.raises(ValueError, match="at most one speaker model"):
        ensure_task_loaded("speaker", overrides)
    # Refused while parsing the load, before anything was built or recorded.
    assert _state.loads.completed() == frozenset()


def test_an_unbuildable_speaker_combination_is_refused_by_the_parse() -> None:
    with pytest.raises(ValueError, match="at most one speaker model"):
        task_load(
            InferTask.SPEAKER, {"speaker": "pyannote", "speaker_embedding": "pyannote"}
        )
