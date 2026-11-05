"""Tests for worker startup and on-demand loading.

Every worker starts with no models loaded, in every bootstrap mode, and loads
each task's models on demand via ensure_task_loaded(), which the Rust pool
sends before a worker's first request for the task.
"""

from __future__ import annotations

import threading

from batchalign.worker._loads import SPEAKER_HOSTED_LOAD
from batchalign.worker._main import build_worker_bootstrap_runtime, parse_worker_args
from batchalign.worker._model_loading.bootstrap import (
    EnsureTaskResponse,
    ensure_task_loaded,
    start_worker,
)
from batchalign.worker._types import TaskLoad, WorkerProfile, _state


def _reset_state() -> None:
    """Reset global worker state between tests."""
    _state.__init__()  # type: ignore[misc]


def _build_gpu_bootstrap():
    """Build a GPU profile bootstrap runtime for testing."""
    args = parse_worker_args(
        ["--profile", "gpu", "--lang", "eng", "--serving", "sequential"]
    )
    return build_worker_bootstrap_runtime(
        args,
        environ={"HOME": "/tmp/test-home"},
    )


class TestWorkerStartup:
    """Verify workers start with no models loaded."""

    def setup_method(self) -> None:
        _reset_state()

    def test_a_worker_signals_ready_without_models(self) -> None:
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        assert _state.ready is True
        assert _state.loads.completed() == frozenset()
        assert _state.command == "profile:gpu"
        assert _state.lang == "eng"

    def test_a_worker_preserves_bootstrap_for_later(self) -> None:
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        assert _state.bootstrap is bootstrap
        assert _state.bootstrap.profile == WorkerProfile.GPU

    def test_a_worker_holds_no_model_at_start(self) -> None:
        """Every model a worker holds is a load's entry, and a started worker
        has none: there is no per-model field to be anything but empty."""
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        assert not any(_state.loads.has(load) for load in TaskLoad)


class TestEnsureTaskLoaded:
    """Verify on-demand model loading via ensure_task_loaded()."""

    def setup_method(self) -> None:
        _reset_state()

    def test_ensure_task_on_unbootstrapped_worker_raises(self) -> None:
        """Cannot load tasks before bootstrap."""
        try:
            ensure_task_loaded("fa")
            raise AssertionError("should have raised RuntimeError")
        except RuntimeError as e:
            assert "before worker bootstrap" in str(e)

    def test_ensure_task_unknown_task_raises(self) -> None:
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        try:
            ensure_task_loaded("nonexistent_task")
            raise AssertionError("should have raised ValueError")
        except ValueError as e:
            assert "Unknown ensure_task name" in str(e)

    def test_ensure_task_idempotent(self) -> None:
        """Second call for same task returns already_loaded immediately."""
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        # A previous ensure_task recorded the hosted speaker load.
        first = ensure_task_loaded("speaker")
        assert first.status == "loaded"
        host = _state.loads.get(SPEAKER_HOSTED_LOAD)

        result = ensure_task_loaded("speaker")
        assert result.status == "already_loaded"
        assert result.task == "speaker"
        # The check is timed, never reported as a constant.
        assert result.elapsed_s >= 0.0
        # The repeat recorded nothing: the entry is the first load's host.
        assert _state.loads.get(SPEAKER_HOSTED_LOAD) is host

    def test_ensure_task_result_fields(self) -> None:
        """EnsureTaskResponse has correct fields."""
        result = EnsureTaskResponse(status="loaded", task="fa", elapsed_s=1.23)
        assert result.status == "loaded"
        assert result.task == "fa"
        assert result.elapsed_s == 1.23

    def test_ensure_task_passes_engine_overrides(self) -> None:
        """Engine overrides from ensure_task reach the task's load: an
        override naming a model the load does not know is refused."""
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        try:
            ensure_task_loaded("speaker", {"speaker": "nemo"})
            raise AssertionError("should have raised ValueError")
        except ValueError as e:
            assert "speaker=" in str(e)
        # Refused before anything loaded, so nothing was recorded.
        assert _state.loads.completed() == frozenset()
        result = ensure_task_loaded("speaker")
        assert result.status == "loaded"
        assert _state.loads.completed() == frozenset({TaskLoad.SPEAKER_HOSTED})


class TestEnsureTaskThreadSafety:
    """Verify concurrent ensure_task calls don't cause double loads."""

    def setup_method(self) -> None:
        _reset_state()

    def test_concurrent_ensure_task_loads_once(self) -> None:
        """Two threads calling ensure_task for the same task should only load once."""
        bootstrap = _build_gpu_bootstrap()
        start_worker(bootstrap)

        results: list[EnsureTaskResponse] = []
        errors: list[Exception] = []

        def call_ensure() -> None:
            try:
                r = ensure_task_loaded("speaker")
                results.append(r)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=call_ensure)
        t2 = threading.Thread(target=call_ensure)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Unexpected errors: {errors}"
        assert len(results) == 2
        # Exactly one load ran; the other call found its entry.
        assert sorted(r.status for r in results) == ["already_loaded", "loaded"]
        assert _state.loads.completed() == frozenset({TaskLoad.SPEAKER_HOSTED})


def test_the_lazy_flag_is_gone() -> None:
    """Every worker starts empty, so there is no lazy mode to ask for."""
    import pytest

    with pytest.raises(SystemExit):
        parse_worker_args(
            ["--profile", "gpu", "--lazy", "--lang", "eng", "--serving", "sequential"]
        )
