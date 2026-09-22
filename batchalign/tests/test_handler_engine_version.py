"""Boundary tests for the engine identities a worker reports.

Background: a capability probe once reported the literal engine *name*
``"stanza"`` as the morphosyntax version, which became ``engine=stanza-stanza``
in provenance comments. A later fallback reported ``"unknown"``, which travelled
on exactly like a version, and the FA entry named the default engine's enum
value before any FA model had loaded. A capability report now names only the
forced-alignment engine this process loaded (or ``None``); every other stage
names its engine on its results. A name is admitted by ``reported_engine_name``
before it can reach a stamp.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from batchalign.inference.translate import LoadedTranslation
from batchalign.worker._handlers import _reported_engine
from batchalign.worker._types import (
    InferTask,
    InvalidReportedEngineName,
    _state,
    reported_engine_name,
)


def test_stanza_version_reads_the_installed_package() -> None:
    """The one accessor reads ``stanza.__version__``, never the engine name."""
    import stanza

    assert _state.stanza_version() == stanza.__version__
    assert _state.stanza_engine() == f"stanza-{stanza.__version__}"


def test_coref_engine_names_the_coref_model_that_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coref names the model that runs (release plus package), not only Stanza.

    This is the identity every resolved item carries; the capability report
    names no coref engine at all.
    """
    from batchalign.inference.coref import coref_engine

    monkeypatch.setitem(sys.modules, "stanza", SimpleNamespace(__version__="1.10.1"))
    assert coref_engine() == "stanza-1.10.1/ontonotes-singletons_roberta-large-lora"


def test_only_forced_alignment_names_an_engine_in_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every task but FA reports ``None`` whatever this process has loaded.

    Those stages name their engines on each result, and the Rust capability
    gate refuses a report that names one, so nothing loaded may leak a name in.
    """
    monkeypatch.setattr(_state, "loaded_tasks", {task.value for task in InferTask})
    monkeypatch.setattr(_state, "utseg_config_builder", lambda langs: (langs, {}))
    for task in InferTask:
        if task is not InferTask.FA:
            assert _reported_engine(task) is None, task


def test_stanza_version_is_none_when_stanza_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No importable Stanza means no version: ``None``, not ``"unknown"``."""
    monkeypatch.setitem(sys.modules, "stanza", None)

    assert _state.stanza_version() is None
    assert _state.stanza_engine() is None


def test_stanza_version_with_a_separator_fails_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version that would break a stamp raises rather than being repaired."""
    monkeypatch.setitem(sys.modules, "stanza", SimpleNamespace(__version__="1.9|2"))

    with pytest.raises(InvalidReportedEngineName):
        _state.stanza_version()


@pytest.mark.parametrize(
    "name",
    ["", "   ", " stanza-1.9.2", "stanza-1.9.2\t", "a|b", "a;b", "a]b", "a\nb", "a\rb"],
)
def test_reported_engine_name_refuses_what_a_stamp_cannot_hold(name: str) -> None:
    with pytest.raises(InvalidReportedEngineName):
        reported_engine_name(name)


@pytest.mark.parametrize(
    "name",
    ["stanza-1.10.1", "facebook/nllb-200-distilled-1.3B", "wave2vec-fa-mms-2.5.1+cpu"],
)
def test_reported_engine_name_admits_real_names_unchanged(name: str) -> None:
    assert reported_engine_name(name) == name


def test_fa_is_unreported_until_an_fa_engine_loads() -> None:
    """Before loading, FA names nothing; after, exactly the loader's string."""
    saved = _state.fa_model_name
    try:
        _state.fa_model_name = None
        assert _reported_engine(InferTask.FA) is None

        _state.fa_model_name = "whisper-fa-large-v2"
        assert _reported_engine(InferTask.FA) == "whisper-fa-large-v2"
    finally:
        _state.fa_model_name = saved


def test_a_loaded_translation_refuses_an_unreportable_engine() -> None:
    with pytest.raises(InvalidReportedEngineName):
        LoadedTranslation(
            engine="models|nllb",
            translate=lambda text, _lang: text,
        )


def test_registry_entry_without_build_identity_reads_as_unknown() -> None:
    """An entry written before the field existed is read, with no identity."""
    from batchalign.worker._registry import _entry_from_json

    entry = _entry_from_json(
        {
            "pid": 1,
            "host": "127.0.0.1",
            "port": 9000,
            "profile": "stanza",
            "lang": "eng",
        }
    )
    assert entry.build_identity is None

    with pytest.raises(TypeError):
        _entry_from_json({"pid": 1, "host": "127.0.0.1"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("v0.24.2-7-gabc1234", "v0.24.2-7-gabc1234"),
    ],
)
def test_build_identity_comes_from_the_spawning_server(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str | None
) -> None:
    from batchalign.worker._protocol import _build_identity_from_env

    if value is None:
        monkeypatch.delenv("BATCHALIGN_BUILD_IDENTITY", raising=False)
    else:
        monkeypatch.setenv("BATCHALIGN_BUILD_IDENTITY", value)
    assert _build_identity_from_env() == expected
