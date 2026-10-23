"""Focused tests for worker-side Stanza language mapping."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from batchalign.inference._domain_types import StanzaCode
from batchalign.worker import _stanza_capabilities, _stanza_loading
from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    StanzaLanguageCapability,
    StanzaNotInstalled,
    build_stanza_capability_table_from_resources,
)
from batchalign.worker._stanza_loading import (
    UnsupportedLanguageError,
    load_stanza_models,
    load_stanza_retokenize_model,
    stanza_code_for,
)
from batchalign_core import ISO_639_1_PAIRS

_CATALOG = Path(__file__).parents[2] / "fixtures" / "stanza_catalog_shape.json"


@pytest.fixture
def catalog_table(monkeypatch: pytest.MonkeyPatch) -> StanzaCapabilityTable:
    """The table a worker builds from the committed Stanza 1.15.0 catalog
    snapshot and Rust's ISO table, installed as the worker's cached table."""
    table = build_stanza_capability_table_from_resources(
        json.loads(_CATALOG.read_text())["resources"], ISO_639_1_PAIRS
    )
    monkeypatch.setattr(
        _stanza_capabilities, "get_cached_capability_table", lambda: table
    )
    return table


def _install(
    monkeypatch: pytest.MonkeyPatch, table: StanzaCapabilityTable | StanzaNotInstalled
) -> None:
    monkeypatch.setattr(
        _stanza_capabilities, "get_cached_capability_table", lambda: table
    )


@pytest.mark.usefixtures("catalog_table")
def test_stanza_code_for_reads_the_capability_table() -> None:
    assert stanza_code_for("eng") == "en"
    # Cantonese routes through the Chinese model by policy; `zho` through
    # ISO `zh`, which the catalog aliases to `zh-hans`.
    assert stanza_code_for("yue") == "zh-hans"
    assert stanza_code_for("zho") == "zh-hans"
    # Kazakh: ISO `kk`; Ancient Greek: the catalog's own three-letter key.
    assert stanza_code_for("kaz") == "kk"
    assert stanza_code_for("grc") == "grc"


# Regression for the 2026-05-06 morphotag failure on a file with
# `@Languages: eng, mar`: the capability table resolved `mar` to `mr`, but the
# loader's own second mapping (an override dict, then pycountry, then the code
# verbatim) did not, and `stanza.Pipeline(lang="mar")` crashed the worker. The
# loader now has no mapping of its own: it reads the table.
@pytest.mark.usefixtures("catalog_table")
def test_marathi_resolves_through_the_one_mapping() -> None:
    assert stanza_code_for("mar") == "mr"


@pytest.mark.usefixtures("catalog_table")
def test_a_language_the_catalog_lacks_is_refused_not_passed_through() -> None:
    """The old mapping passed `zzz` to Stanza verbatim with a warning."""
    with pytest.raises(UnsupportedLanguageError, match="no entry"):
        stanza_code_for("zzz")


def test_no_capability_table_is_a_deploy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, StanzaNotInstalled(import_error="No module named 'stanza'"))
    with pytest.raises(UnsupportedLanguageError, match="not installed"):
        stanza_code_for("eng")


def _table_with(
    stanza_code: str,
    *,
    has_tokenize: bool = True,
    has_pos: bool = True,
    has_lemma: bool = True,
    has_depparse: bool = True,
    has_mwt: bool,
) -> StanzaCapabilityTable:
    """A single-entry capability table for language ``xxx``.

    Pure construction, no Stanza, no resources.json read.
    """
    present = {
        "tokenize": has_tokenize,
        "pos": has_pos,
        "lemma": has_lemma,
        "depparse": has_depparse,
        "mwt": has_mwt,
    }
    cap = StanzaLanguageCapability(
        stanza_code=StanzaCode(stanza_code),
        processors=frozenset(name for name, has in present.items() if has),
    )
    return StanzaCapabilityTable(languages={"xxx": cap})


class TestSupportsMorphosyntax:
    def test_returns_true_for_full_processor_stack(self) -> None:
        table = _table_with("en", has_mwt=True)
        assert table.supports_morphosyntax("xxx") is True

    def test_returns_false_for_partial_processor_entry(self) -> None:
        table = _table_with(
            "pa",
            has_pos=False,
            has_lemma=False,
            has_depparse=False,
            has_mwt=False,
        )
        assert table.supports_morphosyntax("xxx") is False


# MWT is requested exactly when the language's catalog entry ships it. A
# hardcoded `MWT_LANGS` set once said Swedish had MWT; the catalog does not
# ship one, every Swedish worker spawn raised `UnsupportedProcessorError`, and
# the 2026-04-15 overnight run lost a 500-file chunk.
@pytest.mark.parametrize("has_mwt", [False, True])
def test_mwt_is_requested_exactly_when_the_entry_ships_it(
    monkeypatch: pytest.MonkeyPatch, has_mwt: bool
) -> None:
    _install(monkeypatch, _table_with("sv", has_mwt=has_mwt))
    requested: list[str] = []

    def _pipeline(*args: object, **kwargs: object) -> object:
        requested.append(str(kwargs["processors"]))
        return types.SimpleNamespace()

    fake_stanza = types.ModuleType("stanza")
    fake_stanza.DownloadMethod = types.SimpleNamespace(REUSE_RESOURCES=object())
    fake_stanza.Pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "stanza", fake_stanza)
    monkeypatch.setattr(_stanza_loading, "lemma_pretrain_options", lambda *_: {})
    monkeypatch.setattr(
        _stanza_loading,
        "_emit_stanza_lang_download_event_if_missing",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        _stanza_loading, "pipeline_cache", lambda: _stanza_loading.StanzaPipelineCache()
    )

    load_stanza_models("xxx")
    assert len(requested) == 1
    assert ("mwt" in requested[0].split(",")) is has_mwt


def test_load_stanza_models_rejects_partial_processor_language_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = StanzaCapabilityTable(
        languages={
            "pan": StanzaLanguageCapability(
                stanza_code=StanzaCode("pa"),
                processors=frozenset({"tokenize"}),
            )
        },
    )
    _install(monkeypatch, table)

    fake_stanza = types.ModuleType("stanza")
    fake_stanza.DownloadMethod = types.SimpleNamespace(REUSE_RESOURCES=object())

    def _pipeline_should_not_run(*args, **kwargs):
        raise AssertionError(
            "stanza.Pipeline must not be called for partial processor entries"
        )

    fake_stanza.Pipeline = _pipeline_should_not_run
    monkeypatch.setitem(sys.modules, "stanza", fake_stanza)

    with pytest.raises(UnsupportedLanguageError) as exc_info:
        load_stanza_models("pan")

    assert "pan" in str(exc_info.value)
    assert "morphosyntax" in str(exc_info.value).lower()


@pytest.mark.usefixtures("catalog_table")
def test_the_retokenize_guard_admits_every_chinese_code_its_caller_passes() -> None:
    """RED FIRST: the guard compared against a literal the mapping had moved on from.

    The mapping sent ``cmn``, ``zho`` and ``yue`` to ``zh-hans`` while the
    loader's guard was ``alpha2 != "zh"``. Its only caller passes ``zho`` or
    ``cmn``, so every real call logged "non-Chinese lang, skipping" and the
    neural-tokenizer pipeline was never loaded. The guard now compares the
    table's code with ``CHINESE_STANZA_CODE``, the constant the table's own
    Chinese policy uses.
    """
    for iso3 in ("cmn", "zho", "yue"):
        assert stanza_code_for(iso3) == _stanza_loading.CHINESE_STANZA_CODE, (
            f"{iso3} is what the caller passes and must reach the loader"
        )
    assert stanza_code_for("eng") != _stanza_loading.CHINESE_STANZA_CODE


@pytest.mark.usefixtures("catalog_table")
def test_two_threads_loading_one_retokenize_key_build_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED FIRST (2026-09-07 review, item 5): the retokenize load stampedes.

    The load-stampede fix covered one of the three key kinds a worker loads.
    ``load_stanza_retokenize_model`` installed straight into the cache with no
    construction slot, so two request threads that both missed on
    ``{lang}:retok`` each built the neural tokenizer, a separate ~200 MB model,
    and the second install replaced the first: double the peak memory and
    double the wall clock for one usable result.

    This reaches the real loader with a fake ``stanza`` module, so what it pins
    is the SEAM (which thread gets to build) and not the model.
    """
    import threading

    from batchalign.worker._pipeline_cache import (
        StanzaPipelineCache,
        retokenize_key,
    )

    cache = StanzaPipelineCache()
    monkeypatch.setattr(_stanza_loading, "pipeline_cache", lambda: cache)
    monkeypatch.setattr(
        _stanza_loading,
        "_emit_stanza_lang_download_event_if_missing",
        lambda *args, **kwargs: None,
    )

    builds: list[str] = []
    first_is_inside = threading.Event()
    let_first_finish = threading.Event()

    def _pipeline(*args: object, **kwargs: object) -> object:
        builds.append("built")
        if len(builds) == 1:
            first_is_inside.set()
            # Hold the build open so a second thread must block on the slot.
            let_first_finish.wait(timeout=5)
        return types.SimpleNamespace()

    fake_stanza = types.ModuleType("stanza")
    fake_stanza.DownloadMethod = types.SimpleNamespace(REUSE_RESOURCES=object())
    fake_stanza.Pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "stanza", fake_stanza)
    # The lemma pretrain repair reads real model files; the build is faked.
    monkeypatch.setattr(_stanza_loading, "lemma_pretrain_options", lambda *_: {})

    first = threading.Thread(target=load_stanza_retokenize_model, args=("zho",))
    second = threading.Thread(target=load_stanza_retokenize_model, args=("zho",))
    first.start()
    assert first_is_inside.wait(timeout=5)
    second.start()
    # The second thread is blocked on the slot, so nothing new is built yet.
    assert builds == ["built"]
    let_first_finish.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert builds == ["built"], (
        "the second thread must find the first thread's pipeline, not build one"
    )
    assert cache.loaded(retokenize_key("zho")) is not None
    assert len(cache) == 1


@pytest.mark.usefixtures("catalog_table")
def test_two_threads_loading_one_mwt_probe_build_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED FIRST (2026-09-07 review, item 5): the probe check-then-acts.

    ``load_stanza_mwt_probe_model`` read ``cache.get(probe_key)`` and built on
    a miss, which is a check followed by an act with nothing between them
    holding a lock. Two threads could both miss, both build, and both install;
    the loser's pipeline had already been handed back to its own caller, so the
    memory stayed reachable as well as duplicated.
    """
    import threading

    from batchalign.worker._pipeline_cache import StanzaPipelineCache, mwt_probe_key
    from batchalign.worker._stanza_loading import load_stanza_mwt_probe_model

    cache = StanzaPipelineCache()
    monkeypatch.setattr(_stanza_loading, "pipeline_cache", lambda: cache)

    builds: list[str] = []
    first_is_inside = threading.Event()
    let_first_finish = threading.Event()

    def _pipeline(*args: object, **kwargs: object) -> object:
        builds.append("built")
        if len(builds) == 1:
            first_is_inside.set()
            let_first_finish.wait(timeout=5)
        return types.SimpleNamespace()

    fake_stanza = types.ModuleType("stanza")
    fake_stanza.DownloadMethod = types.SimpleNamespace(REUSE_RESOURCES=object())
    fake_stanza.Pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "stanza", fake_stanza)

    probes: list[object] = []

    def _load() -> None:
        probes.append(load_stanza_mwt_probe_model("ita"))

    first = threading.Thread(target=_load)
    second = threading.Thread(target=_load)
    first.start()
    assert first_is_inside.wait(timeout=5)
    second.start()
    assert builds == ["built"]
    let_first_finish.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert builds == ["built"], (
        "the second thread must find the first thread's probe, not build one"
    )
    # Both callers hold the SAME probe, which is what makes the single build
    # observable at the seam rather than only in the cache.
    assert len(probes) == 2
    assert probes[0] is probes[1]
    assert cache.loaded(mwt_probe_key("ita")) is not None
