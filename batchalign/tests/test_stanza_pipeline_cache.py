"""Residency tests for the bounded Stanza pipeline cache.

A loaded Stanza pipeline is a multi-hundred-megabyte object. The worker used
to hold them in a plain dict that only ever grew, so a long-lived process that
saw N languages held N models until it exited. These tests pin the two facts
that make the replacement a CACHE rather than a dict: it has a ceiling, and
crossing the ceiling drops the least recently used entry.
"""

from __future__ import annotations

from typing import Any

from batchalign.worker._pipeline_cache import (
    STANZA_PIPELINE_CAPACITY,
    InstalledEvicting,
    InstalledWithoutEviction,
    StanzaPipelineCache,
    mwt_probe_key,
    retokenize_key,
)


class _FakePipeline:
    """Stand-in for a loaded Stanza pipeline.

    The cache never calls into a pipeline, it only holds and hands back the
    object, so identity is the whole contract under test.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, text: str) -> Any:  # pragma: no cover - never invoked
        raise AssertionError("the cache must not call the pipeline it holds")


def test_capacity_is_two() -> None:
    """The default ceiling is pinned so a widening is a deliberate edit.

    This is a POLICY test, not an invariant a type could carry: two is a
    memory ceiling with a real alternative, NOT a claim that a job only ever
    wants two pipelines. See ``STANZA_PIPELINE_CAPACITY``'s own docstring for
    the working set it does not cover, and the reload it pays instead.
    """
    assert STANZA_PIPELINE_CAPACITY == 2


def test_installing_within_capacity_evicts_nothing() -> None:
    cache = StanzaPipelineCache()
    first = _FakePipeline("eng")
    second = _FakePipeline("fra")

    assert isinstance(cache.install("eng", first), InstalledWithoutEviction)
    assert isinstance(cache.install("fra", second), InstalledWithoutEviction)

    assert len(cache) == 2
    assert cache.get("eng") is first
    assert cache.get("fra") is second


def test_installing_a_third_pipeline_evicts_the_least_recently_used() -> None:
    """The third load drops the pipeline that has gone longest unused.

    "Least recently used" is what makes the ceiling safe rather than merely
    small: the entry dropped is the one the next request is least likely to
    ask for. The `get` below is what makes `fra` the stale one, so a cache
    that evicted by INSERTION order would fail here.
    """
    cache = StanzaPipelineCache()
    eng = _FakePipeline("eng")
    fra = _FakePipeline("fra")
    spa = _FakePipeline("spa")

    cache.install("eng", eng)
    cache.install("fra", fra)
    # Use `eng` again, so `fra` is now the least recently used of the two.
    assert cache.get("eng") is eng

    outcome = cache.install("spa", spa)

    assert isinstance(outcome, InstalledEvicting)
    assert outcome.evicted == "fra"
    assert cache.get("fra") is None
    assert cache.get("eng") is eng
    assert cache.get("spa") is spa
    assert len(cache) == STANZA_PIPELINE_CAPACITY


def test_eviction_collects_garbage_so_the_model_memory_is_returned(monkeypatch) -> None:
    """Dropping the reference is half the job; the collection is the other half.

    Stanza pipelines participate in reference CYCLES (a processor holds its
    pipeline, which holds its processors), so removing the cache's reference
    does not by itself return the memory. Nothing observable to a caller
    distinguishes "collected" from "not collected", which is why this asserts
    on the call rather than on a byte count.
    """
    collected: list[int] = []
    monkeypatch.setattr(
        "batchalign.worker._pipeline_cache.gc.collect",
        lambda: collected.append(1),
    )

    cache = StanzaPipelineCache()
    cache.install("eng", _FakePipeline("eng"))
    cache.install("fra", _FakePipeline("fra"))
    assert collected == []

    cache.install("spa", _FakePipeline("spa"))
    assert len(collected) == 1


def test_evicting_a_pipeline_evicts_its_tokenizer_context() -> None:
    """The context travels with its pipeline, so the two cannot drift.

    They used to be two dicts keyed alike with nothing enforcing the
    correspondence, which is how an entry for an unloaded pipeline could
    survive and be read by the realigner.
    """
    cache = StanzaPipelineCache()
    eng_context = object()
    cache.install("eng", _FakePipeline("eng"), context=eng_context)
    cache.install("fra", _FakePipeline("fra"), context=object())
    cache.install("spa", _FakePipeline("spa"), context=object())

    assert cache.contexts.get("eng") is None
    assert set(cache.contexts) == {"fra", "spa"}


def test_variant_keys_have_one_owner() -> None:
    """The `:retok` / `:mwtprobe` suffixes are built in exactly one place.

    They are an out-of-band meaning carried inside a key string, and they were
    written out by hand at three call sites across two modules.
    """
    assert retokenize_key("zho") == "zho:retok"
    assert mwt_probe_key("ita") == "ita:mwtprobe"


def test_loaded_returns_the_pipeline_and_its_context_together() -> None:
    """One read, one pair: the atomic half of ``PipelineLookup``.

    Reading the pipeline and then reading its context were two separately
    locked lookups, so an install on another thread could evict the entry
    between them and the caller ran the right pipeline with the wrong
    (or missing) realignment context.
    """
    cache = StanzaPipelineCache()
    nlp = _FakePipeline("eng")
    context = object()
    cache.install("eng", nlp, context=context)

    entry = cache.loaded("eng")
    assert entry is not None
    assert entry.nlp is nlp
    assert entry.context is context
    assert cache.loaded("fra") is None


def test_a_read_that_pairs_them_survives_an_eviction_between_the_old_lookups() -> None:
    """RED FIRST (review item 6): the eviction window is gone.

    This reproduces the old two-lookup shape directly. Between reading the
    pipeline and reading its context, another language is installed and
    crosses the ceiling; ``eng`` is the least recently used, so it goes. The
    old code then read ``contexts.get("eng")`` and got ``None``, and the
    realigner ran with no context while holding a perfectly good pipeline.
    The single ``loaded`` call takes both before that can happen.
    """
    cache = StanzaPipelineCache()
    eng_context = object()
    cache.install("eng", _FakePipeline("eng"), context=eng_context)
    cache.install("fra", _FakePipeline("fra"), context=object())

    entry = cache.loaded("eng")
    assert entry is not None

    # The evicting install that used to land between the two reads. `eng` is
    # the least recently used only after `loaded` moved it to the end, so use
    # `fra` first to make `eng` the victim.
    assert cache.loaded("fra") is not None
    cache.install("spa", _FakePipeline("spa"), context=object())

    assert cache.contexts.get("eng") is None, "precondition: eng was evicted"
    assert entry.context is eng_context, (
        "the pair taken in one read must survive an eviction that follows it"
    )


def test_load_slot_serializes_construction_of_one_key() -> None:
    """RED FIRST (review item 7): two threads must not both build one language.

    Loads ran outside any load lock, so two request threads that missed on the
    same language both called the loader and both built a multi-hundred-
    megabyte pipeline. The slot makes the second thread wait and find the
    first thread's install.
    """
    import threading

    cache = StanzaPipelineCache()
    builds: list[str] = []
    first_is_inside = threading.Event()
    let_first_finish = threading.Event()

    def load(name: str, announce: bool) -> None:
        with cache.load_slot("eng") as already:
            if already is not None:
                return
            builds.append(name)
            if announce:
                first_is_inside.set()
                # Hold the slot open so the second thread must block on it.
                let_first_finish.wait(timeout=5)
            cache.install("eng", _FakePipeline(name))

    first = threading.Thread(target=load, args=("first", True))
    second = threading.Thread(target=load, args=("second", False))
    first.start()
    assert first_is_inside.wait(timeout=5)
    second.start()
    # The second thread is blocked on the slot, so nothing new is built yet.
    assert builds == ["first"]
    let_first_finish.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert builds == ["first"], (
        "the second thread must find the first thread's pipeline, not build one"
    )
    assert len(cache) == 1


def test_load_slot_yields_an_already_resident_entry() -> None:
    """A language already loaded is a no-op, which is what makes the
    on-demand loader cheap enough to call per language group."""
    cache = StanzaPipelineCache()
    nlp = _FakePipeline("eng")
    cache.install("eng", nlp)

    with cache.load_slot("eng") as already:
        assert already is not None
        assert already.nlp is nlp
