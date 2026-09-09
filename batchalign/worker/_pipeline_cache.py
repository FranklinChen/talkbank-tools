"""Bounded, thread-safe residency for loaded Stanza pipelines.

A loaded Stanza pipeline is a multi-hundred-megabyte object graph. The worker
used to keep them in a plain dict that only ever grew: every language a
long-lived worker process saw stayed resident until the process exited, so a
multilingual queue drove one worker's resident set up without limit and
nothing ever gave the memory back.

This module replaces that dict with a small LRU cache. It owns three things
the dict could not express:

* a CEILING on how many pipelines stay resident (``STANZA_PIPELINE_CAPACITY``);
* the ORDER of use, so the entry dropped at the ceiling is the one least
  likely to be wanted next;
* the fact that a tokenizer context BELONGS to a pipeline. Those used to be
  two dicts keyed alike, with nothing enforcing the correspondence, so a
  context could outlive the pipeline it was built for.

The cache is a ``Mapping``, so every reader keeps working unchanged, and a
lookup through it is also a USE: reading a pipeline is what keeps it resident.

Eviction is not merely a ``del``. Stanza pipelines participate in reference
cycles (a pipeline holds its processors, each processor holds the pipeline),
so dropping the last dict reference leaves the memory for the cycle collector
to find whenever it next runs. ``gc.collect()`` after an eviction is what
actually returns it, which is the whole point of having a ceiling.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from batchalign.inference._domain_types import LanguageCode

if TYPE_CHECKING:
    from batchalign.inference._tokenizer_realign import TokenizerContext
    from batchalign.inference.types import StanzaNLP

L = logging.getLogger("batchalign.worker")


STANZA_PIPELINE_CAPACITY = 2
"""How many loaded Stanza pipelines one worker process keeps resident.

Two is a CEILING chosen for memory, not a claim about the working set, and
this docstring used to conflate the two by saying a job carries "one primary
language and at most one secondary at a time". It does not. The keys a single
morphotag batch can want are: the primary language; every utterance-level
``[- xxx]`` precode language in the file; the ``:retok`` variant of the
primary, for a Mandarin retokenize job; and the ``:mwtprobe`` variant, for
Italian. A Mandarin retokenize job over a file with one precode language
therefore has a working set of three and WILL thrash at this ceiling.

That is the trade, stated rather than denied: a lookup that misses asks the
loader to bring the language back (see ``batch_infer_morphosyntax``'s
``load_pipeline``), so such a batch still COMPLETES, paying one reload per
language group per batch. The ceiling is not raised to cover it because a
third resident Stanza pipeline is several hundred more megabytes in every
worker on every host, and the thrashing case is the uncommon one. There is
deliberately no configuration seam: this workspace has none for worker memory
policy, and inventing one to avoid choosing a number would leave the ceiling
untested at every value but the default.
"""


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------
#
# Keys are plain strings because that is what the callers already hold: a
# language code for the ordinary pipeline, and a suffixed variant for the two
# special pipelines. The suffixes are an out-of-band meaning inside a key, so
# they get ONE owner here rather than being spelled out at each call site,
# which is how the retokenize key came to be written by hand in two modules.


def retokenize_key(lang: LanguageCode) -> str:
    """Key of the neural-tokenizer pipeline for ``lang`` (Chinese retokenize)."""
    return f"{lang}:retok"


def mwt_probe_key(lang: LanguageCode) -> str:
    """Key of the tokenize+mwt probe pipeline for ``lang`` (Italian MWT)."""
    return f"{lang}:mwtprobe"


# ---------------------------------------------------------------------------
# Entries and install outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedPipeline:
    """One loaded pipeline together with the tokenizer context built for it.

    Held as one value so the two cannot be evicted apart. ``context`` is
    ``None`` for the MWT probe pipeline, which has no realignment context: an
    honest absence rather than an empty context that would read as one.
    """

    nlp: StanzaNLP
    context: TokenizerContext | None


@dataclass(frozen=True, slots=True)
class InstalledWithoutEviction:
    """The install fitted under the ceiling and dropped nothing."""

    key: str


@dataclass(frozen=True, slots=True)
class InstalledEvicting:
    """The install crossed the ceiling and dropped the least recently used entry."""

    key: str
    evicted: str


InstallOutcome: TypeAlias = InstalledWithoutEviction | InstalledEvicting
"""What an install DID, returned rather than logged.

A caller that cannot tell whether a model was just unloaded cannot report it,
and neither can a test double: this is the difference between a seam whose
effects are observable and one whose effects are taken on trust.
"""


class PipelineLookup(Protocol):
    """One atomic read of a pipeline and the context built for it.

    The reason this is a single method and not two mappings: reading the
    pipeline and then reading its context were two separately locked lookups
    with an eviction window between them, so a batch could take a pipeline for
    one language, lose it to an install on another thread, and then read a
    DIFFERENT language's context (or ``None``) for the realigner. Nothing
    would notice: a wrong realignment context produces plausible output.
    ``LoadedPipeline`` exists precisely so the two cannot be separated, and
    this is the interface that stops them being separated at the READ.
    """

    def loaded(self, key: str) -> LoadedPipeline | None:
        """Return the resident entry for ``key``, or ``None``, marking it used."""
        ...


@dataclass(frozen=True, slots=True)
class StaticPipelines:
    """A fixed set of loaded pipelines that never evicts.

    For callers that hold their pipelines for the life of the call: tests, and
    any future in-process embedding that loads once. It satisfies
    [`PipelineLookup`] without pretending to be an LRU, so a test cannot
    accidentally assert eviction behaviour it does not have.
    """

    entries: Mapping[str, LoadedPipeline]

    def loaded(self, key: str) -> LoadedPipeline | None:
        """Return the entry for ``key``; there is no residency to update."""
        return self.entries.get(key)


def static_pipelines(
    nlp_by_key: Mapping[str, StanzaNLP],
    contexts_by_key: Mapping[str, TokenizerContext] | None = None,
) -> StaticPipelines:
    """Pair pipelines with their contexts once, at the boundary."""
    contexts = contexts_by_key or {}
    return StaticPipelines(
        entries={
            key: LoadedPipeline(nlp=nlp, context=contexts.get(key))
            for key, nlp in nlp_by_key.items()
        }
    )


class StanzaPipelineCache(Mapping[str, "StanzaNLP"]):
    """LRU cache of loaded Stanza pipelines, safe to share between threads.

    Thread safety is required, not defensive: the worker serves requests from
    a ``ThreadPoolExecutor`` under the GPU profile and under free-threaded
    Python (see ``_serve_stdio_concurrent``), so two threads can install and
    read at once. An ``OrderedDict`` reorder is several operations and is not
    atomic, so every access takes the lock.
    """

    def __init__(self, capacity: int = STANZA_PIPELINE_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"pipeline cache capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._entries: OrderedDict[str, LoadedPipeline] = OrderedDict()
        # Reentrant so the context view below can read through the same lock
        # without a second lock to order against this one.
        self._lock = threading.RLock()
        # One construction lock per key, created under `_lock` and then held
        # ALONE, so building a pipeline never blocks a read of another one.
        # Bounded in practice by the number of distinct language keys a worker
        # sees; a `threading.Lock` is a few dozen bytes beside a pipeline's
        # hundreds of megabytes.
        self._load_locks: dict[str, threading.Lock] = {}

    # -- Mapping interface, which is also the "use" interface ---------------

    def __getitem__(self, key: str) -> StanzaNLP:
        """Return the pipeline for ``key`` and mark it most recently used.

        Reading IS using: this is what makes eviction order meaningful, so
        there is deliberately no side-effect-free peek in the public surface.
        """
        with self._lock:
            entry = self._entries[key]
            self._entries.move_to_end(key)
            return entry.nlp

    def loaded(self, key: str) -> LoadedPipeline | None:
        """Return the pipeline AND its context in one locked read.

        The atomic half of [`PipelineLookup`]. `__getitem__` hands back only
        the pipeline, so a caller that also wants the context had to look it up
        again through `contexts`, and between the two reads another thread's
        install can evict the entry. Reading IS using here too.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    @contextlib.contextmanager
    def load_slot(self, key: str) -> Iterator[LoadedPipeline | None]:
        """Serialize construction of ONE key across threads.

        Yields whatever is already resident for ``key``, with this thread
        holding that key's construction lock. A caller builds and installs only
        when the yielded value is ``None``.

        Without this, two request threads missing on the same language both
        called the loader and both built a multi-hundred-megabyte Stanza
        pipeline, the second install replacing the first: double the peak
        memory and double the load time for one usable result. The lock is
        PER KEY, so loading Mandarin never blocks a Spanish read, and it is
        taken alone rather than under `_lock`, so a several-minute model
        download does not stop the cache serving every other language.

        It serializes only the callers that TAKE it, and that is the whole of
        its guarantee: a loader which builds and installs without opening this
        slot is not protected, however many other loaders are. The cache cannot
        enforce it, because `install` is also the legitimate way an
        already-serialized build finishes. Until 2026-09-07 only the plain
        language key was routed through here, so the ``:retok`` and
        ``:mwtprobe`` builds still stampeded and the probe additionally
        check-then-acted on a plain `get`.
        """
        with self._lock:
            lock = self._load_locks.setdefault(key, threading.Lock())
        with lock:
            yield self.loaded(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- Mutation ------------------------------------------------------------

    def install(
        self,
        key: str,
        nlp: StanzaNLP,
        context: TokenizerContext | None = None,
    ) -> InstallOutcome:
        """Install a freshly loaded pipeline, evicting if the ceiling is crossed.

        Returns what it did. Re-installing an existing key replaces it in
        place and cannot evict, which is what makes a reload of an already
        resident language free of collateral damage.
        """
        evicted: str | None = None
        with self._lock:
            self._entries[key] = LoadedPipeline(nlp=nlp, context=context)
            self._entries.move_to_end(key)
            if len(self._entries) > self._capacity:
                # popitem(last=False) is the least recently used end.
                evicted, _dropped = self._entries.popitem(last=False)

        if evicted is None:
            return InstalledWithoutEviction(key=key)

        # Outside the lock, deliberately, and this is a policy rather than an
        # oversight: `gc.collect()` walks the whole heap and takes long enough
        # to matter, so holding `_lock` across it would stall every read and
        # every install in every other serving thread for the duration. It is
        # safe outside because the evicted entry is already unreachable FROM
        # THE CACHE by this point, so the only thing the collection races with
        # is another thread's collection, which is harmless. What it cannot
        # promise is that the memory is back before this returns: another
        # thread may still hold a reference it took before the eviction.
        gc.collect()
        L.info(
            "Stanza pipeline cache: installed %s, evicted %s (capacity %d)",
            key,
            evicted,
            self._capacity,
        )
        return InstalledEvicting(key=key, evicted=evicted)

    # -- Contexts ------------------------------------------------------------

    @property
    def contexts(self) -> Mapping[str, TokenizerContext]:
        """Live view of the tokenizer contexts, keyed exactly as the pipelines.

        A VIEW rather than a copy because a caller may load a pipeline (and so
        add its context) in the middle of using it; a snapshot taken at the
        start of a batch would silently miss that, and the realigner would run
        against the wrong language's context.
        """
        return _ContextView(self)

    def _context(self, key: str) -> TokenizerContext | None:
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.context

    def _context_keys(self) -> list[str]:
        with self._lock:
            return [
                key for key, entry in self._entries.items() if entry.context is not None
            ]


class _ContextView(Mapping[str, "TokenizerContext"]):
    """Read-only projection of a cache onto its tokenizer contexts."""

    def __init__(self, cache: StanzaPipelineCache) -> None:
        self._cache = cache

    def __getitem__(self, key: str) -> TokenizerContext:
        context = self._cache._context(key)
        if context is None:
            raise KeyError(key)
        return context

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache._context_keys())

    def __len__(self) -> int:
        return len(self._cache._context_keys())
