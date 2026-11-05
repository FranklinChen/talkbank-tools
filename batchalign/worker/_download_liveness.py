"""Download liveness: a model download proves it is moving, or goes quiet.

A worker downloads its models before it writes its ready signal, and a first
download can take far longer than any fixed deadline should allow: a 1.3 GB
Paraformer composition from ModelScope took 14 minutes on 2026-11-05. The Rust
supervisor therefore does not give a starting worker a fixed time. It gives it
a STALL window, re-armed each time the worker reports that it moved
(``read_ready_line`` in ``crates/batchalign/src/worker/handle/lifecycle.rs``).

This module is the worker's half of that contract. ``download_liveness``
brackets one resolution that may download and, while it runs, measures the
hub's cache trees every ``POLL_INTERVAL_S`` seconds. It reports a
``progress_v2`` event only when they have GROWN by at least
``MIN_REPORTED_GROWTH_BYTES`` since the last report, carrying the KiB
downloaded so far as a ``DownloadGrowthV2`` measure, so a report is an
observation and never a timer: a download that stops growing, or slows to a
trickle, stops reporting, and the supervisor's stall window then expires and
names it. The download is announced at its first observed growth (a
``PhaseV2`` begin on the download's stage, and a complete when it finished),
so a resolution served from the cache says nothing and needs no separate "is
it cached?" guess.

Why whole cache trees rather than one model's directory: Hugging Face
assembles files through its chunk cache (``HF_XET_CACHE``) before they land in
the hub cache (``HF_HUB_CACHE``), and either can be moved by its own
environment variable, so both are watched. The trees are small (a full walk
took under 10 ms on ming), so measuring all of them is cheap and does not
depend on either library's internal layout. The cost is that a concurrent
download into the same cache also counts as growth; a stalled download beside
a live one is then caught only when the live one ends.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from batchalign.worker._protocol import write_load_progress
from batchalign.worker._types_v2 import DownloadGrowthV2, PhaseEdgeV2, PhaseV2

L = logging.getLogger(__name__)

# How often a running download's cache tree is measured. Far inside the
# supervisor's stall windows (`ensure_task_stall_s`, 120 s by default, and
# `worker_startup_stall_s`, 300 s), so a moving download reports many times
# per window.
POLL_INTERVAL_S = 10.0

# Progress counts KiB: the wire's `grown_kib` is a u32, which bytes would
# overflow at 4 GiB (a Qwen composition is about 6 GB).
_KIB = 1 << 10
_U32_MAX = (1 << 32) - 1

# What counts as progress: at least this much new data since the last report.
# Growth below it is not reported, so a download slower than this per stall
# window is a stall. On 2026-11-05 a ModelScope download fell to about
# 0.14 KB/s at 781 MB of 1.3 GB; reporting every few KiB of that kept the
# worker "alive" for a download that would have needed days. One MiB per
# window is about 3.4 KB/s at the 300 s default, far below any usable link,
# and a stalled worker is retired and the load retried, which may reconnect.
MIN_REPORTED_GROWTH_BYTES = 1 << 20


def hf_cache_roots() -> tuple[Path, ...]:
    """The trees a Hugging Face download grows: the hub cache and the chunk
    cache it assembles files in, each where its own environment variable puts
    it."""
    from huggingface_hub.constants import HF_HUB_CACHE, HF_XET_CACHE

    return (Path(HF_HUB_CACHE), Path(HF_XET_CACHE))


def torch_hub_roots() -> tuple[Path, ...]:
    """The tree a torch hub download (torchaudio pipeline bundles) grows."""
    import torch

    return (Path(torch.hub.get_dir()),)


def stanza_roots() -> tuple[Path, ...]:
    """The tree a Stanza download grows: its resources directory."""
    import stanza.resources.common as stanza_common

    return (Path(stanza_common.DEFAULT_MODEL_DIR),)


def modelscope_roots() -> tuple[Path, ...]:
    """The tree a ModelScope download grows: its cache root, which holds both
    versioned (``<root>/models``) and legacy snapshots."""
    from batchalign.worker._model_loading.modelscope_hub import _cache_root

    return (_cache_root(),)


@dataclass(frozen=True, slots=True)
class HfRepo:
    """A Hugging Face repository."""

    model_id: str


@dataclass(frozen=True, slots=True)
class ModelScopeRepo:
    """A ModelScope repository."""

    model_id: str


@dataclass(frozen=True, slots=True)
class TorchHubBundle:
    """A torch hub checkpoint behind a torchaudio pipeline bundle."""

    name: str


@dataclass(frozen=True, slots=True)
class StanzaResources:
    """Stanza resources (a language pack, a lemmatizer's pretrain)."""

    name: str


@dataclass(frozen=True, slots=True)
class StanzaWithTransformer:
    """Stanza resources whose model also pulls a Hugging Face backbone
    (coreference: the Stanza package plus RoBERTa)."""

    name: str


@dataclass(frozen=True, slots=True)
class StanzaCatalog:
    """Stanza's resource catalog (``resources.json``, about 1 MB).

    The one download announced when it starts rather than at first growth:
    its caller fetches it only after finding it absent, so the download is
    known, and at about 1 MB it may never cross ``MIN_REPORTED_GROWTH_BYTES``.
    """


DownloadSource = (
    HfRepo
    | ModelScopeRepo
    | TorchHubBundle
    | StanzaResources
    | StanzaWithTransformer
    | StanzaCatalog
)
"""What a download fetches. It decides the wire stage and the cache trees
watched, so no caller pairs a stage name with roots by hand, and no other code
can spell a ``downloading_`` stage."""


def _slug(name: str) -> str:
    return name.replace("/", "_")


def download_stage(source: DownloadSource) -> str:
    """The wire stage of a download, e.g. ``downloading_hf_openai_whisper``.

    One spelling per source: the supervisor tells stages apart by name when it
    judges whether a load advanced, so two spellings of one download would read
    as two, and two downloads sharing one would share a progress mark.
    """
    match source:
        case HfRepo(model_id=model_id):
            return f"downloading_hf_{_slug(model_id)}"
        case ModelScopeRepo(model_id=model_id):
            return f"downloading_modelscope_{_slug(model_id)}"
        case TorchHubBundle(name=name):
            return f"downloading_torch_hub_{_slug(name)}"
        case StanzaResources(name=name) | StanzaWithTransformer(name=name):
            return f"downloading_stanza_{_slug(name)}"
        case StanzaCatalog():
            return "downloading_stanza_catalog"
        case _ as unreachable:
            assert_never(unreachable)


def _roots(source: DownloadSource) -> tuple[Path, ...]:
    """The cache trees a download from ``source`` grows."""
    match source:
        case HfRepo():
            return hf_cache_roots()
        case ModelScopeRepo():
            return modelscope_roots()
        case TorchHubBundle():
            return torch_hub_roots()
        case StanzaResources() | StanzaCatalog():
            return stanza_roots()
        case StanzaWithTransformer():
            return stanza_roots() + hf_cache_roots()
        case _ as unreachable:
            assert_never(unreachable)


def _announce(stage: str, edge: PhaseEdgeV2, user_message: str) -> None:
    """Write a download's announcement (``BEGIN``) or completion
    (``COMPLETE``) to the progress channel, as a phase on the download's stage.

    Private to this module: a download is announced by ``download_liveness``
    and by nothing else.
    """
    L.info("download: %s %s: %s", stage, edge.value, user_message)
    write_load_progress(stage, PhaseV2(edge=edge))


def tree_bytes(root: Path) -> int:
    """Total size of the regular files under ``root``, symlinks not followed.

    A hub snapshot links to blobs it also stores, so following links would
    count those bytes twice. A file or directory that disappears mid-walk is
    a download renaming its temporary file into place, which is the very
    activity being measured; it is skipped rather than treated as an error.
    """
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            continue
    return total


class GrowthMeter:
    """Measures cache trees and says, each poll, whether they grew.

    The one place the "report only real growth" rule lives, so the reporter
    thread schedules polls and nothing else, and the rule can be tested
    without a thread or a clock.
    """

    def __init__(self, watch: tuple[Path, ...]) -> None:
        self._watch = watch
        self._baseline = self._measure()
        self._reported = self._baseline

    def _measure(self) -> int:
        return sum(tree_bytes(root) for root in self._watch)

    def poll(self) -> int | None:
        """KiB grown since the meter was made, when the trees grew by at least
        ``MIN_REPORTED_GROWTH_BYTES`` since the last answer that was not
        ``None``; ``None`` otherwise.

        A tree that shrinks (a temporary file removed after its rename) does
        not lower the mark, so the reported count never goes backwards.
        """
        observed = self._measure()
        if observed - self._reported < MIN_REPORTED_GROWTH_BYTES:
            return None
        self._reported = observed
        return min((observed - self._baseline) // _KIB, _U32_MAX)


@contextmanager
def download_liveness(source: DownloadSource, what: str) -> Generator[None]:
    """Report the measured growth of a resolution that may download.

    ``source`` names what is fetched, and decides the wire stage and the cache
    trees watched; ``what`` names it for a person ("iic/... for
    punctuation"). Nothing is said until the trees first grow (a
    ``StanzaCatalog`` is announced at once, see there): then the download
    is announced and its growth reported at most every ``POLL_INTERVAL_S``
    seconds, with a last measurement when the body returns, so a download
    shorter than one poll is still reported. A ``COMPLETE`` phase on the stage
    follows only a download that was announced and whose body returned
    normally: a cache hit says nothing, and a failed download is not announced
    as complete, its exception says what happened.
    """
    stage = download_stage(source)
    meter = GrowthMeter(_roots(source))
    stop = threading.Event()
    announced = threading.Event()
    announcement = f"Downloading {what} (one-time; future runs use the local cache)…"

    def announce() -> None:
        if not announced.is_set():
            announced.set()
            _announce(stage, PhaseEdgeV2.BEGIN, announcement)

    def report(grown_kib: int) -> None:
        announce()
        write_load_progress(stage, DownloadGrowthV2(grown_kib=grown_kib))

    match source:
        case StanzaCatalog():
            announce()
        case (
            HfRepo()
            | ModelScopeRepo()
            | TorchHubBundle()
            | StanzaResources()
            | StanzaWithTransformer()
        ):
            pass
        case _ as unreachable:
            assert_never(unreachable)

    def report_growth() -> None:
        while not stop.wait(POLL_INTERVAL_S):
            grown_kib = meter.poll()
            if grown_kib is not None:
                report(grown_kib)

    reporter = threading.Thread(
        target=report_growth, name=f"download-liveness:{stage}", daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop.set()
        reporter.join()
    # One last measurement, on this thread once the reporter has stopped: a
    # download shorter than one poll is still announced and counted.
    grown_kib = meter.poll()
    if grown_kib is not None:
        report(grown_kib)
    if announced.is_set():
        _announce(stage, PhaseEdgeV2.COMPLETE, f"Downloaded {what}.")
