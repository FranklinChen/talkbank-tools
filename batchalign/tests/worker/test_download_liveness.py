"""Download liveness: growth is observed, never assumed, and announced once.

The worker's half of the startup stall contract (``_download_liveness``):
a download reports progress only when the hub's cache tree has grown, so a
download that stops growing goes quiet and the Rust supervisor's stall window
expires. ``GrowthMeter`` holds that rule; the reporter thread only schedules
it, which is why these tests need no thread and no clock.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

import pytest

import batchalign.worker._download_liveness as liveness
from batchalign.worker._download_liveness import GrowthMeter, tree_bytes
from batchalign.worker._types_v2 import DownloadGrowthV2, LoadMeasureV2, PhaseV2


def write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_tree_bytes_counts_files_once_and_ignores_links(tmp_path: Path) -> None:
    """A hub snapshot links to the blob it stores; the blob counts once."""
    write(tmp_path / "blobs" / "abc", 4096)
    snapshot = tmp_path / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to(tmp_path / "blobs" / "abc")
    assert tree_bytes(tmp_path) == 4096


def test_tree_bytes_of_a_missing_root_is_zero(tmp_path: Path) -> None:
    assert tree_bytes(tmp_path / "not-yet-created") == 0


MIB = 1 << 20


def test_the_meter_reports_only_growth_since_its_last_report(tmp_path: Path) -> None:
    write(tmp_path / "already-cached", 10 * 1024)
    meter = GrowthMeter((tmp_path,))

    assert meter.poll() is None, "nothing grew since the meter was made"

    write(tmp_path / "blobs" / "part.incomplete", 3 * MIB)
    assert meter.poll() == 3 * 1024, "3 MiB grew since the baseline, in KiB"
    assert meter.poll() is None, "no growth since that report: stay quiet"

    write(tmp_path / "blobs" / "part.incomplete", 5 * MIB)
    assert meter.poll() == 5 * 1024, (
        "the count is since the baseline, not the last poll"
    )


def test_a_trickle_is_not_progress(tmp_path: Path) -> None:
    """THE 2026-11-05 CASE: a download crawling at about 0.14 KB/s grew a few
    KiB per poll. Growth under ``MIN_REPORTED_GROWTH_BYTES`` since the last
    report is not reported, so a trickle meets the stall window; once a full
    MiB has arrived since the last report, it is reported."""
    meter = GrowthMeter((tmp_path,))
    for kib in range(4, 1024, 4):
        write(tmp_path / "blobs" / "part.incomplete", kib * 1024)
        assert meter.poll() is None, f"{kib} KiB is a trickle, not progress"

    write(tmp_path / "blobs" / "part.incomplete", MIB)
    assert meter.poll() == 1024


def test_a_shrinking_tree_never_lowers_the_count(tmp_path: Path) -> None:
    """A temporary file removed after its rename shrinks the tree; the
    reported count must not go backwards, and nothing is reported until the
    tree passes its previous high mark by a full MiB."""
    meter = GrowthMeter((tmp_path,))
    write(tmp_path / "tmp" / "chunk", 6 * MIB)
    assert meter.poll() == 6 * 1024

    (tmp_path / "tmp" / "chunk").unlink()
    write(tmp_path / "blobs" / "file", 4 * MIB)
    assert meter.poll() is None

    write(tmp_path / "blobs" / "file", 7 * MIB)
    assert meter.poll() == 7 * 1024


STAGE = "downloading_hf_org_x"


def watched_download(root: Path):
    """A Hugging Face download of ``org/x`` whose cache is ``root``."""
    return liveness.download_liveness(liveness.HfRepo("org/x"), "x for ASR")


@pytest.fixture(autouse=True)
def _hf_cache_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hugging Face source watches only the test's directory."""
    monkeypatch.setattr(liveness, "hf_cache_roots", lambda: (tmp_path,))


def _spelled(stage: str, measure: LoadMeasureV2) -> str:
    """One load event as ``<stage>:<edge>`` (a phase) or ``<stage>@<KiB>``
    (growth)."""
    match measure:
        case PhaseV2(edge=edge):
            return f"{stage}:{edge.value}"
        case DownloadGrowthV2(grown_kib=grown_kib):
            return f"{stage}@{grown_kib}"
        case _ as unreachable:
            assert_never(unreachable)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every load event the download wrote, in order (see ``_spelled``).

    Patched at the one writer liveness uses, so the announcement's phase and
    the growth's measure are the ones the real code built."""
    written: list[str] = []
    monkeypatch.setattr(
        liveness,
        "write_load_progress",
        lambda stage, measure: written.append(_spelled(stage, measure)),
    )
    return written


def test_a_cache_hit_says_nothing(tmp_path: Path, events: list[str]) -> None:
    """A resolution served from the cache grows nothing, so nothing is
    announced: there is no separate cached-or-not guess to get wrong."""
    with watched_download(tmp_path):
        pass
    assert events == []


def test_a_download_is_announced_reported_and_completed(
    tmp_path: Path, events: list[str]
) -> None:
    """A download shorter than one poll is still reported, by the last
    measurement taken when the body returns."""
    with watched_download(tmp_path):
        write(tmp_path / "blobs" / "model.pt", 2 * MIB)
    assert events == [f"{STAGE}:begin", f"{STAGE}@2048", f"{STAGE}:complete"]


def test_a_failed_download_is_not_announced_as_complete(
    tmp_path: Path, events: list[str]
) -> None:
    with (
        pytest.raises(OSError, match="network unreachable"),
        watched_download(tmp_path),
    ):
        write(tmp_path / "blobs" / "model.pt.incomplete", 2 * MIB)
        raise OSError("network unreachable")
    assert f"{STAGE}:complete" not in events


def test_every_watched_root_counts(tmp_path: Path) -> None:
    """Hugging Face assembles files in its chunk cache before they reach the
    hub cache; growth in either is the download moving."""
    hub, xet = tmp_path / "hub", tmp_path / "xet"
    meter = GrowthMeter((hub, xet))
    write(xet / "chunk", 2 * MIB)
    assert meter.poll() == 2048


def test_the_catalog_is_announced_when_its_download_starts(
    tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its caller fetches it only once it is known absent, and at about 1 MB
    it may never cross the growth floor, so it is announced at once."""
    monkeypatch.setattr(liveness, "stanza_roots", lambda: (tmp_path,))
    with liveness.download_liveness(liveness.StanzaCatalog(), "the catalog"):
        pass
    assert events == [
        "downloading_stanza_catalog:begin",
        "downloading_stanza_catalog:complete",
    ]


def test_every_source_kind_has_its_own_stage() -> None:
    """The supervisor keys progress marks by stage, so two kinds of download
    of one name must not share one."""
    stages = {
        liveness.download_stage(source)
        for source in (
            liveness.HfRepo("a/b"),
            liveness.ModelScopeRepo("a/b"),
            liveness.TorchHubBundle("a/b"),
            liveness.StanzaResources("a_b"),
            liveness.StanzaCatalog(),
        )
    }
    assert len(stages) == 5
