"""Resolve a pinned Hugging Face snapshot, and observe what was resolved.

# Why this exists

A model the control plane pinned must be loaded at exactly that revision, and
the loader must be able to say which revision it loaded. Neither half can be
taken on trust from the modelling library:

- FunASR's Hugging Face branch IGNORES the revision it is given
  (``get_or_download_model_dir_hf`` calls ``snapshot_download(model)`` with no
  ``revision``), so passing ``model_revision`` there pins nothing at all. The
  only way to hold it to a revision is to resolve the snapshot here and hand
  it a local path.
- Reading ``config._commit_hash`` back out of a loaded ``transformers`` model
  would be an observation of a library internal that may be ``None``, and
  falling back to the revision we asked for would be assuming rather than
  observing, which is the substitution this workstream exists to remove. The
  utterance-boundary model read exactly that field, which is why its revision
  could be absent at all.

So every pinned Hugging Face model goes through one door per family: resolve
the snapshot (``resolve_transformers`` or ``resolve_repository``), then read
the commit off the directory that actually exists on disk. Each door returns
its own snapshot type (``TransformersSnapshot``, ``RepositorySnapshot``), and
each loader accepts only the type it can read: the family is the type of the
result, not a flag beside it.

# Why the commit is read from the directory name

The hub cache stores a revision at ``<cache>/models--<org>--<name>/snapshots/
<commit>/``, so the leaf directory IS the commit, and it describes the bytes
the loader is about to read rather than what any index claimed. A leaf that is
not a 40-character hexadecimal commit means the layout is not what this
function understands, and that is a refusal: continuing would mean reporting a
revision nobody verified.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import assert_never

from huggingface_hub import snapshot_download

from batchalign.worker._types_v2 import (
    LoadedModelV2,
    ObservedCommitV2,
    RequestedCommitV2,
    RequestedContentDigestV2,
    RequestedModelV2,
    RequestedProviderParameterV2,
    RequestedTagV2,
    RequestedUnpinnedV2,
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class PinnedSnapshotError(RuntimeError):
    """A pinned snapshot could not be resolved, or could not be identified.

    A ``RuntimeError`` so the worker's executor taxonomy classifies it the way
    it classifies every other load failure, and so it reaches the operator as a
    named failure instead of a wrong model quietly loading.
    """


def unpinned(model_id: str) -> RequestedModelV2:
    """A request for ``model_id`` at whatever the hub's default resolves to.

    For loaders the plan does not pin; the snapshot still records the commit
    that landed, so the load is identified even though nobody chose it.
    """
    return RequestedModelV2(id=model_id, revision=RequestedUnpinnedV2())


def hub_commit_of(member: RequestedModelV2) -> str | None:
    """The commit a Hugging Face member is pinned to, or ``None`` if unpinned.

    ``None`` is a real case (see ``RequestedRevisionV2.Unpinned``): the hub
    default is resolved and whatever it lands on is reported. Any other
    revision kind means the manifest and this loader disagree about which hub
    the model lives on, which is a refusal rather than something to paper
    over: a tag or a provider parameter cannot address a Hugging Face
    snapshot, and guessing one would load an unknown revision.
    """
    match member.revision:
        case RequestedCommitV2(commit=commit):
            return commit
        case RequestedUnpinnedV2():
            return None
        case (
            RequestedTagV2()
            | RequestedContentDigestV2()
            | RequestedProviderParameterV2()
        ) as other:
            raise ValueError(
                f"{member.id} is a Hugging Face model, so it cannot be loaded "
                f"from a {other.kind!r} revision"
            )
        case _ as unreachable:
            assert_never(unreachable)


_TRANSFORMERS_SUPPORT = ("*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken")
"""Configs, tokenizer vocabularies and chat templates: small, and which of
them a model reads varies by architecture, so every one is fetched."""


class WeightLayout(Enum):
    """A default-variant weight layout ``from_pretrained`` can read.

    Declared in ``from_pretrained``'s own precedence for a local directory
    (``transformers.modeling_utils._get_resolved_checkpoint_files``, read on
    5.22.1): a single safetensors file, then a safetensors index, then a
    single PyTorch file, then a PyTorch index. Iteration order IS that
    precedence, and a snapshot is given exactly one layout: some repositories
    ship the same weights more than once (``nvidia/audio-flamingo-3-hf``
    holds a 16.5 GB ``model.safetensors`` and the same weights in four
    shards), and fetching every layout present would download the model
    twice to read it once. Each value is the patterns that layout's files
    match; a layout is present when every pattern matches a file.
    """

    SAFETENSORS = ("model.safetensors",)
    SAFETENSORS_SHARDED = ("model.safetensors.index.json", "model-*-of-*.safetensors")
    PYTORCH = ("pytorch_model.bin",)
    PYTORCH_SHARDED = ("pytorch_model.bin.index.json", "pytorch_model-*-of-*.bin")

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        """Everything a Transformers loader reads with this layout: the
        support files and this layout's weights. What the resolver fetches,
        and what ``scripts/check_hub_file_selection.py`` reports."""
        weights: tuple[str, ...] = self.value
        return _TRANSFORMERS_SUPPORT + weights


def default_weights(names: Iterable[str]) -> WeightLayout | None:
    """The layout ``from_pretrained`` would read from files named ``names``,
    or ``None`` if none is present.

    Shared by the snapshot's own admission (over the files on disk) and
    ``scripts/check_hub_file_selection.py`` (over a repository listing), so
    the rule the checker reports is the rule a snapshot is admitted by.
    """
    present = tuple(names)
    for layout in WeightLayout:
        if all(fnmatch.filter(present, pattern) for pattern in layout.value):
            return layout
    return None


def _snapshot_names(snapshot: Path) -> list[str]:
    """Every file in a materialized snapshot, as repository-relative paths."""
    return [
        entry.relative_to(snapshot).as_posix()
        for entry in snapshot.rglob("*")
        if entry.is_file()
    ]


def _admitted_commit(requested: RequestedModelV2, path: Path) -> str:
    """The commit the snapshot directory ``path`` holds, or a refusal.

    The hub cache stores a revision at ``.../snapshots/<commit>/``, so the leaf
    IS the commit, and it describes the bytes a loader is about to read. A leaf
    that is not a 40-character commit cannot be reported as a revision, and one
    other than the commit the plan pinned is a revision the plan did not
    choose.
    """
    if not path.is_dir():
        raise PinnedSnapshotError(f"{requested.id}: {path} is not a directory")
    observed = path.name
    if not _COMMIT_PATTERN.match(observed):
        raise PinnedSnapshotError(
            f"resolved {requested.id} to {path}, whose leaf {observed!r} is not "
            f"a 40-character commit; this worker cannot say which revision it "
            f"loaded, so it refuses rather than report an unverified one"
        )
    pinned = hub_commit_of(requested)
    if pinned is not None and observed != pinned:
        raise PinnedSnapshotError(
            f"asked the hub for {requested.id} at {pinned} and it resolved to "
            f"{observed}; refusing rather than load a revision the plan did "
            f"not choose"
        )
    return observed


@dataclass(frozen=True, slots=True)
class TransformersSnapshot:
    """A Transformers model on local disk, at one verified commit, holding a
    weight layout ``from_pretrained`` reads.

    The constructor is the admission: it refuses a path that is not a
    snapshot directory, whose leaf is not a commit, whose commit is not the
    one ``requested`` pins, or that holds no default-variant weights. So
    every route to a value is a proof, not only ``resolve_transformers``. The
    only snapshot a Transformers loader accepts, so a loader cannot be handed
    a FunASR repository, and the only source of the id and revision a loader
    records, so neither can disagree with the files.
    """

    requested: RequestedModelV2
    """What the plan asked for: the hub id and its pinned (or unpinned) revision."""

    path: str
    """The snapshot directory a loader is pointed at (a string, which every
    loading library accepts)."""

    commit: str = field(init=False)
    """The commit ``path`` holds, derived by the constructor."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "commit", _admitted_commit(self.requested, Path(self.path))
        )
        if default_weights(_snapshot_names(Path(self.path))) is None:
            raise PinnedSnapshotError(
                f"{self.requested.id} at {self.commit} holds no default-variant "
                f"weights (safetensors or PyTorch, single or sharded), so there "
                f"is nothing a Transformers loader would read"
            )

    @property
    def model_id(self) -> str:
        """The hub id this snapshot was resolved from."""
        return self.requested.id

    @property
    def loaded(self) -> LoadedModelV2:
        """What loading this snapshot loads: the request, and the commit seen."""
        return LoadedModelV2(
            id=self.requested.id,
            requested=self.requested.revision,
            observed=ObservedCommitV2(commit=self.commit),
        )


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """Every file of a repository on local disk, at one verified commit.

    For FunASR models, which read whatever their ``config.yaml`` names
    (``model.pt``, ``am.mvn``, BPE models); a hand-written list would track
    FunASR's internals. These repositories are small (SenseVoice about
    0.9 GB, its VAD about 4 MB). The constructor admits it as
    ``TransformersSnapshot``'s does, less the weight-layout rule.
    """

    requested: RequestedModelV2
    """What the plan asked for: the hub id and its pinned (or unpinned) revision."""

    path: str
    """The snapshot directory a loader is pointed at (a string, which every
    loading library accepts)."""

    commit: str = field(init=False)
    """The commit ``path`` holds, derived by the constructor."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "commit", _admitted_commit(self.requested, Path(self.path))
        )

    @property
    def model_id(self) -> str:
        """The hub id this snapshot was resolved from."""
        return self.requested.id

    @property
    def loaded(self) -> LoadedModelV2:
        """What loading this snapshot loads: the request, and the commit seen."""
        return LoadedModelV2(
            id=self.requested.id,
            requested=self.requested.revision,
            observed=ObservedCommitV2(commit=self.commit),
        )


def _download(
    model_id: str, commit: str | None, patterns: tuple[str, ...] | None
) -> Path:
    """One hub call; every way it can fail becomes one named error.

    A missing repository, a gated one, a bad token and an offline cache miss
    are all "this revision could not be materialized", and what the operator
    needs is the reason, not the exception class.
    """
    try:
        return Path(
            snapshot_download(
                model_id,
                revision=commit,
                allow_patterns=list(patterns) if patterns is not None else None,
            )
        )
    except Exception as exc:
        pinned = commit or "the hub default"
        raise PinnedSnapshotError(
            f"could not resolve {model_id} at {pinned}: {exc}"
        ) from exc


@contextmanager
def _watched(model_id: str, kind: str) -> Iterator[None]:
    """Run a resolution under ``download_liveness``.

    A snapshot served from the cache grows nothing and says nothing; one that
    downloads is announced (named by ``kind``) and reports its measured
    growth, which is what keeps a slow first download inside the worker's
    startup stall window. There is no "is it cached?" probe to get wrong: one
    that checked only ``config.json`` would send a download whose weights an
    interruption left partial down the unwatched branch.
    """
    from batchalign.worker._download_liveness import HfRepo, download_liveness

    with download_liveness(HfRepo(model_id), what=f"{model_id} for {kind}"):
        yield


def resolve_transformers(
    requested: RequestedModelV2, *, kind: str
) -> TransformersSnapshot:
    """Materialize the files a Transformers loader reads of ``requested``.

    ``snapshot_download`` with no filter fetches every file a repository
    holds, and a Transformers repository commonly carries its weights in
    several formats at once: ``openai/whisper-large-v3`` holds Flax, PyTorch
    ``.bin``, fp32 shards and safetensors, 23 GB, of which the loader reads
    the 3 GB ``model.safetensors``. So layouts are asked for one at a time,
    in ``from_pretrained``'s precedence, until one is present: the first is
    the one it would read, and no later one is fetched. A layout the
    revision lacks matches no file, so asking for it downloads nothing.

    The first call fixes the commit and every later one names it. For an
    unpinned model a call without a commit asks the hub what its default
    branch is, one network round trip per layout tried, and a branch that
    moved between calls would assemble one snapshot from two revisions.
    """
    model_id = requested.id
    layouts = iter(WeightLayout)
    with _watched(model_id, kind):
        path = _download(
            model_id, hub_commit_of(requested), next(layouts).allow_patterns
        )
        commit = _admitted_commit(requested, path)
        for layout in layouts:
            if default_weights(_snapshot_names(path)) is not None:
                break
            path = _download(model_id, commit, layout.allow_patterns)
    return TransformersSnapshot(requested=requested, path=str(path))


def resolve_repository(requested: RequestedModelV2, *, kind: str) -> RepositorySnapshot:
    """Materialize every file of ``requested``, for FunASR."""
    with _watched(requested.id, kind):
        path = _download(requested.id, hub_commit_of(requested), None)
    return RepositorySnapshot(requested=requested, path=str(path))


__all__ = [
    "PinnedSnapshotError",
    "RepositorySnapshot",
    "TransformersSnapshot",
    "WeightLayout",
    "default_weights",
    "hub_commit_of",
    "resolve_repository",
    "resolve_transformers",
    "unpinned",
]
