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

So every pinned Hugging Face model goes through one door: resolve the snapshot,
then read the commit off the directory that actually exists on disk.

# Why the commit is read from the directory name

The hub cache stores a revision at ``<cache>/models--<org>--<name>/snapshots/
<commit>/``, so the leaf directory IS the commit, and it describes the bytes
the loader is about to read rather than what any index claimed. A leaf that is
not a 40-character hexadecimal commit means the layout is not what this
function understands, and that is a refusal: continuing would mean reporting a
revision nobody verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

from huggingface_hub import snapshot_download

from batchalign.worker._types_v2 import RequestedModelV2

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class PinnedSnapshotError(RuntimeError):
    """A pinned snapshot could not be resolved, or could not be identified.

    A ``RuntimeError`` so the worker's executor taxonomy classifies it the way
    it classifies every other load failure, and so it reaches the operator as a
    named failure instead of a wrong model quietly loading.
    """


@dataclass(frozen=True, slots=True)
class ResolvedSnapshot:
    """One revision on disk: where it is, and which commit it actually is."""

    path: str
    """Local directory a loader should be pointed at."""

    commit: str
    """The commit that directory holds, read from the cache layout."""


def hub_commit_of(member: RequestedModelV2) -> str | None:
    """The commit a Hugging Face member is pinned to, or ``None`` if unpinned.

    Shared by every pinned Hugging Face loader, because "which commit did the
    plan choose for this member" has one answer and three callers. ``None`` is
    a real case (see ``RequestedRevisionV2.Unpinned``): the hub default is
    resolved and whatever it lands on is reported.

    Any other revision kind means the manifest and this loader disagree about
    which hub the model lives on, which is a refusal rather than something to
    paper over: a tag or a provider parameter cannot address a Hugging Face
    snapshot, and guessing one would load an unknown revision.
    """
    revision = member.revision
    if revision.kind == "commit":
        return revision.commit
    if revision.kind == "unpinned":
        return None
    raise ValueError(
        f"{member.id} is a Hugging Face model, so it cannot be loaded from a "
        f"{revision.kind!r} revision"
    )


class HubFiles(Enum):
    """Which of a repository's files a loader reads, so only those download.

    ``snapshot_download`` with no filter fetches every file a repository
    holds. A Transformers repository commonly carries its weights in several
    formats at once: ``openai/whisper-large-v3`` holds Flax, PyTorch ``.bin``,
    fp32 shards and safetensors, 23 GB, of which the loader reads the 3 GB
    ``model.safetensors``. Each loader states its family; there is no
    default, so a new loader cannot inherit the whole-repository fetch by
    omission.
    """

    TRANSFORMERS = "transformers"
    """A Transformers model: its configs, tokenizer and template files, and
    the default-variant weights, safetensors when the revision has them and
    PyTorch ``.bin`` otherwise (the precedence ``from_pretrained`` applies).
    Weight variants (``model.fp32-*``) and other frameworks' formats (Flax,
    TensorFlow, ONNX) are never fetched."""

    WHOLE_REPOSITORY = "whole-repository"
    """Everything in the repository. For FunASR models, which read whatever
    their ``config.yaml`` names (``model.pt``, ``am.mvn``, BPE models); a
    hand-written list would track FunASR's internals. These repositories are
    small (SenseVoice about 0.9 GB, its VAD about 4 MB)."""


_TRANSFORMERS_SUPPORT = ("*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken")
"""Configs, tokenizer vocabularies and chat templates: small, and which of
them a model reads varies by architecture, so every one is fetched."""

_SAFETENSORS_WEIGHTS = (
    "model.safetensors",
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
)
_PYTORCH_WEIGHTS = (
    "pytorch_model.bin",
    "pytorch_model-*-of-*.bin",
    "pytorch_model.bin.index.json",
)


def _has_default_weights(snapshot: Path, patterns: tuple[str, ...]) -> bool:
    """Whether a materialized snapshot holds weights matching ``patterns``."""
    return any(any(snapshot.glob(pattern)) for pattern in patterns)


def _download(
    model_id: str, commit: str | None, patterns: tuple[str, ...] | None
) -> str:
    """One hub call; every way it can fail becomes one named error.

    A missing repository, a gated one, a bad token and an offline cache miss
    are all "this revision could not be materialized", and what the operator
    needs is the reason, not the exception class.
    """
    try:
        return snapshot_download(
            model_id,
            revision=commit,
            allow_patterns=list(patterns) if patterns is not None else None,
        )
    except Exception as exc:
        pinned = commit or "the hub default"
        raise PinnedSnapshotError(
            f"could not resolve {model_id} at {pinned}: {exc}"
        ) from exc


def _materialize(model_id: str, commit: str | None, files: HubFiles) -> str:
    """Download exactly the files ``files`` names and return the snapshot."""
    match files:
        case HubFiles.WHOLE_REPOSITORY:
            return _download(model_id, commit, None)
        case HubFiles.TRANSFORMERS:
            path = _download(
                model_id, commit, _TRANSFORMERS_SUPPORT + _SAFETENSORS_WEIGHTS
            )
            if _has_default_weights(Path(path), _SAFETENSORS_WEIGHTS):
                return path
            # No safetensors at this revision: the PyTorch weights are the
            # ones `from_pretrained` would load, so those are fetched.
            path = _download(model_id, commit, _PYTORCH_WEIGHTS)
            if _has_default_weights(Path(path), _PYTORCH_WEIGHTS):
                return path
            pinned = commit or "the hub default"
            raise PinnedSnapshotError(
                f"{model_id} at {pinned} has neither safetensors nor PyTorch "
                f"weights in the default variant, so there is nothing a "
                f"Transformers loader would read"
            )
        case _ as unreachable:
            assert_never(unreachable)


def resolve_pinned_snapshot(
    model_id: str, commit: str | None, files: HubFiles
) -> ResolvedSnapshot:
    """Materialize the ``files`` a loader reads of ``model_id`` at ``commit``,
    and report what landed.

    ``commit`` of ``None`` means the plan did not pin this model (see
    ``RequestedRevisionV2.Unpinned``): the hub default is resolved and the
    commit it resolved to is reported, which is the whole point of leaving a
    model unpinned rather than refusing the job.
    """
    path = _materialize(model_id, commit, files)

    observed = Path(path).name
    if not _COMMIT_PATTERN.match(observed):
        raise PinnedSnapshotError(
            f"resolved {model_id} to {path}, whose leaf {observed!r} is not a "
            f"40-character commit; this worker cannot say which revision it "
            f"loaded, so it refuses rather than report an unverified one"
        )
    if commit is not None and observed != commit:
        raise PinnedSnapshotError(
            f"asked the hub for {model_id} at {commit} and it resolved to "
            f"{observed}; refusing rather than load a revision the plan did "
            f"not choose"
        )
    return ResolvedSnapshot(path=path, commit=observed)


__all__ = [
    "HubFiles",
    "PinnedSnapshotError",
    "ResolvedSnapshot",
    "hub_commit_of",
    "resolve_pinned_snapshot",
]
