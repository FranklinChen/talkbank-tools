"""Materialize a pinned ModelScope snapshot before any loader runs.

The ModelScope counterpart of ``pinned_hub.resolve_repository``. FunASR's
Paraformer composition lives on ModelScope, and FunASR will download it itself
when handed a model id. It used to: the download then happened inside the
first transcription request, under a time budget derived from the audio's
duration, and on 2026-11-05 a 14-minute first download timed that request out,
was retried by a fresh worker, and started again from nothing.

Resolving the snapshot here, at worker bootstrap, moves the download to where
it belongs: before the worker reports ready, under ``download_liveness``, so
the supervisor's startup stall window sees it moving. The loader is then
handed the resolved snapshot only, the same arrangement the Hugging Face
members have.

Two ModelScope behaviours this module does not take on trust, both observed on
2026-11-05 (ModelScope 1.40):

- **A cache lookup returns a directory, not a model.** The local lookup
  returns a snapshot directory whether or not its files finished downloading.
  A download a startup stall interrupted left ``model.pt.incomplete``; the
  retry's lookup returned that directory, FunASR loaded without a checkpoint,
  and the transcript was confident nonsense with no error anywhere. A
  ``ModelScopeSnapshot`` is therefore admitted only complete.
- **A legacy cache answers for any tag.** ModelScope reuses an unversioned
  "legacy" directory for whatever revision is asked, and even downloads a new
  tag into it, over the old revision's files. A pinned member is therefore
  always resolved in its own versioned directory (``local_dir``), which the
  lookup and the download both honour, so the tag is the directory by
  construction.

ModelScope reports no commit behind a tag, so the identity recorded for these
members stays the declared one (``_declared_member``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from batchalign.worker._model_loading.pinned_hub import PinnedSnapshotError


class SnapshotRefusal(Enum):
    """Why a directory is not admitted as a model."""

    MISSING = "the directory does not exist"
    PARTIAL = "it holds a partial download (*.incomplete)"
    LACKS_REQUIRED = "it lacks a file the loader reads"


class SnapshotRefused(PinnedSnapshotError):
    """A directory was refused as a ModelScope snapshot, with the reason."""

    def __init__(self, path: Path, refusal: SnapshotRefusal) -> None:
        super().__init__(f"{path}: {refusal.value}")
        self.refusal = refusal


def _refusal(path: Path, requires: tuple[str, ...]) -> SnapshotRefusal | None:
    """Why ``path`` is not a complete model, or ``None`` if it is."""
    if not path.is_dir():
        return SnapshotRefusal.MISSING
    if any(path.rglob("*.incomplete")):
        return SnapshotRefusal.PARTIAL
    if not all((path / required).is_file() for required in requires):
        return SnapshotRefusal.LACKS_REQUIRED
    return None


@dataclass(frozen=True, slots=True)
class ModelScopeSnapshot:
    """A COMPLETE ModelScope model on local disk.

    The constructor is the admission: it refuses a missing directory, one
    holding any ``*.incomplete`` file, and one missing a file in
    ``requires`` (raising ``SnapshotRefused``). So every route to a value is
    a proof that the loader will read finished files, including a test's.
    """

    path: Path
    requires: tuple[str, ...]
    """The files the loader reads, which the constructor checked are there."""

    def __post_init__(self) -> None:
        refusal = _refusal(self.path, self.requires)
        if refusal is not None:
            raise SnapshotRefused(self.path, refusal)

    @classmethod
    def admit(
        cls, path: Path, requires: tuple[str, ...]
    ) -> ModelScopeSnapshot | SnapshotRefusal:
        """The snapshot at ``path`` if it is complete, else why it is not."""
        try:
            return cls(path=path, requires=requires)
        except SnapshotRefused as refused:
            return refused.refusal


def _cache_root() -> Path:
    """ModelScope's download root: ``MODELSCOPE_CACHE``, else
    ``~/.cache/modelscope``, by ModelScope's own precedence. Versioned
    snapshots land under ``<root>/models`` and legacy ones under
    ``<root>/hub`` or ``<root>`` itself, so watching the root covers both."""
    from modelscope.hub.utils.utils import _modelscope_hub_cache_root

    return Path(_modelscope_hub_cache_root())


def _versioned_dir(model_id: str, tag: str) -> Path:
    """Where ModelScope keeps ``model_id`` at ``tag``:
    ``<root>/models/<owner>--<name>/snapshots/<tag>``, its own layout for
    versioned downloads, so existing downloads are reused."""
    return _cache_root() / "models" / model_id.replace("/", "--") / "snapshots" / tag


def resolve_modelscope_snapshot(
    model_id: str, tag: str | None, *, kind: str, requires: tuple[str, ...]
) -> ModelScopeSnapshot:
    """Materialize ``model_id`` at ``tag`` and return it, complete.

    ``tag`` of ``None`` resolves the hub default, for a member the plan left
    unpinned (ModelScope then chooses the directory). ``requires`` names the
    files the loader reads (``model.pt`` for a FunASR model). A complete
    snapshot already on disk is returned without a network call. Anything else
    (nothing cached, or a snapshot an interrupted download left partial) is
    fetched under ``download_liveness``, which completes partial files, and the
    result must then be complete or the load is refused.
    """
    from modelscope import snapshot_download
    from modelscope_hub.errors import CacheNotFound

    from batchalign.worker._download_liveness import (
        ModelScopeRepo,
        download_liveness,
    )

    local_dir = None if tag is None else str(_versioned_dir(model_id, tag))
    pinned = tag or "the hub default"
    try:
        cached = snapshot_download(
            model_id,
            revision=tag,
            local_dir=local_dir,
            local_files_only=True,  # audit: allow cache probe; a miss downloads below
        )
    except CacheNotFound:
        cached = None
    except Exception as exc:
        raise PinnedSnapshotError(
            f"could not look up {model_id} at {pinned} in the ModelScope cache: {exc}"
        ) from exc
    if cached is not None:
        complete = ModelScopeSnapshot.admit(Path(cached), requires)
        if isinstance(complete, ModelScopeSnapshot):
            return complete

    try:
        with download_liveness(ModelScopeRepo(model_id), what=f"{model_id} for {kind}"):
            downloaded = snapshot_download(model_id, revision=tag, local_dir=local_dir)
    except Exception as exc:
        raise PinnedSnapshotError(
            f"could not resolve {model_id} at {pinned} from ModelScope: {exc}"
        ) from exc
    complete = ModelScopeSnapshot.admit(Path(downloaded), requires)
    match complete:
        case ModelScopeSnapshot():
            return complete
        case SnapshotRefusal() as refusal:
            raise PinnedSnapshotError(
                f"ModelScope returned {model_id} at {pinned} incomplete: "
                f"{downloaded}: {refusal.value} (requires {', '.join(requires)}); "
                f"refusing to load a model whose files did not finish"
            )


__all__ = [
    "ModelScopeSnapshot",
    "SnapshotRefusal",
    "SnapshotRefused",
    "resolve_modelscope_snapshot",
]
