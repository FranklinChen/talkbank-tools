"""Resolve the word-vector file a Stanza lemmatizer's contextual models need.

The defect (Stanza 1.15.0, Greek)
---------------------------------
A Stanza lemmatizer checkpoint can carry "contextual" sub-models (lemma
classifiers) that each need a word-vector pretrain file. Stanza finds that file
in one of two ways: the pipeline option ``lemma_pretrain_path``, which Stanza
fills from the lemma package's ``pretrain`` dependency in ``resources.json``,
or, when that option is absent, the path stored inside the checkpoint at
training time.

Stanza 1.15.0's Greek lemmatizer (``el/lemma/gdt_nocharlm``) lists no pretrain
dependency, and its checkpoint stores the build machine's path,
``/nlp/scr/<user>/stanza_resources/el/pretrain/conll17.pt``. Every Greek
pipeline with a ``lemma`` processor therefore fails to load with
``FileNotFoundError``, although the very file it wants, ``el/pretrain/
conll17.pt``, is installed in our resources directory. A scan of all 28
installed 1.15.0 lemmatizers found this in Greek only (2026-10-01).

The repair
----------
Make sure the lemma model is on disk (downloading it if needed, so a fresh
install is repaired too, not skipped), then read the checkpoint's stored paths.
A stored path that exists is used as is. A missing one is relocated: its ``<lang>/pretrain/<file>`` tail is joined onto our
resources directory, which is the file the model was trained with, under our
install root. The relocated path must exist, or loading stops with
:class:`StanzaLemmaPretrainError` naming the model, the stored path and the
place we looked. There is no fallback to another pretrain: a contextual model
given different word vectors from the ones it was trained on would still load
and would silently mislabel.

This is keyed on the defect, not on Greek, so a later model packaged the same
way is repaired, and one packaged some other broken way stops loudly.
"""

from __future__ import annotations

import functools
from pathlib import Path, PurePosixPath

# The checkpoint key that lists contextual sub-models, and the key inside each
# sub-model's saved arguments that names its word-vector file.
_CONTEXTUAL_KEY = "contextual"
_PRETRAIN_ARG = "wordvec_pretrain_file"
# The resources-directory layout a stored pretrain path must end in to be
# relocatable: `<lang>/pretrain/<file>.pt`.
_PRETRAIN_DIR = "pretrain"
# The `stanza.Pipeline` option that supplies a contextual model's pretrain.
_PIPELINE_OPTION = "lemma_pretrain_path"
# Stanza's name for its model-free lemmatizer (the lemma is the word).
_IDENTITY_LEMMATIZER = "identity"


class StanzaLemmaPretrainError(RuntimeError):
    """A lemmatizer's contextual model needs a word-vector file we cannot find."""


def _stored_pretrain_paths(checkpoint: object, model_file: Path) -> list[str]:
    """Every word-vector path the checkpoint's contextual sub-models name.

    The checkpoint is Stanza's own format; a shape other than the one Stanza
    itself reads is not something to work around, so it stops loading.
    """
    try:
        sub_models = checkpoint.get(_CONTEXTUAL_KEY, [])  # type: ignore[attr-defined]
        stored = [model["args"].get(_PRETRAIN_ARG) for model in sub_models]
    except (AttributeError, KeyError, TypeError) as error:
        raise StanzaLemmaPretrainError(
            f"{model_file}: unexpected lemma checkpoint layout ({error!r})"
        ) from error
    return [path for path in stored if path]


def _relocate(stored: str, resources_dir: Path, model_file: Path) -> Path:
    """The stored path's `<lang>/pretrain/<file>` tail, under ``resources_dir``."""
    parts = PurePosixPath(stored).parts
    if len(parts) < 3 or parts[-2] != _PRETRAIN_DIR:
        raise StanzaLemmaPretrainError(
            f"{model_file}: contextual lemma model names {stored!r}, which does not "
            f"exist and is not a `<lang>/{_PRETRAIN_DIR}/<file>` path we can relocate"
        )
    candidate = resources_dir.joinpath(*parts[-3:])
    if not candidate.is_file():
        raise StanzaLemmaPretrainError(
            f"{model_file}: contextual lemma model names {stored!r}, which does not "
            f"exist, and its file is not installed at {candidate} either"
        )
    return candidate


def relocated_pretrain(model_file: Path, resources_dir: Path) -> Path | None:
    """The pretrain to pass for the lemmatizer at ``model_file``.

    ``None`` when every stored path exists (or none is stored), so Stanza reads
    them itself; otherwise the one installed file every missing path relocates
    to, which this function has checked exists.
    """
    import torch

    # `weights_only=True`, as Stanza's own lemma loader reads these files.
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
    stored = _stored_pretrain_paths(checkpoint, model_file)
    missing = [path for path in stored if not Path(path).is_file()]
    if not missing:
        return None
    relocated = {_relocate(path, resources_dir, model_file) for path in missing}
    if len(relocated) != 1:
        # One Stanza option serves every contextual sub-model, so sub-models
        # trained on different pretrains cannot all be repaired with it.
        raise StanzaLemmaPretrainError(
            f"{model_file}: contextual lemma models need different pretrains "
            f"{sorted(map(str, relocated))}; one {_PIPELINE_OPTION} cannot serve them"
        )
    return relocated.pop()


@functools.cache
def lemma_pretrain_options(
    alpha2: str, lemma_package: str | None = None
) -> dict[str, str]:
    """``stanza.Pipeline`` options for ``alpha2``'s lemmatizer.

    ``lemma_package`` is the package the pipeline requests for ``lemma``, or
    ``None`` for the language's default in Stanza's catalog. A language whose
    catalog entry has no lemmatizer needs no option.

    Cached per process: the answer depends only on installed files, and the
    utseg config builder asks on every request.

    Downloads the lemma model first when it is absent, because the repair has
    to read it before the pipeline that would otherwise download it is built.
    """
    import json

    import stanza
    from stanza.resources.common import DEFAULT_MODEL_DIR, maintain_processor_list

    resources_dir = Path(DEFAULT_MODEL_DIR)
    catalog = json.loads((resources_dir / "resources.json").read_text(encoding="utf-8"))
    if lemma_package is None:
        lemma_package = (
            catalog.get(alpha2, {}).get("default_processors", {}).get("lemma")
        )
        if lemma_package is None:
            return {}
    if lemma_package == _IDENTITY_LEMMATIZER:
        # Stanza's built-in lemmatizer for languages without a trained one
        # (Thai, Vietnamese): no checkpoint, so nothing to repair.
        return {}
    # A requested package may name a bundle (Japanese `combined` installs
    # `combined_nocharlm`); Stanza's own resolver says which model file the
    # pipeline will load, so this reads that file and no other.
    resolved = maintain_processor_list(
        catalog, alpha2, None, {"lemma": lemma_package}, maybe_add_mwt=False
    )
    lemma_model = next(specs[0].package for name, specs in resolved if name == "lemma")
    model_file = resources_dir / alpha2 / "lemma" / f"{lemma_model}.pt"
    if not model_file.is_file():
        from batchalign.worker._progress import emit_download_event

        emit_download_event(
            stage=f"downloading_stanza_lemma_{alpha2}",
            user_message=(
                f"Downloading the Stanza {alpha2} lemmatizer (one-time; future "
                "runs will use the local cache)…"
            ),
        )
        stanza.download(
            lang=alpha2,
            package=None,
            processors={"lemma": lemma_model},
            download_json=False,
        )
    if not model_file.is_file():
        raise StanzaLemmaPretrainError(
            f"Stanza did not install the {alpha2} lemmatizer at {model_file}"
        )
    path = relocated_pretrain(model_file, resources_dir)
    return {} if path is None else {_PIPELINE_OPTION: str(path)}
