"""The lemmatizer pretrain repair decides from the checkpoint, never from the language.

Fake checkpoints carry only what the resolver reads: the ``contextual`` list
and each sub-model's ``args.wordvec_pretrain_file``. The real defect (Stanza
1.15.0 Greek) is covered by the golden Greek morphotag tests, which fail to
load the pipeline without this repair.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from batchalign.worker._stanza_lemma_pretrain import (
    StanzaLemmaPretrainError,
    relocated_pretrain,
)

BUILD_MACHINE = "/nlp/scr/someone/stanza_resources"


def _checkpoint(tmp_path: Path, *stored: str) -> Path:
    path = tmp_path / "lemma.pt"
    contextual = [
        {"model_type": "LSTM", "args": {"wordvec_pretrain_file": p}} for p in stored
    ]
    torch.save({"model": {}, "contextual": contextual}, path)
    return path


def _install(resources: Path, relative: str) -> Path:
    path = resources / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"vectors")
    return path


def test_no_contextual_model_needs_no_option(tmp_path: Path) -> None:
    model = _checkpoint(tmp_path)
    assert relocated_pretrain(model, tmp_path / "resources") is None


def test_a_stored_path_that_exists_is_left_to_stanza(tmp_path: Path) -> None:
    installed = _install(tmp_path / "elsewhere", "el/pretrain/conll17.pt")
    model = _checkpoint(tmp_path, str(installed))
    assert relocated_pretrain(model, tmp_path / "resources") is None


def test_a_missing_build_machine_path_is_relocated_to_the_same_file(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    installed = _install(resources, "el/pretrain/conll17.pt")
    model = _checkpoint(tmp_path, f"{BUILD_MACHINE}/el/pretrain/conll17.pt")
    assert relocated_pretrain(model, resources) == installed


def test_a_missing_path_whose_file_is_not_installed_is_refused(tmp_path: Path) -> None:
    model = _checkpoint(tmp_path, f"{BUILD_MACHINE}/el/pretrain/conll17.pt")
    with pytest.raises(StanzaLemmaPretrainError, match="not installed at"):
        relocated_pretrain(model, tmp_path / "resources")


def test_a_missing_path_outside_the_pretrain_layout_is_refused(tmp_path: Path) -> None:
    model = _checkpoint(tmp_path, "/somewhere/vectors.pt")
    with pytest.raises(StanzaLemmaPretrainError, match="path we can relocate"):
        relocated_pretrain(model, tmp_path / "resources")


def test_sub_models_needing_different_pretrains_are_refused(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    _install(resources, "el/pretrain/conll17.pt")
    _install(resources, "el/pretrain/fasttext.pt")
    model = _checkpoint(
        tmp_path,
        f"{BUILD_MACHINE}/el/pretrain/conll17.pt",
        f"{BUILD_MACHINE}/el/pretrain/fasttext.pt",
    )
    with pytest.raises(StanzaLemmaPretrainError, match="different pretrains"):
        relocated_pretrain(model, resources)


def test_an_unexpected_checkpoint_layout_stops_loading(tmp_path: Path) -> None:
    path = tmp_path / "lemma.pt"
    torch.save({"contextual": [{"model_type": "LSTM"}]}, path)
    with pytest.raises(
        StanzaLemmaPretrainError, match="unexpected lemma checkpoint layout"
    ):
        relocated_pretrain(path, tmp_path / "resources")


def test_the_identity_lemmatizer_needs_no_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thai and Vietnamese lemmatize by identity: no checkpoint, no download."""
    import stanza.resources.common

    from batchalign.worker._stanza_lemma_pretrain import (
        lemma_pretrain_options,
        resolve_lemma_pretrain,
    )

    (tmp_path / "resources.json").write_text(
        '{"th": {"default_processors": {"lemma": "identity"}}}', encoding="utf-8"
    )
    monkeypatch.setattr(stanza.resources.common, "DEFAULT_MODEL_DIR", str(tmp_path))
    resolve_lemma_pretrain.cache_clear()
    try:
        assert lemma_pretrain_options("th") == {}
    finally:
        resolve_lemma_pretrain.cache_clear()
