"""Prove each hub model loads from only the files its family selects.

``check_hub_file_selection`` shows what WOULD download; this downloads it.
Each model is resolved by the real resolve edge (``resolve_transformers``)
into an EMPTY Hugging Face cache, so nothing but the selected files exists,
then loaded by BA3's own loader, which accepts only that snapshot. A file the
selection missed fails the load here instead of in a user's job. (The checks
used to repeat each loader's ``from_pretrained`` calls by hand; with every
loader taking a snapshot, they call the loaders themselves.)

The cache must be empty, so run with a fresh ``HF_HUB_CACHE`` (read when
``huggingface_hub`` is imported, so it has to be set in the environment):

    HF_HUB_CACHE=$(mktemp -d) uv run --no-sync python -m scripts.verify_hub_selection_loads [NAME ...]

With no NAME every check runs, which downloads tens of GB (Audio Flamingo 3
alone is 16.5 GB); name checks to run a subset. Delete the cache after.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from batchalign.worker._model_loading.pinned_hub import (
    TransformersSnapshot,
    resolve_transformers,
    unpinned,
)


def resolve(model_id: str) -> TransformersSnapshot:
    """The hub default of ``model_id``, resolved the way every loader's is."""
    snapshot = resolve_transformers(unpinned(model_id), kind="selection check")
    print(f"  {model_id} @ {snapshot.commit}")
    return snapshot


def whisper_asr() -> None:
    from batchalign.inference.asr import load_whisper_asr
    from batchalign.worker._model_loading.asr import STOCK_WHISPER_ID
    from batchalign.worker._types_v2 import DetectWhisperLanguageV2

    load_whisper_asr(resolve(STOCK_WHISPER_ID), decoding=DetectWhisperLanguageV2())


def whisper_fa() -> None:
    from batchalign.inference.fa import WHISPER_FA_MODEL_ID, load_whisper_fa

    load_whisper_fa(resolve(WHISPER_FA_MODEL_ID))


def utterance_bert() -> None:
    from batchalign.models.utterance.infer import BertUtteranceModel

    BertUtteranceModel(snapshot=resolve("talkbank/CHATUtterance-en"), lang="eng")


def qwen_aligner() -> None:
    import torch

    from batchalign.inference.qwen_forced_alignment import (
        QWEN_FORCED_ALIGNER_MODEL_ID,
        QwenForcedAligner,
    )

    QwenForcedAligner.load(
        resolve(QWEN_FORCED_ALIGNER_MODEL_ID), device="cpu", dtype=torch.float32
    )


def qwen_asr() -> None:
    import torch

    from batchalign.inference.languages.cantonese._qwen_common import (
        LoadedQwen,
        QwenSnapshots,
    )
    from batchalign.inference.qwen_forced_alignment import (
        QWEN_FORCED_ALIGNER_MODEL_ID,
    )
    from batchalign.worker._model_loading.asr import DEFAULT_QWEN_ASR_ID

    LoadedQwen.load(
        QwenSnapshots(
            asr=resolve(DEFAULT_QWEN_ASR_ID),
            aligner=resolve(QWEN_FORCED_ALIGNER_MODEL_ID),
        ),
        device="cpu",
        dtype=torch.float32,
    )


def nllb() -> None:
    from batchalign.worker._model_loading.translation import _load_nllb_translate

    _load_nllb_translate()


def seamless() -> None:
    from batchalign.worker._model_loading.translation import _load_seamless_translate

    _load_seamless_translate()


def machine_ear() -> None:
    from batchalign.inference.machine_ear import (
        CALIBRATED_MODEL_ID,
        EarFamily,
        load_machine_ear,
    )

    load_machine_ear(resolve(CALIBRATED_MODEL_ID), EarFamily.AUDIO_FLAMINGO_3)


CHECKS: dict[str, Callable[[], None]] = {
    "whisper-asr": whisper_asr,
    "whisper-fa": whisper_fa,
    "utterance-bert": utterance_bert,
    "qwen-aligner": qwen_aligner,
    "qwen-asr": qwen_asr,
    "nllb": nllb,
    "seamless": seamless,
    "machine-ear": machine_ear,
}


def main(names: list[str]) -> int:
    cache = os.environ.get("HF_HUB_CACHE", "")
    if not cache or any(Path(cache).iterdir()):
        print("HF_HUB_CACHE must name an empty directory", file=sys.stderr)
        return 2
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        print(f"unknown checks {unknown}; known: {sorted(CHECKS)}", file=sys.stderr)
        return 2
    failed = False
    for name in names or list(CHECKS):
        print(name)
        try:
            CHECKS[name]()
        # Every load failure is the finding, whatever its type.
        except Exception as error:
            print(f"  LOAD FAILED: {error}")
            failed = True
        else:
            print("  loaded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
