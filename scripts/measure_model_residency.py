"""Measure what one model holds while in use, and what a release returns.

The host-memory lease an on-demand model load takes is sized from these
numbers (book/src/batchalign/developer/memory-safety.md), so they are
measurements, never guesses. Each run loads one model in a fresh process,
after importing its libraries (so the libraries' own footprint is not counted
as the model's), runs ``--batches`` inferences on it (one by default), each on an input
``--input-scale`` times the base size (one by default: the English document
below, or 30 seconds of audio), and reports:

- resident memory before the load, after it, after each inference, and after
  releasing it and collecting garbage;
- the process's peak resident memory, which is what an out-of-memory kill
  sees.

The figure a lease is sized from is the in-use one, after an inference, never
the built one. Building a model can leave most of its weights unread, so they
are not yet resident: the Stanza coreference pipeline measured +520 MiB built
and +2406 MiB after resolving one two-sentence document (2026-11-06, ming).
A build-only measurement under-reserves every lease it sizes.

Downloads are off unless ``--allow-download`` is given: a model not already
cached is reported as not measured, never fetched behind the operator's back.

Usage:
    uv run --no-sync python scripts/measure_model_residency.py baseline
    uv run --no-sync python scripts/measure_model_residency.py coref [--allow-download] [--batches N]
    uv run --no-sync python scripts/measure_model_residency.py pyannote
    uv run --no-sync python scripts/measure_model_residency.py embedding
    uv run --no-sync python scripts/measure_model_residency.py morphosyntax
    uv run --no-sync python scripts/measure_model_residency.py utseg

``baseline`` is the other half: what a worker holds before any load, the
Python runtime with the worker package and the libraries its loads import
(torch, Stanza, transformers). Every model figure above is measured on top of
those imports, so the two add up to what a loaded worker holds.

``morphosyntax`` and ``utseg`` build one English Stanza pipeline through the
worker's own loaders (the morphosyntax pipeline, and utseg's constituency
pipeline) and annotate a short English document on it. Stanza fetches a
missing language pack itself, so they need the English pack already
installed to measure without a download. ``pyannote`` and ``embedding`` run
on a synthesized waveform held in memory; no media file is written.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MIB = 1024 * 1024
COREF_PACKAGE = "ontonotes-singletons_roberta-large-lora"
"""The coreference package BA3 loads (``batchalign.inference.coref``)."""

ENGLISH_DOCUMENT = (
    "The teacher gave the children a book . She asked them to read it .\n\n"
    "Then the children told their parents about the story they had read ."
)
"""Plain English prose (not CHAT) that each Stanza probe's inference reads."""

AUDIO_SECONDS = 30
"""Length of the synthesized waveform the audio probes run on, per unit of
``--input-scale``."""


def english_document(scale: int) -> str:
    """The English document, repeated `scale` times."""
    return "\n\n".join([ENGLISH_DOCUMENT] * scale)


def rss_mb() -> float:
    """Current resident set size, in MiB."""
    import psutil

    return float(psutil.Process().memory_info().rss) / MIB


def peak_mb() -> float:
    """Peak resident set size so far, in MiB (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / MIB if sys.platform == "darwin" else peak / 1024


def import_worker_libraries() -> None:
    """What a worker imports before or during its first load: the worker
    package and the libraries every model load here builds on."""
    import stanza  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401

    import batchalign.worker  # noqa: F401


def synthetic_waveform(sample_rate_hz: int, scale: int) -> Any:
    """Low-level noise as a mono float32 waveform, in memory: signal enough for
    a model's stages to run, with nothing read from disk."""
    import numpy as np

    rng = np.random.default_rng(0)
    samples = rng.standard_normal(sample_rate_hz * AUDIO_SECONDS * scale) * 0.05
    return samples.astype(np.float32)


@dataclass(frozen=True)
class Probe:
    """One model to measure: how the worker builds it, and one inference
    shaped like a request's, run on what was built with an input of the given
    scale (``--input-scale``)."""

    build: Callable[[bool], Any]
    exercise: Callable[[Any, int], None]


def build_morphosyntax(_allow_download: bool) -> Any:
    """The English morphosyntax pipeline's cache entry, through the worker's
    loader, which installs it in the worker's bounded Stanza cache."""
    from batchalign.worker._stanza_loading import load_stanza_models, pipeline_cache

    load_stanza_models("eng")
    return pipeline_cache()["eng"]


def build_utseg(_allow_download: bool) -> Any:
    """The English utseg constituency pipeline, through the worker's loader."""
    from batchalign.worker._stanza_loading import load_utseg_pipeline

    return load_utseg_pipeline("eng")


def run_stanza(entry: Any, scale: int) -> None:
    """Annotate the English document on a cached Stanza pipeline entry."""
    with entry.calling() as nlp:
        nlp(english_document(scale))


def build_coref(allow_download: bool) -> Any:
    """The Stanza English coreference pipeline BA3 builds for a coref batch."""
    import stanza
    from stanza.pipeline.core import DownloadMethod

    return stanza.Pipeline(
        lang="en",
        processors="tokenize,coref",
        package={"coref": COREF_PACKAGE},
        tokenize_pretokenized=True,
        download_method=(
            DownloadMethod.REUSE_RESOURCES if allow_download else DownloadMethod.NONE
        ),
    )


def run_coref(pipeline: Any, scale: int) -> None:
    """Resolve the English document, as one coref batch item does."""
    pipeline(english_document(scale))


def build_pyannote(_allow_download: bool) -> Any:
    """BA3's local pyannote diarization pipeline, through its own loader (Hugging
    Face downloads are governed by ``HF_HUB_OFFLINE``)."""
    from batchalign.inference.pyannote_local import load_local_pyannote_pipeline

    return load_local_pyannote_pipeline()


def run_pyannote(loaded: Any, scale: int) -> None:
    """Diarize the synthesized waveform through BA3's own call."""
    from batchalign.inference.speaker import infer_pyannote_speaker_prepared_audio

    infer_pyannote_speaker_prepared_audio(
        synthetic_waveform(16_000, scale), 16_000, None, pipeline=loaded
    )


def build_embedding(_allow_download: bool) -> Any:
    """BA3's speaker-embedding model, through its own loader (Hugging Face
    downloads are governed by ``HF_HUB_OFFLINE``)."""
    from batchalign.inference import speaker_embedding

    return speaker_embedding.load_speaker_embedding_model()


def run_embedding(model: Any, scale: int) -> None:
    """Embed one window of the synthesized waveform, shaped as BA3 hands it
    (``(batch, channel, sample)``)."""
    import torch

    window = synthetic_waveform(model.sample_rate_hz, scale)
    model.inference(torch.from_numpy(window).reshape(1, 1, -1))


PROBES: dict[str, Probe] = {
    "coref": Probe(build_coref, run_coref),
    "pyannote": Probe(build_pyannote, run_pyannote),
    "embedding": Probe(build_embedding, run_embedding),
    "morphosyntax": Probe(build_morphosyntax, run_stanza),
    "utseg": Probe(build_utseg, run_stanza),
}


def measure_baseline() -> int:
    """Report the worker baseline: resident memory once the worker package and
    its loads' libraries are imported, before any model."""
    started = time.monotonic()
    import_worker_libraries()
    gc.collect()
    print(
        f"baseline: {rss_mb():.0f} MiB resident after imports "
        f"({time.monotonic() - started:.1f} s), peak {peak_mb():.0f} MiB"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure what one model holds while in use."
    )
    parser.add_argument("model", choices=sorted([*PROBES, "baseline"]))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--batches",
        type=int,
        default=1,
        help="inferences to run on the one built model (default 1)",
    )
    parser.add_argument(
        "--input-scale",
        type=int,
        default=1,
        help="base units per inference input: document repeats or 30 s spans",
    )
    args = parser.parse_args()
    if args.input_scale < 1:
        parser.error("--input-scale must be at least 1")
    if args.batches < 1:
        parser.error("--batches must be at least 1: the in-use figure needs one")
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.model == "baseline":
        return measure_baseline()

    # Counted as the libraries (the baseline), not the model.
    import_worker_libraries()
    gc.collect()
    base = rss_mb()
    probe = PROBES[args.model]
    started = time.monotonic()
    try:
        model = probe.build(args.allow_download)
    # A failure is the measurement's answer, reported, never hidden.
    except Exception as error:
        print(f"{args.model}: NOT MEASURED ({type(error).__name__}: {error})")
        return 1
    built = rss_mb()
    print(
        f"{args.model}: base {base:.0f} MiB, built {built:.0f} MiB "
        f"(+{built - base:.0f} MiB in {time.monotonic() - started:.1f} s)"
    )
    in_use = built
    for batch in range(1, args.batches + 1):
        started = time.monotonic()
        try:
            probe.exercise(model, args.input_scale)
        # As above: a failed inference is reported, and nothing is in use.
        except Exception as error:
            print(
                f"{args.model} inference {batch}: NOT MEASURED "
                f"({type(error).__name__}: {error})"
            )
            return 1
        gc.collect()
        in_use = rss_mb()
        print(
            f"{args.model} inference {batch}: {in_use:.0f} MiB "
            f"(+{in_use - base:.0f} MiB in use, {time.monotonic() - started:.1f} s)"
        )
    peak_in_use = peak_mb()
    del model
    gc.collect()
    released = rss_mb()
    print(
        f"{args.model}: in use +{in_use - base:.0f} MiB, peak {peak_in_use:.0f} MiB, "
        f"after release {released:.0f} MiB (returned {in_use - released:.0f} MiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
