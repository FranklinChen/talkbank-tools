#!/usr/bin/env python3
"""Prove the `qwen3_fa` forced-alignment engine on REAL weights and REAL audio.

The standing rule is to build only what we can prove, and a unit test with a
fake aligner proves the wiring, not the engine. This runs the production seam,
`batchalign.inference.qwen_forced_alignment.load_qwen_fa`, over a real clip
with the words of its own hand transcript, and checks the timings a reviewer
would check:

  1. THE FOLD: how many of OUR words the aligner's own units covered, and how
     many came back untimed. This is the check the first proof did not make.
     That run used a CHARACTER-tokenized fixture, where the aligner's own
     segmentation happened to equal ours, so it proved the engine on the one
     input shape where the fold is the identity. Run it on a multi-character
     word list too, which is what most real CJK looks like.
  2. the timings are monotone and non-overlapping,
  3. every timing lies inside the clip, which is what a window-versus-file
     coordinate error would break, and
  4. the zero-length rate.

It is a SCRIPT and not a test because it downloads or loads two 0.6B models
and takes minutes; the suite must stay fast. Run it when the engine changes.

    uv run --no-sync python scripts/prove_qwen3_fa.py \\
        --audio batchalign/tests/languages/cantonese/fixtures/05b_clip.wav \\
        --words '<the words, in order>' --lang yue

Exit 0 only if every check passes; the failures are printed, never summarised
away.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Clip:
    """The audio and the words to align against it."""

    samples: object
    duration_ms: int
    words: list[str]


def load_clip(audio: Path, words: list[str]) -> Clip:
    """Read the audio; the words are GIVEN, never parsed out of CHAT here.

    Hand-parsing CHAT in Python is banned in this workspace, and the ban has
    teeth: a first version of this script split the main tier on whitespace,
    kept the bullet `350_1375` as if it were a word, and the aligner refused
    the mismatch its own tokenizer produced (`3501375`). The refusal was
    correct and the harness was wrong. So the caller passes the words, taken
    from the transcript by eye or by a Rust tool, and this file contains no
    CHAT knowledge at all.
    """
    import soundfile  # type: ignore[import-not-found]

    samples, rate = soundfile.read(str(audio), dtype="float32")
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    duration_ms = int(round(len(samples) / float(rate) * 1000.0))
    return Clip(samples=samples, duration_ms=duration_ms, words=words)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--words",
        required=True,
        help="the spoken words, whitespace separated, in order; NOT parsed from CHAT here",
    )
    parser.add_argument("--lang", default="yue")
    args = parser.parse_args(argv)

    from batchalign.inference.qwen_forced_alignment import (
        QwenAlignmentMismatch,
        QwenSegmentationDrift,
        QwenSegmentationNotDecomposable,
        QwenUnitNotTimed,
        TimedChatWord,
        UntimedChatWord,
        load_qwen_fa,
    )

    clip = load_clip(args.audio, args.words.split())
    print(f"clip: {args.audio.name}, {clip.duration_ms} ms, {len(clip.words)} words")
    print(f"words: {' '.join(clip.words)}")

    host = load_qwen_fa(args.lang)
    try:
        fold = host.align_words(clip.samples, clip.words)
    except (
        QwenAlignmentMismatch,
        QwenSegmentationDrift,
        QwenSegmentationNotDecomposable,
        QwenUnitNotTimed,
    ) as refusal:
        # A refusal is a RESULT: the engine declines rather than padding, and
        # that is the documented behaviour. It still means this clip is not
        # proof that the engine aligns, so it exits non-zero.
        print(f"REFUSED: {refusal}")
        return 1

    # THE FOLD. The aligner segments the transcript into its OWN units (CJK
    # characters individually, `black+bird` as one unit spelled `blackbird`),
    # and the host folds those spans back onto our words. Reporting the two
    # counts is the whole point of this pass: a run where every word is
    # covered on a character-tokenized clip says nothing about a run on real
    # multi-character words.
    segmentation = host.segment(clip.words)
    covered = sum(1 for word in fold.words if isinstance(word, TimedChatWord))
    untimed = [word for word in fold.words if isinstance(word, UntimedChatWord)]
    print(
        f"fold: {len(clip.words)} CHAT words -> "
        f"{len(segmentation.units)} aligner units; "
        f"{covered} covered, {len(untimed)} untimed"
    )
    for word in untimed:
        print(f"  untimed: {word.word!r} ({word.reason.value})")

    # `to_indexed_timings` is the same lowering the worker uses, so this reads
    # exactly what the wire would carry. A first version of this harness probed
    # for `.start`/`.end`, found neither, and reported 26 untimed words on a
    # run that had in fact aligned all 26; read the fields by name.
    timings = list(fold.to_indexed_timings())
    print(f"aligned: {covered} timed words of {len(timings)} rows")

    failures: list[str] = []
    zero_length: list[str] = []
    previous_end = -1
    for index, word in enumerate(timings):
        text = word.word
        if word.interval_ms is None:
            # NOT a failure by itself: an untimed word is a reported outcome,
            # already counted above. It becomes a failure only if the whole
            # clip is untimed, which the coverage check below catches.
            print(f"  {index:>3}  {'':>6} {'':>6}  {text}  UNTIMED")
            continue
        start, end = word.interval_ms
        start, end = int(start), int(end)
        print(f"  {index:>3}  {start:>6} {end:>6}  {text}")
        if start < 0 or end < 0:
            failures.append(f"{index} ({text}): negative timing")
            continue
        if end < start:
            failures.append(f"{index} ({text}): negative length {start}..{end}")
        elif end == start:
            # An OBSERVATION, not a failure. The model answers on an 80 ms
            # frame grid, so a short syllable can land with its start and end
            # in one frame. The Rust postprocess heals anything under
            # MIN_HEALABLE_WORD_DURATION_MS (40) by rebalancing with a
            # neighbour and records that as `Origin::RebalancedWithNeighbour`,
            # so the guess stays visible in the provenance tally. Counted here
            # because the RATE is worth knowing before recommending the engine.
            zero_length.append(f"{index} ({text}) at {start}")
        if start < previous_end:
            failures.append(
                f"{index} ({text}): starts {start} before the previous end {previous_end}"
            )
        if end > clip.duration_ms:
            failures.append(
                f"{index} ({text}): ends {end} past the clip's {clip.duration_ms} ms, "
                "which is what a window-versus-file coordinate error looks like"
            )
        previous_end = end

    if len(timings) != len(clip.words):
        failures.append(f"{len(clip.words)} words in, {len(timings)} rows out")
    if covered == 0:
        failures.append("the fold covered none of the requested words")

    if zero_length:
        rate = len(zero_length) / covered
        print(
            f"\nOBSERVED: {len(zero_length)} of {covered} timed words "
            f"({rate:.0%}) came back zero-length on the model's frame grid; "
            "the postprocess heals each by rebalancing with a neighbour and "
            "records Origin::RebalancedWithNeighbour."
        )
        for observation in zero_length:
            print(f"  {observation}")

    if failures:
        print(f"\nFAILED, {len(failures)} check(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        f"\nPASSED: {covered}/{len(clip.words)} words covered by the fold, "
        f"{len(untimed)} untimed, monotone, inside the clip."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
