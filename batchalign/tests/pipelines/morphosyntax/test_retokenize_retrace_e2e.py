"""End-to-end regression test: morphotag --retokenize on CHAT with retraces.

This test runs the actual batchalign3 CLI on a CHAT file containing
a retrace marker and verifies the pipeline completes without error.

Bug: MOST corpus 40415b.cha utterance 36 failed with "MOR item count (5)
does not match alignable word count (6)" when --retokenize was used.

Source: data/childes-other-data/Chinese/Cantonese/MOST/10002/40415b.cha
"""

from __future__ import annotations

import pytest

from ._managed_morphotag import ManagedMorphotag, Tokenization

CHAT_WITH_RETRACE = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\tyue\n"
    "@Participants:\tPAR0 Participant\n"
    "@ID:\tyue|MOST|PAR0|||||Participant|||\n"
    "@Media:\t40415b, audio\n"
    # Exact utterance from MOST 40415b.cha line 46 (with NAK bullet delimiters)
    "*PAR0:\t呢 度 <下 次> [/] 下 次 食 飯 啦 . \x1510245"
    "0_112560\x15\n"
    "@End\n"
)


@pytest.mark.golden
@pytest.mark.integration
def test_morphotag_retokenize_with_retrace_succeeds() -> None:
    """morphotag --retokenize must not crash on utterances with retraces.

    Guards: morphotag retokenize with retrace succeeds for the supplied linguistic source.
    Kind: journey
    Not a type: ManagedMorphotag excludes direct-mode composition, but cannot prove subprocess success or these model-derived serialized MOR and GRA bytes. The residual observation is morphotag retokenize with retrace succeeds for the supplied linguistic source.
    State: The shared managed CLI fixture stages one CHAT source and reads its real output file.
    Concurrency: Each submission has private temporary paths while the server owns shared model scheduling.
    """
    output_text = ManagedMorphotag.from_environment().run(
        CHAT_WITH_RETRACE,
        Tokenization.RETOKENIZE,
    )
    assert "%mor:" in output_text, f"Output should have %mor tiers:\n{output_text}"
    assert "[/]" in output_text, f"Retrace should be preserved:\n{output_text}"
