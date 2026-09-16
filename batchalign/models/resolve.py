"""Resolve model shortcodes + language to HuggingFace IDs."""

from __future__ import annotations

from batchalign.inference._domain_types import LanguageCode

_RESOLVER: dict[str, dict[LanguageCode, str]] = {
    # The ``utterance`` family is deliberately absent. The utterance-boundary
    # model a language loads is named by the Rust manifest
    # (``model_manifest::UTSEG_BOUNDARY_MODELS``) and sent to this worker with
    # the spawn, because an id must be known BEFORE a load in order to pin its
    # revision. A copy here could only disagree with that one, and the copy was
    # the half that decided which model actually loaded.
    #
    # Per-language default HF model_id for the ``whisper_hub`` ASR engine.
    # Entries are added reactively from empirical evaluation, not
    # speculatively, and each one carries a dated provenance comment so
    # a maintainer can re-evaluate it when Rev.AI / stock Whisper change.
    #
    # Absent languages are handled by
    # ``batchalign.inference.whisper_hub.resolve_whisper_hub_model_id``,
    # which raises ``WhisperHubModelNotFoundError`` telling the user to
    # pass an explicit ``model_id`` via ``--engine-overrides``. This
    # surfaces the gap loudly rather than silently falling back to an
    # inappropriate default.
    "whisper_hub": {
        # 2026-04-22: seeded from empirical eval. thennal/whisper-medium-ml
        # produced 100 % Malayalam-script coherent output on a 73-second
        # failing sample where stock openai/whisper-medium and
        # openai/whisper-large-v3 both collapsed into Khmer / Gurmukhi
        # character loops and hallucinated "Thank you for watching." See
        # batchalign3/book/src/reference/whisper-hub-asr.md for the full
        # comparison and the eval artifacts on the operational workspace.
        "mal": "thennal/whisper-medium-ml",
    },
}


def resolve(model_class: str, lang_code: LanguageCode) -> str | None:
    """Resolve one model family/language pair to a concrete model identifier."""
    return _RESOLVER.get(model_class, {}).get(lang_code)
