"""Translation-engine bootstrap helpers for worker startup.

Two backends are supported:

* ``TranslationBackend.GOOGLE`` — the historical default. Wraps the
  ``googletrans`` library, which calls the public Google Translate
  endpoint. Requires reachability to ``translate.google.com`` and is
  therefore unsuitable for hosts behind the Great Firewall unless a VPN
  is active.
* ``TranslationBackend.SEAMLESS`` — Meta's SeamlessM4T, loaded locally
  from HuggingFace. No outbound network at inference time.

Selection is driven by the same ``engine_overrides`` dict ASR and FA use
(see ``asr.py::resolve_asr_engine``). The Rust control plane decides which
backend a worker pool loads and passes the choice through
``WorkerBootstrapRuntime.engine_overrides``.
"""

from __future__ import annotations

import logging

from batchalign.inference._domain_types import LanguageCode, TranslationBackend
from batchalign.worker._types import WorkerBootstrapRuntime, _state

L = logging.getLogger("batchalign.worker")


def load_translation_engine(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the translation engine for this worker.

    Dispatches on the resolved ``TranslationBackend`` so adding a new
    variant later forces a missing-arm error rather than silently
    falling through to Google.
    """
    backend = resolve_translate_engine(bootstrap.engine_overrides or None)
    if backend is TranslationBackend.GOOGLE:
        _load_google_translate()
    elif backend is TranslationBackend.SEAMLESS:
        _load_seamless_translate()
    else:
        # Exhaustive — if a new TranslationBackend variant is added and
        # not wired in here, we raise rather than leaving translate_fn
        # unset (the batch-infer handler would later fail opaquely).
        raise RuntimeError(f"unhandled translation backend: {backend!r}")


def resolve_translate_engine(
    engine_overrides: dict[str, str] | None,
) -> TranslationBackend:
    """Pick the translation backend from the engine-overrides dict.

    Precedence:

    1. An explicit ``"translate"`` entry selects that backend. Unknown
       values raise ``ValueError`` rather than silently falling back —
       a typo in a per-host config would otherwise produce silently-
       wrong translations.
    2. Default is Google, preserving historical behavior for hosts that
       never set a translate override.
    """
    if engine_overrides and "translate" in engine_overrides:
        choice = engine_overrides["translate"]
        if choice == "google":
            return TranslationBackend.GOOGLE
        if choice == "seamless":
            return TranslationBackend.SEAMLESS
        raise ValueError(
            f"unknown translate engine {choice!r}; "
            f"expected one of: google, seamless"
        )
    return TranslationBackend.GOOGLE


def _load_google_translate() -> None:
    """Bind ``_state.translate_fn`` to a googletrans-backed translator."""
    from googletrans import Translator

    async def _do_translate(translator: Translator, text: str) -> str:
        result = await translator.translate(text)
        return str(getattr(result, "text", result))

    def translate_fn(text: str, src_lang: LanguageCode) -> str:
        """Run the async translator behind the worker's synchronous IPC seam."""
        import asyncio

        translator = Translator()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do_translate(translator, text))
        finally:
            loop.close()

    _state.translate_backend = TranslationBackend.GOOGLE
    _state.translate_fn = translate_fn


def _load_seamless_translate() -> None:
    """Bind ``_state.translate_fn`` to a locally-loaded SeamlessM4T model.

    Model is downloaded from HuggingFace on first load and cached
    thereafter. Operators on hosts where the public HF endpoint is slow
    or blocked can point at a mirror via ``HF_ENDPOINT`` before the
    worker starts.
    """
    from transformers import AutoProcessor, SeamlessM4TModel

    from batchalign.worker._progress import (
        HF_ARTIFACTS_SEAMLESS,
        emit_hf_download_if_missing,
    )

    emit_hf_download_if_missing(
        "facebook/hf-seamless-m4t-medium",
        kind="translation",
        artifacts=HF_ARTIFACTS_SEAMLESS,
    )

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        "facebook/hf-seamless-m4t-medium"
    )
    model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
    # torch.nn.Module.eval() — sets the module to inference mode,
    # unrelated to Python's builtin eval().
    if hasattr(model, "eval"):
        model.eval()  # type: ignore[no-untyped-call]

    def seamless_fn(text: str, src_lang: LanguageCode) -> str:
        """Translate one text payload through SeamlessM4T."""
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
        output = model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        return str(processor.decode(output[0].tolist()[0], skip_special_tokens=True))

    _state.translate_backend = TranslationBackend.SEAMLESS
    _state.translate_fn = seamless_fn


__all__ = [
    "load_translation_engine",
    "resolve_translate_engine",
]
