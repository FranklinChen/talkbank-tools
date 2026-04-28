"""Stanza and language-code loading helpers for the worker process.

This module exists to keep Stanza-specific bootstrap policy out of the generic
worker entrypoint and the request-time inference routers. It owns:

- ISO language-code normalization for Stanza
- the MWT/non-MWT processor policy (capability-driven; see ``should_request_mwt``)
- installation of preloaded Stanza pipelines into worker runtime state
- the utseg-specific stanza-config builder used by inference dispatch
"""

from __future__ import annotations

import logging
import threading

from batchalign.inference._domain_types import LanguageCode, LanguageCode2
from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    get_cached_capability_table,
)
from batchalign.worker._types import _state

L = logging.getLogger("batchalign.worker")


class UnsupportedLanguageError(ValueError):
    """Stanza has no usable pipeline for the requested language.

    Distinct from a configuration error: the request itself cannot be
    served by this worker, so callers should reject the job upstream
    rather than retry. Surfaced as a typed error so downstream code
    can branch on it cleanly instead of pattern-matching the deep
    ``KeyError`` Stanza would otherwise raise from
    ``maintain_processor_list``.
    """


def should_request_mwt(
    alpha2: LanguageCode2, table: StanzaCapabilityTable | None
) -> bool:
    """Decide whether to request the ``mwt`` processor for ``alpha2``.

    Single source of truth: the Stanza capability table built from the
    installed catalog's ``resources.json`` (see ``_stanza_capabilities``).
    A previous hardcoded ``MWT_LANGS`` set drifted from the catalog and
    requested ``mwt`` for languages Stanza no longer ships it for (e.g.
    Swedish on Stanza 1.11), crashing the worker at bootstrap.

    Returns False when the table is unavailable: the conservative choice
    is to omit ``mwt`` and let Stanza tokenize/POS/lemma/depparse only,
    rather than guess and risk an ``UnsupportedProcessorError``.
    """
    if table is None:
        return False
    for cap in table.languages.values():
        if cap.alpha2 == alpha2:
            return cap.has_mwt
    return False


def iso3_to_alpha2(iso3: LanguageCode) -> LanguageCode2:
    """Convert ISO-639-3 language code to ISO-639-1 for Stanza.

    Batchalign uses ISO-639-3 broadly, but Stanza is configured with mostly
    ISO-639-1-style identifiers plus a few special cases. This function is the
    canonical bridge so the rest of the worker code does not embed ad hoc
    language-code fallbacks or guess at unsupported codes.
    """
    mapping: dict[str, str] = {
        "eng": "en", "spa": "es", "fra": "fr", "deu": "de",
        "ita": "it", "por": "pt", "nld": "nl", "zho": "zh",
        "jpn": "ja", "kor": "ko", "ara": "ar", "heb": "he",
        "tur": "tr", "fin": "fi", "dan": "da", "swe": "sv",
        "nor": "nb", "pol": "pl", "ces": "cs", "ron": "ro",
        "hun": "hu", "bul": "bg", "hrv": "hr", "slk": "sk",
        "slv": "sl", "ukr": "uk", "ell": "el", "fas": "fa",
        "hin": "hi", "urd": "ur", "ben": "bn", "tam": "ta",
        "tel": "te", "kan": "kn", "mal": "ml", "tha": "th",
        "vie": "vi", "ind": "id", "msa": "ms", "tgl": "tl",
        "kat": "ka", "hye": "hy", "cat": "ca", "glg": "gl",
        "eus": "eu", "cym": "cy", "gle": "ga", "gla": "gd",
        "mlt": "mt", "est": "et", "lav": "lv", "lit": "lt",
        "isl": "is", "yue": "zh",
        "cmn": "zh",
        "rus": "ru", "afr": "af", "lat": "la", "ltz": "lb",
    }
    if iso3 in mapping:
        return mapping[iso3]
    if len(iso3) == 2:
        return iso3
    L.warning(
        "Unknown ISO-639-3 code %r - passing through unchanged for Stanza",
        iso3,
    )
    return iso3


def load_stanza_models(lang: LanguageCode) -> None:
    """Load Stanza morphosyntax models for one language.

    The resulting pipeline, tokenizer context, and lock are installed into the
    shared worker state so request handlers can do pure inference routing
    without rebuilding Stanza pipelines on every call.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import (
        TokenizerContext,
        make_tokenizer_postprocessor,
    )

    # Preflight gate: consult the capability table BEFORE calling
    # stanza.Pipeline. The capability table is built from the installed
    # Stanza catalog (resources.json) and is the only source of truth
    # that stays correct across Stanza upgrades. Hardcoded lists
    # (Rust SUPPORTED_STANZA_CODES, the iso3_to_alpha2 mapping below)
    # have drifted multiple times and are now treated as advisory.
    # Without this gate, an unsupported language reaches stanza.Pipeline
    # which raises KeyError('packages') deep in maintain_processor_list —
    # the worker dies before emitting its ready signal, the daemon sees a
    # generic IPC error, and the user gets "transcription failed" with
    # the linguistic root cause buried in stderr.
    table = get_cached_capability_table()
    if table is None:
        raise UnsupportedLanguageError(
            f"Cannot load Stanza for {lang!r}: capability table is "
            "unavailable (Stanza not installed or resources.json "
            "could not be read). The worker must not load any pipeline."
        )
    if lang not in table.languages:
        sample = sorted(table.languages.keys())[:8]
        raise UnsupportedLanguageError(
            f"Stanza has no processor packages for language {lang!r}. "
            f"It may appear in Stanza's resources.json as a stub "
            f"(charlm-only) entry, but no usable Pipeline can be built. "
            f"Supported languages include: {sample} (and "
            f"{len(table.languages) - len(sample)} more)."
        )

    alpha2 = iso3_to_alpha2(lang)

    # MWT availability comes from Stanza's installed resources.json — never
    # from a hardcoded list. A stale list silently crashes the worker when
    # upstream drops a model (see the 2026-04-15 Swedish bootstrap failure).
    has_mwt = should_request_mwt(alpha2, table)
    processors = "tokenize,pos,lemma,depparse"
    if has_mwt:
        processors += ",mwt"

    ctx = TokenizerContext()
    lock = threading.Lock()

    # The Stanza pipeline shape varies by language because tokenization and MWT
    # support are not uniform across the supported languages.
    if alpha2 == "ja":
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_pretokenized=True,
            package={
                "tokenize": "combined",
                "pos": "combined",
                "lemma": "combined",
                "depparse": "combined",
            },
        )
    elif not has_mwt:
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_pretokenized=True,
        )
    elif alpha2 == "en":
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx, alpha2),
            package={"mwt": "gum"},
        )
    else:
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx, alpha2),
        )

    # Preserve any pipelines already loaded for other languages in this worker.
    existing_pipelines = _state.stanza_pipelines or {}
    existing_contexts = _state.stanza_contexts or {}
    existing_pipelines[lang] = nlp
    existing_contexts[lang] = ctx
    _state.stanza_pipelines = existing_pipelines
    _state.stanza_contexts = existing_contexts
    _state.stanza_nlp_lock = lock

    try:
        _state.stanza_version = stanza.__version__
    except AttributeError:
        _state.stanza_version = "unknown"


def load_stanza_retokenize_model(lang: LanguageCode) -> None:
    """Lazy-load a Stanza pipeline with neural tokenization for Chinese retokenize.

    Unlike the default Chinese pipeline (which uses ``tokenize_pretokenized=True``),
    this variant lets Stanza's neural tokenizer segment the text into words.
    Used when ``--retokenize`` is requested for Mandarin (``cmn``/``zho``).

    The pipeline is stored under key ``"{lang}:retok"`` in worker state so it
    coexists with the standard pretokenized pipeline.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import TokenizerContext

    alpha2 = iso3_to_alpha2(lang)
    if alpha2 != "zh":
        L.warning(
            "load_stanza_retokenize_model called for non-Chinese lang %s — skipping",
            lang,
        )
        return

    processors = "tokenize,pos,lemma,depparse"
    ctx = TokenizerContext()

    nlp = stanza.Pipeline(
        lang=alpha2,
        processors=processors,
        download_method=DownloadMethod.REUSE_RESOURCES,
        tokenize_no_ssplit=True,
        tokenize_pretokenized=False,
    )

    retok_key = f"{lang}:retok"
    existing_pipelines = _state.stanza_pipelines or {}
    existing_contexts = _state.stanza_contexts or {}
    existing_pipelines[retok_key] = nlp
    existing_contexts[retok_key] = ctx
    _state.stanza_pipelines = existing_pipelines
    _state.stanza_contexts = existing_contexts

    L.info("Loaded Stanza retokenize pipeline for %s (key=%s)", lang, retok_key)


def load_utseg_builder(lang: LanguageCode) -> None:
    """Load the utseg config builder for one primary language.

    Utterance segmentation uses a lighter-weight configuration boundary than
    morphosyntax. Instead of preloading full pipelines here, the worker stores a
    callable that can derive the necessary Stanza config bundle from a set of
    languages at inference time.
    """
    alpha2 = iso3_to_alpha2(lang)
    mwt_exclude = {"zh", "ja", "ko", "th", "vi", "my"}
    has_mwt = alpha2 not in mwt_exclude

    def build_stanza_config_from_langs(
        langs: list[str],
    ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
        """Build the Stanza config payload expected by utseg inference.

        Processor selection is per-language: only request processors that
        Stanza actually supports for each language (from the capability
        table). Languages without constituency get sentence-boundary
        segmentation instead.
        """
        from batchalign.worker._stanza_capabilities import get_cached_capability_table

        table = get_cached_capability_table()

        lang_alpha2: list[str] = []
        configs: dict[str, dict[str, str | bool]] = {}
        for language in langs:
            alpha2_code = iso3_to_alpha2(language)
            if alpha2_code == "zh":
                alpha2_code = "zh-hans"
            lang_alpha2.append(alpha2_code)

            processors: set[str] = {"tokenize", "pos", "lemma"}

            # Only add constituency if the language explicitly supports it.
            # When capability data is unavailable, prefer the safe
            # sentence-boundary fallback over guessing and crashing.
            lang_caps = table.languages.get(language) if table else None
            if lang_caps is not None and lang_caps.has_constituency:
                processors.add("constituency")

            # Only add MWT if the language supports it.
            if lang_caps is not None and lang_caps.has_mwt:
                processors.add("mwt")
            elif table is None and has_mwt:
                processors.add("mwt")

            configs[alpha2_code] = {
                "processors": ",".join(sorted(processors)),
                "tokenize_pretokenized": True,
            }
        return lang_alpha2, configs

    _state.utseg_config_builder = build_stanza_config_from_langs

    try:
        import stanza

        _state.utseg_version = stanza.__version__
    except (ImportError, AttributeError):
        _state.utseg_version = "unknown"
