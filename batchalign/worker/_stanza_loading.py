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
from collections.abc import Sequence

from batchalign.inference._domain_types import LanguageCode, LanguageCode2
from batchalign.inference._italian_mwt import (
    INFINITIVE_ENDINGS,
    ItalianLexicon,
    ItalianMwtPolicy,
)
from batchalign.inference.types import StanzaNLP
from batchalign.worker._pipeline_cache import (
    StanzaPipelineCache,
    mwt_probe_key,
    retokenize_key,
)
from batchalign.worker._stanza_capabilities import (
    _ISO3_OVERRIDES,
    StanzaCapabilityTable,
    get_cached_capability_table,
)
from batchalign.worker._types import _state

L = logging.getLogger("batchalign.worker")

# Every Stanza alpha-2 code a Chinese ISO-639-3 code resolves to, DERIVED from
# the shared override table rather than spelled out.
#
# The retokenize loader's guard was the literal `"zh"` while `_ISO3_OVERRIDES`
# maps `cmn`, `zho` and `yue` all to `"zh-hans"`, so it rejected every code its
# only caller passes and the neural-tokenizer pipeline could not be loaded at
# all: a `--retokenize` Mandarin job logged "non-Chinese lang zho, skipping" and
# silently fell back to the pretokenized pipeline. A literal beside a mapping it
# has to agree with is the drift this module's own `iso3_to_alpha2` docstring
# warns about, one function further down.
_STANZA_CHINESE_ALPHA2: frozenset[str] = frozenset(
    _ISO3_OVERRIDES[iso3] for iso3 in ("cmn", "zho", "yue")
)

# One process-wide creation of the pipeline cache, ordered by a lock because
# two serving threads can reach a first load at the same time. The previous
# read-modify-write of `_state.stanza_pipelines` could lose one thread's
# pipeline entirely; `None` still means "no Stanza bootstrap has run".
_CACHE_CREATION_LOCK = threading.Lock()


def pipeline_cache() -> StanzaPipelineCache:
    """Return this worker's Stanza pipeline cache, creating it once."""
    with _CACHE_CREATION_LOCK:
        cache = _state.stanza_pipelines
        if cache is None:
            cache = StanzaPipelineCache()
            _state.stanza_pipelines = cache
        return cache


# Stanza's key for pos-independent lemmatizer entries (`_POS_INDEPENDENT` in
# `stanza/models/lemma/trainer.py`). Mirrored rather than imported because it is
# private there; a rename upstream surfaces as the empty-lexicon raise below,
# which is the fail-closed path, not a silent shrink.
_STANZA_POS_INDEPENDENT = "*"


class StanzaLexiconUnavailableError(RuntimeError):
    """Stanza's shipped lexicon could not be read off a loaded pipeline.

    The lexicon lives on the lemmatizer's trainer, which is a PRIVATE Stanza
    attribute (``processors["lemma"]._trainer.word_dict``). Reaching into it is
    a deliberate, isolated choice: it is the only route to the ~50k-form Italian
    word list Stanza already ships, and the alternative is a hand-maintained
    table of our own, which is exactly what the April 2026 MWT audit retired
    after five such tables drifted out of sync with the model.

    Isolating the access in one place means a Stanza upgrade that moves the
    attribute produces THIS error, loudly, at one seam, instead of silently
    degrading the Italian analysis. ``test_stanza_lexicon_extraction`` pins the
    shape so the break surfaces in CI rather than in a corpus.
    """


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
    ISO-639-1-style identifiers plus a few special cases. This function is
    the canonical bridge so the rest of the worker code does not embed ad
    hoc language-code fallbacks or guess at unsupported codes.

    Resolution order:

    1. **Special-case overrides**, for codes where ISO 639-3 disagrees with
       Stanza's own catalog labelling (e.g. ``yue``/``cmn`` both routing to
       Stanza ``zh``, ``nor``→``nb`` for Norwegian Bokmål as the default).
       These cases are *not* recoverable via pycountry because pycountry
       only encodes ISO-639's standard 1-to-1 mapping, which would point
       at codes Stanza does not ship.
    2. **pycountry**, for every other code with a standard ISO 639-1
       counterpart (``mar``→``mr``, ``swa``→``sw``, ...). This must be the
       fallback rather than a duplicate hardcoded dict, otherwise the
       hardcoded list inevitably drifts out of sync, Stanza adds a
       language, the capability table picks it up via pycountry, but
       ``iso3_to_alpha2`` returns the iso3 verbatim and ``stanza.Pipeline``
       crashes with "Language X is currently unsupported".
       (2026-05-06: Marathi ``mar`` failed exactly this way on
       ``childes-other-data/Biling/Gelman/Bystander/25.cha``.)
    3. **Pass-through with warning**, for genuinely unmapped codes
       (length-2 codes assumed to already be alpha-2; everything else
       hits the warning path).
    """
    # 1. Stanza-specific overrides: single source of truth shared with
    #    the capability-table builder. A second independent override dict
    #    here is the drift hazard that caused the 2026-05-06 ``mar``
    #    failure; do not reintroduce one.
    if iso3 in _ISO3_OVERRIDES:
        return _ISO3_OVERRIDES[iso3]

    # 2. pycountry for the standard ISO 639-3 ↔ ISO 639-1 cases.
    #    The capability table already uses pycountry; this keeps the two
    #    code paths honest about which languages they each understand.
    try:
        import pycountry

        lang = pycountry.languages.get(alpha_3=iso3)
        if lang is not None:
            alpha2 = getattr(lang, "alpha_2", None)
            if isinstance(alpha2, str) and alpha2:
                return alpha2
    except ImportError:
        # Fall through to the warning path.
        pass

    # 3. Already alpha-2? Pass through silently, common when callers feed a
    #    Stanza-style code straight in.
    if len(iso3) == 2:
        return iso3

    L.warning(
        "Unknown ISO-639-3 code %r - passing through unchanged for Stanza",
        iso3,
    )
    return iso3


def load_stanza_models(lang: LanguageCode) -> None:
    """Load Stanza morphosyntax models for one language, once per language.

    Owns the cache's per-language construction slot and nothing else, so the
    serialization is visible in four lines instead of buried at the bottom of
    a hundred-line build. See :func:`_build_stanza_models` for the build, and
    the docstring below for why the slot exists.
    """
    with pipeline_cache().load_slot(lang) as already_resident:
        if already_resident is not None:
            return
        _build_stanza_models(lang)


def _build_stanza_models(lang: LanguageCode) -> None:
    """Build and install one language's Stanza pipeline.

    Called only by :func:`load_stanza_models`, and only with that language's
    construction slot held.

    The resulting pipeline and tokenizer context are installed into the
    shared worker state so request handlers can do pure inference routing
    without rebuilding Stanza pipelines on every call. The inference lock is
    NOT created here: it is one per process, created with the state, so a
    reload can never hand a running handler a different lock.

    Residency is BOUNDED: the pipeline goes into ``StanzaPipelineCache``, which
    evicts the least recently used pipeline once it is full, so calling this
    for a third language returns the second-oldest one's memory rather than
    keeping every language a long-lived worker has ever seen.

    Construction is SERIALIZED PER LANGUAGE by the cache's ``load_slot``, and
    a language already resident when the slot opens is a no-op. Two request
    threads that missed on the same language used to call this concurrently
    and both build a multi-hundred-megabyte Stanza pipeline, the second install
    replacing the first: double the peak memory and double the wall clock for
    one usable result. The second thread now waits and finds the first's.
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
    # which raises KeyError('packages') deep in maintain_processor_list
    # the worker dies before emitting its ready signal, the daemon sees a
    # generic IPC error, and the user gets "transcription failed" with
    # the linguistic root cause buried in stderr.
    table = get_cached_capability_table()
    if table is None:
        # Post-2026-05-06: ``get_cached_capability_table()`` returns ``None``
        # ONLY when the Stanza Python package is missing from the worker
        # venv. The historical "resources.json missing" path now bootstraps
        # the catalog automatically and either returns a populated table or
        # raises ``StanzaCatalogDownloadError``. So if we reach here, it's a
        # genuine deploy-config error: BA3 was installed without Stanza.
        raise UnsupportedLanguageError(
            f"Cannot load Stanza for {lang!r}: the Stanza Python package "
            "is not installed in the worker environment. Reinstall "
            "batchalign3 with the morphosyntax extras enabled, or contact "
            "an operator to fix the deploy."
        )
    if not table.supports_morphosyntax(lang):
        sample = sorted(table.languages.keys())[:8]
        raise UnsupportedLanguageError(
            f"Stanza lacks the core morphosyntax processors for language {lang!r}. "
            f"It may appear in Stanza's resources.json as a stub or partial "
            f"entry, but no usable morphotag Pipeline can be built. "
            f"Languages with full morphosyntax support include: {sample} (and "
            f"{len(table.languages) - len(sample)} more)."
        )

    alpha2 = iso3_to_alpha2(lang)

    # MWT availability comes from Stanza's installed resources.json, never
    # from a hardcoded list. A stale list silently crashes the worker when
    # upstream drops a model (see the 2026-04-15 Swedish bootstrap failure).
    has_mwt = should_request_mwt(alpha2, table)
    processors = "tokenize,pos,lemma,depparse"
    if has_mwt:
        processors += ",mwt"

    ctx = TokenizerContext()

    # Italian needs a policy for single-word multi-word tokens; every other
    # language keeps Stanza's own behavior. Built here rather than inside the
    # postprocessor so the dependency is visible at the seam, and resolved
    # lazily because it reads from the pipeline this call is about to create.
    italian_policy: ItalianMwtPolicy | None = None
    if alpha2 == "it":
        provider = ItalianMwtPolicyProvider(lang)
        italian_policy = ItalianMwtPolicy(
            propose_splits=provider.propose_splits,
            lexicon=provider.lexicon,
        )

    # If the language pack for ``alpha2`` is not yet on disk, ``stanza.Pipeline``
    # will block while downloading several hundred MB of model files. Surface
    # that wait via the progress channel so every UI shows it to the user.
    _emit_stanza_lang_download_event_if_missing(lang, alpha2)

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
            tokenize_postprocessor=make_tokenizer_postprocessor(
                ctx, alpha2, italian_policy
            ),
        )

    # Install into the bounded cache, which may evict another language's
    # pipeline to stay under its ceiling. The context travels with it.
    pipeline_cache().install(lang, nlp, context=ctx)

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

    Owns that key's construction slot and nothing else, exactly as
    :func:`load_stanza_models` owns the plain language key's; the build is
    :func:`_build_stanza_retokenize_model`. The slot was missing here until
    2026-09-07, so the load-stampede fix covered one of the three key kinds a
    worker loads: two request threads that missed on ``{lang}:retok`` both
    built the neural tokenizer (a separate ~200 MB model), and the second
    install replaced the first.
    """
    alpha2 = iso3_to_alpha2(lang)
    if alpha2 not in _STANZA_CHINESE_ALPHA2:
        L.warning(
            "load_stanza_retokenize_model called for non-Chinese lang %s, skipping",
            lang,
        )
        return

    retok_key = retokenize_key(lang)
    with pipeline_cache().load_slot(retok_key) as already_resident:
        if already_resident is not None:
            return
        _build_stanza_retokenize_model(lang, alpha2, retok_key)


def _build_stanza_retokenize_model(
    lang: LanguageCode, alpha2: str, retok_key: str
) -> None:
    """Build and install the Mandarin retokenize pipeline.

    Called only by :func:`load_stanza_retokenize_model`, and only with
    ``retok_key``'s construction slot held.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import TokenizerContext

    processors = "tokenize,pos,lemma,depparse"
    ctx = TokenizerContext()

    # The Mandarin retokenize pipeline uses the neural tokenizer (a separate
    # ~200 MB model from the ``tokenize_pretokenized`` variant). Surface the
    # wait if it's about to download.
    _emit_stanza_lang_download_event_if_missing(lang, alpha2)

    nlp = stanza.Pipeline(
        lang=alpha2,
        processors=processors,
        download_method=DownloadMethod.REUSE_RESOURCES,
        tokenize_no_ssplit=True,
        tokenize_pretokenized=False,
    )

    pipeline_cache().install(retok_key, nlp, context=ctx)

    L.info("Loaded Stanza retokenize pipeline for %s (key=%s)", lang, retok_key)


# The POS tags that can host an Italian enclitic. AUX counts because Italian
# auxiliaries take them too (`avercelo`, `esserci`).
_ENCLISIS_HOSTING_POS = ("VERB", "AUX")


def _lexicon_from_legacy(
    *,
    word_dict: dict[str, str],
    composite_dict: dict[tuple[str, str], str],
) -> ItalianLexicon:
    """Read a pre-1.14 lemmatizer, which records verb attestation COMPLETELY.

    `composite_dict` lists every (surface, upos) pair the lemmatizer knows, so
    the answer is a filter and nothing needs reconstructing. Doing so anyway is
    not harmless: the heuristic the 1.14 reader needs admits ~400 non-verbs
    whose lemma merely ends in `-are`, and adding those to data that was already
    right stopped `dammela` expanding under 1.13.0 on 2026-07-31.
    """
    return ItalianLexicon(
        surface_forms=frozenset(word_dict),
        verb_forms=frozenset(
            surface
            for (surface, upos) in composite_dict
            if upos in _ENCLISIS_HOSTING_POS
        ),
    )


def _lexicon_from_pos_dict(pos_dict: dict[str, dict[str, str]]) -> ItalianLexicon:
    """Read a 1.14+ lemmatizer, which records verb attestation PARTIALLY.

    The repack keeps a POS entry only where its lemma differs from the
    pos-independent one, so Italian VERB+AUX holds 547 surfaces where 1.13.0
    held 13,248 (both measured). Taking the POS keys alone silently stops
    `giralo` and `chiamalo` expanding.

    What the repack dropped is recoverable without the discarded data: a word
    whose pos-independent lemma is an Italian infinitive is a form of that verb,
    and that lemma survives. `gira` is absent from the POS keys but maps to
    `girare`. A heuristic, and measurably imprecise against 1.13.0's complete
    data, which is why it lives HERE and never runs on the legacy reader.
    """
    surface_forms = pos_dict.get(_STANZA_POS_INDEPENDENT, {})
    return ItalianLexicon(
        surface_forms=frozenset(surface_forms),
        verb_forms=frozenset(
            {
                surface
                for upos in _ENCLISIS_HOSTING_POS
                for surface in pos_dict.get(upos, {})
            }
            | {
                surface
                for surface, lemma in surface_forms.items()
                if isinstance(lemma, str) and lemma.endswith(INFINITIVE_ENDINGS)
            }
        ),
    )


def extract_stanza_lexicon(nlp: object) -> ItalianLexicon:
    """Read Stanza's shipped lexicon off a loaded pipeline's lemmatizer.

    Answers the two questions the Italian MWT decision asks: is this surface
    form a known word, and is it attested as a verb.

    Two readers, because the supported Stanza range spans the 1.14 lemmatizer
    repack and the two layouts record verb attestation to DIFFERENT
    completeness. Reading whichever is present also keeps the version
    constraint and this code independent: a version-brittle reader forces them
    to move together, and a constraint change then disables the policy through
    the provider's resolve-to-None path instead of failing.

    Raises ``StanzaLexiconUnavailableError`` rather than returning an empty
    lexicon, because an empty one would silently answer "no" to every question
    and quietly change which words get split.
    """
    try:
        lemma_processor = nlp.processors["lemma"]  # type: ignore[attr-defined]
        trainer = lemma_processor._trainer
        if hasattr(trainer, "pos_dict"):
            lexicon = _lexicon_from_pos_dict(trainer.pos_dict)
        else:
            lexicon = _lexicon_from_legacy(
                word_dict=trainer.word_dict,
                composite_dict=trainer.composite_dict,
            )
    except (AttributeError, KeyError, TypeError) as exc:
        raise StanzaLexiconUnavailableError(
            "Stanza's lemmatizer exposes neither pos_dict nor "
            "word_dict/composite_dict at processors['lemma']._trainer. The "
            "Italian single-word MWT policy depends on that lexicon; find its "
            "new home or the policy must be disabled deliberately, not by "
            "accident."
        ) from exc

    if not lexicon.surface_forms or not lexicon.verb_forms:
        raise StanzaLexiconUnavailableError(
            f"Stanza's lemmatizer dictionaries are empty "
            f"(surface_forms={len(lexicon.surface_forms)}, "
            f"verb_forms={len(lexicon.verb_forms)}); an empty lexicon would "
            "silently suppress every multi-word token."
        )
    return lexicon


class ItalianMwtPolicyProvider:
    """Lazily resolves the probe pipeline and lexicon for one language.

    One instance per loaded language pipeline, so the caches live exactly as
    long as the pipeline they describe. Both resolutions are deferred because
    the tokenizer postprocessor closure is constructed BEFORE ``stanza.Pipeline``
    returns, and the lexicon can only be read from the finished pipeline.

    Every failure resolves to ``None``, which the decision treats as "do not
    intervene". Degrading to Stanza's own behavior is the honest failure mode;
    the alternative would suppress real multi-word tokens on the strength of a
    check that never ran.
    """

    def __init__(self, lang: LanguageCode) -> None:
        self._lang = lang
        # Read by the probe pipeline's postprocessor during propose_splits.
        # Thread-local because Stanza calls the postprocessor synchronously on
        # the calling thread, and the worker may run languages concurrently.
        self._forcing = threading.local()
        self._lexicon: ItalianLexicon | None = None
        self._lexicon_resolved = False
        self._probe: StanzaNLP | None = None
        self._probe_resolved = False

    def lexicon(self) -> ItalianLexicon | None:
        if self._lexicon_resolved:
            return self._lexicon
        self._lexicon_resolved = True
        pipelines = _state.stanza_pipelines
        nlp = None if pipelines is None else pipelines.get(self._lang)
        if nlp is None:
            L.warning(
                "Italian MWT policy: no loaded pipeline for %s, leaving Stanza's "
                "own multi-word tokenization untouched",
                self._lang,
            )
            return None
        try:
            self._lexicon = extract_stanza_lexicon(nlp)
        except StanzaLexiconUnavailableError as exc:
            L.warning("Italian MWT policy disabled: %s", exc)
            return None
        L.info(
            "Italian MWT lexicon: %d surface forms, %d verb forms",
            len(self._lexicon.surface_forms),
            len(self._lexicon.verb_forms),
        )
        return self._lexicon

    def propose_splits(
        self, utterances: Sequence[Sequence[str]], force: frozenset[str]
    ) -> list[dict[str, tuple[str, ...]]] | None:
        """Ask Stanza what its MWT processor would do to each utterance.

        This has to be a second pipeline because the decision is made in the
        tokenize_postprocessor, which Stanza runs BEFORE the MWT processor: at
        that point the split we need to judge does not exist yet.

        Whole utterances, not isolated words. Stanza's MWT is context-sensitive
        (``hai`` alone stays whole, but becomes *ha* + *i* mid-sentence), so a
        probe that drops the context answers a question nobody asked.

        Every name in ``force`` is hinted ``(text, True)`` to the probe. For a
        token Stanza was already going to expand that changes nothing, and for
        one it declines to expand it reveals the split it would otherwise
        withhold: ``aprilo`` is left whole by default and yields *apri* + *lo*
        with the real lemma *aprire* when hinted. One pass therefore serves both
        directions of the policy.

        Returns one text-to-split mapping per input utterance. Where the same
        surface occurs twice in one utterance the first analysis wins, which is
        safe because Stanza gives the same surface the same treatment within a
        sentence in every case observed.
        """
        probe = self._probe_pipeline()
        if probe is None or not utterances:
            return None
        self._forcing.targets = {w.lower() for w in force}
        try:
            import stanza

            docs = probe([stanza.Document([], text=" ".join(u)) for u in utterances])
        except Exception as exc:
            L.warning("Italian MWT split probe failed, leaving analysis alone: %s", exc)
            return None
        finally:
            self._forcing.targets = set()
        if not isinstance(docs, list):
            docs = [docs]

        per_utterance: list[dict[str, tuple[str, ...]]] = []
        for doc in docs:
            splits: dict[str, tuple[str, ...]] = {}
            for sent in doc.sentences:
                for token in sent.tokens:
                    splits.setdefault(token.text, tuple(w.text for w in token.words))
            per_utterance.append(splits)
        # A short probe result would silently misalign utterance to mapping, so
        # pad rather than let zip-by-index quietly judge the wrong sentence.
        while len(per_utterance) < len(utterances):
            per_utterance.append({})
        return per_utterance

    def _force_postprocessor(
        self, batch: list[list[str | tuple[str, bool]]]
    ) -> list[list[str | tuple[str, bool]]]:
        """Hint every forcing target as an MWT for the probe pipeline only.

        Deliberately does NOT touch the production pipeline: this shapes what
        the probe REPORTS, and the policy then decides what the real pipeline is
        told. Keeping the two separate is what lets a forced split be inspected
        and rejected without ever reaching the output.
        """
        targets: set[str] = getattr(self._forcing, "targets", set())
        if not targets:
            return batch
        forced: list[list[str | tuple[str, bool]]] = []
        for sentence in batch:
            row: list[str | tuple[str, bool]] = []
            for token in sentence:
                text = token[0] if isinstance(token, tuple) else token
                row.append((text, True) if text.lower() in targets else token)
            forced.append(row)
        return forced

    def _probe_pipeline(self) -> StanzaNLP | None:
        if self._probe_resolved:
            return self._probe
        self._probe_resolved = True
        try:
            self._probe = load_stanza_mwt_probe_model(
                self._lang, self._force_postprocessor
            )
        except Exception as exc:
            L.warning("Could not load the Italian MWT probe pipeline: %s", exc)
            self._probe = None
        return self._probe


def load_stanza_mwt_probe_model(
    lang: LanguageCode, postprocessor: object | None = None
) -> StanzaNLP:
    """Load a minimal tokenize+mwt pipeline used to preview MWT proposals.

    Stored under ``"{lang}:mwtprobe"`` in worker state, alongside the standard
    and retokenize pipelines. Loaded lazily: only batches that actually contain
    a single-word Italian utterance ever pay for it, and it carries just the
    tokenizer and the small MWT model, no POS, lemma or depparse.

    NOTE the deliberate absence of ``tokenize_pretokenized``. That flag makes
    Stanza accept the given token boundaries, which means the tokenizer never
    marks anything as a multi-word candidate and the MWT processor expands
    nothing. A pretokenized probe would answer "never splits" for every word,
    which reads as a clean result and is a measurement that never ran.

    Construction is SERIALIZED on ``{lang}:mwtprobe`` by the cache's
    ``load_slot``. Until 2026-09-07 this was a check-then-act on
    ``cache.get(probe_key)``: two threads could both miss, both build a
    tokenizer plus MWT model, and both install, with the loser's pipeline
    already handed out to its caller.
    """
    import stanza
    from stanza import DownloadMethod

    probe_key = mwt_probe_key(lang)
    cache = pipeline_cache()
    with cache.load_slot(probe_key) as already_resident:
        if already_resident is not None:
            return already_resident.nlp

        alpha2 = iso3_to_alpha2(lang)
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors="tokenize,mwt",
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=postprocessor,
        )
        # No tokenizer context: the probe has no realignment layer, and an empty
        # one here would read as "a context exists" to anything that looked.
        cache.install(probe_key, nlp, context=None)
        L.info("Loaded Stanza MWT probe pipeline for %s (key=%s)", lang, probe_key)
        # `stanza.Pipeline` is untyped (no stubs ship with it, and
        # `ignore_missing_imports` makes it `Any`), so this is the boundary where
        # an `Any` becomes a typed value. Narrowed explicitly rather than
        # returned bare: a bare return would silently re-infect every caller with
        # `Any`, which is exactly how the untyped probe seam went unnoticed.
        probe: StanzaNLP = nlp
        return probe


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


# ---------------------------------------------------------------------------
# Language-pack download notification helper.
# ---------------------------------------------------------------------------


def _emit_stanza_lang_download_event_if_missing(
    lang: LanguageCode, alpha2: LanguageCode2
) -> None:
    """Emit a user-visible event if Stanza needs to download ``alpha2``.

    Probes the configured Stanza model directory for the presence of any
    files under ``<model_dir>/<alpha2>/``. Absence implies ``stanza.Pipeline``
    will block on a multi-hundred-MB download. The notification surfaces
    that wait through the same progress channel UIs already render for
    model loading.

    Best-effort: if the probe fails for any reason, we emit anyway. False-
    positive notifications are a much smaller UX cost than silent waits.
    """
    import os

    from batchalign.worker._progress import emit_download_event

    is_present = False
    try:
        import stanza.resources.common as src

        lang_dir = os.path.join(src.DEFAULT_MODEL_DIR, alpha2)
        # Stanza scatters language packs across subdirectories named after
        # processor packages (e.g. ``en/tokenize/combined.pt``); presence of
        # *any* file under the language directory means at least some pack
        # has been seeded and the download is partial-or-done. The Pipeline
        # call may still pull a few small files; that's fine, the user has
        # already been informed via past events.
        if os.path.isdir(lang_dir):
            for _root, _dirs, files in os.walk(lang_dir):
                if files:
                    is_present = True
                    break
    except Exception as probe_exc:
        L.debug("Stanza lang-pack probe failed for %s: %s", alpha2, probe_exc)

    if is_present:
        return

    emit_download_event(
        stage=f"downloading_stanza_lang_{alpha2}",
        user_message=(
            f"Downloading Stanza language pack for {lang} ({alpha2}) "
            "(one-time, ~250-500 MB; future runs will use the local cache)…"
        ),
    )
