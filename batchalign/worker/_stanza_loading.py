"""Stanza and language-code loading helpers for the worker process.

This module exists to keep Stanza-specific bootstrap policy out of the generic
worker entrypoint and the request-time inference routers. It owns:

- resolving a language to the catalog entry that serves it (``stanza_code_for``)
- the MWT/non-MWT processor policy (capability-driven, from that entry)
- installation of preloaded Stanza pipelines into worker runtime state
- utseg's Stanza constituency pipeline, loaded once per language
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from batchalign.inference._domain_types import LanguageCode, StanzaCode
from batchalign.inference._italian_mwt import (
    INFINITIVE_ENDINGS,
    ItalianLexicon,
    ItalianMwtPolicy,
)
from batchalign.worker import _stanza_capabilities
from batchalign.worker._language_errors import UnsupportedLanguageError
from batchalign.worker._pipeline_cache import (
    LoadedPipeline,
    StanzaPipelineCache,
    mwt_probe_key,
    retokenize_key,
    utseg_key,
)
from batchalign.worker._stanza_capabilities import (
    CHINESE_STANZA_CODE,
    StanzaLanguageCapability,
    StanzaRequirementName,
    processors_argument,
    requirement_processors,
)
from batchalign.worker._stanza_lemma_pretrain import lemma_pretrain_options
from batchalign.worker._types import StanzaProcessorName, _state

L = logging.getLogger("batchalign.worker")

# The one Stanza package Japanese is loaded with, for every processor. Named
# once because the lemma pretrain repair must read the same lemma model the
# pipeline loads.
_JA_PACKAGE = "combined"

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


def _pipeline_processors(requirement: StanzaRequirementName, *, with_mwt: bool) -> str:
    """The processors a command's pipeline is built with: Rust's requirement
    for that command, plus ``mwt`` when the language's catalog entry ships it."""
    wanted: set[StanzaProcessorName] = set(requirement_processors(requirement))
    if with_mwt:
        wanted.add("mwt")
    return processors_argument(wanted)


def _morphosyntax_processors(*, with_mwt: bool) -> str:
    """The processors a morphotag pipeline is built with."""
    return _pipeline_processors("morphosyntax", with_mwt=with_mwt)


def language_capability(lang: LanguageCode) -> StanzaLanguageCapability:
    """The catalog entry that serves ``lang``, from the capability table.

    The table is the worker's ONE ISO 639-3 to Stanza mapping (Rust's ISO
    639-1 table plus the installed catalog; see ``_stanza_capabilities``), the
    same table the server admits languages from, so a pipeline can only be
    built for a language the server could have admitted. Raises
    ``UnsupportedLanguageError`` when the table is unavailable or has no entry
    for ``lang``; nothing is passed through to Stanza unresolved.

    It replaced ``iso3_to_alpha2``, a second resolution (an override dict,
    then ``pycountry``, then the code verbatim with a warning) that agreed
    with the table only by construction and handed Stanza codes it had no
    entry for.
    """
    # Read through the module so a test that replaces the table replaces it
    # for every loader.
    table = _stanza_capabilities.get_cached_capability_table()
    if isinstance(table, _stanza_capabilities.StanzaNotInstalled):
        # A deploy error, not a language Stanza lacks.
        raise UnsupportedLanguageError(
            f"Cannot load Stanza for {lang!r}: the Stanza Python package "
            f"is not installed in the worker environment ({table.import_error}). "
            "Reinstall batchalign3 with the morphosyntax extras enabled, or "
            "contact an operator to fix the deploy."
        )
    cap = table.languages.get(lang)
    if cap is None:
        raise UnsupportedLanguageError(
            f"The installed Stanza catalog has no entry for language {lang!r}."
        )
    return cap


def stanza_code_for(lang: LanguageCode) -> StanzaCode:
    """The Stanza catalog code that serves ``lang`` (``kaz`` is ``kk``)."""
    return language_capability(lang).stanza_code


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
    from batchalign_core import RealignLanguage

    # Preflight gate: consult the capability table BEFORE calling
    # stanza.Pipeline. Without it, an unsupported language reaches
    # stanza.Pipeline, which raises KeyError('packages') deep in
    # maintain_processor_list: the worker dies before emitting its ready
    # signal, the daemon sees a generic IPC error, and the user gets
    # "transcription failed" with the linguistic root cause buried in stderr.
    cap = language_capability(lang)
    if not cap.supports_morphosyntax:
        raise UnsupportedLanguageError(
            f"Stanza's catalog entry {cap.stanza_code!r} for language {lang!r} "
            "lacks the core morphosyntax processors "
            f"({', '.join(sorted(requirement_processors('morphosyntax')))}), so no "
            "morphotag Pipeline can be built."
        )

    alpha2 = cap.stanza_code

    # MWT availability comes from Stanza's installed resources.json, never
    # from a hardcoded list. A stale list silently crashes the worker when
    # upstream drops a model (see the 2026-04-15 Swedish bootstrap failure).
    has_mwt = cap.has("mwt")
    processors = _morphosyntax_processors(with_mwt=has_mwt)

    ctx = TokenizerContext(alpha2)

    # Italian needs a policy for single-word multi-word tokens; every other
    # language keeps Stanza's own behavior. Built here rather than inside the
    # postprocessor so the dependency is visible at the seam, and resolved
    # lazily because it reads from the pipeline this call is about to create.
    italian_policy: ItalianMwtPolicy | None = None
    if ctx.language == RealignLanguage.Italian:
        provider = ItalianMwtPolicyProvider(lang)
        italian_policy = ItalianMwtPolicy(
            propose_splits=provider.propose_splits,
            lexicon=provider.lexicon,
        )

    # If the language pack for ``alpha2`` is not yet on disk, ``stanza.Pipeline``
    # will block while downloading several hundred MB of model files. Surface
    # that wait via the progress channel so every UI shows it to the user.
    _emit_stanza_lang_download_event_if_missing(lang, alpha2)

    # Options every branch shares. The lemma pretrain repair (see
    # `_stanza_lemma_pretrain`) is in here, rather than repeated per branch,
    # so a new branch cannot be written without it.
    lemma_package = _JA_PACKAGE if alpha2 == "ja" else None
    common = {
        "lang": alpha2,
        "processors": processors,
        "download_method": DownloadMethod.REUSE_RESOURCES,
        "tokenize_no_ssplit": True,
        **lemma_pretrain_options(alpha2, lemma_package),
    }

    # The Stanza pipeline shape varies by language because tokenization and MWT
    # support are not uniform across the supported languages.
    if alpha2 == "ja":
        nlp = stanza.Pipeline(
            **common,
            tokenize_pretokenized=True,
            package={
                "tokenize": _JA_PACKAGE,
                "pos": _JA_PACKAGE,
                "lemma": _JA_PACKAGE,
                "depparse": _JA_PACKAGE,
            },
        )
    elif not has_mwt:
        nlp = stanza.Pipeline(**common, tokenize_pretokenized=True)
    elif alpha2 == "en":
        nlp = stanza.Pipeline(
            **common,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx),
            package={"mwt": "gum"},
        )
    else:
        nlp = stanza.Pipeline(
            **common,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx, italian_policy),
        )

    # Install into the bounded cache, which may evict another language's
    # pipeline to stay under its ceiling. The context travels with it.
    pipeline_cache().install(lang, nlp, context=ctx)
    # No version is recorded here: `_state.stanza_version()` is the one
    # accessor, and it reads the installed package this pipeline came from.


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
    alpha2 = stanza_code_for(lang)
    if alpha2 != CHINESE_STANZA_CODE:
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
    lang: LanguageCode, alpha2: StanzaCode, retok_key: str
) -> None:
    """Build and install the Mandarin retokenize pipeline.

    Called only by :func:`load_stanza_retokenize_model`, and only with
    ``retok_key``'s construction slot held.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import TokenizerContext

    processors = _morphosyntax_processors(with_mwt=False)
    ctx = TokenizerContext(alpha2)

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
        **lemma_pretrain_options(alpha2),
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
        self._probe: LoadedPipeline | None = None
        self._probe_resolved = False

    def lexicon(self) -> ItalianLexicon | None:
        if self._lexicon_resolved:
            return self._lexicon
        self._lexicon_resolved = True
        pipelines = _state.stanza_pipelines
        entry = None if pipelines is None else pipelines.loaded(self._lang)
        if entry is None:
            L.warning(
                "Italian MWT policy: no loaded pipeline for %s, leaving Stanza's "
                "own multi-word tokenization untouched",
                self._lang,
            )
            return None
        try:
            # Under the pipeline's guard like any other use of it. This runs
            # from the postprocessor, inside a call that already holds it on
            # this thread, which the reentrant guard admits.
            with entry.calling() as nlp:
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

            with probe.calling() as nlp:
                docs = nlp([stanza.Document([], text=" ".join(u)) for u in utterances])
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

    def _probe_pipeline(self) -> LoadedPipeline | None:
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
) -> LoadedPipeline:
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
            return already_resident

        alpha2 = stanza_code_for(lang)
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors="tokenize,mwt",
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=postprocessor,
        )
        # No tokenizer context: the probe has no realignment layer, and an empty
        # one here would read as "a context exists" to anything that looked.
        # The installed entry is returned, so the caller calls the probe
        # through the one guard the cache gave it.
        installed = cache.install(probe_key, nlp, context=None)
        L.info("Loaded Stanza MWT probe pipeline for %s (key=%s)", lang, probe_key)
        return installed.entry


def prepare_utseg_stanza(lang: LanguageCode) -> None:
    """Resolve utseg's primary language against the Stanza catalog.

    Loads nothing: the constituency pipeline is built on the first request
    that names the Stanza segmenter and has an item of more than one word,
    by :func:`load_utseg_pipeline` (which the utseg handler calls directly,
    in every worker language mode), and reused after that. The language is
    resolved now so an unservable primary fails at worker load, not at the
    first request.
    """
    language_capability(lang)


def load_utseg_pipeline(lang: LanguageCode) -> LoadedPipeline:
    """The Stanza constituency pipeline for ``lang``, built once and reused.

    Built through the pipeline cache under ``{lang}:utseg``, with that key's
    construction slot held, exactly as morphotag's pipelines are, so a worker
    that segments many batches loads the model once rather than once per
    batch, and every call goes through the entry's own guard.

    The processors are utterance segmentation's requirement (Rust's
    ``StanzaRequirement::UtteranceSegmentation``: tokenize, pos and
    constituency, whose trees are all the route reads), plus ``mwt`` where the
    catalog entry ships it. No lemmatizer: segmentation never reads lemmas.

    Raises ``UnsupportedLanguageError`` when the language's catalog entry
    lacks a processor of that requirement. The server refuses such a language
    before dispatch (`UtsegRoute::admit`); this is the worker's own refusal
    for a request that reached it anyway, never a pipeline without a parser
    whose every item would fail.
    """
    cap = language_capability(lang)
    if not cap.serves("utterance_segmentation"):
        raise UnsupportedLanguageError(
            f"The installed Stanza catalog entry for {lang!r} lacks the utterance "
            f"segmentation processors ("
            f"{_pipeline_processors('utterance_segmentation', with_mwt=False)}"
            f"), so it cannot segment with Stanza."
        )

    key = utseg_key(lang)
    cache = pipeline_cache()
    with cache.load_slot(key) as already_resident:
        if already_resident is not None:
            return already_resident

        import stanza
        from stanza import DownloadMethod

        _emit_stanza_lang_download_event_if_missing(lang, cap.stanza_code)
        nlp = stanza.Pipeline(
            lang=cap.stanza_code,
            processors=_pipeline_processors(
                "utterance_segmentation", with_mwt=cap.has("mwt")
            ),
            tokenize_pretokenized=True,
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
        # No tokenizer context: utseg has no realignment layer.
        installed = cache.install(key, nlp, context=None)
        L.info("Loaded Stanza utseg pipeline for %s (key=%s)", lang, key)
        return installed.entry


# ---------------------------------------------------------------------------
# Language-pack download notification helper.
# ---------------------------------------------------------------------------


def _emit_stanza_lang_download_event_if_missing(
    lang: LanguageCode, alpha2: StanzaCode
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
