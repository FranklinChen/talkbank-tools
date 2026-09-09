"""Stanza morphosyntax inference: words -> POS/dep/lemma.

Pure inference, no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError, model_validator

from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._pipeline_cache import (
    LoadedPipeline,
    PipelineLookup,
    retokenize_key,
)

if TYPE_CHECKING:
    from batchalign.inference._tokenizer_realign import TokenizerContext

from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    WorkerJSONValue,
)

L = logging.getLogger("batchalign.worker")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Terminator(StrEnum):
    """A CHAT utterance terminator, by its surface form.

    The closed set, mirroring `talkbank_model::Terminator`'s CHAT surface
    forms, which is what the Rust side serializes at the IPC boundary. Naming
    each member keeps the vocabulary visible: a reader here sees that CHAT has
    thirteen ways to end an utterance, not that "the terminator is a string".

    Membership is enforced by parsing, so an unrecognized terminator fails the
    item's validation rather than travelling on as an unknown token. That is
    not hypothetical: it immediately caught a fixture asserting a CJK full
    stop, which CHAT does not use and Rust cannot send.

    KNOWN DUPLICATION, and it is NOT blocked. This is the third in-repo copy
    of the vocabulary, after `talkbank_model::Terminator::try_from_chat_str`
    and `chat_punct_chars()` in `batchalign-transform/src/translate.rs`, which
    already enumerates all thirteen. The removal is entirely local: give the
    Rust side one list, declare it as the schema's closed set in place of
    `#[schemars(with = "String")]` on `MorphosyntaxBatchItem::terminator`, and
    let `scripts/generate_ipc_types.sh` emit this enum, as it already emits
    `AsrBackendV2` and friends. Not done here only because it reaches outside
    this change; it is a task, not an obstacle. A conformance test asserting
    the copies stay equal is deliberately NOT the answer: it would
    institutionalize the second copy rather than remove it.
    """

    PERIOD = "."
    QUESTION = "?"
    EXCLAMATION = "!"
    TRAILING_OFF = "+..."
    INTERRUPTION = "+/."
    SELF_INTERRUPTION = "+//."
    INTERRUPTED_QUESTION = "+/?"
    BROKEN_QUESTION = "+!?"
    QUOTED_NEW_LINE = '+"/.'
    QUOTED_PERIOD_SIMPLE = '+".'
    SELF_INTERRUPTED_QUESTION = "+//?"
    TRAILING_OFF_QUESTION = "+..?"
    BREAK_FOR_CODING = "+."


class MorphosyntaxBatchItem(BaseModel):
    """A single item in the batch morphosyntax payload from Rust."""

    words: list[str]
    # Required, with no default: a default period would be a sentinel that is
    # also a legal value, making "the caller sent no terminator"
    # indistinguishable from "the utterance ended in a period".
    terminator: Terminator
    # Required, with no defaults, because the Rust schema declares both
    # required and always sends them. An empty-string language is not a
    # language: defaulting it would let a caller that forgot to route by
    # language reach Stanza looking well-formed, which is the failure mode
    # that silently mixes languages rather than reporting anything.
    special_forms: list[list[str | None]]
    lang: LanguageCode

    @model_validator(mode="after")
    def _words_must_not_contain_the_terminator(self) -> MorphosyntaxBatchItem:
        """The terminator is not a main-tier word, and must not arrive as one.

        Silently dropping a duplicate would hide a caller that has confused
        the two, and keeping them apart is the entire purpose of this
        boundary: `words` are CHAT content that must map 1-to-1 onto `%mor`
        items, while the terminator is a cue for Stanza whose `%mor` and
        `%gra` representation the Rust side synthesizes from the typed model.
        """
        if self.words and self.words[-1] in Terminator:
            raise ValueError(
                f"words must not end with the utterance terminator "
                f"{self.words[-1]!r}: the terminator travels in its own "
                f"field and is not a main-tier word"
            )
        return self


@dataclass(frozen=True, slots=True)
class StanzaInput:
    """One utterance as Stanza receives it, with the two kinds kept apart.

    `chat_words` are the CHAT main-tier words, which must map 1-to-1 onto
    `%mor` items. The terminator is EVIDENCE for the model and never data:
    Stanza's analysis changes without it (the Italian model reads `dammela` as
    an ADJ and declines to MWT-expand it), so it must be present in the text,
    and it must not survive into the payload that becomes `%mor`.

    Both the text and the realigner's boundary list are DERIVED here rather
    than stored, so they cannot disagree with each other or with `chat_words`.
    That also makes the terminator's position knowable by construction, which
    is how it is removed on the way back: no inspection of what Stanza made of
    it, and therefore nothing to get wrong when Stanza tags it unexpectedly.
    """

    item_index: int
    chat_words: tuple[str, ...]
    terminator: Terminator

    @property
    def boundaries(self) -> tuple[str, ...]:
        """The token boundaries the realigner holds Stanza to."""
        return (*self.chat_words, self.terminator.value)

    @property
    def text(self) -> str:
        """The text handed to Stanza."""
        return " ".join(self.boundaries)

    def without_terminator(self, sentence: list[JSONObject]) -> list[JSONObject]:
        """The sentence with the cue we appended taken back off.

        Symmetric with `boundaries`: the type that added the boundary is the
        type that removes it, so the two cannot be changed apart. It is the
        LAST boundary, hence the last UD word, which is why nothing here
        inspects what Stanza made of it.

        Only valid where we chose the boundaries; the caller gates on that.
        """
        if len(sentence) <= 1:
            # The realigner did not hold, so there is no trustworthy last
            # word to remove. Leave it and let the count-mismatch machinery
            # downstream report the misalignment, but say so here: a silent
            # special case is how this class of bug hides.
            L.warning(
                "morphotag: item %d produced %d UD words for %d boundaries; "
                "leaving the terminator in place for the misalignment audit",
                self.item_index,
                len(sentence),
                len(self.boundaries),
            )
            return sentence
        return sentence[:-1]


@dataclass(frozen=True, slots=True)
class Realigned:
    """Our tokenization: the realigner holds Stanza to boundaries we supplied.

    This is the only mode in which the terminator's position in the output is
    known by construction, because it is the only mode in which we chose the
    boundaries.
    """

    context: TokenizerContext
    inputs: list[StanzaInput]


@dataclass(frozen=True, slots=True)
class StanzaOwnsTokenization:
    """Retokenize requested: Stanza segments freely and its MWT passes through."""


@dataclass(frozen=True, slots=True)
class UnrealignedFallback:
    """Normal mode with no realignment context: the degraded, warned-about state.

    Stanza's neural tokenizer is free to split or merge CHAT words, silently
    breaking the 1-to-1 invariant the Rust injection assumes. Count mismatches
    from this mode surface as MisalignmentBug decisions.
    See `book/src/architecture/morphotag-invariants.md`.
    """

    lang_code: LanguageCode


# Named `RealignmentMode`, not `TokenizationMode`, because Rust already owns
# that name for a COARSER and different fact: `TokenizationMode::{Preserve,
# StanzaRetokenize}` is what the CALLER asked for. This is the resolved answer
# to "who tokenizes this batch, and can we hold Stanza to our boundaries",
# which refines `Preserve` into the case where we have a realignment context
# and the case where we asked for it and do not. Two concepts, two names.
RealignmentMode = Realigned | StanzaOwnsTokenization | UnrealignedFallback


def _realignment_mode(
    *,
    context: TokenizerContext | None,
    stanza_owns_tokenization: bool,
    inputs: list[StanzaInput],
    lang_code: LanguageCode,
) -> RealignmentMode:
    """Resolve the three modes ONCE, so the illegal one cannot be built.

    These were two booleans and a `None` check evaluated at three separate
    places, which made "normal mode, but no realignment context" a
    representable state that the code could only warn about after the fact.
    As a sum type it is a named variant instead, and every consumer must say
    what it does in that case rather than falling through a condition.
    """
    if stanza_owns_tokenization:
        return StanzaOwnsTokenization()
    if context is None:
        return UnrealignedFallback(lang_code=lang_code)
    return Realigned(context=context, inputs=inputs)


@contextlib.contextmanager
def _realignment_applied(mode: RealignmentMode) -> Iterator[None]:
    """Install the realigner's boundaries for the duration of one Stanza call."""
    match mode:
        case Realigned(context=context, inputs=inputs):
            context.original_words = [list(item.boundaries) for item in inputs]
            try:
                yield
            finally:
                context.original_words = []
        case StanzaOwnsTokenization():
            yield
        case UnrealignedFallback(lang_code=lang_code):
            L.warning(
                "morphotag: realignment context missing for language %r, "
                "Stanza will own tokenization on this batch, which may "
                "violate the 1-to-1 invariant. This batch's count mismatches "
                "will surface as MisalignmentBug decisions.",
                lang_code,
            )
            yield


def _drops_appended_terminator(mode: RealignmentMode) -> bool:
    """Whether the terminator we appended must be taken back off the output.

    Only under `Realigned`, because that is the only mode in which we chose
    the boundaries and therefore know where the terminator went. In the other
    two nothing knows, and the Rust side's recognition-based filter is their
    answer.

    Resolved once per language batch rather than per utterance: the mode is
    fixed for the whole group.
    """
    match mode:
        case Realigned():
            return True
        case StanzaOwnsTokenization() | UnrealignedFallback():
            return False


# The 37 Universal Dependencies relation heads (UD v2). Subtypes after a
# colon are open and language-specific, so only the head is checked. This
# mirrors the closed set chatter's E761 enforces on the reading side; the two
# must not drift apart.
UD_RELATIONS: frozenset[str] = frozenset(
    {
        "acl",
        "advcl",
        "advmod",
        "amod",
        "appos",
        "aux",
        "case",
        "cc",
        "ccomp",
        "clf",
        "compound",
        "conj",
        "cop",
        "csubj",
        "dep",
        "det",
        "discourse",
        "dislocated",
        "expl",
        "fixed",
        "flat",
        "goeswith",
        "iobj",
        "list",
        "mark",
        "nmod",
        "nsubj",
        "nummod",
        "obj",
        "obl",
        "orphan",
        "parataxis",
        "punct",
        "reparandum",
        "root",
        "vocative",
        "xcomp",
    }
)

# Known non-UD labels observed from Stanza, mapped to their UD equivalent.
# `iob` is emitted by the Italian model for clitic pronouns and is
# unambiguously `iobj`; it is the defect that put IOB into the corpora.
UD_DEPREL_ALIASES: dict[str, str] = {
    "iob": "iobj",
}


class UdWord(BaseModel, extra="allow"):
    """A single UD word/token: mirrors Rust ``UdWord`` in types.rs."""

    id: int | list[int] | float
    text: str
    lemma: str = ""
    upos: str = "X"
    xpos: str | None = None
    feats: str | None = None
    head: int = 0
    deprel: str = "dep"
    deps: str | None = None
    misc: str | None = None

    @model_validator(mode="after")
    def _default_lemma_to_text(self) -> UdWord:
        if not self.lemma and not isinstance(self.id, list):
            self.lemma = self.text
        return self

    @model_validator(mode="after")
    def _sanitize_pad_deprel(self) -> UdWord:
        if self.deprel.startswith("<") and self.deprel.endswith(">"):
            L.warning(
                "Stanza emitted deprel=%r for word %r, replacing with 'dep'",
                self.deprel,
                self.text,
            )
            self.deprel = "dep"
        return self

    @model_validator(mode="after")
    def _normalize_deprel_to_ud(self) -> UdWord:
        """Force the relation HEAD into the Universal Dependencies closed set.

        Stanza does not guarantee UD-conformant labels. Its Italian model
        emits ``iob`` (verified against stanza 1.13.0 on "attenzione ."),
        which is not a UD relation; UD defines ``iobj``. Passing it through
        wrote ``2|1|IOB`` into ``%gra`` across the published corpora, where it
        went undetected for months because nothing on either side validated
        the label: CLAN CHECK does not check relations at all, and chatter
        only gained the rule (E761) in v0.4.0.

        Only the HEAD is closed. UD defines SUBTYPES as open and
        language-specific, and the corpora legitimately use many
        (``nmod:poss``, ``acl:relcl``, ``flat:foreign``), so the subtype is
        preserved verbatim and never validated.

        An unrecognised head degrades to ``dep``, a real UD relation, rather
        than reaching the transcript. Silent pass-through is exactly how the
        original defect escaped.
        """
        head, sep, subtype = self.deprel.partition(":")
        lowered = head.lower()
        if lowered in UD_RELATIONS:
            if head != lowered:
                self.deprel = lowered + sep + subtype
            return self

        replacement = UD_DEPREL_ALIASES.get(lowered)
        if replacement is not None:
            L.warning(
                "Stanza emitted non-UD deprel=%r for word %r, normalizing to %r",
                self.deprel,
                self.text,
                replacement,
            )
            self.deprel = replacement + sep + subtype
            return self

        L.warning(
            "Stanza emitted unrecognized deprel=%r for word %r, replacing with 'dep'",
            self.deprel,
            self.text,
        )
        self.deprel = "dep"
        return self


UdWordRaw = dict[str, str | int | float | list[int] | tuple[int, ...] | None]
JSONObject = dict[str, WorkerJSONValue]


# ---------------------------------------------------------------------------
# CJK word segmentation
# ---------------------------------------------------------------------------


def _segment_cantonese(words: list[str]) -> list[str]:
    """Segment Cantonese per-character tokens into words using PyCantonese.

    Only re-segments contiguous runs of single-CJK-character tokens.
    Existing multi-character tokens are preserved as-is to avoid breaking
    word boundaries that are already correct (e.g., from Tencent ASR or
    hand-transcribed corpora).

    This prevents the bug where joining all words into one string causes
    PyCantonese to merge tokens across word boundaries (e.g., 啦+飯+啦
    becoming 啦飯啦).
    """
    if not words:
        return []
    import pycantonese

    # Only re-segment if the input looks like per-character ASR output:
    # all CJK tokens are single characters. If any multi-char CJK token
    # exists, the input already has some word boundaries, preserve them.
    cjk_words = [w for w in words if any("\u4e00" <= c <= "\u9fff" for c in w)]
    has_multichar_cjk = any(len(w) > 1 for w in cjk_words)

    if has_multichar_cjk:
        # Input already has word boundaries, don't re-segment.
        # This prevents merging tokens across existing boundaries.
        return list(words)

    # All CJK tokens are single characters, safe to join and segment.
    text = "".join(words)
    if not text:
        return []
    return pycantonese.segment(text)


def _override_pos_with_pycantonese(
    ud_words: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Override Stanza POS tags with PyCantonese POS for Cantonese words.

    Stanza's Mandarin-trained model misclassifies core Cantonese vocabulary
    (~50% accuracy). PyCantonese's POS tagger scores ~94% on the same words.
    This function replaces ``upos`` in each UD word dict while preserving
    all other fields (lemma, deprel, head, etc.) from Stanza.

    Called as a post-processing step when ``retokenize=True`` and ``lang=yue``.
    """
    import pycantonese

    texts = [w.get("text", "") for w in ud_words]
    if not texts:
        return ud_words

    tagged = pycantonese.pos_tag(texts)
    tag_map = {word: pos for word, pos in tagged}

    result = []
    for w in ud_words:
        text = w.get("text", "")
        pyc_pos = tag_map.get(text)
        if pyc_pos is not None:
            w = {**w, "upos": pyc_pos}
        result.append(w)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_bogus_lemma(text: str, lemma: str) -> bool:
    """Detect when Stanza returns a lemma that's pure punctuation for a word."""
    if text == lemma or not lemma:
        return False
    text_has_letters = any(unicodedata.category(c).startswith("L") for c in text)
    lemma_all_punct = all(unicodedata.category(c).startswith(("P", "S")) for c in lemma)
    return text_has_letters and lemma_all_punct


def validate_ud_words(sents: list[list[UdWordRaw]]) -> None:
    """Validate and normalize every token through the UdWord model.

    Mutates *sents* in place.
    """
    for sent in sents:
        for word_idx in range(len(sent)):
            raw = sent[word_idx]
            raw_id = raw.get("id")
            if isinstance(raw_id, tuple):
                raw["id"] = list(raw_id)

            validated = UdWord.model_validate(raw)

            if not isinstance(validated.id, list) and _is_bogus_lemma(
                validated.text, validated.lemma
            ):
                L.warning(
                    "Stanza returned bogus lemma %r for word %r, falling back to surface form",
                    validated.lemma,
                    validated.text,
                )
                validated.lemma = validated.text

            sent[word_idx] = validated.model_dump()


# ---------------------------------------------------------------------------
# Failure vocabulary
# ---------------------------------------------------------------------------


class MorphosyntaxFailure(StrEnum):
    """Why one language group of a batch produced no morphosyntactic analysis.

    A closed set rather than an ad-hoc string per site: each member is a
    distinct operational situation with a distinct fix (install the language,
    investigate a crash, investigate a tokenization drift), and naming them
    here is what stops the next such path being written as a bare
    ``L.warning`` with nothing returned.
    """

    PIPELINE_RAISED = "Stanza pipeline raised"
    PIPELINE_MISSING = "no Stanza pipeline loaded"
    SENTENCE_COUNT_MISMATCH = "Stanza sentence count mismatch"


@dataclass(frozen=True, slots=True)
class LanguageGroupFailure:
    """One language group's failure, carrying where and why it happened.

    Every item in the group gets the SAME failure, because the group is the
    unit Stanza is invoked on: when it fails, nothing is known about any of
    its utterances.
    """

    kind: MorphosyntaxFailure
    lang: LanguageCode
    detail: str

    def message(self) -> str:
        """The operator-facing sentence, used for both the log and the wire.

        One renderer, so the line an operator reads in the worker log and the
        string the Rust side reports for the failed file cannot disagree.
        """
        return f"{self.kind.value} for language {self.lang}: {self.detail}"

    def response(self) -> InferResponse:
        """The per-item response carrying this failure to the Rust consumer.

        ``result`` is deliberately left unset. An empty UD document is a
        LEGAL analysis (it is what an utterance with no words gets), so
        returning one here would be indistinguishable from success: the Rust
        side would inject empty ``%mor`` and ``%gra`` tiers and write the
        file. With an error it fails the file instead, which is what
        ``crates/batchalign/src/morphosyntax/worker.rs`` does with any
        per-item error.
        """
        return InferResponse(error=self.message(), elapsed_s=0.0)


_UNDECIDED_ERROR = (
    "morphosyntax decided nothing for this item; refusing to report an empty "
    "analysis as success"
)
"""Fail-closed placeholder for an item no code path decided.

Reaching it is a defect in ``batch_infer_morphosyntax`` rather than a fact
about the utterance, which is exactly why it must not be an empty result: an
unforeseen path then fails the file loudly instead of silently emptying its
tiers.
"""


def _finalized(
    results: list[InferResponse | None],
    *,
    elapsed_s: float | None,
) -> BatchInferResponse:
    """Settle every item's decision into the batch response.

    ``None`` means "no path decided this item" and becomes an error (see
    ``_UNDECIDED_ERROR``). ``elapsed_s`` is stamped on the first item only,
    which is the batch-level timing convention the Rust side reads; ``None``
    leaves the timings alone, for the early return that did no work.
    """
    settled = [
        r if r is not None else InferResponse(error=_UNDECIDED_ERROR, elapsed_s=0.0)
        for r in results
    ]
    if settled and elapsed_s is not None:
        first = settled[0]
        settled[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed_s
        )
    return BatchInferResponse(results=settled)


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def batch_infer_morphosyntax(
    req: BatchInferRequest,
    pipelines: PipelineLookup,
    nlp_lock: threading.Lock,
    free_threaded: bool,
    mwt_lexicon: dict[str, list[str]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    load_pipeline: Callable[[LanguageCode], None] | None = None,
) -> BatchInferResponse:
    """Batch Stanza inference: (words, lang) -> UdResponse.

    Parameters
    ----------
    req : BatchInferRequest
        Batch of MorphosyntaxBatchItem payloads.
    pipelines : PipelineLookup
        One atomic read of a loaded pipeline AND the tokenizer context built
        for it, keyed by ISO-3 code (or a suffixed variant key). In the worker
        this is the BOUNDED pipeline cache, so a lookup is also a use: it
        decides what stays resident from what is read here. It is ONE lookup
        rather than a pipeline mapping beside a context mapping because those
        were two separately locked reads with an eviction window between them:
        the second could miss, and `tok_ctx` silently became `None` or another
        language's context while the pipeline in hand was the right one.
    nlp_lock : threading.Lock
        Lock guarding Stanza calls on GIL-enabled Python.
    free_threaded : bool
        Whether to skip the lock (free-threaded Python).
    mwt_lexicon : dict, optional
        Custom multi-word token lexicon mapping surface forms to
        expansion tokens (e.g. ``{"gonna": ["going", "to"]}``).
        When provided, matching tokens in Stanza's output are
        expanded according to this lexicon.
    load_pipeline : callable, optional
        Loader invoked with a language code when ``pipelines`` has no
        pipeline for it, before that is reported as a failure. The worker
        supplies ``load_stanza_models``; a bounded cache can evict a language
        loaded earlier in the same batch, and this is what brings it back.
        Absent (the default, used by unit tests), a miss is reported straight
        away rather than reaching for a model download.
    """

    @contextlib.contextmanager
    def _maybe_lock() -> Iterator[None]:
        if free_threaded:
            yield
        else:
            with nlp_lock:
                yield

    t0 = time.monotonic()

    n = len(req.items)
    items: list[MorphosyntaxBatchItem | None] = []
    for raw_item in req.items:
        try:
            items.append(MorphosyntaxBatchItem.model_validate(raw_item))
        except ValidationError:
            items.append(None)

    # UNDECIDED until some path below decides. This used to be pre-filled with
    # an empty UD document, which made "Stanza never ran" identical on the wire
    # to "Stanza found nothing", so a whole language group could fail and the
    # file was still written with empty %mor and %gra tiers. Every path now
    # writes its own answer, and anything left undecided fails closed in
    # `_finalized`.
    results: list[InferResponse | None] = [None] * n

    # The ONE legitimately empty analysis: an utterance with no words has no
    # morphology, and that is a fact about the utterance rather than a failure.
    no_words_result: JSONObject = {"sentences": []}

    by_lang: dict[LanguageCode, list[StanzaInput]] = {}
    for i, item in enumerate(items):
        if item is None:
            results[i] = InferResponse(error="Invalid batch item", elapsed_s=0.0)
            continue
        if not item.words:
            results[i] = InferResponse(result=no_words_result, elapsed_s=0.0)
            continue

        words = list(item.words)
        item_lang = item.lang or req.lang

        # Apply PyCantonese word segmentation for Cantonese retokenize
        if req.retokenize and item_lang in ("yue",):
            words = _segment_cantonese(words)

        # Rust cleaned_text() already handles CHAT notation. Stripping parens
        # here silently drops bare "(" / ")" words, causing MOR count
        # mismatches in the retokenize inject path.
        by_lang.setdefault(item_lang, []).append(
            StanzaInput(
                item_index=i,
                chat_words=tuple(words),
                terminator=item.terminator,
            )
        )

    if not by_lang:
        return _finalized(results, elapsed_s=None)

    for lang_code, lang_items in by_lang.items():
        indices = [item.item_index for item in lang_items]

        # Mandarin retokenize: use Stanza neural tokenizer instead of pretokenized.
        # Only activate when the JOB language is Mandarin, per-utterance language
        # codes (e.g., [- zho] in a Cantonese file) must NOT trigger retokenization.
        use_retok_pipeline = (
            req.retokenize
            and lang_code in ("zho", "cmn")
            and req.lang in ("zho", "cmn")
        )
        entry: LoadedPipeline | None = None
        if use_retok_pipeline:
            retok_key = retokenize_key(lang_code)
            entry = pipelines.loaded(retok_key)
            if entry is None:
                # Lazy-load the retokenize pipeline on first request
                from batchalign.worker._stanza_loading import (
                    load_stanza_retokenize_model,
                )

                load_stanza_retokenize_model(lang_code)
                entry = pipelines.loaded(retok_key)
            if entry is None:
                L.warning(
                    "Failed to load retokenize pipeline for %s",
                    lang_code,
                )
                use_retok_pipeline = False
        if not use_retok_pipeline:
            entry = pipelines.loaded(lang_code)
        # Why the pipeline is missing, when we know: a failed reload says
        # something the bare absence does not, and it is the operator's actual
        # lead. Kept rather than logged and dropped.
        missing_reason = "this worker has no loaded pipeline for the language"
        if entry is None and load_pipeline is not None:
            # The worker's pipeline cache is BOUNDED, so a language loaded
            # earlier in this same batch can have been evicted to make room for
            # a later one. Ask the loader once and look again; only a second
            # miss is a real failure. A raise here is not one either: it is one
            # more way to have no pipeline, and the miss below reports it.
            try:
                load_pipeline(lang_code)
            except Exception as reload_error:
                missing_reason = f"loading it failed: {reload_error}"
                L.warning(
                    "Reloading the Stanza pipeline for %s failed: %s",
                    lang_code,
                    reload_error,
                )
            entry = pipelines.loaded(lang_code)
        if entry is None:
            failure = LanguageGroupFailure(
                kind=MorphosyntaxFailure.PIPELINE_MISSING,
                lang=lang_code,
                detail=missing_reason,
            )
            L.warning("%s", failure.message())
            for idx in indices:
                results[idx] = failure.response()
            continue

        # Space-joined for every mode. Stanza's neural tokenizer
        # (tokenize_pretokenized=False) re-segments regardless of spacing, and
        # a no-space join would merge Latin+CJK words ("hello你好" as one
        # token) in code-switched utterances. This used to be two arms that
        # computed the same string by different routes; `StanzaInput.text` is
        # defined as the space-joined boundaries, which is what the retokenize
        # arm was rebuilding by hand.
        combined = "\n\n".join(item.text for item in lang_items)
        # The context comes off the ENTRY WE ARE ABOUT TO RUN, not from a
        # second lookup by the same key. Those were two locked reads with an
        # eviction window between them, so under concurrent installs the
        # realigner could be handed `None`, or another language's context,
        # alongside the right pipeline. `LoadedPipeline` cannot be split, so
        # this pairing is now the type's, not the caller's.
        tok_ctx = entry.context
        if tok_ctx is None:
            # A genuinely different key, so a genuinely separate lookup: the
            # retokenize pipeline has no realignment context of its own and
            # borrows the plain pipeline's. A miss here means no realignment,
            # which `_realignment_mode` handles.
            for fallback_key in (lang_code, req.lang):
                fallback = pipelines.loaded(fallback_key)
                if fallback is not None and fallback.context is not None:
                    tok_ctx = fallback.context
                    break

        # `retokenize` is the whole condition: `use_retok_pipeline` is a
        # narrowing of it (Mandarin, both job and utterance), so it adds
        # nothing here. Under it Stanza owns tokenization and we want its MWT
        # expansion (gonna -> gon+na, don't -> do+n't) to pass through.
        mode = _realignment_mode(
            context=tok_ctx,
            stanza_owns_tokenization=req.retokenize,
            inputs=lang_items,
            lang_code=lang_code,
        )

        try:
            with _maybe_lock():
                with _realignment_applied(mode):
                    doc = entry.nlp(combined)

            sents = doc.to_dict()

            # Validate and normalize BEFORE anything consumes the result.
            #
            # This call is the whole point of the validators above, and its
            # absence is why they were dead code: `validate_ud_words` and its
            # `<PAD>` sanitizer were unit-tested for months while `PAD` and
            # `IOB` flowed into the published corpora, because `doc.to_dict()`
            # went straight into the response. Stanza does not promise
            # UD-conformant labels, so nothing downstream may assume it.
            validate_ud_words(sents)

            if len(sents) != len(indices):
                # Stanza returned a different number of sentences than the
                # utterances we sent, so no sentence can be attributed to any
                # utterance. That is a failure of the whole group, not a
                # licence to publish empty tiers for it.
                failure = LanguageGroupFailure(
                    kind=MorphosyntaxFailure.SENTENCE_COUNT_MISMATCH,
                    lang=lang_code,
                    detail=f"expected {len(indices)} sentences, got {len(sents)}",
                )
                L.warning("%s", failure.message())
                for idx in indices:
                    results[idx] = failure.response()
            else:
                # For Cantonese, override Stanza POS with PyCantonese.
                # Stanza's Mandarin model scores ~50% on Cantonese vocabulary;
                # PyCantonese scores ~94%. We keep Stanza's dependency parse
                # (deprel, head) and lemma, only upos is replaced.
                # Applied to ALL Cantonese morphotag, not just retokenize,
                # because the POS accuracy problem affects all Cantonese output.
                apply_pyc_pos = lang_code in ("yue",)
                # Both decisions are fixed for the whole language group.
                drop_terminator = _drops_appended_terminator(mode)

                for i, idx in enumerate(indices):
                    # The terminator was a cue for the model, not content, so
                    # it leaves here rather than travelling on to `%mor`.
                    sent = (
                        lang_items[i].without_terminator(sents[i])
                        if drop_terminator
                        else sents[i]
                    )
                    if apply_pyc_pos:
                        sent = _override_pos_with_pycantonese(sent)
                    results[idx] = InferResponse(
                        result={"raw_sentences": [sent]},
                        elapsed_s=0.0,
                    )
        except Exception as e:
            # The narration stays, but it is no longer the only place the fact
            # goes: a log line is where lost information looks like it was
            # handled. Every item of the group carries the failure home, so
            # the Rust side fails the file instead of writing empty tiers.
            failure = LanguageGroupFailure(
                kind=MorphosyntaxFailure.PIPELINE_RAISED,
                lang=lang_code,
                detail=str(e),
            )
            L.warning("%s (%d items)", failure.message(), len(indices))
            for idx in indices:
                results[idx] = failure.response()

        # Report progress: how many items have been decided so far
        # (across all language groups), whether they succeeded or failed.
        if progress_callback is not None:
            completed_so_far = sum(1 for r in results if r is not None)
            progress_callback(completed_so_far, n)

    elapsed = time.monotonic() - t0
    L.info("batch_infer morphosyntax: %d items, %.3fs", n, elapsed)
    return _finalized(results, elapsed_s=elapsed)
