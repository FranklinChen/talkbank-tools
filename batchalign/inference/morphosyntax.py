"""Stanza morphosyntax inference: words -> POS/dep/lemma.

Pure inference, no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import time
import unicodedata
from collections.abc import Callable, Generator
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, assert_never

from pydantic import BaseModel, ValidationError, model_validator

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference._english_verb_reading import annotate_verb_reading_lemmas
from batchalign.inference._morphosyntax_graph import (
    AnalyzableSentence,
    DependencyGraph,
    DependencyProjectionError,
    ProjectionFailure,
)
from batchalign.worker._pipeline_cache import (
    LoadedPipeline,
    PipelineLookup,
    retokenize_key,
)
from batchalign.worker._types_v2 import (
    MorphosyntaxPipelineV2,
    UdFeatureRepairKindV2,
    UdRelationRepairKindV2,
)

if TYPE_CHECKING:
    from batchalign.inference._tokenizer_realign import TokenizerContext

from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    ItemFailed,
    ItemOutcome,
    ItemProduced,
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
    # Where a tokenizer may merge the words: run lengths partitioning
    # `words`, each run a stretch with nothing (a retrace, a pause, a
    # bracket edge) between its words on the main tier. Required: a missing
    # partition defaulting to "one run" is exactly the joined string that
    # let the Cantonese segmenter merge across retraces.
    runs: list[int]
    lang: LanguageCode

    @model_validator(mode="after")
    def _runs_must_partition_the_words(self) -> MorphosyntaxBatchItem:
        """Every run is nonempty and together they cover the words exactly."""
        if any(run < 1 for run in self.runs) or sum(self.runs) != len(self.words):
            raise ValueError(
                f"runs {self.runs} do not partition the {len(self.words)} words"
            )
        return self

    def word_runs(self) -> list[list[str]]:
        """The words grouped into their runs, in order."""
        grouped: list[list[str]] = []
        start = 0
        for run in self.runs:
            grouped.append(self.words[start : start + run])
            start += run
        return grouped

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
    The realigned output must admit these logical boundaries before its cue
    can be projected out of the dependency graph, regardless of its POS tag.
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


@dataclass(frozen=True, slots=True)
class Realigned:
    """Our tokenization: the realigner holds Stanza to boundaries we supplied.

    This is the only mode in which the terminator's position in the output is
    known by construction, because it is the only mode in which we chose the
    boundaries.
    """

    context: TokenizerContext
    inputs: tuple[StanzaInput, ...]


@dataclass(frozen=True, slots=True)
class CueFreeRealigned:
    """One bounded retry: lexical boundaries only, with no cue to remove."""

    context: TokenizerContext
    words: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.words)


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


class TokenizationOwner(StrEnum):
    """Who decides a language group's word boundaries, resolved once per group.

    `--retokenize` asks for model word boundaries, but WHICH model draws them
    differs by language, and the difference decides how the terminator cue is
    taken back out:

    - CHAT: no retokenize; the transcript's words are the boundaries.
    - PYCANTONESE: Cantonese retokenize. PyCantonese segments the words and
      the Stanza pipeline is pretokenized, so the boundaries handed to Stanza
      are still ours and the cue is the last one by construction.
    - STANZA: any other retokenize. Stanza's own tokenizer (and MWT) draws
      them, so only Stanza knows where the cue landed.

    PYCANTONESE used to be treated as STANZA. The cue then came back as an
    ordinary token, the PyCantonese POS override relabelled `+...` as a noun
    or verb, the punctuation test that was guessing the cue no longer matched,
    and the terminator was written into the transcript as a word.
    """

    CHAT = "chat"
    PYCANTONESE = "pycantonese"
    STANZA = "stanza"

    @classmethod
    def of(cls, *, retokenize: bool, lang: LanguageCode) -> TokenizationOwner:
        """The owner for a group in `lang`; one answer for segmentation and mode."""
        if not retokenize:
            return cls.CHAT
        if lang == "yue":
            return cls.PYCANTONESE
        return cls.STANZA


# Named `RealignmentMode`, not `TokenizationMode`, because Rust already owns
# that name for a COARSER and different fact: `TokenizationMode::{Preserve,
# StanzaRetokenize}` is what the CALLER asked for. This is the resolved answer
# to "who tokenizes this batch, and can we hold Stanza to our boundaries",
# which refines `Preserve` into the case where we have a realignment context
# and the case where we asked for it and do not. Two concepts, two names.
RealignmentMode = (
    Realigned | CueFreeRealigned | StanzaOwnsTokenization | UnrealignedFallback
)


def _realignment_mode(
    *,
    context: TokenizerContext | None,
    owner: TokenizationOwner,
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
    match owner:
        case TokenizationOwner.STANZA:
            return StanzaOwnsTokenization()
        case TokenizationOwner.CHAT | TokenizationOwner.PYCANTONESE:
            # Our boundaries either way: the transcript's, or PyCantonese's
            # segments of it. Holding Stanza to them is what lets the cue be
            # projected out by position rather than recognised by its tag.
            if context is None:
                return UnrealignedFallback(lang_code=lang_code)
            return Realigned(context=context, inputs=tuple(inputs))
        case _ as unreachable:
            assert_never(unreachable)


@contextlib.contextmanager
def _realignment_applied(mode: RealignmentMode) -> Generator[None]:
    """Install the realigner's boundaries for the duration of one Stanza call."""
    match mode:
        case Realigned(context=context, inputs=inputs):
            context.original_words = [list(item.boundaries) for item in inputs]
            try:
                yield
            finally:
                context.original_words = []
        case CueFreeRealigned(context=context, words=words):
            context.original_words = [list(words)]
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

# Known misspellings of UD feature NAMES observed from Stanza, mapped to the UD
# name. Stanza's Italian model writes `Verbform=Fin` (lower-case f) on verbs
# such as the `da` of `dammela` (verified against stanza 1.14.0 and 1.15.0,
# 2026-10-01). Every consumer looks for `VerbForm`, so the verb read as having
# no form at all. The value is untouched: only the name is respelled, so the
# analysis Stanza made is the analysis that arrives.
UD_FEATURE_NAME_ALIASES: dict[str, str] = {
    "Verbform": "VerbForm",
}


def _canonical_feature_names(feats: str | None) -> str | None:
    """``feats`` with every known misspelled feature name respelled."""
    if not feats:
        return feats
    pairs = []
    for pair in feats.split("|"):
        name, sep, value = pair.partition("=")
        pairs.append(UD_FEATURE_NAME_ALIASES.get(name, name) + sep + value)
    return "|".join(pairs)


class UdWord(BaseModel, extra="allow"):
    """A single UD word/token: mirrors Rust ``UdWord`` in types.rs.

    Fills defaults for what Stanza omits, and nothing else. It does NOT repair
    relations: a validator can only return itself, so a repair made here could
    be reported nowhere, which is how relation rewrites came to exist only as
    log lines. That work belongs to ``_repaired_relation`` below, which hands
    back the repair as a value, and to ``RepairedSentence``, which carries it.
    """

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


UdWordRaw = dict[str, str | int | float | list[int] | tuple[int, ...] | None]
JSONObject = dict[str, WorkerJSONValue]


# ---------------------------------------------------------------------------
# CJK word segmentation
# ---------------------------------------------------------------------------


def _segment_cantonese(runs: list[list[str]]) -> list[str]:
    """Segment Cantonese per-character tokens into words using PyCantonese.

    ``runs`` are the utterance's words grouped by the main tier's runs
    (``MorphosyntaxBatchItem.word_runs``). Each run is segmented on its own,
    so a segmented word can only join characters with nothing between them
    in the transcript: PyCantonese given the whole utterance as one string
    merged characters on either side of a retrace or untranscribed material,
    and the rebuild had to refuse the utterance.

    Only per-character input is re-segmented. An utterance that already has
    a multi-character CJK word keeps its words as written, so word
    boundaries a transcriber or ASR engine drew are never redrawn.
    """
    words = [word for run in runs for word in run]
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

    # All CJK tokens are single characters: segment each run on its own.
    return [
        word
        for run in runs
        if (text := "".join(run))
        for word in pycantonese.segment(text)
    ]


def _override_pos_with_pycantonese(
    ud_words: list[JSONObject],
) -> list[JSONObject]:
    """Override Stanza POS tags with PyCantonese POS for Cantonese words.

    Stanza's Mandarin-trained model misclassifies core Cantonese vocabulary
    (~50% accuracy). PyCantonese's POS tagger scores ~94% on the same words.
    This function replaces ``upos`` in each UD word dict while preserving
    all other fields (lemma, deprel, head, etc.) from Stanza.

    Applied to the admitted lexical rows of the Cantonese pipeline variant.
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


# The Python wire owner supplies the vocabulary; inference cannot add a third
# copy beside the Rust schema and its Python boundary model.
RelationRepairKind = UdRelationRepairKindV2


@dataclass(frozen=True, slots=True)
class RelationRepair:
    """One relation rewrite, as a value a caller can count and attribute.

    Carries the word and both relations, not just a kind: once the analysis is
    injected, the relation Stanza produced exists nowhere, so this is the only
    record of what was changed and where.

    Refuses the two things a repair cannot be, so nothing downstream re-checks
    either: a rewrite to something that is not a UD relation, and a rewrite
    that changed nothing. Both would be defects in this module rather than
    facts about an utterance, and a stamp counting them would overstate what
    happened to the transcript.
    """

    kind: RelationRepairKind
    word: str
    from_relation: str
    to_relation: str

    def __post_init__(self) -> None:
        if self.from_relation == self.to_relation:
            raise ValueError(
                f"a repair must change the relation, but {self.from_relation!r} "
                f"was recorded as repaired to itself"
            )
        head, _, _ = self.to_relation.partition(":")
        if head not in UD_RELATIONS:
            raise ValueError(
                f"a repair must produce a UD relation, but {self.to_relation!r} "
                f"has head {head!r}, which is not one"
            )

    def wire(self) -> JSONObject:
        """The repair as the Rust side reads it (``UdRelationRepairV2``)."""
        return {
            "kind": self.kind.value,
            "word": self.word,
            "from_relation": self.from_relation,
            "to_relation": self.to_relation,
        }


def _repair(
    kind: RelationRepairKind, word: str, from_relation: str, to_relation: str
) -> tuple[str, RelationRepair]:
    """Narrate one repair and build it. The narration is no longer the record."""
    L.warning(
        "morphotag: Stanza emitted deprel=%r for word %r, applying %r (%s)",
        from_relation,
        word,
        to_relation,
        kind.value,
    )
    return to_relation, RelationRepair(
        kind=kind, word=word, from_relation=from_relation, to_relation=to_relation
    )


def _repaired_relation(deprel: str, word: str) -> tuple[str, RelationRepair | None]:
    """The relation to apply for *word*, and the repair if one was needed.

    Total, and the ONE place a relation is rewritten. Stanza does not
    guarantee UD-conformant labels: its Italian model emits ``iob`` (verified
    against stanza 1.13.0 on "attenzione ."), which is not a UD relation,
    while UD defines ``iobj``. Passing it through wrote ``2|1|IOB`` into
    ``%gra`` across the published corpora, where it went undetected for months
    because nothing on either side validated the label: CLAN CHECK does not
    check relations at all, and chatter only gained the rule (E761) in v0.4.0.

    Only the HEAD is closed. UD defines SUBTYPES as open and
    language-specific, and the corpora legitimately use many (``nmod:poss``,
    ``acl:relcl``, ``flat:foreign``), so the subtype is preserved verbatim and
    never validated.

    Returning the repair rather than applying it silently is the point. An
    unrecognised head still degrades to ``dep``, a real UD relation, rather
    than reaching the transcript, but the degradation is now a value the
    caller carries to the server, which counts it into the file's provenance.
    """
    if deprel.startswith("<") and deprel.endswith(">"):
        return _repair(RelationRepairKind.PAD_RELATION, word, deprel, "dep")

    head, sep, subtype = deprel.partition(":")
    lowered = head.lower()
    if lowered in UD_RELATIONS:
        if head == lowered:
            return deprel, None
        return _repair(
            RelationRepairKind.RELATION_CASE, word, deprel, lowered + sep + subtype
        )

    replacement = UD_DEPREL_ALIASES.get(lowered)
    if replacement is not None:
        return _repair(
            RelationRepairKind.RELATION_ALIAS, word, deprel, replacement + sep + subtype
        )

    return _repair(RelationRepairKind.UNKNOWN_RELATION, word, deprel, "dep")


# The Python wire owner supplies the feature-repair vocabulary too.
FeatureRepairKind = UdFeatureRepairKindV2


@dataclass(frozen=True, slots=True)
class FeatureRepair:
    """One feature value removed from an analysis, as a value to count.

    Carries the word, the feature and the removed value: once the analysis is
    injected, the value Stanza produced exists nowhere else. Built only from
    Rust's repair (``batchalign_core.repair_feats``), which removes padding
    and nothing else; the server refuses a reported removal of anything else.
    """

    kind: FeatureRepairKind
    word: str
    feature: str
    removed_value: str

    def wire(self) -> JSONObject:
        """The repair as the Rust side reads it (``UdFeatureRepairV2``)."""
        return {
            "kind": self.kind.value,
            "word": self.word,
            "feature": self.feature,
            "removed_value": self.removed_value,
        }


def _repaired_features(
    feats: str | None, word: str
) -> tuple[str | None, tuple[FeatureRepair, ...]]:
    """``feats`` without padding values, and one repair per removed value.

    The repair is Rust's (``batchalign_core.repair_feats``, the same test the
    server's FEATS admission refuses an unrepaired value by): each
    comma-separated value is judged on its own, so ``Case=Nom,<PAD>`` keeps
    ``Case=Nom``; a feature left with no value is dropped, and FEATS left
    with no feature is ``None``. This function only logs and records what it
    removed.
    """
    if not feats:
        return feats, ()
    from batchalign_core import repair_feats

    kept, removed = repair_feats(feats)
    repairs: list[FeatureRepair] = []
    for name, value in removed:
        L.warning(
            "morphotag: Stanza emitted padding %r as %s of word %r; removing it",
            value,
            name,
            word,
        )
        repairs.append(
            FeatureRepair(
                kind=FeatureRepairKind.PAD_FEATURE_VALUE,
                word=word,
                feature=name,
                removed_value=value,
            )
        )
    return kept, tuple(repairs)


UdRepair = RelationRepair | FeatureRepair
"""Every repair a worker reports: a relation rewritten or a feature value removed."""


@dataclass(frozen=True, slots=True)
class _RowRepair:
    """A repair bound to its producing row, not its surface string."""

    position: int
    repair: UdRepair


@dataclass(frozen=True, slots=True, init=False)
class RepairedSentence:
    """One Stanza sentence, validated, with its relations and features repaired.

    THE REPAIR IS THE CONSTRUCTOR, and it is the only one. This type is built
    from raw Stanza words and from nothing else, so a repaired sentence cannot
    exist without the repair having run, the step cannot be skipped, and no
    caller can mint one that claims an empty repair list beside words nothing
    checked. That is what lets ``_analysis`` take the repairs from the
    sentence rather than from a caller who could pass another sentence's.

    It replaces a validator that returned ``None`` and mutated its argument in
    place, which left no proof it had run: the pipeline called it for months
    while ``PAD`` and ``IOB`` flowed into the corpora, and nothing in a
    signature could have said so.
    """

    _words: tuple[JSONObject, ...]
    repair_rows: tuple[_RowRepair, ...]

    @property
    def words(self) -> tuple[JSONObject, ...]:
        """Return copies; consumers cannot mutate row admission evidence."""
        return deepcopy(self._words)

    @property
    def repairs(self) -> tuple[UdRepair, ...]:
        return tuple(entry.repair for entry in self.repair_rows)

    def __init__(self, raw_words: list[UdWordRaw]) -> None:
        words: list[JSONObject] = []
        repairs: list[_RowRepair] = []
        for position, raw in enumerate(raw_words):
            raw_id = raw.get("id")
            # Copied rather than coerced in place: Stanza's MWT ranges arrive
            # as tuples, and the caller's document is not ours to rewrite.
            if isinstance(raw_id, tuple):
                raw = {**raw, "id": list(raw_id)}

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

            feats, feature_repairs = _repaired_features(
                _canonical_feature_names(validated.feats), validated.text
            )
            validated.feats = feats
            repairs.extend(_RowRepair(position, entry) for entry in feature_repairs)

            relation, repair = _repaired_relation(validated.deprel, validated.text)
            if repair is not None:
                validated.deprel = relation
                repairs.append(_RowRepair(position, repair))

            words.append(validated.model_dump())

        object.__setattr__(self, "_words", tuple(words))
        object.__setattr__(self, "repair_rows", tuple(repairs))


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
        return InferResponse.unexecuted(ItemFailed(error=self.message()))


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


_NO_STANZA_VERSION_ERROR = (
    "morphosyntax cannot name the Stanza version that would analyze this item; "
    "refusing to report an analysis without its model identity"
)
"""Failure for an item whose analysis could not name its model."""


_NO_WORDS_RESULT: JSONObject = {"kind": "no_words"}
"""The one legitimately empty outcome: an utterance with no words.

It has no morphology, which is a fact about the utterance rather than a
failure, and no model ran on it, so it names none. It used to be an empty
analysis stamped with the batch's Stanza version, which provenance then
counted as that model having analyzed something.
"""


def _pipeline_variant(
    *, lang_code: LanguageCode, mandarin_retokenize: bool
) -> MorphosyntaxPipelineV2:
    """Which procedure analyzes one language group.

    The value decides the procedure as well as naming it: the PyCantonese
    override below runs exactly when this says Cantonese, so the reported
    variant and the executed one cannot drift apart.
    """
    if mandarin_retokenize:
        return MorphosyntaxPipelineV2.MANDARIN_RETOKENIZE
    if lang_code == "yue":
        return MorphosyntaxPipelineV2.CANTONESE_PYCANTONESE_POS
    return MorphosyntaxPipelineV2.STANDARD


def _analysis(
    sentence: AnalyzableSentence,
    *,
    lang: LanguageCode,
    stanza_version: str | None,
    pipeline: MorphosyntaxPipelineV2,
) -> ItemOutcome:
    """One analyzed item, carrying the model identity every analysis names.

    Takes the graph-owned emission capability, never raw/repaired rows or a
    separate repair list. Projection cannot leave a deleted target behind.

    Built as the wire's tagged ``analyzed`` shape
    (``MorphosyntaxAnalyzedItemV2``) directly rather than through the pydantic
    model: this runs once per utterance, and validating every Stanza sentence
    a second time would cost more than the whole response is worth. The Rust
    bridge parses the item into its tagged type either way.

    With no Stanza version there is no honest identity, so the item fails.
    """
    if stanza_version is None:
        return ItemFailed(error=_NO_STANZA_VERSION_ERROR)
    # Each repair goes to the wire list its kind belongs to; exhaustive, so a
    # third kind of repair cannot be added without deciding where it is sent.
    relation_repairs: list[WorkerJSONValue] = []
    feature_repairs: list[WorkerJSONValue] = []
    for repair in sentence.repairs:
        match repair:
            case RelationRepair():
                relation_repairs.append(repair.wire())
            case FeatureRepair():
                feature_repairs.append(repair.wire())
            case _ as unreachable:
                assert_never(unreachable)
    return ItemProduced(
        result={
            "kind": "analyzed",
            "raw_sentences": [sentence.wire_rows()],
            "model": {
                "stanza_version": stanza_version,
                "lang": lang,
                "pipeline": pipeline.value,
            },
            "repairs": relation_repairs,
            "feature_repairs": feature_repairs,
        }
    )


def _analyzed_item(
    sentence_index: int,
    sentence: list[UdWordRaw],
    *,
    mode: RealignmentMode,
    apply_pyc_pos: bool,
    lang: LanguageCode,
    stanza_version: str | None,
    pipeline: MorphosyntaxPipelineV2,
    retry_cue_free: Callable[[StanzaInput], ItemOutcome] | None = None,
) -> ItemOutcome:
    """One item's own post-Stanza work, as a single call that can be timed.

    Everything here is attributable to THIS item: taking back the terminator
    cue we appended, the Cantonese POS override, the relation repair and the
    analysis. The initial Stanza call runs once per language group; its cost
    belongs to the group. A bounded lexical-only retry belongs to this item
    and is included in its elapsed time.
    """
    try:
        graph = DependencyGraph(RepairedSentence(sentence))
        match mode:
            case Realigned(inputs=inputs):
                admitted = graph.project_cue(len(inputs[sentence_index].boundaries))
            case CueFreeRealigned(words=words):
                admitted = graph.preserve_boundaries(len(words))
            case StanzaOwnsTokenization() | UnrealignedFallback():
                admitted = graph.preserve()
            case _ as unreachable:
                assert_never(unreachable)
        if apply_pyc_pos:
            admitted = admitted.with_pos(_override_pos_with_pycantonese)
    except (DependencyProjectionError, ValidationError) as error:
        if (
            isinstance(error, DependencyProjectionError)
            and error.kind is ProjectionFailure.AMBIGUOUS_ROOT
            and isinstance(mode, Realigned)
            and retry_cue_free is not None
        ):
            L.warning(
                "morphotag: cue-root ambiguity; requesting one lexical-only parse"
            )
            retried = retry_cue_free(mode.inputs[sentence_index])
            if isinstance(retried, ItemFailed):
                return ItemFailed(
                    error=f"morphosyntax dependency projection: {error}; cue-free retry: {retried.error}"
                )
            return retried
        # This item's graph failed; do not relabel the entire language group.
        return ItemFailed(error=f"morphosyntax dependency projection: {error}")
    return _analysis(
        admitted,
        lang=lang,
        stanza_version=stanza_version,
        pipeline=pipeline,
    )


def _reanalyze_without_cue(
    source: StanzaInput,
    *,
    mode: Realigned,
    entry: LoadedPipeline,
    apply_pyc_pos: bool,
    lang: LanguageCode,
    stanza_version: str | None,
    pipeline: MorphosyntaxPipelineV2,
) -> ItemOutcome:
    """Ask the same model for lexical heads once, with no removable cue."""
    lexical = CueFreeRealigned(mode.context, source.chat_words)
    try:
        with entry.calling() as nlp:
            with _realignment_applied(lexical):
                retry_rows = nlp(lexical.text).to_dict()
            if lang == "eng":
                annotate_verb_reading_lemmas(retry_rows, nlp)
        if len(retry_rows) != 1:
            return ItemFailed(error="cue-free model returned the wrong sentence count")
        return _analyzed_item(
            0,
            retry_rows[0],
            mode=lexical,
            apply_pyc_pos=apply_pyc_pos,
            lang=lang,
            stanza_version=stanza_version,
            pipeline=pipeline,
        )
    except Exception as error:
        return ItemFailed(error=f"cue-free model failed: {error}")


def _finalized(results: list[InferResponse | None]) -> BatchInferResponse:
    """Settle every item's decision into the batch response.

    ``None`` means "no path decided this item" and becomes an error (see
    ``_UNDECIDED_ERROR``).

    Nothing is stamped onto an item here. Every response that reports a
    duration got it from ``InferResponse.timed``, which timed that item's own
    work, and the rest report none (``null``, built by
    ``InferResponse.unexecuted``) because no work is attributable to them: an
    item whose payload never parsed, an utterance with no words, and every item
    of a language group that failed before per-item work began. Until
    2026-10-01 those reported ``0.0``, which reads as a measurement.

    Until 2026-09-16 this function took the batch's total elapsed time and
    wrote it onto ``results[0]``. That inflated the first item by every other
    item's work, left every other item at 0.0, and made both indistinguishable
    from an item that genuinely took no measurable time, inside
    provenance-bearing evidence. The docstring used to justify it as "the
    batch-level timing convention the Rust side reads"; checked 2026-09-16,
    nothing reads it. The Rust control plane measures the whole response itself
    in ``worker_execute::execute_request_v2``, and
    ``normalize_morphosyntax_result`` never looks at the per-item field. The
    batch total is a fact about the batch, so it is reported where a batch fact
    belongs, this module's log line, and it is no longer a parameter any caller
    can hand to an item.
    """
    return BatchInferResponse(
        results=[
            r
            if r is not None
            else InferResponse.unexecuted(ItemFailed(error=_UNDECIDED_ERROR))
            for r in results
        ]
    )


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def batch_infer_morphosyntax(
    req: BatchInferRequest,
    pipelines: PipelineLookup,
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
        Every call on a pipeline goes through its entry's guard
        (``LoadedPipeline.calling``), so one pipeline is never called by two
        threads at once, whatever the interpreter.
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

    # The version every analyzed item names as its model, resolved once
    # through the worker's one accessor. Imported here rather than at module
    # load so this module stays importable without the worker state.
    from batchalign.worker._types import _state

    stanza_version = _state.stanza_version()

    by_lang: dict[LanguageCode, list[StanzaInput]] = {}
    for i, item in enumerate(items):
        if item is None:
            results[i] = InferResponse.unexecuted(
                ItemFailed(error="Invalid batch item")
            )
            continue
        if not item.words:
            results[i] = InferResponse.unexecuted(ItemProduced(result=_NO_WORDS_RESULT))
            continue

        words = list(item.words)
        item_lang = item.lang or req.lang

        # The same owner the group's mode is resolved from below, so the
        # words cannot be segmented by one rule and realigned by another.
        match TokenizationOwner.of(retokenize=req.retokenize, lang=item_lang):
            case TokenizationOwner.PYCANTONESE:
                words = _segment_cantonese(item.word_runs())
            case TokenizationOwner.CHAT | TokenizationOwner.STANZA:
                pass
            case _ as unreachable:
                assert_never(unreachable)

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
        return _finalized(results)

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

        # The owner, not `retokenize` alone: `use_retok_pipeline` is a
        # narrowing of STANZA (Mandarin, both job and utterance), so it adds
        # nothing here. Under STANZA we want its MWT expansion (gonna ->
        # gon+na, don't -> do+n't) to pass through; under PYCANTONESE the
        # boundaries are ours and Stanza is held to them.
        mode = _realignment_mode(
            context=tok_ctx,
            owner=TokenizationOwner.of(retokenize=req.retokenize, lang=lang_code),
            inputs=lang_items,
            lang_code=lang_code,
        )
        # Decided after the retokenize fallback above, so a Mandarin group
        # whose retokenize pipeline failed to load reports the standard
        # procedure it actually ran.
        pipeline = _pipeline_variant(
            lang_code=lang_code, mandarin_retokenize=use_retok_pipeline
        )

        try:
            with entry.calling() as nlp:
                with _realignment_applied(mode):
                    doc = nlp(combined)
                sents = doc.to_dict()
                if lang_code == "eng":
                    # Evidence for the Rust finite-verb rescue, computed by the
                    # same pipeline under the same guard (see the module).
                    annotate_verb_reading_lemmas(sents, nlp)

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
                apply_pyc_pos = (
                    pipeline is MorphosyntaxPipelineV2.CANTONESE_PYCANTONESE_POS
                )

                retry_cue_free = (
                    partial(
                        _reanalyze_without_cue,
                        mode=mode,
                        entry=entry,
                        apply_pyc_pos=apply_pyc_pos,
                        lang=lang_code,
                        stanza_version=stanza_version,
                        pipeline=pipeline,
                    )
                    if isinstance(mode, Realigned)
                    else None
                )

                for i, idx in enumerate(indices):
                    # Measured as it runs, so the duration this item reports is
                    # the one it earned. `_analyzed_item` keeps the terminator
                    # strip and the relation repair on the only route to a
                    # response: Stanza does not promise UD-conformant labels,
                    # and for months nothing on the production path checked,
                    # because `doc.to_dict()` went straight into the response
                    # while the validators sat unit-tested and uncalled, and
                    # `PAD` and `IOB` reached the published corpora. A step the
                    # type system requires cannot be dropped again.
                    results[idx] = InferResponse.timed(
                        partial(
                            _analyzed_item,
                            i,
                            sents[i],
                            mode=mode,
                            apply_pyc_pos=apply_pyc_pos,
                            lang=lang_code,
                            stanza_version=stanza_version,
                            pipeline=pipeline,
                            retry_cue_free=retry_cue_free,
                        )
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

    # The batch total is a fact about the batch, so it is reported where a
    # batch fact belongs: this log line. It is deliberately written onto no
    # item, because no item performed it.
    elapsed = time.monotonic() - t0
    L.info("batch_infer morphosyntax: %d items, %.3fs", n, elapsed)
    return _finalized(results)
