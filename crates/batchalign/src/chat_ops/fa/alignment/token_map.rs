//! Attributing an onset-only engine's LABELS to the transcript's WORDS.
//!
//! # The problem this module owns
//!
//! Whisper-style forced alignment answers with a stream of LABELS: little
//! pieces of text, each with the instant it began. The transcript answers with
//! WORDS. Nothing guarantees the two agree about where one unit stops and the
//! next starts: the engine splits on its own byte-pair vocabulary, spells
//! numerals out ("nineteen ninety five" for `1995`), and drops or adds material
//! the transcriber did not. So a mapping has to be computed, and the only thing
//! the engine actually MEASURED is a label's onset.
//!
//! # Two routes, and why the second one exists
//!
//! 1. **The in-order stitch.** Walk labels and words together, concatenating
//!    whole labels until they spell the whole word. Exact when it works, and
//!    it works for the overwhelming majority of groups.
//! 2. **The character-level DP remap.** When the stitch cannot identify a word,
//!    everything from that word onwards is the RESIDUE, and the residue's
//!    characters are aligned against the remaining labels' characters with the
//!    Hirschberg aligner this workspace already owns
//!    ([`batchalign_transform::dp_align::align_chars`]). Each label then goes
//!    to whichever word matched the most of its characters, and a word's span
//!    runs from its first label's onset to the onset of the first label it did
//!    not get.
//!
//! Until 2026-09-07 there was no second route: a mismatch ABORTED the group,
//! so every later word lost a timing whose onset was sitting unread in the same
//! response. One tokenizer disagreement cost the whole utterance.
//!
//! # What the DP does NOT do
//!
//! It never splits a label. A label is one measured interval, and cutting it in
//! two to give a boundary to each of two words invents a number nothing
//! observed, which is the failure mode the whole [`crate::chat_ops::fa::timing`]
//! vocabulary exists to prevent. So when one label merges two transcript words,
//! one of them is timed and the other is [`UntimedReason::NoLabelSpan`], said
//! out loud rather than silently absent.
//!
//! # Why the route is part of the RESULT, and of the NUMBER
//!
//! A word placed by exact concatenation and a word placed by an edit-distance
//! guess are not the same claim, even when the milliseconds are identical, and
//! `Option<WordTiming>` cannot tell them apart: `Some` said "timed" and `None`
//! said nothing at all about why. [`WordTimingOutcome`] is the sum of what can
//! happen to a word, so the caller inside this crate matches on it.
//!
//! That is not enough on its own, and saying it was is the mistake this
//! paragraph used to make. `LabelMapping::into_timings` lowers to
//! `Vec<Option<WordTiming>>` at the module edge, so a route recorded ONLY in
//! the outcome would be dead the moment it left, and a remapped word would
//! reach a reviewer byte-identical to a stitched one. So the route also goes
//! into the NUMBER'S OWN provenance: BOTH ends of a DP-remapped word are
//! wrapped in [`Origin::AttributedByCharAlignment`], which keeps the earlier
//! origin underneath, carries the [`CharEdits`] that say how badly the
//! characters fitted, and classifies as ASSUMED, so
//! [`crate::chat_ops::fa::origin::ProvenanceTally::needs_review`] sees it.
//!
//! Both routes still get their milliseconds from the SAME
//! [`LabelTrack::span_for`], so the origin UNDERNEATH an end is
//! `DerivedFromNextOnset` or `FallbackDuration` on both routes. What the DP
//! decides is not how a label's onset was obtained but WHICH WORD the run of
//! labels belongs to, and that decision fixes both of the word's boundaries at
//! once: the run's first onset is its start and the run's last label settles
//! its end. Until 2026-09-07 only the START was wrapped, so a reviewer reading
//! a remapped word's end saw `DerivedFromNextOnset` and could not tell that
//! WHICH label ended the word was the same guess as which label began it.

use crate::chat_ops::fa::coordinates::{
    Clamped, FaWindow, Ms, OutsideWindow, RecordedInstant, Recording, WindowMs,
};
use unicode_normalization::UnicodeNormalization;

use crate::chat_ops::fa::origin::{CharEdits, EngineId, Origin};
use crate::chat_ops::fa::timing::{SpanFault, WordSpan};

use super::DiscardedTimings;
use super::residue::remap_residue;
use crate::chat_ops::fa::{FaWord, LAST_WORD_FALLBACK_MS, WordTiming};
use crate::chat_ops::nlp::FaRawToken;

/// The comparison key for one word or one label.
///
/// Lower-cased, COMPOSED, and stripped to alphanumerics, because neither
/// side's punctuation, casing, nor choice of Unicode encoding is evidence
/// about the audio: an engine that writes `", world"` and a transcript that
/// writes `world` are saying the same thing, and so are the two legal
/// spellings of an accented letter.
///
/// # Why composition comes before the filter, and after the lowercasing
///
/// Unicode encodes "e with an acute accent" two ways: NFC, one code point
/// U+00E9, and NFD, `e` followed by the combining mark U+0301. The filter
/// keeps alphanumerics, and a combining mark is not one, so the NFD spelling
/// silently lost its accent and keyed as `cafe` while the NFC spelling keyed
/// as `café`. Two keys for one word, and the stitch then failed on a
/// difference that exists only in the encoding. Composing FOLDS the mark into
/// its base instead of dropping it beside it.
///
/// It runs after the lowercasing rather than before because lowercasing can
/// itself DECOMPOSE: U+0130 (capital I with dot above) lowercases to `i` plus
/// U+0307, so composing first would leave a fresh combining mark for the
/// filter to eat.
///
/// # The residual limit, stated rather than hidden
///
/// Composition only helps where a composed form EXISTS. A script whose marks
/// have no precomposed code point (most Indic scripts, and many rarer
/// combinations in Latin) still loses them here. That is a narrower gap than
/// the one this closes, and closing it means keeping marks rather than
/// composing them, which changes the key for every language at once; it should
/// be done deliberately with corpus evidence, not as a side effect of this fix.
pub(crate) fn normalize_fa_alignment_unit(text: &str) -> String {
    text.chars()
        .flat_map(|ch| ch.to_lowercase())
        .nfc()
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

/// Position of a transcript word within ONE FA group.
///
/// A newtype because this module carries four different `usize` sequences at
/// once (words, surviving labels, word characters, label characters) and
/// passing one where another belongs type-checks perfectly when they are all
/// bare integers. Careful variable naming is what people do instead, and the
/// careful naming is the tell that the type is owed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WordSlot(pub(crate) usize);

impl WordSlot {
    /// The word at `index` within its group.
    ///
    /// An index, not a proof: it asserts nothing except which sequence it
    /// belongs to, which is the whole reason it is a newtype.
    ///
    /// Private to this module, so the only slots in the program are the ones
    /// [`WordTrack`] handed out and they cannot name a word that is not in the
    /// track. [`super::residue`] used to build its own from a bare `usize`
    /// cursor walked against a `&[String]`, which is the pairing-by-care shape
    /// the newtype exists to remove.
    const fn new(index: usize) -> Self {
        Self(index)
    }

    /// A slot for a hand-built sequence in a test.
    ///
    /// The second and last route to a `WordSlot`, enumerated here as the
    /// weakest-constructor rule requires. It exists for
    /// [`super::residue::owners_are_monotone`], whose whole subject is an
    /// owner sequence no real alignment produces, so there is no track to get
    /// the slots from.
    #[cfg(test)]
    pub(crate) const fn for_test(index: usize) -> Self {
        Self(index)
    }
}

/// The transcript's words for one group, normalized, in order.
///
/// # Why this is a type
///
/// The mirror of [`LabelTrack`] on the transcript side, and it exists for the
/// same reason: the normalized forms were a bare `&[String]` passed beside a
/// `WordSlot`, and [`super::residue`] walked it with a raw `usize` cursor,
/// minting a `WordSlot` from that cursor at five call sites. Nothing said the
/// cursor and the slice belonged together, so an index into a DIFFERENT
/// sequence type-checked. Here the slots come FROM the track that answers for
/// them.
pub(crate) struct WordTrack {
    norms: Vec<String>,
}

impl WordTrack {
    /// The only way to obtain a track: normalize the group's own words.
    ///
    /// Refuses the whole group when a word has no alphanumeric content: it has
    /// no identity to stitch and no characters to align, and it would silently
    /// join its neighbour in the concatenated character stream.
    fn of_words(words: &[FaWord]) -> Result<Self, UntimedReason> {
        let mut norms = Vec::with_capacity(words.len());
        for word in words {
            let norm = normalize_fa_alignment_unit(word.text.as_str());
            if norm.is_empty() {
                return Err(UntimedReason::NoLexicalContent);
            }
            norms.push(norm);
        }
        Ok(Self { norms })
    }

    pub(crate) fn len(&self) -> usize {
        self.norms.len()
    }

    pub(crate) fn norm(&self, slot: WordSlot) -> &str {
        // Every `WordSlot` comes from this track's own `slots_from`, or from
        // `0..len` inside this module, so the access is total by construction.
        self.norms[slot.0].as_str()
    }

    /// Every slot from `from` to the end, in order.
    pub(crate) fn slots_from(&self, from: WordSlot) -> impl Iterator<Item = WordSlot> + use<> {
        (from.0..self.norms.len()).map(WordSlot::new)
    }

    /// The slot after `slot`, or `None` at the end of the track.
    pub(crate) fn after(&self, slot: WordSlot) -> Option<WordSlot> {
        (slot.0 + 1 < self.norms.len()).then(|| WordSlot::new(slot.0 + 1))
    }

    /// The slot before `slot`, or `None` at the start of the track.
    pub(crate) fn before(&self, slot: WordSlot) -> Option<WordSlot> {
        slot.0.checked_sub(1).map(WordSlot::new)
    }
}

/// Position of a SURVIVING label within one FA group.
///
/// "Surviving" matters: labels the engine placed outside the audio it was given
/// are dropped before a `LabelSlot` exists, so this is never an index into the
/// engine's own token list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LabelSlot(pub(crate) usize);

/// A non-empty, contiguous run of labels attributed to ONE word.
///
/// Contiguous by construction rather than by convention: both routes can only
/// produce a run by naming a first and a last slot, so "the labels for this
/// word" is never a set that might have a hole in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LabelRun {
    pub(crate) first: LabelSlot,
    pub(crate) last: LabelSlot,
}

impl LabelRun {
    /// The run covering `first..=last`, or `None` if that is empty.
    pub(crate) fn through(first: LabelSlot, last: LabelSlot) -> Option<Self> {
        (last >= first).then_some(Self { first, last })
    }

    /// The first label after this run, which is where the word's end comes from.
    fn after(self) -> LabelSlot {
        LabelSlot(self.last.0 + 1)
    }
}

/// The labels that survived, in order, with the instants the engine measured.
///
/// # Why this is a type and not two parallel `Vec`s
///
/// The norms and the onsets are indexed by the same [`LabelSlot`] and nothing
/// but care kept them the same length while they were two locals in one long
/// function. Here they are private to a struct with ONE constructor, so the
/// pairing is a property of the value rather than a thing to remember.
pub(crate) struct LabelTrack {
    norms: Vec<String>,
    onsets: Vec<RecordedInstant>,
}

impl LabelTrack {
    /// The only way to obtain a track: filter the engine's own tokens.
    ///
    /// Three kinds of token never become a label. Whisper's special markers
    /// (`<|notimestamps|>` and friends) are protocol, not speech. A token whose
    /// normalized form is empty (a bare comma) carries no identity to match.
    /// And a token the engine placed PAST the audio it was handed is the shape
    /// that once wrote word timings 28 seconds beyond the end of a recording:
    /// Whisper pads its input to a fixed 30 second window and duly reports
    /// onsets across the padding, so containment is checked here, once, before
    /// any onset can reach a span.
    fn from_engine_tokens(
        tokens: &[FaRawToken],
        window: &FaWindow,
        engine: &EngineId,
        discarded: &mut DiscardedTimings,
    ) -> Self {
        let mut norms = Vec::new();
        let mut onsets = Vec::with_capacity(tokens.len());
        for token in tokens {
            let token_text = token.text.trim();
            if token_text.starts_with("<|") && token_text.ends_with("|>") {
                continue;
            }
            let norm = normalize_fa_alignment_unit(token_text);
            if norm.is_empty() {
                continue;
            }
            match window.to_file(WindowMs::reported((token.time_s * 1000.0) as u64), engine) {
                Ok(onset) => {
                    norms.push(norm);
                    onsets.push(onset);
                }
                Err(fault) => discarded.record_outside(fault),
            }
        }
        Self { norms, onsets }
    }

    pub(crate) fn len(&self) -> usize {
        self.norms.len()
    }

    fn is_empty(&self) -> bool {
        self.norms.is_empty()
    }

    pub(crate) fn norm(&self, slot: LabelSlot) -> &str {
        // Every `LabelSlot` this module builds comes from a range over
        // `0..self.len()`, so the slice access is total by construction.
        self.norms[slot.0].as_str()
    }

    /// The extent of the word that owns `run`.
    ///
    /// THE shared step, and the reason the control test can assert that a
    /// cleanly stitched group is unchanged: both routes reach their
    /// milliseconds here, so there is no second arithmetic to drift.
    ///
    /// An onset-only engine says when speech STARTS and never when it stops, so
    /// a word's end is always somebody else's number, and which somebody is a
    /// fact about the value:
    ///
    /// * a label follows the run -> the end is DERIVED from that label's onset
    /// * none does -> the end is ASSUMED, and may then be CLAMPED
    fn span_for(
        &self,
        run: LabelRun,
        recording: &Recording,
    ) -> Result<Clamped<WordSpan>, SpanFault> {
        let start = self.onsets[run.first.0].clone();
        match self.onsets.get(run.after().0) {
            Some(next_onset) => {
                WordSpan::end_from_next_onset(start, next_onset).map(Clamped::AsGiven)
            }
            None => WordSpan::end_assumed(start, Ms(LAST_WORD_FALLBACK_MS), recording),
        }
    }
}

/// Why a word has no timing.
///
/// A sum rather than `None`, because an operator's next action differs by
/// reason: a group refused for lack of lexical content is a transcript
/// question, a span refused for having no width is an engine question, and a
/// word the DP could not place is a tokenization question. Deliberately not
/// counted in this sentence; the list below is the list, and a prose count
/// beside it is one more thing to keep true.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UntimedReason {
    /// The HOST returned no timing for this word at all.
    ///
    /// Not a refusal of ours: an indexed-word-level response carries one
    /// `Option` per input word, and a Qwen3 group answers `None` for a word
    /// whose characters its tokenizer keeps none of. Distinct from every
    /// reason below, which are all timings the engine DID return and we then
    /// found unusable, and the distinction is the operator's next action:
    /// this one is a question for the engine, the others are questions about
    /// its answer.
    NoTimingFromHost,
    /// The engine reported this word's extent outside the audio it was handed.
    ///
    /// The shape that put up to 28.2 seconds of phantom speech into six of 226
    /// screened sessions: half a span is not a span, so one end outside the
    /// window condemns the pair rather than being repaired from the other.
    ReportedOutsideWindow(OutsideWindow),
    /// A word in this group normalized to nothing, so the whole group has no
    /// character stream to stitch or align against.
    NoLexicalContent,
    /// No label survived filtering, so there is nothing to attribute.
    NoUsableLabels,
    /// No label was attributed to this word.
    ///
    /// Either the labels ran out before the word did, or one label merged this
    /// word with its neighbour and the neighbour matched more of it. Splitting
    /// that label is what this module refuses to do.
    NoLabelSpan,
    /// Labels were attributed and the span they imply cannot exist.
    ///
    /// Whisper reports equal adjacent onsets for single-frame backchannels; a
    /// word cannot start and end at the same instant, so the honest answer is
    /// no timing rather than an invented millisecond.
    SpanRefused(SpanFault),
    /// A span that [`WordSpan`] proved positive was still refused by
    /// [`WordTiming::new`].
    ///
    /// Unreachable while `WordSpan` guarantees `end > start`. It is a variant
    /// rather than an `expect` so that weakening that guarantee shows up as a
    /// reported word rather than as a panic in a long-running pipeline.
    TimingRefusedProvenSpan,
    /// The residue was longer than [`MAX_RESIDUE_ALIGN_CHARS`] on one side, so
    /// no character alignment was attempted for it.
    ///
    /// Refused rather than truncated: aligning a prefix would attribute labels
    /// from a comparison that never saw the words they belong to.
    ResidueTooLongToAlign {
        /// Transcript characters in the residue.
        transcript_chars: usize,
        /// Label characters in the residue.
        label_chars: usize,
        /// The budget each had to fit inside.
        budget: usize,
    },
    /// Character alignment attributed labels to words OUT OF ORDER.
    ///
    /// A monotone alignment cannot do this, so it means an assumption this
    /// module rests on has stopped holding. The residue is refused rather than
    /// timed from a contradiction; see [`owners_are_monotone`].
    NonMonotoneAttribution,
}

impl UntimedReason {
    /// The sentence a log line leads with when this reason refused a WHOLE
    /// group, or `None` for the reasons that only ever apply to one word.
    ///
    /// Exists because the log line used to open "labels did not tokenize as
    /// the transcript does" for every refusal, including a group with no
    /// lexical content and a group with no surviving labels, neither of which
    /// ever reached the tokenization question. A stated cause that is not the
    /// cause is worse than no cause.
    pub(crate) const fn group_headline(&self) -> Option<&'static str> {
        match self {
            Self::NoLexicalContent => Some(
                "a word in this group has no alphanumeric content, so nothing could be \
                 matched against the engine's labels",
            ),
            Self::NoUsableLabels => {
                Some("no engine label survived filtering, so this group had nothing to attribute")
            }
            Self::ResidueTooLongToAlign { .. } => Some(
                "the residue is longer than the character-alignment budget; those words \
                 are left untimed rather than aligned from a truncated comparison",
            ),
            Self::NonMonotoneAttribution => Some(
                "character alignment attributed labels out of order; the residue is \
                 refused rather than timed from a contradiction",
            ),
            // Per-word outcomes: they say nothing about the group as a whole,
            // so the counts in the log line speak for them.
            Self::NoTimingFromHost
            | Self::ReportedOutsideWindow(_)
            | Self::NoLabelSpan
            | Self::SpanRefused(_)
            | Self::TimingRefusedProvenSpan => None,
        }
    }
}

/// What happened to one word, and how well its labels accounted for it.
///
/// # Why there is one timed variant and not two
///
/// There were two, `TimedByStitch` and `TimedByDpRemap`, and they recorded
/// WHICH CODE RAN. That is not a fact about the word: the stitch is the case
/// of an edit-distance fold with zero unreconciled characters, so a residue
/// word the DP reconciled exactly is evidenced precisely as well as a stitched
/// one, and splitting them made the weaker-looking variant carry words nothing
/// was guessed about. [`CharEdits::is_exact`] is the fact worth keeping, and
/// it is the same question on both former arms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WordTimingOutcome {
    /// The word has a timing, whose milliseconds are label onsets.
    Timed {
        /// The timing.
        timing: WordTiming,
        /// How much of the word and of its labels went unreconciled.
        ///
        /// [`CharEdits::ZERO`] means the labels spell the word exactly, which
        /// is what the in-order stitch proves and what an exact fold of the
        /// residue proves equally.
        edits: CharEdits,
    },
    /// The word has no timing, and says why.
    Untimed {
        /// What went wrong.
        reason: UntimedReason,
    },
}

impl WordTimingOutcome {
    /// The timing, for a caller that only needs the milliseconds.
    ///
    /// Test-only today, and gated rather than left dead: the production
    /// lowering wants the timing BY VALUE ([`Self::into_timing`]), so a
    /// borrowing accessor beside it would be a second way to ask the same
    /// question with no caller.
    #[cfg(test)]
    pub(crate) fn timing(&self) -> Option<&WordTiming> {
        match self {
            Self::Timed { timing, .. } => Some(timing),
            Self::Untimed { .. } => None,
        }
    }

    fn into_timing(self) -> Option<WordTiming> {
        match self {
            Self::Timed { timing, .. } => Some(timing),
            Self::Untimed { .. } => None,
        }
    }
}

/// Lower a group's outcomes to the vector the FA pipeline still speaks in.
///
/// THE lossy step, and the only one: every reason a word has no timing is
/// dropped here. Both response shapes go through it, so a future consumer that
/// wants the reason has one place to stop calling rather than two.
pub(crate) fn lower_outcomes(outcomes: Vec<WordTimingOutcome>) -> Vec<Option<WordTiming>> {
    outcomes
        .into_iter()
        .map(WordTimingOutcome::into_timing)
        .collect()
}

/// One group's worth of attributions, plus what the engine reported unusably.
pub(crate) struct LabelMapping {
    outcomes: Vec<WordTimingOutcome>,
    discarded: DiscardedTimings,
    /// How many tokens the engine sent, for the log line's denominator.
    tokens_reported: usize,
}

impl LabelMapping {
    /// One outcome per input word, in input order.
    pub(crate) fn outcomes(&self) -> &[WordTimingOutcome] {
        &self.outcomes
    }

    /// Lower to the vector the FA pipeline still speaks in.
    ///
    /// The lossy step, and the only one: everything the route said is dropped
    /// here. It is a single call site so that a future consumer wanting the
    /// route has one place to stop calling.
    pub(crate) fn into_timings(self) -> Vec<Option<WordTiming>> {
        lower_outcomes(self.outcomes)
    }

    fn all_untimed(
        words: usize,
        reason: UntimedReason,
        discarded: DiscardedTimings,
        tokens_reported: usize,
    ) -> Self {
        Self {
            outcomes: vec![WordTimingOutcome::Untimed { reason }; words],
            discarded,
            tokens_reported,
        }
    }

    /// Say, once, what this group could not do.
    pub(crate) fn report(&self, window: &FaWindow, engine: &EngineId) {
        self.discarded
            .warn_if_any(&self.outcomes, self.tokens_reported, engine, window);

        let mut exact = 0usize;
        let mut edited = 0usize;
        let mut untimed = 0usize;
        for outcome in self.outcomes() {
            match outcome {
                WordTimingOutcome::Timed { edits, .. } if edits.is_exact() => exact += 1,
                WordTimingOutcome::Timed { .. } => edited += 1,
                WordTimingOutcome::Untimed { .. } => untimed += 1,
            }
        }
        if edited + untimed == 0 {
            return;
        }
        // The CAUSE leads, not a guess at it. A group-level refusal names
        // itself; otherwise the tokenization sentence is the true one, and its
        // second half says whether the group was rescued or still lost words.
        let group_refusal = self.outcomes().iter().find_map(|outcome| match outcome {
            WordTimingOutcome::Untimed { reason } => reason.group_headline(),
            WordTimingOutcome::Timed { .. } => None,
        });
        let headline = match (group_refusal, untimed) {
            (Some(stated), _) => stated,
            (None, 0) => {
                "labels did not tokenize as the transcript does; the residue was \
                 remapped by character alignment"
            }
            (None, _) => {
                "labels did not tokenize as the transcript does; some words could \
                 not be given a label span and are left untimed"
            }
        };
        tracing::warn!(
            // Counted by what the characters said, not by which code ran: an
            // exactly reconciled fold belongs with the stitched words.
            exactly_reconciled_words = exact,
            partly_reconciled_words = edited,
            untimed_words = untimed,
            total_words = self.outcomes.len(),
            token_count = self.tokens_reported,
            "{headline}"
        );
    }
}

/// One word's labels and how completely they accounted for it.
///
/// There used to be a `Route { Stitch, DpRemap(CharEdits) }` beside the run,
/// naming the code path. `Stitch` was `DpRemap(CharEdits::ZERO)` under another
/// name, so the same word could be described two ways and only one of the two
/// carried a number; the count is kept and the route is not.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Attribution {
    pub(crate) run: LabelRun,
    /// [`CharEdits::ZERO`] for the in-order stitch, which reconciles every
    /// character by construction, and for a residue fold that did the same.
    pub(crate) edits: CharEdits,
}

/// The words the in-order stitch identified, and where it stopped.
struct StitchPrefix {
    /// One run per word, covering words `0..runs.len()` and no others.
    ///
    /// A prefix rather than a sparse map because the stitch is strictly
    /// sequential: it identifies word `n` only after word `n - 1`, so the words
    /// it covers can only ever be an initial run.
    runs: Vec<LabelRun>,
    /// The first label the prefix did not consume.
    ///
    /// Labels a FAILED word partly consumed are NOT here. That is the fix: they
    /// go back into the residue for the DP, instead of being eaten by a word
    /// that was never identified.
    next_label: LabelSlot,
}

/// Map an onset-only engine's labels onto the transcript's words.
///
/// The seam. Takes what the engine said and what the transcript says, and
/// answers with one [`WordTimingOutcome`] per word; nothing about the
/// attribution is left for a caller to re-derive.
pub(crate) fn map_labels_to_words(
    original: &[FaWord],
    tokens: &[FaRawToken],
    window: &FaWindow,
    engine: &EngineId,
) -> LabelMapping {
    let mut discarded = DiscardedTimings::default();

    // A word with no alphanumeric content refuses the whole group, as it
    // always did, rather than producing an attribution nobody can check.
    let words = match WordTrack::of_words(original) {
        Ok(words) => words,
        Err(reason) => {
            return LabelMapping::all_untimed(original.len(), reason, discarded, tokens.len());
        }
    };

    let labels = LabelTrack::from_engine_tokens(tokens, window, engine, &mut discarded);
    if labels.is_empty() {
        return LabelMapping::all_untimed(
            original.len(),
            UntimedReason::NoUsableLabels,
            discarded,
            tokens.len(),
        );
    }

    let mut attributions: Vec<Option<Attribution>> = vec![None; original.len()];
    let prefix = stitch_in_order(&words, &labels);
    for (idx, run) in prefix.runs.iter().enumerate() {
        attributions[idx] = Some(Attribution {
            run: *run,
            // The stitch only accepts a run of labels that spells the word
            // exactly, so there is nothing unreconciled to count.
            edits: CharEdits::ZERO,
        });
    }

    // The residue: every word the stitch did not identify, against every label
    // it did not consume. Either being empty means there is nothing to remap,
    // which is also the fast path's guarantee that it is not disturbed.
    let residue_words = WordSlot::new(prefix.runs.len());
    // Why an unplaced word has no span, which is NOT always the same fact. It
    // is `NoLabelSpan` when the DP ran and could not place this particular
    // word; it is the refusal's own reason when the DP declined to run at all,
    // and reporting the first for the second would send an operator hunting a
    // tokenization problem that never happened.
    let mut unplaced = UntimedReason::NoLabelSpan;
    if residue_words.0 < words.len()
        && prefix.next_label.0 < labels.len()
        && let Err(refusal) = remap_residue(
            &words,
            &labels,
            residue_words,
            prefix.next_label,
            &mut attributions,
        )
    {
        // The stitched prefix is untouched: it proved its own words, and a
        // refusal downstream of it is no reason to drop timings that hold.
        attributions[residue_words.0..].fill(None);
        unplaced = refusal;
    }

    let recording = window.recording();
    let outcomes = attributions
        .into_iter()
        .map(|attribution| {
            lower_to_outcome(attribution, &unplaced, &labels, &recording, &mut discarded)
        })
        .collect();

    LabelMapping {
        outcomes,
        discarded,
        tokens_reported: tokens.len(),
    }
}

/// Walk labels and words together, concatenating whole labels into whole words.
///
/// Stops at the first word it cannot spell. Everything from there is the
/// residue; this function deliberately knows nothing about what happens to it.
fn stitch_in_order(words: &WordTrack, labels: &LabelTrack) -> StitchPrefix {
    let mut runs = Vec::with_capacity(words.len());
    let mut next = 0usize;

    for slot in words.slots_from(WordSlot::new(0)) {
        let word_norm = words.norm(slot);
        if next >= labels.len() {
            break;
        }
        // How many BYTES of `word_norm` the labels taken so far spell. The
        // concatenation of those labels is exactly `word_norm[..matched]`, so
        // a cursor says everything an accumulated `String` said and allocates
        // nothing. This loop runs once per label of every group, and it used
        // to clone the accumulator on every iteration, so the old shape cost
        // one allocation per label EXAMINED, including the ones it rejected.
        let mut matched = 0usize;
        let mut cursor = next;
        let mut spelled = false;
        while cursor < labels.len() {
            // `matched` is the length of a prefix already verified against
            // `word_norm`, so it is a char boundary and this is always `Some`.
            // A `break` rather than a panic, because refusing to guess is this
            // module's job.
            let Some(rest) = word_norm.get(matched..) else {
                break;
            };
            let label = labels.norm(LabelSlot(cursor));
            // Only a PREFIX of the word may be extended: a label that overshoots
            // is a disagreement, not a longer word.
            if !rest.starts_with(label) {
                break;
            }
            matched += label.len();
            cursor += 1;
            if matched == word_norm.len() {
                spelled = true;
                break;
            }
        }
        if !spelled {
            break;
        }
        let Some(run) = LabelRun::through(LabelSlot(next), LabelSlot(cursor - 1)) else {
            // `cursor > next` whenever a word was spelled, since spelling it
            // consumed at least one label. Unreachable, and a `break` rather
            // than a panic, because refusing to guess is this module's job.
            break;
        };
        runs.push(run);
        next = cursor;
    }

    StitchPrefix {
        runs,
        next_label: LabelSlot(next),
    }
}

/// Turn one word's attribution into the outcome that leaves this module.
fn lower_to_outcome(
    attribution: Option<Attribution>,
    unplaced: &UntimedReason,
    labels: &LabelTrack,
    recording: &Recording,
    discarded: &mut DiscardedTimings,
) -> WordTimingOutcome {
    let Some(attribution) = attribution else {
        return WordTimingOutcome::Untimed {
            reason: unplaced.clone(),
        };
    };
    match labels.span_for(attribution.run, recording) {
        Ok(outcome) => {
            discarded.note_clamp(&outcome);
            let span = outcome.value();
            // Both ends carry their own provenance across, which is the whole
            // point on this path: an onset-only engine MEASURED the start and
            // only the end is inferred, so collapsing the two would report a
            // half-observed word as wholly derived.
            // THE one construction site for a DP-remapped word's provenance,
            // and it is here rather than at the module edge so that nothing
            // downstream has to rebuild a `WordTiming` to add it, and so the
            // outcome and the number cannot disagree about the route.
            //
            // It wraps rather than replaces: each instant is still whatever the
            // label track made it, and what we added is the claim that this
            // word is the one that RUN of labels belongs to.
            //
            // BOTH ends are wrapped, because that claim settles both. The run
            // the DP handed this word decides which label starts it and which
            // label ends it, so an end reported as a bare `DerivedFromNextOnset`
            // would present the second half of one guess as an inference from a
            // neighbour nobody disputed. Only the start was wrapped until
            // 2026-09-07.
            //
            // Both ends are taken BY VALUE. The closure used to borrow each
            // origin and clone it, so the exact route (which is most words)
            // paid two clones of a boxed provenance chain to produce the same
            // value it was handed.
            let (start, end) = span.into_parts();
            let (start_at, start_origin) = start.into_parts();
            let (end_at, end_origin) = end.into_parts();
            let attributed = |origin: Origin| match attribution.edits.is_exact() {
                true => origin,
                false => Origin::AttributedByCharAlignment {
                    was: Box::new(origin),
                    edits: attribution.edits,
                },
            };
            let timing = WordTiming::new(
                start_at.get(),
                end_at.get(),
                attributed(start_origin),
                attributed(end_origin),
            );
            match timing {
                Some(timing) => WordTimingOutcome::Timed {
                    timing,
                    edits: attribution.edits,
                },
                None => WordTimingOutcome::Untimed {
                    reason: UntimedReason::TimingRefusedProvenSpan,
                },
            }
        }
        // A zero-width or inverted span is refused, never nudged: a word cannot
        // start and end at the same instant, so an invented millisecond would
        // turn the engine's admission that it found nothing into a measurement.
        // Counted from the outcome, not beside it: the reason IS the word's
        // outcome, and recording it a second time on `discarded` was the same
        // fact in two representations (and the only reason for the clone).
        Err(fault) => WordTimingOutcome::Untimed {
            reason: UntimedReason::SpanRefused(fault),
        },
    }
}
