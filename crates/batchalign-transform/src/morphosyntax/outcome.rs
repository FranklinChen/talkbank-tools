//! Per-utterance morphotag outcome: the typed statement of what
//! happened on each utterance the pipeline visited
//! ([`MorOutcome`] / [`MorOutcomeKind`]), the structured "no Mor
//! content" reason ([`NotApplicableReason`]), the classifier that picks
//! that reason from CHAT content ([`classify_not_applicable`]), and the
//! adapter that converts an outcome into a [`DecisionRecord`] for tracing
//! and structured evidence.
//!
//! The split exists because these types form a cohesive
//! "outcome reporting" boundary that callers consume independently of
//! the payload-collection or UD-typing layers.

use crate::decisions::LineIdx;
use talkbank_model::alignment::helpers::{TierDomain, WordItem, walk_words};
use talkbank_model::model::{SpeakerCode, Utterance, WordCategory};

use crate::decisions::{DecisionRecord, DecisionStrategy, MorphosyntaxStrategy};
use crate::inject::MisalignmentDiagnostic;

/// One morphotag outcome for one utterance.
///
/// Carries enough information (line index, speaker, kind) to be converted
/// into a [`DecisionRecord`] without further context from the caller.
#[derive(Debug, Clone)]
pub struct MorOutcome {
    /// Index into `ChatFile.lines` identifying the utterance.
    pub line_idx: usize,
    /// Speaker code for the affected utterance.
    pub speaker: SpeakerCode,
    /// What happened on this utterance.
    pub kind: MorOutcomeKind,
}

/// The three possible morphotag outcomes per utterance.
#[derive(Debug, Clone)]
pub enum MorOutcomeKind {
    /// The utterance had zero Mor-alignable words under CHAT policy.
    /// No `%mor`/`%gra` was produced, and that is correct behavior
    /// there is no morphological content to analyze.
    NotApplicable {
        /// Which class of non-linguistic content the utterance held.
        reason: NotApplicableReason,
    },

    /// Stanza returned N tokens for N CHAT words after MWT reassembly;
    /// `%mor`/`%gra` were injected successfully.
    Aligned {
        /// The alignable-word count that was matched on both sides.
        n_words: usize,
    },

    /// The `|stanza_tokens| = |chat_words|` invariant was violated.
    /// Always a bug in extraction, realignment, MWT reassembly, or the
    /// terminator filter. Never silently absorbed.
    MisalignmentBug(MisalignmentDiagnostic),
}

/// Why an utterance had no Mor-alignable content.
///
/// These reasons are mutually exclusive at the point of classification:
/// when an utterance yields zero Mor-alignable words, exactly one of
/// these describes what was there instead. The classifier walks the
/// utterance content once and picks the most specific variant that
/// matches every non-separator word in the utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotApplicableReason {
    /// The utterance body was empty after parsing (no words, no
    /// separators, no annotations that carry content).
    Empty,
    /// Every word in the utterance is a filler (`&-um`, `&-hmm`, …).
    /// `%mor` does not annotate paralinguistic fillers.
    FillerOnly,
    /// Every word in the utterance is a phonological fragment (`&+le`).
    FragmentOnly,
    /// Every word in the utterance is a nonword (`&~ach`, `&~uh`).
    NonwordOnly,
    /// Every word in the utterance is untranscribed (`xxx`, `yyy`, `www`).
    UntranscribedOnly,
    /// Every word in the utterance is an omission (`0is`, or a CA `(word)`).
    OmissionOnly,
    /// The utterance has words, but all of them are inside retrace
    /// groups (`<...> [/]`, `<...> [//]`), which Mor excludes.
    AllRetraced,
    /// The utterance contains a mix of non-linguistic categories
    /// (e.g. fillers + fragments + untranscribed) where no single
    /// narrower reason above fully describes it.
    MixedNonLinguistic,
}

impl NotApplicableReason {
    /// Stable evidence label used after `not_applicable:`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::FillerOnly => "filler_only",
            Self::FragmentOnly => "fragment_only",
            Self::NonwordOnly => "nonword_only",
            Self::UntranscribedOnly => "untranscribed_only",
            Self::OmissionOnly => "omission_only",
            Self::AllRetraced => "all_retraced",
            Self::MixedNonLinguistic => "mixed_nonlinguistic",
        }
    }
}

/// Inspect an utterance whose Mor-domain extraction yielded zero words,
/// and classify the [`NotApplicableReason`] that best describes why.
///
/// The classifier walks the content once, recording which word
/// categories it sees. If every content-bearing element is of one
/// category, that specific variant is returned; otherwise
/// [`NotApplicableReason::MixedNonLinguistic`] is used.
///
/// Callers must only invoke this for utterances where
/// `extract::collect_utterance_content(…, TierDomain::Mor, …)` returned
/// an empty vector; otherwise the classification is meaningless.
#[must_use]
pub fn classify_not_applicable(utterance: &Utterance) -> NotApplicableReason {
    // Two walks: domain-free sees all content (including retrace);
    // Mor-domain sees everything except retrace. If Mor sees nothing
    // but domain-free saw words, retrace is the reason. Otherwise
    // classify by category from the Mor walk. `walk_words` doesn't
    // expose retrace context to the closure, so a single-walk fusion
    // would require a custom walker; the per-call cost is tiny (invoked
    // only on empty-payload utterances, ~3K across a 54-file corpus).
    let mut total = ContentKinds::default();
    walk_words(&utterance.main.content.content, None, &mut |item| {
        total.observe(item)
    });

    let mut mor_only = ContentKinds::default();
    walk_words(
        &utterance.main.content.content,
        Some(TierDomain::Mor),
        &mut |item| mor_only.observe(item),
    );

    match (total.any_word(), mor_only.any_word()) {
        (false, _) => NotApplicableReason::Empty,
        // Domain-free saw words but the Mor walk saw none: all retraced.
        (true, false) => NotApplicableReason::AllRetraced,
        (true, true) => mor_only.reason_when_nothing_alignable(),
    }
}

/// What one transcribed-as-written word is, for this classification.
///
/// Closed and matched exhaustively from `WordCategory`, so a category chatter
/// adds breaks this match instead of being counted as linguistic, which is
/// how omissions were once miscounted (a `_` arm sent them there).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum WordKind {
    /// Ordinary speech: what `%mor` analyzes.
    Linguistic,
    Filler,
    Fragment,
    Nonword,
    Untranscribed,
    Omission,
}

impl WordKind {
    fn of(word: &talkbank_model::model::Word) -> Self {
        if word.untranscribed().is_some() {
            return Self::Untranscribed;
        }
        match &word.category {
            None => Self::Linguistic,
            Some(WordCategory::Filler) => Self::Filler,
            Some(WordCategory::PhonologicalFragment) => Self::Fragment,
            Some(WordCategory::Nonword) => Self::Nonword,
            Some(WordCategory::Omission | WordCategory::CAOmission) => Self::Omission,
        }
    }

    /// The reason an utterance made only of this kind has no `%mor`.
    const fn only_reason(self) -> NotApplicableReason {
        match self {
            // The caller's precondition (nothing Mor-alignable) is violated
            // if a linguistic word is present; answer conservatively.
            Self::Linguistic => NotApplicableReason::MixedNonLinguistic,
            Self::Filler => NotApplicableReason::FillerOnly,
            Self::Fragment => NotApplicableReason::FragmentOnly,
            Self::Nonword => NotApplicableReason::NonwordOnly,
            Self::Untranscribed => NotApplicableReason::UntranscribedOnly,
            Self::Omission => NotApplicableReason::OmissionOnly,
        }
    }
}

/// The distinct word kinds one walk saw.
#[derive(Default, Debug)]
struct ContentKinds(std::collections::BTreeSet<WordKind>);

impl ContentKinds {
    /// Record one leaf. A replaced word is classified by what the
    /// transcriber wrote; separators carry no kind (a tag-marker separator
    /// would have made extraction non-empty, so classify would not run).
    fn observe(&mut self, item: WordItem<'_>) {
        let word = match item {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if !word.cleaned_text().is_empty() {
            self.0.insert(WordKind::of(word));
        }
    }

    fn any_word(&self) -> bool {
        !self.0.is_empty()
    }

    /// The narrowest reason that explains "nothing alignable": one kind
    /// names its own reason; several are mixed.
    fn reason_when_nothing_alignable(&self) -> NotApplicableReason {
        let mut kinds = self.0.iter();
        match (kinds.next(), kinds.next()) {
            (None, _) => NotApplicableReason::Empty,
            (Some(only), None) => only.only_reason(),
            (Some(_), Some(_)) => NotApplicableReason::MixedNonLinguistic,
        }
    }
}

impl MorOutcome {
    /// Convert this outcome into a [`DecisionRecord`] for tracing and
    /// structured evidence.
    ///
    /// [`MorOutcomeKind::Aligned`] outcomes return `None` because the
    /// happy path is not review-worthy and would produce one record per
    /// successfully-morphotagged utterance: noise, not signal.
    /// Callers that want to surface aligned counts should aggregate
    /// separately.
    #[must_use]
    pub fn to_decision_record(&self) -> Option<DecisionRecord> {
        match &self.kind {
            MorOutcomeKind::Aligned { .. } => None,
            MorOutcomeKind::NotApplicable { reason } => Some(DecisionRecord {
                line_idx: LineIdx::new(self.line_idx),
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::NotApplicable),
                reason: format!("reason={}", reason.as_str()),
                // NotApplicable is correct behavior, not a failure,
                // so it does not require review.
                needs_review: false,
            }),
            MorOutcomeKind::MisalignmentBug(diag) => Some(DecisionRecord {
                line_idx: LineIdx::new(self.line_idx),
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::MisalignmentBug),
                reason: format!(
                    "class={} expected={} actual={} chat_words={:?} stanza_tokens={:?}",
                    diag.suspected_class.as_str(),
                    diag.expected,
                    diag.actual,
                    diag.chat_words,
                    diag.stanza_tokens_after_mapping,
                ),
                // Misalignment bugs always want human attention
                // they indicate something the pipeline got wrong.
                needs_review: true,
            }),
        }
    }
}
