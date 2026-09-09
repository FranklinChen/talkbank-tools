//! Response parsing and deterministic alignment for FA results.

use crate::chat_ops::nlp::{FaIndexedTiming, FaRawResponse, FaRawToken};

use super::coordinates::{Clamped, FaWindow, Ms, OutsideWindow, WindowMs};
use super::origin::{CharEdits, EngineId};
use super::timing::{SpanRejections, WordSpan};
use super::{FaWord, ModelAlignmentScore, WordTiming};
use token_map::{UntimedReason, WordTimingOutcome};

pub(crate) mod residue;
pub(crate) mod token_map;

/// Typed error returned by [`parse_fa_response`].
///
/// Wave 5 of the morphotag reconciliation architecture replaced the
/// previous `Result<_, String>` return with this enum so failure modes
/// can be discriminated at the call site without string parsing. The
/// two variants correspond to structurally distinct problems:
///
/// - `JsonParse`: worker returned text that isn't a valid FA response
///   payload. This is a worker-protocol bug.
/// - `IndexedCountMismatch`: worker returned the wrong number of
///   per-word timings (the FA equivalent of morphotag's
///   `MisalignmentBug`). Always a worker-contract bug, the Python FA
///   worker is supposed to emit one `Option<FaIndexedTiming>` per input
///   `FaWord` for the indexed-word-level response shape.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FaAlignmentError {
    /// The worker's JSON response could not be deserialized into
    /// [`FaRawResponse`].
    #[error("failed to parse raw FA response: {message}")]
    JsonParse {
        /// Underlying serde error rendered as a string (preserved
        /// through the `Clone` boundary; `serde_json::Error` itself is
        /// not `Clone`).
        message: String,
    },
    /// The worker returned an indexed-word-level response whose length
    /// disagrees with the number of input words.
    #[error(
        "FA indexed-response length mismatch: expected {expected} timings for \
         {expected} words, got {actual}"
    )]
    IndexedCountMismatch {
        /// Number of words sent to the worker (expected count).
        expected: usize,
        /// Number of timings actually returned.
        actual: usize,
    },
}

/// Parse the JSON response from the FA callback and align it with original words.
///
/// # Why this takes a window rather than a start offset
///
/// An engine reports positions relative to the AUDIO IT WAS GIVEN. Until
/// 2026-08-15 this function took a bare `audio_start_ms: u64`, added it, and
/// checked only that each resulting end exceeded its start. That check is a
/// relation among the engine's own numbers and says nothing about the audio.
///
/// The failure it allowed, measured on a real session: a 2.263 second group
/// (`1257705..1259968`) was rejected by Wave2Vec for exceeding a CTC target
/// limit and retried on Whisper, which pads its input to a fixed 30 second
/// window and duly reported tokens out to 29.98 seconds. Offsetting those gave
/// word timings at 1287685 in a recording 1259968 milliseconds long. Six of 226
/// screened sessions carried up to 28.2 seconds of such phantom speech.
///
/// Taking the window makes containment askable, and [`FaWindow::to_file`] makes
/// it unskippable: it is the only route from window coordinates into file
/// coordinates, so neither response shape can bypass it.
///
/// # Errors
///
/// Returns [`FaAlignmentError::JsonParse`] if the response isn't valid
/// FA JSON, or [`FaAlignmentError::IndexedCountMismatch`] if the
/// indexed-word-level variant returned the wrong count. The second is raised
/// by [`IndexedWordTimings::pair`], which is where the per-word cardinality
/// stops being a check and becomes a value the alignment below is given.
pub fn parse_fa_response(
    json_str: &str,
    original_words: &[FaWord],
    window: &FaWindow,
    engine: &EngineId,
) -> Result<Vec<Option<WordTiming>>, FaAlignmentError> {
    let raw_resp: FaRawResponse =
        serde_json::from_str(json_str).map_err(|e| FaAlignmentError::JsonParse {
            message: e.to_string(),
        })?;

    match raw_resp {
        FaRawResponse::IndexedWordLevel { indexed_timings } => {
            let paired = IndexedWordTimings::pair(original_words, &indexed_timings)?;
            let reported = paired.word_count();
            let (outcomes, discarded) = apply_indexed_timings(paired, window, engine);
            discarded.warn_if_any(&outcomes, reported, engine, window);
            Ok(token_map::lower_outcomes(outcomes))
        }
        FaRawResponse::TokenLevel { tokens } => {
            Ok(align_token_timings(original_words, &tokens, window, engine))
        }
    }
}

/// An indexed FA response paired with the words it answers about, proven to
/// carry exactly one timing slot per word.
///
/// # Why the pairing is a type rather than a check somewhere else
///
/// [`apply_indexed_timings`] answers with one [`WordTimingOutcome`] per SLOT,
/// and the caller reads those positionally against its own word list. A
/// response one slot short therefore does not merely lose the last word: it
/// leaves a shorter vector than there are words, and every warn-line counter
/// is derived from that vector, so the group under-reports by the difference
/// without saying so.
///
/// The equality was checked by an `if` inside [`parse_fa_response`] that
/// returned nothing, while `apply_indexed_timings` went on taking the two
/// slices separately. Its signature could not say that the pair belonged
/// together, its doc comment asserted the per-word promise in prose, and its
/// four in-crate test call sites could hand it any two slices at all. This is
/// that check turned into a value: it is the only route from two loose slices
/// to the paired form, so the mismatch is refused where it is born and cannot
/// be reconstructed downstream.
pub(super) struct IndexedWordTimings<'a> {
    /// The words the outcomes will be read against.
    words: &'a [FaWord],
    /// One slot per word, `None` where the host declined to time it.
    timings: &'a [Option<FaIndexedTiming>],
}

impl<'a> IndexedWordTimings<'a> {
    /// The only constructor, and it is the length check.
    ///
    /// # Errors
    ///
    /// [`FaAlignmentError::IndexedCountMismatch`] when the host returned a
    /// number of slots other than the number of words it was sent. Always a
    /// worker-contract bug.
    pub(super) fn pair(
        words: &'a [FaWord],
        timings: &'a [Option<FaIndexedTiming>],
    ) -> Result<Self, FaAlignmentError> {
        if timings.len() != words.len() {
            return Err(FaAlignmentError::IndexedCountMismatch {
                expected: words.len(),
                actual: timings.len(),
            });
        }
        Ok(Self { words, timings })
    }

    /// How many words this response answers about.
    ///
    /// Also how many timing slots the host returned, because the pairing is
    /// exactly the proof that those two numbers are one number.
    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }
}

/// Apply index-aligned word timings (no DP remapping required).
///
/// Answers with one [`WordTimingOutcome`] per word, exactly as the token path
/// does: "this word has no timing" is one fact and had two models, a typed
/// per-word reason there and a `None` plus a hand-maintained counter here. The
/// counters the warn line prints are now COUNTED FROM these outcomes, so a
/// reason cannot be reported in one place and tallied in another.
///
/// The per-word promise is carried by [`IndexedWordTimings`] and not by this
/// paragraph: the loop walks the timing slots, of which the pairing proves
/// there is one per word.
///
/// What comes back beside them is the residue that is genuinely not a per-word
/// absence: adjustments to words that ARE timed.
pub(super) fn apply_indexed_timings(
    paired: IndexedWordTimings<'_>,
    window: &FaWindow,
    engine: &EngineId,
) -> (Vec<WordTimingOutcome>, DiscardedTimings) {
    let mut outcomes = Vec::with_capacity(paired.words.len());
    let mut discarded = DiscardedTimings::default();
    for maybe_timing in paired.timings.iter() {
        let Some(timing) = maybe_timing else {
            // The HOST returned no timing for this word. A reason on the word,
            // not a counter beside it: downstream it is otherwise
            // indistinguishable from a word whose timing we rejected.
            outcomes.push(WordTimingOutcome::Untimed {
                reason: UntimedReason::NoTimingFromHost,
            });
            continue;
        };
        let model_score = timing.confidence.and_then(|score| {
            ModelAlignmentScore::try_from_f64(score)
                .map_err(|_| discarded.record_invalid_model_score())
                .ok()
        });
        // A word-interval engine reports both ends, so this is the model's own
        // answer, in the coordinates of the audio it was handed. Both ends must
        // land inside that audio: half a span is not a span, so one end outside
        // condemns the pair rather than being repaired from the other.
        outcomes.push(
            match (
                window.to_file(WindowMs::reported(timing.start_ms), engine),
                window.to_file(WindowMs::reported(timing.end_ms), engine),
            ) {
                // Both ends measured, so this is the one constructor that yields a
                // fully observed span. Routed through `WordSpan` rather than
                // straight to `WordTiming` so that both response shapes classify
                // their failures identically: an unusable timing is worse than
                // none, because it reads as a real measurement downstream.
                (Ok(start), Ok(end)) => match WordSpan::measured(start, end) {
                    Ok(span) => {
                        // The span's ends already carry their provenance; the
                        // timing takes the END's, because that is the one a later
                        // pass may replace and the one a consumer asks about.
                        match WordTiming::new(
                            span.start().at().get(),
                            span.end().at().get(),
                            span.start().origin().clone(),
                            span.end().origin().clone(),
                        )
                        .map(|timing| match model_score {
                            Some(score) => timing.with_model_score(score),
                            None => timing,
                        }) {
                            Some(timing) => WordTimingOutcome::Timed {
                                timing,
                                // The engine reported this word's own extent, so
                                // nothing was reconciled and nothing was guessed.
                                edits: CharEdits::ZERO,
                            },
                            None => WordTimingOutcome::Untimed {
                                reason: UntimedReason::TimingRefusedProvenSpan,
                            },
                        }
                    }
                    Err(fault) => WordTimingOutcome::Untimed {
                        reason: UntimedReason::SpanRefused(fault),
                    },
                },
                (Err(fault), _) | (Ok(_), Err(fault)) => WordTimingOutcome::Untimed {
                    reason: UntimedReason::ReportedOutsideWindow(fault),
                },
            },
        );
    }

    (outcomes, discarded)
}

/// The per-word classes of the FA warn line, counted from the outcomes.
///
/// Derived rather than accumulated. Each of these used to be a field on
/// [`DiscardedTimings`] incremented on the line that produced the word's
/// `None`, which is the same fact in two representations: the reason on the
/// word, and a number beside it that nothing tied to the word.
#[derive(Default)]
struct UntimedWords {
    /// The classes this path shares with UTR, so both report them alike.
    rejected: SpanRejections,
    /// Words the host returned no timing for at all.
    no_timing_from_host: usize,
    /// Words whose span was PROVEN positive and whose timing was refused
    /// anyway.
    ///
    /// Its own field because BOTH response shapes produce it: the token path
    /// at `token_map::time_word`, and the indexed path above. It is not an
    /// attribution fact, so the ignored arm below has no business holding it,
    /// and on the indexed path there is no second log line to pick it up: a
    /// word refused this way was reported by nothing at all.
    timing_refused_proven_span: usize,
}

/// Sort one group's untimed words into the warn line's fields.
///
/// Exhaustive over [`UntimedReason`], and that is STRONGER than the running
/// `notable()` total it partly replaces: a new reason cannot be added without
/// deciding here where it is reported, and the decision is a compile error
/// rather than a counter that silently stays zero.
fn untimed_words(outcomes: &[WordTimingOutcome]) -> UntimedWords {
    let mut tally = UntimedWords::default();
    for outcome in outcomes {
        let WordTimingOutcome::Untimed { reason } = outcome else {
            continue;
        };
        match reason {
            UntimedReason::NoTimingFromHost => tally.no_timing_from_host += 1,
            UntimedReason::SpanRefused(fault) => tally.rejected.record_span_fault(fault.clone()),
            UntimedReason::ReportedOutsideWindow(fault) => tally.rejected.record_outside(*fault),
            UntimedReason::TimingRefusedProvenSpan => tally.timing_refused_proven_span += 1,
            // Reasons only the TOKEN path produces, and only about
            // ATTRIBUTION rather than about a timing: no lexical content to
            // match, no label to attribute, a residue refused as a whole. This
            // line has no field for them and must not invent one, because the
            // token path is the one path that also emits
            // `LabelMapping::report`, which counts them as `untimed` and can
            // additionally say WHY.
            UntimedReason::NoLexicalContent
            | UntimedReason::NoUsableLabels
            | UntimedReason::NoLabelSpan
            | UntimedReason::ResidueTooLongToAlign { .. }
            | UntimedReason::NonMonotoneAttribution => {}
        }
    }
    tally
}

/// What an FA group's warn line is ABOUT.
///
/// A named subject rather than a `&str` chosen inline, because the headline
/// was chosen from two of the facts and printed beside all six. A group whose
/// only notable facts were a clamped assumption or a bad model score fell
/// through to the first arm and announced that the engine had returned no
/// timing for some words, on a group in which every word was timed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaWarnSubject {
    /// The engine reported about audio it was never given. The class that used
    /// to corrupt output rather than merely lose a timing, so it leads.
    OutsideWindow,
    /// Words lost a timing because the answer could not be used.
    UnusableTimings,
    /// The host simply declined to time some words.
    HostDeclined,
    /// Every word is timed. Only already-timed words were adjusted.
    AdjustmentsOnly,
}

impl FaWarnSubject {
    /// The sentence this subject leads with.
    const fn headline(self) -> &'static str {
        match self {
            Self::OutsideWindow => {
                "engine reported word timings past the end of the audio it was given; \
                 those words are left unaligned rather than placed outside the recording"
            }
            Self::UnusableTimings => {
                "some FA word timings were unusable and those words are left unaligned"
            }
            // Saying "unusable" here would blame us for a decision the engine
            // made.
            Self::HostDeclined => {
                "the engine returned no timing for some words; they are left unaligned"
            }
            // Every word is timed, so no sentence about missing timings is
            // true. This arm did not exist and its cases printed one anyway.
            Self::AdjustmentsOnly => {
                "every word was timed, but some timings were adjusted before use"
            }
        }
    }
}

/// One group's warn line, tallied once.
///
/// Both halves in one value: the per-word absences counted from the outcomes,
/// and the residue counted on [`DiscardedTimings`]. It exists so the line is
/// built from ONE walk of the outcomes rather than three (the gate, the
/// headline, and the fields each did their own), and so the subject is chosen
/// from exactly the facts the fields print.
///
/// `pub(super)` because it is what a test asks about the group: the same value
/// the warn line is rendered from, rather than a second accessor that could
/// answer differently.
pub(super) struct FaGroupWarning {
    /// Per-word absences.
    untimed: UntimedWords,
    /// Engine LABELS placed outside the window, which are not word absences.
    labels_rejected: SpanRejections,
    /// Assumed word ends cut back to the recording.
    assumed_then_clamped: usize,
    /// Model scores removed as non-finite or out of range.
    invalid_model_score: usize,
}

impl FaGroupWarning {
    /// Out-of-window reports at either level.
    ///
    /// Word-level and label-level share one printed field, as they always
    /// have. The sum is the honest count whichever path ran: each path
    /// populates one of the two halves and leaves the other at zero, so the
    /// number is the same one either half would have printed alone.
    fn outside_window(&self) -> usize {
        self.untimed.rejected.outside_window + self.labels_rejected.outside_window
    }

    /// Whether anything was reported about audio the engine was not given.
    fn any_outside_window(&self) -> bool {
        self.untimed.rejected.any_outside_window() || self.labels_rejected.any_outside_window()
    }

    /// How far the worst out-of-window report exceeded its audio.
    fn worst_overshoot(&self) -> Ms {
        self.untimed
            .rejected
            .worst_overshoot
            .max(self.labels_rejected.worst_overshoot)
    }

    /// Timings that existed and could not be used, at either level.
    fn unusable(&self) -> usize {
        self.untimed.rejected.total()
            + self.labels_rejected.total()
            + self.untimed.timing_refused_proven_span
    }

    /// How many facts in this group are worth an operator's attention.
    ///
    /// One total still decides whether anything is said, so the gate cannot
    /// enumerate the ways "nothing happened" can be true and miss one.
    pub(super) fn notable(&self) -> usize {
        self.unusable() + self.untimed.no_timing_from_host + self.adjustments()
    }

    /// Facts about words that ARE timed.
    fn adjustments(&self) -> usize {
        self.assumed_then_clamped + self.invalid_model_score
    }

    /// Which fact leads the line, chosen from the facts PRESENT.
    ///
    /// Exhaustive over the cross product of the three absence kinds, with no
    /// catch-all: the last arm is reached only when nothing is absent, and
    /// [`warn_if_any`] has already established that something is notable, so
    /// the only thing left is an adjustment.
    ///
    /// [`warn_if_any`]: DiscardedTimings::warn_if_any
    fn subject(&self) -> FaWarnSubject {
        match (
            self.any_outside_window(),
            self.unusable() > 0,
            self.untimed.no_timing_from_host > 0,
        ) {
            (true, _, _) => FaWarnSubject::OutsideWindow,
            (false, true, _) => FaWarnSubject::UnusableTimings,
            (false, false, true) => FaWarnSubject::HostDeclined,
            (false, false, false) => FaWarnSubject::AdjustmentsOnly,
        }
    }
}

/// What the engine said that no word could keep.
///
/// # What is here and what is NOT
///
/// Everything on this struct is a fact that is not a per-word absence, which
/// is the partition the previous version did not make. It held four hand-
/// maintained counters, each with its own `record_*`, its own accessor and its
/// own line in a format string, and they were three different kinds of fact:
///
/// - **Absence**, per word: the host declined, the span had no width, the
///   report fell outside the window, a proven span was still refused. Those
///   are now [`UntimedReason`]s on the word itself and are counted by
///   [`untimed_words`].
/// - **Adjustment**, about a word that IS timed: an assumed end cut back to
///   the recording, a model score outside its promised range. The word keeps
///   its timing, so there is no absence to carry the fact and it stays a
///   counter here.
/// - **Neither**, because it is not about a word at all: a LABEL the token
///   path placed outside the window and threw away before any word was
///   attributed. It cannot be derived from the outcomes, so it stays a
///   counter too, and it is named as label-level rather than sharing a field
///   with the word-level one.
#[derive(Default)]
pub(crate) struct DiscardedTimings {
    /// Engine LABELS placed outside the audio the engine was handed, dropped
    /// before any word was attributed.
    ///
    /// Token path only, and NOT a word count: a word that lost its labels this
    /// way reports [`UntimedReason::NoLabelSpan`] separately. It shares the
    /// warn line's `outside_window` field with the word-level class; see
    /// [`FaGroupWarning::outside_window`] for why the sum is right either way.
    labels_rejected: SpanRejections,
    /// Assumed word ends that had to be cut back to the recording.
    ///
    /// Not an absence: the word keeps a timing. It is reported alongside them
    /// because it is the same KIND of fact, a place where the output is not
    /// what the engine said, and an operator reading one line wants all of
    /// them together.
    assumed_then_clamped: usize,
    /// Model scores that were non-finite or outside the promised 0..=1 range.
    ///
    /// Also not an absence: the interval remains usable, and corrupt optional
    /// metadata is removed rather than laundered or allowed to condemn an
    /// independent measurement.
    invalid_model_score: usize,
}

impl DiscardedTimings {
    /// A LABEL the engine placed outside the audio it was handed.
    ///
    /// The one surviving `record_*` for a rejection, because it is the one
    /// rejection that is not about a word.
    pub(crate) fn record_outside(&mut self, fault: OutsideWindow) {
        self.labels_rejected.record_outside(fault);
    }

    /// Record whether a span's end had to be cut back to the recording.
    ///
    /// Takes the [`Clamped`] outcome itself rather than an `Option<Ms>` teased
    /// out of it. Both are two-case values, but only one of them says what the
    /// cases MEAN: `AsGiven` and `ClampedTo` name the fact, while `Some`/`None`
    /// makes the caller remember which way round the question was asked.
    pub(crate) fn note_clamp(&mut self, outcome: &Clamped<WordSpan>) {
        match outcome {
            Clamped::AsGiven(_) => {}
            Clamped::ClampedTo { .. } => self.assumed_then_clamped += 1,
        }
    }

    pub(crate) fn record_invalid_model_score(&mut self) {
        self.invalid_model_score += 1;
    }

    /// Both halves of this group's warn line in one value.
    ///
    /// The single walk of the outcomes. Every question the line asks is a
    /// method on what this returns, so the headline cannot be decided from
    /// facts the fields do not print, and the outcomes are not re-tallied per
    /// question.
    pub(super) fn tally(&self, outcomes: &[WordTimingOutcome]) -> FaGroupWarning {
        FaGroupWarning {
            untimed: untimed_words(outcomes),
            labels_rejected: self.labels_rejected,
            assumed_then_clamped: self.assumed_then_clamped,
            invalid_model_score: self.invalid_model_score,
        }
    }

    pub(crate) fn warn_if_any(
        &self,
        outcomes: &[WordTimingOutcome],
        total: usize,
        engine: &EngineId,
        window: &FaWindow,
    ) {
        // One total decides whether anything is worth saying; the fields then
        // say what. Enumerating the ways "nothing happened" can be true is how
        // a new counter gets forgotten.
        let warning = self.tally(outcomes);
        if warning.notable() == 0 {
            return;
        }
        // Field names come from `SpanRejections` so this line and UTR's can be
        // aggregated together; only `assumed_then_clamped` is this path's own.
        tracing::warn!(
            no_extent = warning.untimed.rejected.no_extent,
            inverted = warning.untimed.rejected.inverted,
            outside_window = warning.outside_window(),
            worst_overshoot_ms = warning.worst_overshoot().0,
            assumed_then_clamped = warning.assumed_then_clamped,
            invalid_model_score = warning.invalid_model_score,
            untimed_by_host = warning.untimed.no_timing_from_host,
            timing_refused_proven_span = warning.untimed.timing_refused_proven_span,
            window_len_ms = window.len().0,
            total,
            %engine,
            "{}",
            warning.subject().headline()
        );
    }
}

/// Align token-level onset times (typical for Whisper) with original CHAT words.
///
/// A thin lowering over [`token_map::map_labels_to_words`], which owns the
/// attribution and answers with a route per word. Everything interesting is
/// there; this function exists because the FA pipeline still speaks
/// `Vec<Option<WordTiming>>`, and it is the single place that discards what the
/// route said.
///
/// # Two routes, not one
///
/// Until 2026-09-07 this path was deterministic ONLY: it stitched normalized
/// labels onto normalized words in order and, on the first disagreement, left
/// every remaining word `None`. A word the engine had measured perfectly well
/// lost its onset because an earlier word had been tokenized differently. The
/// residue of a broken stitch now goes through a character-level Hirschberg
/// remap; the exact in-order stitch is unchanged and still runs first.
fn align_token_timings(
    original: &[FaWord],
    tokens: &[FaRawToken],
    window: &FaWindow,
    engine: &EngineId,
) -> Vec<Option<WordTiming>> {
    let mapping = token_map::map_labels_to_words(original, tokens, window, engine);
    mapping.report(window, engine);
    mapping.into_timings()
}

#[cfg(test)]
mod warn_line_tests {
    use super::*;
    use crate::chat_ops::fa::coordinates::{FileMs, Recording};

    fn engine() -> EngineId {
        EngineId::new("test-fa")
    }

    /// A one-second span inside a ten-second recording, for a clamp outcome
    /// that needs a real value to carry.
    fn a_span() -> WordSpan {
        let recording = Recording::of_duration(Ms(10_000)).expect("a non-zero duration");
        let window = FaWindow::within(&recording, FileMs::new(0), recording.duration())
            .expect("the whole recording is a window over itself");
        let start = window
            .to_file(WindowMs::reported(1_000), &engine())
            .expect("inside the window");
        let end = window
            .to_file(WindowMs::reported(2_000), &engine())
            .expect("inside the window");
        WordSpan::measured(start, end).expect("ordered and non-empty")
    }

    /// A group whose only notable facts are ADJUSTMENTS says so.
    ///
    /// The headline used to be chosen from two of the six facts, so a group in
    /// which every word was timed and one end had been cut back to the
    /// recording announced that "the engine returned no timing for some words;
    /// they are left unaligned". Both adjustment kinds are exercised, because
    /// either alone reaches the same wrong arm.
    #[test]
    fn a_group_whose_only_facts_are_adjustments_does_not_claim_missing_timings() {
        let mut clamped = DiscardedTimings::default();
        clamped.note_clamp(&Clamped::ClampedTo { bound: a_span() });
        assert_eq!(
            clamped.tally(&[]).subject(),
            FaWarnSubject::AdjustmentsOnly,
            "a clamped assumption is not a missing timing"
        );

        let mut bad_score = DiscardedTimings::default();
        bad_score.record_invalid_model_score();
        assert_eq!(
            bad_score.tally(&[]).subject(),
            FaWarnSubject::AdjustmentsOnly,
            "a removed model score is not a missing timing"
        );
    }

    /// A refused proven span is REPORTED, on the path that has no second line.
    ///
    /// `TimingRefusedProvenSpan` is produced by both response shapes, and the
    /// tally ignored it as a token-path attribution fact. On the indexed path
    /// there is no `LabelMapping::report` to pick it up, so such a word
    /// appeared in no field and in no log line at all: the group could be
    /// entirely silent while a word lost its timing.
    #[test]
    fn a_refused_proven_span_is_counted_and_leads_the_line() {
        let outcomes = vec![WordTimingOutcome::Untimed {
            reason: UntimedReason::TimingRefusedProvenSpan,
        }];
        let discarded = DiscardedTimings::default();
        assert_eq!(
            discarded.tally(&outcomes).notable(),
            1,
            "a word refused this way is worth an operator's attention"
        );
        assert_eq!(
            discarded.tally(&outcomes).subject(),
            FaWarnSubject::UnusableTimings
        );
    }

    /// The three absence kinds keep their own precedence.
    ///
    /// Out-of-window leads whatever else happened, because it is the class that
    /// used to place words outside the recording rather than merely lose them.
    #[test]
    fn an_out_of_window_report_leads_over_every_other_fact() {
        let outcomes = vec![
            WordTimingOutcome::Untimed {
                reason: UntimedReason::ReportedOutsideWindow(OutsideWindow::BeyondWindowEnd {
                    reported: WindowMs::reported(30_000),
                    window_len: Ms(10_000),
                    exceeds_by: Ms(20_000),
                }),
            },
            WordTimingOutcome::Untimed {
                reason: UntimedReason::NoTimingFromHost,
            },
        ];
        let mut discarded = DiscardedTimings::default();
        discarded.record_invalid_model_score();
        assert_eq!(
            discarded.tally(&outcomes).subject(),
            FaWarnSubject::OutsideWindow
        );

        let host_only = vec![WordTimingOutcome::Untimed {
            reason: UntimedReason::NoTimingFromHost,
        }];
        assert_eq!(
            DiscardedTimings::default().tally(&host_only).subject(),
            FaWarnSubject::HostDeclined
        );
    }
}
