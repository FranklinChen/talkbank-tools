//! The life of a word's timing, as a sequence of types.
//!
//! # The thesis
//!
//! Every time this pipeline learns something new about a number, or decides
//! something on the number's behalf, the number becomes a DIFFERENT VALUE OF A
//! DIFFERENT TYPE. Not a mutated integer, not an integer with a comment: a new
//! type, produced by a named transition, carrying the story of how it got
//! there.
//!
//! The alternative is what this module replaces. A word's end used to be
//! computed like this:
//!
//! ```ignore
//! let end_ms = token_starts_ms
//!     .get(token_idx)
//!     .copied()
//!     .unwrap_or(start_ms + LAST_WORD_FALLBACK_MS)
//!     .max(start_ms + 1);
//! ```
//!
//! Three separate decisions are taken in that expression and none of them
//! leaves a trace. The `unwrap_or` invents 500 milliseconds when no successor
//! exists. The `.max` nudges a span that had no width. A `.min` elsewhere cuts
//! a value down to fit the recording. Downstream, all four possible histories,
//! measured, inherited from a neighbour, invented, nudged, are the same `u64`,
//! and a reader of the resulting CHAT file cannot tell them apart. Neither can
//! a program. That is how a corpus comes to be 37 percent interpolated with
//! nothing marking which 37 percent.
//!
//! # The vocabulary
//!
//! * [`WordSpan`] is a word's extent. It cannot be built from integers: both
//!   ends must be [`RecordedInstant`]s, which only a [`Recording`] issues, and
//!   which each carry an [`Origin`]. Provenance is therefore not something a
//!   caller remembers to attach; it is the only way to construct the value.
//! * Every way of obtaining an end is a NAMED CONSTRUCTOR
//!   ([`WordSpan::measured`], [`WordSpan::end_from_next_onset`],
//!   [`WordSpan::end_assumed`]), so choosing one is a decision written in the
//!   source rather than a fallback buried in an expression.
//! * Every span that cannot exist is a [`SpanFault`], not a nudge. A word that
//!   starts and ends at the same instant is not a word with a one millisecond
//!   duration; it is an engine that failed, and saying so is more useful than
//!   papering over it.
//!
//! # What a reader gains
//!
//! `WordSpan::end_assumed(start, LAST_WORD_FALLBACK_MS, &recording)` says, in
//! its own name, that the end of this word was invented and by how much. No
//! comment is required, no `if` has to be traced, and the type of the result
//! ([`Clamped`]) forces the caller to acknowledge that the invention may have
//! been cut down to fit. That is the whole point: the logic lives in the types,
//! where it can be read, rather than in control flow, where it must be
//! reconstructed.

use super::coordinates::{Clamped, FileMs, Ms, OutsideWindow, RecordedInstant, Recording};
use super::origin::Origin;

/// Why a proposed word span cannot exist.
///
/// A sum rather than a silent correction. Each variant is a real thing that
/// happened to a real engine response, and a caller that wants to count them,
/// or refuse the file, or leave the word unaligned, can tell them apart.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpanFault {
    /// The end precedes the start.
    ///
    /// Never repaired by swapping: an inverted span means the engine's two
    /// answers disagree, and choosing an order for it fabricates a measurement
    /// out of a contradiction.
    #[error("word ends at {end} before it starts at {start}")]
    Inverted {
        /// Proposed start.
        start: FileMs,
        /// Proposed end.
        end: FileMs,
    },

    /// The span has no width.
    ///
    /// Deliberately NOT nudged to one millisecond. A word cannot start and end
    /// at the same instant, so a zero-width answer is an engine reporting that
    /// it found nothing, and an invented millisecond would turn that admission
    /// into a measurement. Whisper's DTW timestamps land here for single-frame
    /// backchannels; the honest result is no timing.
    #[error("word has no extent at {at}")]
    NoExtent {
        /// The instant both ends landed on.
        at: FileMs,
    },
}

/// A word's extent within a recording, with the provenance of both ends.
///
/// # Why it cannot be built from integers
///
/// The constructors take [`RecordedInstant`]s, which only a [`Recording`]
/// issues and which each carry their [`Origin`]. There is deliberately no
/// `WordSpan::new(start_ms: u64, end_ms: u64)`: such a function would let any
/// caller assert both containment and provenance from raw parts, which is the
/// difference between a proof and a label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSpan {
    start: RecordedInstant,
    end: RecordedInstant,
}

impl WordSpan {
    /// The private gate every constructor passes through.
    ///
    /// Ordering and width are checked once, here, so no public constructor can
    /// forget and no caller can re-implement the check. Both faults are
    /// returned rather than corrected, because correcting either one invents a
    /// number.
    fn between(start: RecordedInstant, end: RecordedInstant) -> Result<Self, SpanFault> {
        match end.at().cmp(&start.at()) {
            std::cmp::Ordering::Greater => Ok(Self { start, end }),
            std::cmp::Ordering::Equal => Err(SpanFault::NoExtent { at: start.at() }),
            std::cmp::Ordering::Less => Err(SpanFault::Inverted {
                start: start.at(),
                end: end.at(),
            }),
        }
    }

    /// Both ends measured by the engine.
    ///
    /// The strongest span available: a word-interval engine reported where the
    /// word began and where it ended, and both survived containment.
    pub fn measured(start: RecordedInstant, end: RecordedInstant) -> Result<Self, SpanFault> {
        Self::between(start, end)
    }

    /// The end taken from the NEXT word's onset.
    ///
    /// Onset-only engines (Whisper) report when a word starts and never when it
    /// stops, so the only end available is the following word's beginning. That
    /// is a reasonable inference and it is not a measurement: the speaker may
    /// have finished long before the next word began, and a pause is
    /// indistinguishable from a drawl in the result.
    ///
    /// Naming it makes the difference legible in the output. The end's origin
    /// becomes [`Origin::DerivedFromNextOnset`], wrapping nothing, because the
    /// next onset's own measurement belongs to the next word rather than this
    /// one.
    pub fn end_from_next_onset(
        start: RecordedInstant,
        next_onset: &RecordedInstant,
    ) -> Result<Self, SpanFault> {
        let end = next_onset.relabelled(Origin::DerivedFromNextOnset);
        Self::between(start, end)
    }

    /// No successor exists, so a duration is ASSUMED.
    ///
    /// The last word of a group has no following onset to borrow from. This is
    /// the weakest thing the pipeline produces and it used to be the least
    /// visible: `unwrap_or(start + 500)` wrote an invented number in the same
    /// shape as a measured one.
    ///
    /// Returns [`Clamped`] because the assumption can run past the end of the
    /// recording, and a caller must acknowledge that: an assumed duration must
    /// not be permitted to do what the engine itself was just forbidden from
    /// doing. Both the clamping and the assumption appear in the resulting
    /// end's [`Origin`].
    pub fn end_assumed(
        start: RecordedInstant,
        assumed: Ms,
        recording: &Recording,
    ) -> Result<Clamped<Self>, SpanFault> {
        let proposed = FileMs::new(start.at().get() + assumed.0);
        let clamped_end = recording.clamp(proposed, Origin::FallbackDuration { assumed });
        match clamped_end {
            Clamped::AsGiven(end) => Self::between(start, end).map(Clamped::AsGiven),
            // The overshoot rides on `bound`'s own origin, so forwarding the
            // span is all this arm has to do.
            Clamped::ClampedTo { bound } => {
                Self::between(start, bound).map(|span| Clamped::ClampedTo { bound: span })
            }
        }
    }

    /// Where the word begins.
    pub const fn start(&self) -> &RecordedInstant {
        &self.start
    }

    /// Where the word ends.
    pub const fn end(&self) -> &RecordedInstant {
        &self.end
    }

    /// Take both ends, consuming the span.
    ///
    /// For the caller that OWNS its span and is building the value that
    /// replaces it: `alignment::token_map`'s lowering needs both origins by
    /// value, and reaching them through the borrowing accessors above meant
    /// cloning each one back out, twice per word, on every aligned group. An
    /// `Origin` is a boxed chain rather than a scalar, so that is not a free
    /// copy.
    ///
    /// The borrowing accessors stay for readers that keep the span; this adds
    /// no way to BUILD one, so the constructor gate above is untouched.
    pub fn into_parts(self) -> (RecordedInstant, RecordedInstant) {
        (self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::coordinates::{Containment, FaWindow, WindowMs};
    use crate::chat_ops::fa::origin::EngineId;

    fn recording() -> Recording {
        Recording::of_duration(Ms(10_000)).expect("non-zero")
    }

    fn engine() -> EngineId {
        EngineId::new("whisper-fa-large-v2")
    }

    /// An instant the engine measured, at `at` milliseconds into the file.
    fn measured_at(recording: &Recording, at: u64) -> RecordedInstant {
        let window = FaWindow::within(recording, FileMs::new(0), recording.duration())
            .expect("the whole recording is a window over itself");
        window
            .to_file(WindowMs::reported(at), &engine())
            .expect("inside the window")
    }

    #[test]
    fn a_measured_span_is_fully_observed() {
        let rec = recording();
        let span = WordSpan::measured(measured_at(&rec, 1_000), measured_at(&rec, 1_500))
            .expect("ordered and non-empty");
        assert_eq!(span.end().at().since(span.start().at()), Ms(500));
        assert!(span.start().origin().is_observation());
        assert!(span.end().origin().is_observation());
    }

    #[test]
    fn an_end_borrowed_from_the_next_onset_is_not_an_observation() {
        // The distinction the old `u64` could not express: the START is a real
        // measurement and the END is an inference, so the span as a whole is
        // not an observation of the word's extent.
        let rec = recording();
        let span =
            WordSpan::end_from_next_onset(measured_at(&rec, 1_000), &measured_at(&rec, 1_800))
                .expect("ordered");
        assert!(span.start().origin().is_observation());
        assert!(!span.end().origin().is_observation());
        assert_eq!(span.end().origin(), &Origin::DerivedFromNextOnset);
    }

    #[test]
    fn an_assumed_end_says_it_was_assumed_and_how_long_for() {
        let rec = recording();
        let Ok(Clamped::AsGiven(span)) =
            WordSpan::end_assumed(measured_at(&rec, 1_000), Ms(500), &rec)
        else {
            panic!("500ms after 1000ms fits inside a 10000ms recording")
        };
        assert_eq!(span.end().at().since(span.start().at()), Ms(500));
        assert_eq!(
            span.end().origin(),
            &Origin::FallbackDuration { assumed: Ms(500) }
        );
        assert!(!span.end().origin().is_observation());
    }

    #[test]
    fn an_assumption_that_runs_past_the_recording_is_clamped_and_says_so() {
        // The case `unwrap_or(start + 500).min(recording_end)` handled in
        // silence. Both facts survive into the type: it was assumed, and it was
        // then cut down.
        let rec = recording();
        let Ok(Clamped::ClampedTo { bound }) =
            WordSpan::end_assumed(measured_at(&rec, 9_800), Ms(500), &rec)
        else {
            panic!("9800 + 500 overshoots a 10000ms recording")
        };
        assert_eq!(bound.end().at(), FileMs::new(10_000));
        // How far it overshot is on the value, not beside it.
        assert!(matches!(
            bound.end().origin(),
            Origin::ClampedTo {
                overshoot: Ms(300),
                ..
            }
        ));
        assert_eq!(
            bound.end().origin().underlying(),
            &Origin::FallbackDuration { assumed: Ms(500) }
        );
    }

    #[test]
    fn a_zero_width_span_is_refused_rather_than_nudged() {
        // `.max(start_ms + 1)` used to turn this into a one millisecond word.
        // A word cannot start and end at the same instant, so the honest answer
        // is that the engine found nothing here.
        let rec = recording();
        assert_eq!(
            WordSpan::measured(measured_at(&rec, 2_000), measured_at(&rec, 2_000)),
            Err(SpanFault::NoExtent {
                at: FileMs::new(2_000)
            })
        );
    }

    #[test]
    fn an_inverted_span_is_refused_rather_than_swapped() {
        let rec = recording();
        assert_eq!(
            WordSpan::measured(measured_at(&rec, 3_000), measured_at(&rec, 2_000)),
            Err(SpanFault::Inverted {
                start: FileMs::new(3_000),
                end: FileMs::new(2_000),
            })
        );
    }

    #[test]
    fn a_span_cannot_be_built_from_an_instant_outside_the_recording() {
        // Not a property of `WordSpan` at all, which is the point: there is no
        // route from raw milliseconds to a `RecordedInstant`, so containment is
        // already settled before a span is even proposed.
        let rec = recording();
        let Containment::Beyond { exceeds_by, .. } =
            rec.locate(FileMs::new(12_000), Origin::TranscriptBullet)
        else {
            panic!("12000ms is past a 10000ms recording")
        };
        assert_eq!(exceeds_by, Ms(2_000));
    }
}

/// Timings an engine produced that could not be used, counted by reason.
///
/// # Why this is one type and not two
///
/// The FA path and the UTR path both take engine output, both convert it
/// through [`WordSpan`], and both classify the same failures: a span with no
/// width, a span running backwards, and a position past the audio the engine
/// was given. They each grew their own counter struct, with their own field
/// names and their own `warn!` keys:
///
/// | fact | FA path was | UTR path was |
/// |---|---|---|
/// | zero-width span | `no_extent` | `dropped_degenerate` |
/// | inverted span | `inverted` | `dropped_inverted` |
/// | past window end | `outside_window` | `dropped_outside_window` |
///
/// So one incident produced two dashboards, and an operator could not ask "how
/// many out-of-window rejections did this file have" in a single query. The UTR
/// side even carried a comment claiming that routing through `WordSpan` had made
/// the two comparable; it unified the CLASSIFICATION and left the counters and
/// the log keys apart.
///
/// Each path keeps its own extra fields (the FA path counts clamped
/// assumptions, the UTR path counts tokens the engine gave no span for at all);
/// what they share lives here.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpanRejections {
    /// The engine answered, but its answer had no width.
    pub no_extent: usize,
    /// The engine's two ends contradict each other.
    pub inverted: usize,
    /// The engine answered about audio it was never given.
    ///
    /// The shape that put 28 seconds of phantom speech into six sessions.
    pub outside_window: usize,
    /// How far the worst out-of-window report exceeded its audio.
    pub worst_overshoot: Ms,
}

impl SpanRejections {
    /// A span that could not exist, kept apart by WHY.
    ///
    /// Exhaustive rather than a catch-all: a new [`SpanFault`] variant must be
    /// given a home here instead of silently joining a bucket.
    pub fn record_span_fault(&mut self, fault: SpanFault) {
        match fault {
            SpanFault::NoExtent { .. } => self.no_extent += 1,
            SpanFault::Inverted { .. } => self.inverted += 1,
        }
    }

    /// A report the engine placed outside the audio it was handed.
    pub fn record_outside(&mut self, fault: OutsideWindow) {
        self.outside_window += 1;
        let OutsideWindow::BeyondWindowEnd { exceeds_by, .. } = fault;
        self.worst_overshoot = self.worst_overshoot.max(exceeds_by);
    }

    /// How many timings were rejected, for any reason.
    pub fn total(self) -> usize {
        self.no_extent + self.inverted + self.outside_window
    }

    /// Whether the engine reported about audio it was not given.
    ///
    /// Named because it is the one class that used to corrupt output rather
    /// than merely losing a timing, so it earns the headline in a log line.
    pub fn any_outside_window(self) -> bool {
        self.outside_window > 0
    }
}

#[cfg(test)]
mod rejection_tests {
    use super::*;
    use crate::chat_ops::fa::coordinates::WindowMs;

    #[test]
    fn the_two_span_faults_are_counted_apart() {
        let mut tally = SpanRejections::default();
        tally.record_span_fault(SpanFault::NoExtent {
            at: FileMs::new(100),
        });
        tally.record_span_fault(SpanFault::Inverted {
            start: FileMs::new(200),
            end: FileMs::new(100),
        });
        assert_eq!(tally.no_extent, 1);
        assert_eq!(tally.inverted, 1);
        assert_eq!(tally.total(), 2);
        assert!(!tally.any_outside_window());
    }

    #[test]
    fn the_worst_overshoot_is_kept_not_the_last() {
        let mut tally = SpanRejections::default();
        for exceeds_by in [Ms(500), Ms(28_217), Ms(12)] {
            tally.record_outside(OutsideWindow::BeyondWindowEnd {
                reported: WindowMs::reported(0),
                window_len: Ms(1_000),
                exceeds_by,
            });
        }
        assert_eq!(tally.outside_window, 3);
        assert_eq!(tally.worst_overshoot, Ms(28_217));
        assert!(tally.any_outside_window());
    }
}
