//! Coordinate spaces, containment proofs and explicit clamping for forced
//! alignment timings.
//!
//! # The defect this module exists to make unrepresentable
//!
//! A forced aligner is handed a WINDOW of audio and reports word timings
//! measured from the start of that window. Those times are then offset back
//! into whole-file coordinates before they are written into a transcript. Until
//! this module existed both coordinate spaces were `u64`, so
//!
//! ```ignore
//! timing.start_ms + audio_offset_ms
//! ```
//!
//! type-checked whether the offset was the right one, the wrong one, or omitted
//! entirely, and the sum was written out with no further question asked. The
//! only validation on the result was that its end exceeded its start, which is a
//! relation among the transcript's own numbers and says nothing about the
//! recording.
//!
//! Nothing in the pipeline carried the recording's LENGTH, so no code was in a
//! position to notice that a word had been placed after the audio stopped. The
//! consequence, measured on real deliveries: alignment output reaching 28.2
//! seconds past the end of the file it was aligned against, on 6 of 226 screened
//! sessions, producing utterances that name a moment no recording contains.
//! Downstream every one of those reads as a measurement.
//!
//! # The shape of the cure
//!
//! Three separate facts get three separate types, and the transitions between
//! them are the only routes:
//!
//! * [`WindowMs`] is what an ENGINE reports. It cannot be written into a
//!   transcript, because a transcript's timings are not measured from a window.
//! * [`FileMs`] is a point measured from the start of the recording. The only
//!   way to obtain one from a [`WindowMs`] is [`FaWindow::to_file`], which is
//!   also the only place the containment question can be asked, so it cannot be
//!   forgotten.
//! * [`RecordedInstant`] is a `FileMs` that a specific [`Recording`] has agreed
//!   lies inside itself. Only [`Recording::locate`] produces one.
//!
//! # Why clamping is a type and not a `.min()` call
//!
//! `x.min(bound)` is correct arithmetic and invisible behaviour: a caller
//! cannot tell a value that FITTED from one that was silently cut down to fit,
//! and "we had to cut this down" is usually the signal that a transcript is
//! describing more speech than the recording holds. [`Clamped`] makes the
//! difference a variant, so a caller that wants to count, log or refuse the
//! clamped case has to acknowledge it exists.
//!
//! **Where this type is and is not applied, adjudicated 2026-08-15 rather than
//! inherited.** An earlier draft of this paragraph listed four files as though
//! every `.min()` in them were a pending conversion. Nine sites were then read
//! individually, and only one was this shape:
//!
//! * `grouping.rs::estimate_untimed_boundaries` computed an aligner window end
//!   and cut it to the audio with `.min(total_audio_ms)`. That IS the shape,
//!   and it now goes through [`Recording::clamp_bound`], which returns
//!   [`Clamped`] so the call site names both arms and counts the second. A
//!   window running past the end of the recording is a fact about our own gap
//!   arithmetic, and it was unobservable.
//! * `postprocess.rs` has two, and NEITHER is a defect. The fallback cap at the
//!   next onset already records what it did in the value's [`Origin`]
//!   (`ClampedTo { bound: NextOnset, .. }` wrapping the assumption), which is
//!   the whole point of the type; and the utterance-boundary pair feeds
//!   `clamped_to_bullet`, which is the named operation. Converting either would
//!   move information that is already carried.
//! * `grouping.rs::GroupWindow::extend_into_trailing_gap` and `expand_for_fillers.rs` cap an
//!   INCREMENT (`(gap / 2).min(MAX_..)`), not a position cut down to a domain
//!   bound. Nothing upstream is wrong when that cap binds; it is the policy
//!   working. A `Clamped` here would report a non-event on most utterances.
//! * `rescue_narrow_bullets.rs` selects among three candidate ends rather than
//!   trimming one to fit. Being capped by the next utterance is the expected
//!   case, not a signal.
//!
//! The distinction that decides it: [`Clamped`] is for a value cut down to fit
//! a bound where being cut MEANS something went wrong upstream. A maximum on an
//! increment, and a choice among candidates, are ordinary policy and stay
//! arithmetic.

use std::cmp::Ordering;

use crate::media::window::MediaWindow;

// One definition, in `crate::time`; this spelling is the one FA call sites
// already use. See that module for why it is not this one.
pub use crate::time::{FileMs, Ms, WindowMs};

use super::origin::{ClampBound, EngineId, Origin};

/// A value that may have been reduced to fit a bound, carrying WHICH happened.
///
/// Replaces a bare `.min(bound)`. The two cases mean different things to an
/// operator: `AsGiven` is a transcript that fits its recording, `ClampedTo` is
/// one that claimed more than the recording holds and was cut down. Collapsing
/// them to a single number is what let the second case pass unremarked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clamped<T> {
    /// The value was already within the bound and is unchanged.
    AsGiven(T),
    /// The value exceeded the bound and was reduced to it.
    ///
    /// Carries no `overshoot`. It used to, and `Recording::clamp` wrote the
    /// same number twice: once here and once into the clamped value's own
    /// [`Origin`], where it travels with the value instead of being lost the
    /// moment `value()` is called. Nothing kept the two equal, and the outer
    /// copy had no production reader; the one consumer matched `{ .. }`.
    ///
    /// That argument is ORIGIN-PATH ONLY. [`Recording::clamp_bound`] is
    /// deliberately provenance-free, so for its values the magnitude really is
    /// gone: 30 ms past the end of the audio and 30 minutes past it are the
    /// same variant, and that difference is what the 28.2-second incident
    /// turned on. Ask [`Recording::overshoot_of`] first where the size matters.
    ClampedTo {
        /// The bound, which is now the value.
        bound: T,
    },
}

impl<T> Clamped<T> {
    /// The resulting value, whichever case applied.
    pub fn value(self) -> T {
        match self {
            Self::AsGiven(v) => v,
            Self::ClampedTo { bound, .. } => bound,
        }
    }
}

/// The recording that alignment is being performed against.
///
/// # Why there is no `Option<Recording>` and no `Default`
///
/// Forced alignment always has an audio file: it cannot run without one, and
/// its duration can always be probed. The previous representation
/// (`total_audio_ms: Option<u64>`) made "we do not know how long the audio is"
/// a legal state, and every consumer then invented its own fallback for the
/// `None` arm. One skipped untimed utterances entirely, one declined to extend
/// the final window, and neither could bound anything. Making the duration a
/// required field deletes those arms rather than choosing better behaviour for
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recording {
    duration: FileMs,
}

/// Why a recording could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NotARecording {
    /// A zero-length file cannot contain speech, and treating it as a bound
    /// would place every word outside it.
    #[error("recording has zero duration")]
    ZeroDuration,
}

/// Where a point falls relative to a recording.
///
/// A sum rather than a `bool` plus a number, so the failing case carries the
/// evidence needed to report it and cannot be reached by a caller that only
/// tested the happy one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Containment {
    /// The point lies within the recording, and the proof is inside.
    Inside(RecordedInstant),
    /// The point lies after the recording ends. This is the state that used to
    /// be written into transcripts unremarked.
    Beyond {
        /// The point as computed.
        reported: FileMs,
        /// How far past the end of the recording it falls.
        exceeds_by: Ms,
        /// Where the offending number came from. Carried so that a report can
        /// say whether an engine measured it or we computed it, which decides
        /// whether the fix is to re-run alignment or to fix our arithmetic.
        origin: Origin,
    },
}

impl Recording {
    /// How far a point falls past the end of this recording, if it does.
    ///
    /// The containment question with no provenance attached, because a window
    /// bound is a decision of OURS about which utterances to group, not a
    /// measurement of anything, and nothing is ever written to a transcript
    /// from one.
    ///
    /// It exists because [`FaWindow::within`] previously reached this answer
    /// through [`Recording::locate`], which demands an `Origin`, and so passed
    /// `Origin::TranscriptBullet` for a value that came from no transcript: a
    /// fabricated provenance, inside the module whose purpose is to make
    /// provenance honest.
    pub fn overshoot_of(&self, at: FileMs) -> Option<Ms> {
        match at <= self.duration {
            true => None,
            false => Some(at.since(self.duration)),
        }
    }

    /// Cut a WINDOW BOUND down to the end of the recording, saying so.
    ///
    /// The provenance-free sibling of [`Recording::clamp`], and it exists for
    /// the same reason [`Recording::overshoot_of`] does: a window bound is a
    /// decision of OURS about which audio to hand an aligner, not a measurement
    /// of anything, and nothing is ever written into a transcript from one. So
    /// there is no honest `Origin` to attach, and the alternative was what this
    /// module was built to stop, a caller inventing one.
    ///
    /// Returns [`Clamped`] rather than a bare [`FileMs`] because "the window we
    /// computed runs past the end of the audio" is a fact about OUR arithmetic
    /// that a caller should be able to count. `.min()` made it invisible.
    pub fn clamp_bound(&self, at: FileMs) -> Clamped<FileMs> {
        // Built on `overshoot_of` rather than restating `at <= self.duration`.
        // This module's whole argument is that the containment question gets
        // asked in ONE place; three copies of the comparison inside one `impl`
        // is that argument failing at home. `clamp` composes on `locate` for
        // exactly the same reason.
        match self.overshoot_of(at) {
            None => Clamped::AsGiven(at),
            Some(_) => Clamped::ClampedTo {
                bound: self.duration,
            },
        }
    }

    /// A recording of a known duration.
    pub fn of_duration(duration: Ms) -> Result<Self, NotARecording> {
        match duration {
            Ms(0) => Err(NotARecording::ZeroDuration),
            Ms(ms) => Ok(Self {
                duration: FileMs::new(ms),
            }),
        }
    }

    /// The recording's length.
    pub const fn duration(&self) -> FileMs {
        self.duration
    }

    /// Ask whether a point falls inside this recording.
    ///
    /// The ONLY producer of [`RecordedInstant`]. A caller holding raw
    /// milliseconds cannot assert containment for itself, which is what keeps
    /// the proof honest, and it must say where the number came from, which is
    /// what keeps provenance unforgettable: there is no constructor that omits
    /// the origin.
    pub fn locate(&self, at: FileMs, origin: Origin) -> Containment {
        match at <= self.duration {
            true => Containment::Inside(RecordedInstant {
                at,
                recording: *self,
                origin,
            }),
            false => Containment::Beyond {
                reported: at,
                exceeds_by: at.since(self.duration),
                origin,
            },
        }
    }

    /// Reduce a point to the end of the recording when it overshoots, saying so
    /// in both the return type and the resulting value's provenance.
    ///
    /// The explicit form of `.min(total_audio_ms)`. Use where cutting the value
    /// down is genuinely the right behaviour and the fact that it happened is
    /// worth reporting; use [`Recording::locate`] where an overshoot should be
    /// refused instead.
    ///
    /// The clamped instant's origin WRAPS the origin it was given rather than
    /// replacing it, so a clamped measurement stays distinguishable from a
    /// clamped estimate all the way downstream.
    pub fn clamp(&self, at: FileMs, origin: Origin) -> Clamped<RecordedInstant> {
        match self.locate(at, origin) {
            Containment::Inside(instant) => Clamped::AsGiven(instant),
            Containment::Beyond {
                reported,
                exceeds_by,
                origin,
            } => Clamped::ClampedTo {
                bound: RecordedInstant {
                    at: self.duration,
                    recording: *self,
                    // The overshoot lives HERE, on the value, not beside it.
                    origin: Origin::ClampedTo {
                        bound: ClampBound::RecordingEnd,
                        was: Box::new(origin),
                        original: reported,
                        overshoot: exceeds_by,
                    },
                },
            },
        }
    }
}

/// A point that a specific [`Recording`] has agreed lies inside itself.
///
/// Constructible only by [`Recording::locate`] or [`Recording::clamp`], so
/// possession is the proof. It carries the recording it was checked against
/// rather than only the number, because a point checked against one recording
/// proves nothing about another and a bare `FileMs` cannot tell the two apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedInstant {
    at: FileMs,
    recording: Recording,
    origin: Origin,
}

impl RecordedInstant {
    /// The point, in file coordinates.
    pub const fn at(&self) -> FileMs {
        self.at
    }

    /// How this number was produced.
    ///
    /// Travels with the value through every later stage, so the question "is
    /// this a measurement or our own arithmetic?" is answerable at the point of
    /// use rather than reconstructed from a pipeline log.
    pub const fn origin(&self) -> &Origin {
        &self.origin
    }

    /// Keep the position, restate what the number MEANS here.
    ///
    /// The narrow case where a value changes role without changing value: the
    /// next word's measured onset, reused as THIS word's end, is no longer a
    /// measurement of anything about this word. Containment is untouched: the
    /// instant is already inside the recording it came from, so this is a
    /// relabelling rather than a move, and it needs no re-check.
    pub fn relabelled(&self, origin: Origin) -> Self {
        Self {
            at: self.at,
            recording: self.recording,
            origin,
        }
    }
}

/// Why a stretch of audio is not a window over its recording.
///
/// Both variants carry the numbers that produced them, so a caller reporting a
/// refusal does not have to re-derive them from the inputs it was about to
/// throw away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WindowFault {
    /// The end precedes the start. Refused before containment is considered:
    /// an inverted window has a negative length, and every later question
    /// asked of it (its length, whether a report fits inside it) would return
    /// an answer that looks reasonable and means nothing.
    #[error("window start {start} is after its end {end}")]
    Inverted {
        /// The proposed start.
        start: FileMs,
        /// The proposed end, which precedes it.
        end: FileMs,
    },
    /// The window extends past the end of the recording it is cut from, so
    /// audio the engine would be told about does not exist.
    #[error("window ends at {end}, {exceeds_by} past the end of the recording")]
    PastRecording {
        /// The proposed end.
        end: FileMs,
        /// How far past the recording's end it falls.
        exceeds_by: Ms,
    },
}

/// Why an engine's report cannot be placed in the file.
///
/// A single variant today, and still an enum: the alternative is a unit struct
/// whose meaning lives in its name, and adding the second reason later would
/// then change every call site rather than adding an arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutsideWindow {
    /// The engine reported a position past the end of the audio it was handed.
    ///
    /// This is the defect that produced this module. Whisper pads its input to
    /// a fixed 30 seconds and reports positions inside the padding; added to a
    /// window offset unchecked, those became file positions up to 28.2 seconds
    /// past the end of the recording, and were written into transcripts as
    /// measurements.
    #[error("engine reported {reported} in a {window_len} window, {exceeds_by} beyond its end")]
    BeyondWindowEnd {
        /// The engine's report, in window coordinates.
        reported: WindowMs,
        /// The length of the audio the engine was actually given.
        window_len: Ms,
        /// How far past that length the report falls.
        exceeds_by: Ms,
    },
}

/// A stretch of a [`Recording`] that was handed to an aligner as its input.
///
/// # Why this exists
///
/// An engine is given a slice of audio and answers in coordinates relative to
/// THAT SLICE, while a transcript is written in coordinates relative to the
/// FILE. Both were `u64`, so `engine_report + offset` type-checked whether the
/// offset was right, wrong, or omitted, and nothing could ask whether the sum
/// was still inside the recording.
///
/// Constructible only through [`FaWindow::within`], which refuses a window that
/// is inverted or extends past its recording, so possession of one is proof
/// that the audio it names exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaWindow {
    start: FileMs,
    end: FileMs,
    recording: Recording,
}

impl FaWindow {
    /// The only constructor: a window over a recording, checked against it.
    ///
    /// Ordering is checked before containment, because an inverted window's
    /// length is meaningless and every containment answer computed from it
    /// would be as well.
    pub fn within(recording: &Recording, start: FileMs, end: FileMs) -> Result<Self, WindowFault> {
        match start.cmp(&end) {
            Ordering::Greater => Err(WindowFault::Inverted { start, end }),
            Ordering::Less | Ordering::Equal => match recording.overshoot_of(end) {
                Some(exceeds_by) => Err(WindowFault::PastRecording { end, exceeds_by }),
                None => Ok(Self {
                    start,
                    end,
                    recording: *recording,
                }),
            },
        }
    }

    /// A window over a recording, from a [`MediaWindow`] that already proved
    /// its own ordering.
    ///
    /// The single owner of the `MediaWindow` to `FaWindow` conversion. Callers
    /// used to destructure the media window into two `u64`s and hand them to
    /// [`FaWindow::within`], which re-proved ordering the media window had
    /// already established and passed the containment question through raw
    /// integers: the proof was discarded and rebuilt, and nothing typed the gap
    /// between the two. One conversion means one place where the extra fact
    /// (does this window fit the recording?) is added to the one the media
    /// window already carries.
    pub fn over(recording: &Recording, window: MediaWindow) -> Result<Self, WindowFault> {
        Self::within(recording, window.start(), window.end())
    }

    /// The recording this window is cut from.
    ///
    /// Exposed so a caller holding a window can ask the recording for the
    /// clamping and containment decisions that belong to it, rather than
    /// re-deriving a bound from `self.end()` and getting a window-shaped answer
    /// to a recording-shaped question.
    pub const fn recording(&self) -> Recording {
        self.recording
    }

    /// The window's end, in file coordinates.
    ///
    /// Engine reports cross into file coordinates through `to_file`; audio
    /// request construction uses the crate-private `audio_start` accessor.
    pub const fn end(&self) -> FileMs {
        self.end
    }

    /// Start coordinate used to build the audio request for this admitted window.
    pub(crate) const fn audio_start(&self) -> FileMs {
        self.start
    }

    /// Extend coverage without leaving the recording that admitted this window.
    pub(crate) fn extend_by(self, extension: Ms) -> Self {
        Self {
            end: FileMs::new(
                self.end
                    .get()
                    .saturating_add(extension.0)
                    .min(self.recording.duration().get()),
            ),
            ..self
        }
    }

    /// The window's length.
    pub const fn len(&self) -> Ms {
        self.end.since(self.start)
    }

    /// THE transition from engine coordinates into file coordinates.
    ///
    /// This is the single place where a window-relative time becomes a
    /// file-relative one, which is what makes the containment question
    /// unforgettable: there is no other constructor, so every engine report in
    /// the codebase passes through this check.
    ///
    /// A report beyond the window's own end is refused rather than clamped. The
    /// engine was given exactly this audio; a time past its end is not a
    /// measurement that needs trimming, it is evidence that the engine did not
    /// align, and the honest result is no timing for that word.
    pub fn to_file(
        &self,
        reported: WindowMs,
        engine: &EngineId,
    ) -> Result<RecordedInstant, OutsideWindow> {
        let window_len = self.len();
        match reported.offset_from_window_start() > window_len.0 {
            true => Err(OutsideWindow::BeyondWindowEnd {
                reported,
                window_len,
                exceeds_by: Ms(reported.offset_from_window_start() - window_len.0),
            }),
            false => {
                let at = FileMs::new(self.start.get() + reported.offset_from_window_start());
                // Cloned, not moved: the caller keeps the id for the rest of
                // the group. Free on the live path, where the name is a
                // borrowed literal.
                let measured = Origin::EngineMeasured {
                    engine: engine.clone(),
                };
                // Cannot be `Beyond`: the window's end is inside the recording
                // and `at` is at most the window's end. Matched rather than
                // unwrapped so the impossible case has a written-out answer.
                match self.recording.locate(at, measured) {
                    Containment::Inside(instant) => Ok(instant),
                    Containment::Beyond { exceeds_by, .. } => Err(OutsideWindow::BeyondWindowEnd {
                        reported,
                        window_len,
                        exceeds_by,
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(ms: u64) -> Recording {
        Recording::of_duration(Ms(ms)).expect("non-zero")
    }

    fn engine() -> EngineId {
        EngineId::new("whisper-fa-large-v2")
    }

    #[test]
    fn a_zero_length_recording_is_not_a_recording() {
        assert_eq!(
            Recording::of_duration(Ms(0)),
            Err(NotARecording::ZeroDuration)
        );
    }

    #[test]
    fn the_defect_that_produced_this_module_is_now_refused() {
        // The real shape, in miniature. A window sits at the very end of the
        // recording with little audio left, and the engine reports a position
        // far beyond the audio it was handed. Before this module the sum was
        // written into a transcript as a measurement.
        let rec = recording(1_259_968);
        let window = FaWindow::within(&rec, FileMs::new(1_255_165), FileMs::new(1_259_968))
            .expect("window inside the recording");
        let absurd = WindowMs::reported(33_020);
        assert_eq!(
            window.to_file(absurd, &engine()),
            Err(OutsideWindow::BeyondWindowEnd {
                reported: absurd,
                window_len: Ms(4_803),
                exceeds_by: Ms(28_217),
            })
        );
    }

    #[test]
    fn a_report_inside_the_window_lands_inside_the_recording_and_says_who_measured_it() {
        let rec = recording(10_000);
        let window =
            FaWindow::within(&rec, FileMs::new(4_000), FileMs::new(6_000)).expect("inside");
        let placed = window
            .to_file(WindowMs::reported(500), &engine())
            .expect("inside");
        assert_eq!(placed.at(), FileMs::new(4_500));
        assert_eq!(
            placed.origin(),
            &Origin::EngineMeasured { engine: engine() }
        );
        assert!(placed.origin().is_observation());
    }

    #[test]
    fn a_report_exactly_at_the_window_end_is_accepted() {
        // The boundary is inclusive: an engine reporting the final instant of
        // its audio has said something true, and refusing it would discard a
        // legitimate alignment.
        let rec = recording(10_000);
        let window =
            FaWindow::within(&rec, FileMs::new(4_000), FileMs::new(6_000)).expect("inside");
        let placed = window
            .to_file(WindowMs::reported(2_000), &engine())
            .expect("boundary");
        assert_eq!(placed.at(), FileMs::new(6_000));
    }

    #[test]
    fn a_window_cannot_be_opened_past_the_recording() {
        let rec = recording(10_000);
        assert_eq!(
            FaWindow::within(&rec, FileMs::new(9_000), FileMs::new(12_000)),
            Err(WindowFault::PastRecording {
                end: FileMs::new(12_000),
                exceeds_by: Ms(2_000),
            })
        );
    }

    #[test]
    fn an_inverted_window_is_refused_before_containment_is_considered() {
        let rec = recording(10_000);
        assert_eq!(
            FaWindow::within(&rec, FileMs::new(6_000), FileMs::new(4_000)),
            Err(WindowFault::Inverted {
                start: FileMs::new(6_000),
                end: FileMs::new(4_000),
            })
        );
    }

    #[test]
    fn clamping_says_that_it_clamped_and_by_how_much() {
        // The behaviour `.min(total_audio_ms)` had, plus the fact it discarded.
        let rec = recording(10_000);
        let fits = rec.clamp(FileMs::new(9_000), Origin::TranscriptBullet);
        assert!(matches!(fits, Clamped::AsGiven(_)));
        assert_eq!(fits.value().at(), FileMs::new(9_000));

        let cut = rec.clamp(FileMs::new(12_500), Origin::TranscriptBullet);
        assert!(matches!(cut, Clamped::ClampedTo { .. }));
        let instant = cut.value();
        // The overshoot is read off the value's own provenance, which is the
        // only copy now and the one that survives `value()`.
        assert_eq!(
            instant.origin(),
            &Origin::ClampedTo {
                bound: ClampBound::RecordingEnd,
                was: Box::new(Origin::TranscriptBullet),
                original: FileMs::new(12_500),
                overshoot: Ms(2_500),
            }
        );
        assert_eq!(instant.at(), FileMs::new(10_000));
        // A clamped value is no longer an observation, and still remembers it
        // once was: both halves matter, and neither was expressible before.
        assert!(!instant.origin().is_observation());
        assert_eq!(instant.origin().underlying(), &Origin::TranscriptBullet);
    }
}
