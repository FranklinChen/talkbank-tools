//! The one owner of ASR interval rounding and range admission.
//!
//! Every provider reports word and segment boundaries as its own numbers:
//! fractional seconds (Rev, Whisper, the HK provider elements), whole
//! milliseconds (FunASR, Aliyun), or a millisecond OFFSET that only means
//! something once a segment start is added to it (Tencent). Before this module
//! each of those was converted where it happened, with `as i64` casts, `round()`
//! calls and `unwrap_or(0)` defaults spread across the bridge and the pipeline.
//! Two defects lived in that spread:
//!
//! - **A fabricated zero.** An absent bound became `0`, which is a legal time,
//!   so a word missing only its start claimed to begin at the start of the
//!   recording and passed every downstream check.
//! - **An unchecked cast.** `as i64` on a non-finite or astronomically large
//!   float saturates silently, so a provider sending epoch milliseconds, or a
//!   NaN, produced a number rather than a refusal.
//!
//! The graph here makes both unrepresentable:
//!
//! ```text
//! provider numbers --admit--> AdmittedInterval        (proof: finite, ordered, in range)
//!                  --refuse-> IntervalRefusal         (which bound, and why)
//! absence          ---------> WordTiming::Untimed(cause)
//! ```
//!
//! [`AdmittedInterval`] (`batchalign_types::interval`, re-exported here) has
//! private fields, and every constructor (a pair of integers through
//! `admit_millis` included) checks the range and the order, so a value of
//! this type is a proof that the numbers inside it were rounded once and
//! admitted against its range policy. [`WordTiming`] is the other half: a word is timed
//! with an admitted interval, or untimed with a NAMED cause. There is no third
//! state and no way to spell "untimed" as a number.

use batchalign_types::domain::AudioPositionSeconds;
pub use batchalign_types::interval::{
    AdmittedInterval, IntervalBound, IntervalRefusal, UntimedCause,
};
use serde::{Deserialize, Serialize};

/// One word's timing: an admitted interval, or an absence with a named cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WordTiming {
    /// The provider timed this word, and the interval was admitted.
    Timed(AdmittedInterval),
    /// The word has no interval, for this reason.
    Untimed(UntimedCause),
}

impl WordTiming {
    /// Admit optional audio positions into a timing, refusing a bad pair.
    ///
    /// Absence is a state, never a zero. A zero-width span is admitted by
    /// [`AdmittedInterval`] and then reported here as untimed, because a span
    /// that covers no time cannot locate a word in audio.
    pub fn from_positions(
        start: Option<AudioPositionSeconds>,
        end: Option<AudioPositionSeconds>,
    ) -> Result<Self, IntervalRefusal> {
        match (start, end) {
            (None, None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoTiming)),
            (None, Some(_)) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoStart)),
            (Some(_), None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoEnd)),
            (Some(start), Some(end)) => {
                let interval = AdmittedInterval::admit_positions(start, end)?;
                Ok(Self::from_admitted(interval))
            }
        }
    }

    /// Admit optional millisecond bounds into a timing, refusing bad numbers.
    pub fn from_millis(
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> Result<Self, IntervalRefusal> {
        match (start_ms, end_ms) {
            (None, None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoTiming)),
            (None, Some(_)) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoStart)),
            (Some(_), None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoEnd)),
            (Some(start_ms), Some(end_ms)) => {
                let interval = AdmittedInterval::admit_millis(start_ms, end_ms)?;
                Ok(Self::from_admitted(interval))
            }
        }
    }

    /// Wrap an already-admitted interval, demoting a zero-width one.
    pub const fn from_admitted(interval: AdmittedInterval) -> Self {
        if interval.is_empty() {
            Self::Untimed(UntimedCause::ZeroLengthSpan)
        } else {
            Self::Timed(interval)
        }
    }

    /// Admit optional audio positions, recording a refusal as an untimed cause.
    ///
    /// For the callers that have no error channel to refuse through: the ASR
    /// post-processing pipeline runs inside a total transform, and a word whose
    /// bounds are inadmissible must carry no time rather than a fabricated one.
    /// The cause is still named, so the fact is not lost the way a silent
    /// `(None, None)` lost it.
    ///
    /// The bounds are positions, already proven finite and non-negative where
    /// they were born, so the `NotFinite` and `Negative` refusals cannot arise
    /// here. What can still be refused is the PAIR: an end before its start,
    /// or a bound beyond [`AdmittedInterval::MAX_MS`].
    pub fn admit_positions_or_untimed(
        start: Option<AudioPositionSeconds>,
        end: Option<AudioPositionSeconds>,
    ) -> Self {
        match Self::from_positions(start, end) {
            Ok(timing) => timing,
            Err(_) => Self::Untimed(UntimedCause::RefusedByAdmission),
        }
    }

    /// The admitted interval, when there is one.
    pub const fn interval(self) -> Option<AdmittedInterval> {
        match self {
            Self::Timed(interval) => Some(interval),
            Self::Untimed(_) => None,
        }
    }

    /// The cause of absence, when the word is untimed.
    pub const fn untimed_cause(self) -> Option<UntimedCause> {
        match self {
            Self::Timed(_) => None,
            Self::Untimed(cause) => Some(cause),
        }
    }

    /// The positive interval this timing proves, or why there is none.
    ///
    /// Total: a `Timed` interval that is somehow empty (the variant is public,
    /// so only the constructors above demote) answers `ZeroLengthSpan` rather
    /// than a bullet that locates nothing.
    pub fn positive(self) -> Result<PositiveInterval, UntimedCause> {
        match self {
            Self::Timed(interval) => PositiveInterval::of(interval),
            Self::Untimed(cause) => Err(cause),
        }
    }

    /// Lower to the `(Option<i64>, Option<i64>)` pair the post-processing
    /// word type still stores.
    ///
    /// The ONE lossy step, named so the place where the cause stops travelling
    /// is a signature rather than a habit.
    pub const fn into_optional_millis(self) -> (Option<i64>, Option<i64>) {
        match self {
            Self::Timed(interval) => (Some(interval.start_ms()), Some(interval.end_ms())),
            Self::Untimed(_) => (None, None),
        }
    }
}

/// An interval that covers real time: admitted, with its end after its start.
///
/// The one input a generated CHAT bullet is built from ([`Self::bullet`]).
/// Its fields are private and its only constructor is [`Self::of`] over an
/// [`AdmittedInterval`], so a value is a proof that its bounds were admitted
/// once, are non-negative, and enclose a positive duration. Two such intervals
/// cover a third ([`Self::covering`]), which is how an utterance's span is
/// formed from its words' without a second check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveInterval {
    start_ms: u64,
    end_ms: u64,
}

impl PositiveInterval {
    /// Admit a non-empty admitted interval. An empty one is the zero-width
    /// span that locates nothing. Admission already proved both bounds
    /// non-negative, so the unsigned conversion cannot refuse; if it ever did,
    /// the cause is named rather than a bound invented.
    pub fn of(interval: AdmittedInterval) -> Result<Self, UntimedCause> {
        let (Ok(start_ms), Ok(end_ms)) = (
            u64::try_from(interval.start_ms()),
            u64::try_from(interval.end_ms()),
        ) else {
            return Err(UntimedCause::RefusedByAdmission);
        };
        if end_ms > start_ms {
            Ok(Self { start_ms, end_ms })
        } else {
            Err(UntimedCause::ZeroLengthSpan)
        }
    }

    /// Start, in recording milliseconds.
    pub const fn start_ms(self) -> u64 {
        self.start_ms
    }

    /// End, in recording milliseconds; always after the start.
    pub const fn end_ms(self) -> u64 {
        self.end_ms
    }

    /// The smallest interval covering both. Positive because both are.
    pub fn covering(self, other: Self) -> Self {
        Self {
            start_ms: self.start_ms.min(other.start_ms),
            end_ms: self.end_ms.max(other.end_ms),
        }
    }

    /// THE construction of a generated bullet: from a positive interval only.
    pub fn bullet(self) -> talkbank_model::model::Bullet {
        talkbank_model::model::Bullet::new(self.start_ms, self.end_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn a_positive_interval_is_only_built_from_real_duration() {
        let admit = |start, end| AdmittedInterval::admit_millis(start, end).expect("admitted");
        let interval = PositiveInterval::of(admit(100, 250)).expect("positive");
        assert_eq!((interval.start_ms(), interval.end_ms()), (100, 250));
        assert_eq!(
            PositiveInterval::of(admit(100, 100)),
            Err(UntimedCause::ZeroLengthSpan)
        );
        assert_eq!(
            WordTiming::from_millis(Some(5), Some(5))
                .expect("admitted")
                .positive(),
            Err(UntimedCause::ZeroLengthSpan)
        );
        assert_eq!(
            WordTiming::from_millis(None, Some(5))
                .expect("admitted")
                .positive(),
            Err(UntimedCause::ProviderReportedNoStart)
        );
        let later = PositiveInterval::of(admit(400, 500)).expect("positive");
        let hull = interval.covering(later);
        assert_eq!((hull.start_ms(), hull.end_ms()), (100, 500));
        let bullet = hull.bullet();
        assert_eq!((bullet.timing.start_ms, bullet.timing.end_ms), (100, 500));
    }

    #[test]
    fn a_zero_length_span_is_admitted_but_is_not_a_timed_word() {
        let interval = AdmittedInterval::admit_millis(100, 100).expect("zero width is admissible");
        assert!(interval.is_empty());
        assert_eq!(
            WordTiming::from_admitted(interval),
            WordTiming::Untimed(UntimedCause::ZeroLengthSpan)
        );
    }

    #[test]
    fn each_absent_bound_names_its_own_cause() {
        let one = AudioPositionSeconds::try_from(1.0).ok();
        assert_eq!(
            WordTiming::from_positions(None, None),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming))
        );
        assert_eq!(
            WordTiming::from_positions(None, one),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoStart))
        );
        assert_eq!(
            WordTiming::from_positions(one, None),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoEnd))
        );
    }

    #[test]
    fn a_refusal_deep_in_the_pipeline_becomes_a_named_absence_not_a_zero() {
        // An end before its start: the refusal a pair of valid positions can
        // still earn.
        let at = |seconds: f64| AudioPositionSeconds::try_from(seconds).ok();
        let timing = WordTiming::admit_positions_or_untimed(at(2.0), at(1.0));
        assert_eq!(
            timing,
            WordTiming::Untimed(UntimedCause::RefusedByAdmission)
        );
        assert_eq!(timing.into_optional_millis(), (None, None));
    }

    proptest! {


        /// A word timing is never both timed and zero-width: a span that covers
        /// no time cannot locate a word, so it comes back untimed with its own
        /// cause rather than as a degenerate interval.
        #[test]
        fn a_timed_word_always_covers_real_time(
            start_ms in 0_i64..1_000_000_i64,
            length_ms in 0_i64..1_000_i64,
        ) {
            let timing = WordTiming::from_millis(Some(start_ms), Some(start_ms + length_ms))
                .expect("non-negative ordered millis are admissible");
            match timing {
                WordTiming::Timed(interval) => {
                    prop_assert!(length_ms > 0);
                    prop_assert_eq!(interval.end_ms() - interval.start_ms(), length_ms);
                }
                WordTiming::Untimed(cause) => {
                    prop_assert_eq!(length_ms, 0);
                    prop_assert_eq!(cause, UntimedCause::ZeroLengthSpan);
                }
            }
        }

    }
}
