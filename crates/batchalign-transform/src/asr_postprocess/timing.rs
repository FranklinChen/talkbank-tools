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
//! [`AdmittedInterval`] has private fields and no constructor that can be
//! handed a bare pair of integers, so a value of this type is a proof that the
//! numbers inside it were rounded once, by this module, and admitted against
//! the range policy below. [`WordTiming`] is the other half: a word is timed
//! with an admitted interval, or untimed with a NAMED cause. There is no third
//! state and no way to spell "untimed" as a number.

use serde::{Deserialize, Serialize};

/// Which end of an interval a refusal is about.
///
/// Carried on the refusal so an operator reads WHICH bound was bad rather than
/// a message true of either one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalBound {
    /// The interval's start.
    Start,
    /// The interval's end.
    End,
}

impl IntervalBound {
    /// The bound's name, for messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::End => "end",
        }
    }
}

impl std::fmt::Display for IntervalBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a pair of provider numbers is not an interval.
///
/// Every variant names the offending value, because the caller that refuses a
/// file is usually not the person who has to explain it afterwards.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum IntervalRefusal {
    /// The bound is NaN or an infinity. `as i64` would have saturated it into a
    /// plausible-looking number.
    #[error("{bound} time is not a finite number")]
    NotFinite {
        /// Which bound was not finite.
        bound: IntervalBound,
    },
    /// The bound is before the start of the recording.
    #[error("{bound} time {value_ms} ms is negative")]
    Negative {
        /// Which bound was negative.
        bound: IntervalBound,
        /// The offending value, in milliseconds.
        value_ms: f64,
    },
    /// The bound is past [`AdmittedInterval::MAX_MS`]. The realistic cause is a
    /// provider (or a translation of one) reporting an absolute epoch
    /// timestamp where a media offset was expected.
    #[error(
        "{bound} time {value_ms} ms is beyond the admitted range of {} ms; a value \
         this large is an absolute timestamp, not a media offset",
        AdmittedInterval::MAX_MS
    )]
    OutOfRange {
        /// Which bound was out of range.
        bound: IntervalBound,
        /// The offending value, in milliseconds.
        value_ms: f64,
    },
    /// The interval ends before it starts.
    #[error("interval ends at {end_ms} ms, before its start at {start_ms} ms")]
    Inverted {
        /// The admitted start, in milliseconds.
        start_ms: i64,
        /// The admitted end, in milliseconds.
        end_ms: i64,
    },
    /// A segment start plus a word offset does not fit in the millisecond
    /// integer. Checked rather than wrapped: the sum is what the word's
    /// absolute time IS, so a wrapped one is a wrong time, not a big one.
    #[error("segment start {segment_start_ms} ms plus offset {offset_ms} ms overflows")]
    OffsetOverflow {
        /// The segment's own start, in milliseconds.
        segment_start_ms: i64,
        /// The offset that was added to it, in milliseconds.
        offset_ms: i64,
    },
}

/// A media interval in whole milliseconds, proven finite, ordered and in range.
///
/// Private fields, and every constructor is fallible, so holding one of these
/// is the proof. Zero length is ADMITTED here (a provider may legitimately
/// report a zero-width unit) and refused separately by the callers for whom a
/// zero-width span locates nothing; see [`WordTiming::from_seconds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AdmittedInterval {
    start_ms: i64,
    end_ms: i64,
}

impl AdmittedInterval {
    /// The largest media offset this module will admit, in milliseconds.
    ///
    /// About 31.7 years, chosen so that no real recording is refused and the
    /// realistic corruption IS: a Unix epoch timestamp in milliseconds is
    /// around 1.7e12 and lands above this line, while the longest continuous
    /// recording anyone has handed batchalign is measured in hours.
    pub const MAX_MS: i64 = 1_000_000_000_000;

    /// Admit a bound given in fractional seconds.
    ///
    /// Rounds half away from zero, which is what `(seconds * 1000.0).round()`
    /// did at every site this replaced, so admitted values are byte-identical
    /// to the previous pipeline's for every input the previous pipeline did not
    /// silently corrupt.
    fn admit_bound_seconds(bound: IntervalBound, seconds: f64) -> Result<i64, IntervalRefusal> {
        if !seconds.is_finite() {
            return Err(IntervalRefusal::NotFinite { bound });
        }
        let millis = seconds * 1000.0;
        Self::admit_bound_millis_f64(bound, millis)
    }

    /// Admit a bound already expressed in (possibly fractional) milliseconds.
    fn admit_bound_millis_f64(bound: IntervalBound, millis: f64) -> Result<i64, IntervalRefusal> {
        if !millis.is_finite() {
            return Err(IntervalRefusal::NotFinite { bound });
        }
        let rounded = millis.round();
        if rounded < 0.0 {
            return Err(IntervalRefusal::Negative {
                bound,
                value_ms: millis,
            });
        }
        // Compared as f64 against the limit BEFORE any cast, so a value beyond
        // i64 cannot reach the cast at all. `MAX_MS` is exactly representable
        // as an f64, so this comparison is exact.
        #[allow(clippy::cast_precision_loss)]
        if rounded > Self::MAX_MS as f64 {
            return Err(IntervalRefusal::OutOfRange {
                bound,
                value_ms: millis,
            });
        }
        // Proven finite, non-negative and <= MAX_MS just above, so the cast is
        // exact rather than saturating.
        #[allow(clippy::cast_possible_truncation)]
        Ok(rounded as i64)
    }

    /// Admit a pair of already-rounded millisecond bounds.
    pub fn admit_millis(start_ms: i64, end_ms: i64) -> Result<Self, IntervalRefusal> {
        #[allow(clippy::cast_precision_loss)]
        for (bound, value) in [
            (IntervalBound::Start, start_ms),
            (IntervalBound::End, end_ms),
        ] {
            if value < 0 {
                return Err(IntervalRefusal::Negative {
                    bound,
                    value_ms: value as f64,
                });
            }
            if value > Self::MAX_MS {
                return Err(IntervalRefusal::OutOfRange {
                    bound,
                    value_ms: value as f64,
                });
            }
        }
        if end_ms < start_ms {
            return Err(IntervalRefusal::Inverted { start_ms, end_ms });
        }
        Ok(Self { start_ms, end_ms })
    }

    /// Admit a pair of bounds given in fractional seconds.
    pub fn admit_seconds(start_s: f64, end_s: f64) -> Result<Self, IntervalRefusal> {
        let start_ms = Self::admit_bound_seconds(IntervalBound::Start, start_s)?;
        let end_ms = Self::admit_bound_seconds(IntervalBound::End, end_s)?;
        Self::admit_millis(start_ms, end_ms)
    }

    /// Admit a pair of bounds given in (possibly fractional) milliseconds.
    ///
    /// FunASR's shape: a `[start_ms, end_ms]` pair of JSON numbers, which may
    /// arrive fractional. The rounding happens HERE, with the range check, so
    /// the bridge that reads the pair has no reason to keep a cast of its own.
    /// That cast is exactly what survived at one provider after the others had
    /// given theirs up.
    pub fn admit_millis_f64(start_ms: f64, end_ms: f64) -> Result<Self, IntervalRefusal> {
        let start = Self::admit_bound_millis_f64(IntervalBound::Start, start_ms)?;
        let end = Self::admit_bound_millis_f64(IntervalBound::End, end_ms)?;
        Self::admit_millis(start, end)
    }

    /// Admit a word interval expressed as OFFSETS from a segment's own start.
    ///
    /// Tencent reports words this way: the segment carries an absolute
    /// `StartMs` and each word an `OffsetStartMs` / `OffsetEndMs` relative to
    /// it. The addition is CHECKED, because a wrapped sum is a wrong time that
    /// looks exactly like a right one.
    pub fn admit_offset_from(
        segment_start_ms: i64,
        offset_start_ms: i64,
        offset_end_ms: i64,
    ) -> Result<Self, IntervalRefusal> {
        let start_ms = segment_start_ms.checked_add(offset_start_ms).ok_or(
            IntervalRefusal::OffsetOverflow {
                segment_start_ms,
                offset_ms: offset_start_ms,
            },
        )?;
        let end_ms =
            segment_start_ms
                .checked_add(offset_end_ms)
                .ok_or(IntervalRefusal::OffsetOverflow {
                    segment_start_ms,
                    offset_ms: offset_end_ms,
                })?;
        Self::admit_millis(start_ms, end_ms)
    }

    /// The admitted start, in whole milliseconds.
    pub const fn start_ms(self) -> i64 {
        self.start_ms
    }

    /// The admitted end, in whole milliseconds.
    pub const fn end_ms(self) -> i64 {
        self.end_ms
    }

    /// Whether the interval covers no time at all.
    ///
    /// A zero-width interval is a legal admission but locates nothing, so
    /// callers that need a span to point at audio treat it as untimed.
    pub const fn is_empty(self) -> bool {
        self.start_ms == self.end_ms
    }

    /// The admitted start and end in fractional seconds, for the provider
    /// shapes whose wire format speaks seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn as_seconds(self) -> (f64, f64) {
        (
            self.start_ms as f64 / 1000.0,
            self.end_ms as f64 / 1000.0,
        )
    }
}

/// Why a word carries no interval.
///
/// A closed set, because "untimed" with no reason is how a fabricated zero got
/// in: once the reason has to be named, there is nowhere to hide a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UntimedCause {
    /// The provider reported neither bound.
    ProviderReportedNoTiming,
    /// The provider reported an end but no start.
    ProviderReportedNoStart,
    /// The provider reported a start but no end.
    ProviderReportedNoEnd,
    /// The enclosing segment carried no start, so the word's offsets cannot be
    /// made absolute. Tencent's `StartMs` is the field in question; before this
    /// state existed, an absent one became the start of the recording.
    SegmentStartAbsent,
    /// The provider reported a zero-width span, which locates nothing.
    ZeroLengthSpan,
    /// The bounds were present but not admissible. The refusal itself is
    /// reported by the caller that owns an error channel; deep inside the
    /// post-processing pipeline, which has none, the word simply carries no
    /// time and says so.
    RefusedByAdmission,
}

impl UntimedCause {
    /// The cause's wire/log name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProviderReportedNoTiming => "provider reported no timing",
            Self::ProviderReportedNoStart => "provider reported no start",
            Self::ProviderReportedNoEnd => "provider reported no end",
            Self::SegmentStartAbsent => "enclosing segment carried no start",
            Self::ZeroLengthSpan => "provider reported a zero-width span",
            Self::RefusedByAdmission => "reported bounds were not admissible",
        }
    }
}

impl std::fmt::Display for UntimedCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

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
    /// Admit optional second bounds into a timing, refusing bad numbers.
    ///
    /// Absence is a state, never a zero. A zero-width span is admitted by
    /// [`AdmittedInterval`] and then reported here as untimed, because a span
    /// that covers no time cannot locate a word in audio.
    pub fn from_seconds(
        start_s: Option<f64>,
        end_s: Option<f64>,
    ) -> Result<Self, IntervalRefusal> {
        match (start_s, end_s) {
            (None, None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoTiming)),
            (None, Some(_)) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoStart)),
            (Some(_), None) => Ok(Self::Untimed(UntimedCause::ProviderReportedNoEnd)),
            (Some(start_s), Some(end_s)) => {
                let interval = AdmittedInterval::admit_seconds(start_s, end_s)?;
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

    /// Admit optional second bounds, recording a refusal as an untimed cause.
    ///
    /// For the callers that have no error channel to refuse through: the ASR
    /// post-processing pipeline runs inside a total transform, and a word whose
    /// bounds are inadmissible must carry no time rather than a fabricated one.
    /// The cause is still named, so the fact is not lost the way a silent
    /// `(None, None)` lost it.
    pub fn admit_seconds_or_untimed(start_s: Option<f64>, end_s: Option<f64>) -> Self {
        Self::from_seconds(start_s, end_s)
            .unwrap_or(Self::Untimed(UntimedCause::RefusedByAdmission))
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

    /// Lower to the `(Option<i64>, Option<i64>)` pair the post-processing
    /// word type still stores.
    ///
    /// The ONE lossy step, named so the place where the cause stops travelling
    /// is a signature rather than a habit.
    pub const fn into_optional_millis(self) -> (Option<i64>, Option<i64>) {
        match self {
            Self::Timed(interval) => (Some(interval.start_ms), Some(interval.end_ms)),
            Self::Untimed(_) => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rounding_matches_the_half_away_from_zero_rule_it_replaced() {
        // The pipeline's previous expression was `(seconds * 1000.0).round()`.
        // These are the boundary cases that distinguish rounding modes.
        for (start_s, end_s, expected) in [
            (0.0015_f64, 0.0025_f64, (2_i64, 3_i64)),
            (1.2345, 1.2355, (1235, 1236)),
            (0.0, 1.0, (0, 1000)),
        ] {
            let interval = AdmittedInterval::admit_seconds(start_s, end_s).expect("admissible");
            assert_eq!((interval.start_ms(), interval.end_ms()), expected);
        }
    }

    #[test]
    fn a_non_finite_bound_is_refused_by_name_instead_of_saturating() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                AdmittedInterval::admit_seconds(bad, 1.0),
                Err(IntervalRefusal::NotFinite {
                    bound: IntervalBound::Start
                })
            );
            assert_eq!(
                AdmittedInterval::admit_seconds(0.0, bad),
                Err(IntervalRefusal::NotFinite {
                    bound: IntervalBound::End
                })
            );
        }
    }

    #[test]
    fn a_negative_bound_is_refused_with_its_value() {
        assert!(matches!(
            AdmittedInterval::admit_seconds(-0.001, 1.0),
            Err(IntervalRefusal::Negative {
                bound: IntervalBound::Start,
                ..
            })
        ));
        assert!(matches!(
            AdmittedInterval::admit_millis(0, -1),
            Err(IntervalRefusal::Negative {
                bound: IntervalBound::End,
                ..
            })
        ));
    }

    /// The realistic corruption this range exists for: a provider reporting an
    /// absolute epoch timestamp where a media offset belongs.
    #[test]
    fn an_epoch_millisecond_timestamp_is_out_of_range() {
        let epoch_ms = 1_789_000_000_000_i64;
        assert!(epoch_ms > AdmittedInterval::MAX_MS);
        assert!(matches!(
            AdmittedInterval::admit_millis(epoch_ms, epoch_ms + 500),
            Err(IntervalRefusal::OutOfRange {
                bound: IntervalBound::Start,
                ..
            })
        ));
    }

    #[test]
    fn an_inverted_interval_is_refused_rather_than_reordered() {
        assert_eq!(
            AdmittedInterval::admit_millis(300, 200),
            Err(IntervalRefusal::Inverted {
                start_ms: 300,
                end_ms: 200
            })
        );
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
        assert_eq!(
            WordTiming::from_seconds(None, None),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming))
        );
        assert_eq!(
            WordTiming::from_seconds(None, Some(1.0)),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoStart))
        );
        assert_eq!(
            WordTiming::from_seconds(Some(1.0), None),
            Ok(WordTiming::Untimed(UntimedCause::ProviderReportedNoEnd))
        );
    }

    /// Tencent's shape: a word's absolute span is its segment's start plus its
    /// own offsets, and the sum is checked.
    #[test]
    fn segment_offsets_add_with_checked_arithmetic() {
        let interval =
            AdmittedInterval::admit_offset_from(4_850, 0, 200).expect("a real Tencent word");
        assert_eq!((interval.start_ms(), interval.end_ms()), (4_850, 5_050));

        assert!(matches!(
            AdmittedInterval::admit_offset_from(i64::MAX, 1, 2),
            Err(IntervalRefusal::OffsetOverflow { .. })
        ));
    }

    #[test]
    fn a_refusal_deep_in_the_pipeline_becomes_a_named_absence_not_a_zero() {
        let timing = WordTiming::admit_seconds_or_untimed(Some(f64::NAN), Some(1.0));
        assert_eq!(
            timing,
            WordTiming::Untimed(UntimedCause::RefusedByAdmission)
        );
        assert_eq!(timing.into_optional_millis(), (None, None));
    }

    proptest! {
        /// No pair of f64s, however hostile, produces an interval that is
        /// negative, inverted or out of range. This is the property the old
        /// `as i64` cast could not state: it always produced a number.
        #[test]
        fn admitted_intervals_are_always_ordered_and_in_range(
            start_s in proptest::num::f64::ANY,
            end_s in proptest::num::f64::ANY,
        ) {
            if let Ok(interval) = AdmittedInterval::admit_seconds(start_s, end_s) {
                prop_assert!(interval.start_ms() >= 0);
                prop_assert!(interval.end_ms() >= interval.start_ms());
                prop_assert!(interval.end_ms() <= AdmittedInterval::MAX_MS);
            }
        }

        /// Every value outside the admitted range is refused, never truncated
        /// into it. Generated across the whole f64 range including the
        /// subnormal, huge and negative regions.
        #[test]
        fn out_of_range_and_non_finite_values_are_always_refused(
            seconds in proptest::num::f64::ANY,
        ) {
            let admitted = AdmittedInterval::admit_seconds(seconds, seconds);
            let millis = seconds * 1000.0;
            #[allow(clippy::cast_precision_loss)]
            let admissible = millis.is_finite()
                && millis.round() >= 0.0
                && millis.round() <= AdmittedInterval::MAX_MS as f64;
            prop_assert_eq!(admitted.is_ok(), admissible);
        }

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

        /// Inverted pairs are refused for every magnitude, not merely the small
        /// ones a hand-written table would have covered.
        #[test]
        fn inverted_pairs_are_always_refused(
            start_ms in 1_i64..1_000_000_000_i64,
            drop_ms in 1_i64..1_000_000_000_i64,
        ) {
            let end_ms = (start_ms - drop_ms).max(0);
            if end_ms < start_ms {
                // Bound first: `prop_assert!` stringifies its argument into a
                // format string, where the `{ .. }` of a struct pattern would
                // be read as a placeholder.
                let refused = matches!(
                    AdmittedInterval::admit_millis(start_ms, end_ms),
                    Err(IntervalRefusal::Inverted { .. })
                );
                prop_assert!(refused);
            }
        }
    }
}
