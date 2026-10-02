//! A media interval in whole milliseconds, admitted once: finite, ordered and
//! in range.
//!
//! Provider numbers reach an [`AdmittedInterval`] through its fallible
//! constructors only, and deserialization goes through the same check
//! (`try_from` a raw pair), so a value of this type anywhere in the workspace
//! is a proof that its bounds were rounded once and admitted against the
//! range policy here. The ASR post-processing pipeline (`batchalign-transform`'s
//! `asr_postprocess::timing`, which re-exports it) and the speaker
//! diarization wire type (`worker_v2::SpeakerSegmentV2`) share this one
//! ordered-interval type.

use serde::{Deserialize, Serialize};

use crate::domain::AudioPositionSeconds;

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

/// Why an ASR word or element carries no interval.
///
/// A closed set, because "untimed" with no reason is how a fabricated zero got
/// in: once the reason has to be named, there is nowhere to hide a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// The bounds were present but not admissible (inverted, or past
    /// [`AdmittedInterval::MAX_MS`]). The refusal itself is logged where the
    /// bounds were read; the word carries no time and says so.
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

/// A media interval in whole milliseconds, proven finite, ordered and in range.
///
/// Private fields, and every constructor is fallible (deserialization
/// included), so holding one of these is the proof. Zero length is ADMITTED here (a provider may legitimately
/// report a zero-width unit) and refused separately by the callers for whom a
/// zero-width span locates nothing (the transform crate's
/// `WordTiming::from_admitted`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(try_from = "IntervalMillis")]
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

    /// Admit a pair of audio POSITIONS.
    ///
    /// The one public seconds route: the bounds are already proven finite and
    /// non-negative where they were born ([`AudioPositionSeconds`]), so a
    /// position pair can still be refused only for its range or its order.
    pub fn admit_positions(
        start: AudioPositionSeconds,
        end: AudioPositionSeconds,
    ) -> Result<Self, IntervalRefusal> {
        Self::admit_seconds(start.get(), end.get())
    }

    /// Admit a pair of bounds given in raw fractional seconds. Private: a
    /// caller holding seconds holds positions, and goes through
    /// [`Self::admit_positions`]; the raw form is this module's rounding step
    /// and the subject of its hostile-input property tests.
    fn admit_seconds(start_s: f64, end_s: f64) -> Result<Self, IntervalRefusal> {
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

    /// The admitted start as an unsigned millisecond count: the same value
    /// as [`Self::start_ms`], which admission proved non-negative.
    pub const fn start_millis(self) -> u64 {
        self.start_ms.unsigned_abs()
    }

    /// The admitted end as an unsigned millisecond count.
    pub const fn end_millis(self) -> u64 {
        self.end_ms.unsigned_abs()
    }

    /// Admit a pair of unsigned millisecond bounds (a count read from a store
    /// that keeps them unsigned); a bound beyond `i64` is out of range.
    pub fn admit_unsigned_millis(start_ms: u64, end_ms: u64) -> Result<Self, IntervalRefusal> {
        #[allow(clippy::cast_precision_loss)]
        let in_i64 = |bound, value: u64| {
            i64::try_from(value).map_err(|_| IntervalRefusal::OutOfRange {
                bound,
                value_ms: value as f64,
            })
        };
        Self::admit_millis(
            in_i64(IntervalBound::Start, start_ms)?,
            in_i64(IntervalBound::End, end_ms)?,
        )
    }

    /// Whether the interval covers no time at all.
    ///
    /// A zero-width interval is a legal admission but locates nothing, so
    /// callers that need a span to point at audio treat it as untimed.
    pub const fn is_empty(self) -> bool {
        self.start_ms == self.end_ms
    }

    /// The admitted start and end as audio positions, for the provider shapes
    /// whose wire format speaks seconds.
    ///
    /// Infallible: both bounds are proven non-negative at admission, so
    /// `unsigned_abs` is the value itself, and every `u64` millisecond count
    /// is a valid position.
    pub fn as_positions(self) -> (AudioPositionSeconds, AudioPositionSeconds) {
        (
            AudioPositionSeconds::from_millis(self.start_millis()),
            AudioPositionSeconds::from_millis(self.end_millis()),
        )
    }
}

/// The wire form of an [`AdmittedInterval`]: the raw pair that deserialization
/// admits through [`AdmittedInterval::admit_millis`].
#[derive(Deserialize, schemars::JsonSchema)]
struct IntervalMillis {
    /// The start, in milliseconds from the start of the media.
    #[schemars(range(min = 0, max = 1_000_000_000_000_i64))]
    start_ms: i64,
    /// The end, in milliseconds; never before the start.
    #[schemars(range(min = 0, max = 1_000_000_000_000_i64))]
    end_ms: i64,
}

impl TryFrom<IntervalMillis> for AdmittedInterval {
    type Error = IntervalRefusal;

    fn try_from(raw: IntervalMillis) -> Result<Self, Self::Error> {
        Self::admit_millis(raw.start_ms, raw.end_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rounding_matches_the_half_away_from_zero_rule_it_replaced()
    -> Result<(), Box<dyn std::error::Error>> {
        // The pipeline's previous expression was `(seconds * 1000.0).round()`.
        // These are the boundary cases that distinguish rounding modes.
        for (start_s, end_s, expected) in [
            (0.0015_f64, 0.0025_f64, (2_i64, 3_i64)),
            (1.2345, 1.2355, (1235, 1236)),
            (0.0, 1.0, (0, 1000)),
        ] {
            let interval = AdmittedInterval::admit_seconds(start_s, end_s)?;
            assert_eq!((interval.start_ms(), interval.end_ms()), expected);
        }
        Ok(())
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
    /// Tencent's shape: a word's absolute span is its segment's start plus its
    /// own offsets, and the sum is checked.
    #[test]
    fn segment_offsets_add_with_checked_arithmetic() -> Result<(), Box<dyn std::error::Error>> {
        let interval = AdmittedInterval::admit_offset_from(4_850, 0, 200)?;
        assert_eq!((interval.start_ms(), interval.end_ms()), (4_850, 5_050));

        assert!(matches!(
            AdmittedInterval::admit_offset_from(i64::MAX, 1, 2),
            Err(IntervalRefusal::OffsetOverflow { .. })
        ));
        Ok(())
    }

    /// Deserialization admits through the same check as every constructor:
    /// a negative, inverted or out-of-range pair is refused, never held.
    #[test]
    fn deserialization_admits_or_refuses() -> Result<(), Box<dyn std::error::Error>> {
        let admitted: AdmittedInterval = serde_json::from_str(r#"{"start_ms":100,"end_ms":250}"#)?;
        assert_eq!((admitted.start_ms(), admitted.end_ms()), (100, 250));
        for refused in [
            r#"{"start_ms":-5,"end_ms":250}"#,
            r#"{"start_ms":300,"end_ms":250}"#,
            r#"{"start_ms":0,"end_ms":1789000000000}"#,
        ] {
            assert!(
                serde_json::from_str::<AdmittedInterval>(refused).is_err(),
                "{refused} must be refused"
            );
        }
        Ok(())
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
