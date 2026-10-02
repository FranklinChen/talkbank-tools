//! [`MachineTime`]: an instant written for programs to read.
//!
//! # The form, and why
//!
//! RFC 3339 in UTC with exactly three fractional digits:
//! `2026-09-28T16:00:00.000Z`. One spelling everywhere (the job API, the
//! cache database, the docs catalog), so no reader meets `Z` in one place and
//! `+00:00` in another. The fixed width matters: with a variable fraction,
//! `16:00:00Z` sorts after `16:00:00.5Z` as text, so anything sorting these
//! strings (the dashboard's job list does) would put a whole second after a
//! later fraction. Milliseconds are what the job store keeps.
//!
//! Reading accepts any RFC 3339 instant (`Z`, `+00:00`, any offset or
//! fraction), so values written before this type still read. A time without
//! an offset names no instant and is refused.
//!
//! For people, [`MachineTime::local_display`] renders the instant in the
//! viewer's local zone with its abbreviation (`2026-09-28 12:00:00 EDT`).

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use jiff::Timestamp;
use jiff::tz::TimeZone;
use serde::{Deserialize, Serialize};

/// An instant written for programs; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MachineTime(Timestamp);

impl MachineTime {
    /// The current instant, to the millisecond.
    pub fn now() -> Self {
        Self::from_timestamp(Timestamp::now())
    }

    /// `timestamp` truncated to the millisecond, the precision written.
    pub fn from_timestamp(timestamp: Timestamp) -> Self {
        // `constant` panics only out of range, and the whole milliseconds of
        // a valid instant are in range.
        let millis = timestamp.subsec_nanosecond() / 1_000_000 * 1_000_000;
        Self(Timestamp::constant(timestamp.as_second(), millis))
    }

    /// Unix seconds as the job store keeps them (`f64` in SQLite), rounded to
    /// the millisecond. Refused when the value names no instant: not finite,
    /// or outside the range of dates. This is the one door from stored
    /// seconds into a time.
    pub fn from_unix_seconds(seconds: f64) -> Result<Self, NotAnInstant> {
        let millis = (seconds * 1000.0).round();
        // `as` saturates, so range-check in floating point first.
        if !millis.is_finite() || millis.abs() >= i64::MAX as f64 {
            return Err(NotAnInstant(seconds));
        }
        Timestamp::from_millisecond(millis as i64)
            .map(Self)
            .map_err(|_| NotAnInstant(seconds))
    }

    /// The Unix seconds the job store writes (`f64` in SQLite); exact for a
    /// millisecond-precision instant. Private: only the SQLite column (and
    /// this module's tests) may turn a time back into a bare number.
    #[cfg(any(feature = "sqlx", test))]
    fn unix_seconds(self) -> f64 {
        self.0.as_millisecond() as f64 / 1000.0
    }

    /// `duration` later, saturating at the last representable instant.
    pub fn plus(self, duration: std::time::Duration) -> Self {
        Self::from_saturated_millis(
            self.0
                .as_millisecond()
                .saturating_add(whole_millis(duration)),
        )
    }

    /// `duration` earlier, saturating at the first representable instant.
    pub fn minus(self, duration: std::time::Duration) -> Self {
        Self::from_saturated_millis(
            self.0
                .as_millisecond()
                .saturating_sub(whole_millis(duration)),
        )
    }

    /// The time from `earlier` to `self`, or zero when `earlier` is later
    /// (as `std::time::Instant::saturating_duration_since`): how long has
    /// passed, or how long is left to wait.
    pub fn saturating_duration_since(self, earlier: Self) -> std::time::Duration {
        if self > earlier {
            self.0.duration_since(earlier.0).unsigned_abs()
        } else {
            std::time::Duration::ZERO
        }
    }

    /// An instant from milliseconds since the Unix epoch, clamped to the range
    /// of dates, so arithmetic never fails.
    fn from_saturated_millis(millis: i64) -> Self {
        let first = Timestamp::MIN.as_millisecond();
        let last = Timestamp::MAX.as_millisecond();
        let millis = millis.clamp(first, last);
        // In range by the clamp; `constant` panics only out of range. The
        // sub-second part is below 1e9, so it fits an `i32`.
        let nanos = (millis.rem_euclid(1000) * 1_000_000) as i32;
        Self(Timestamp::constant(millis.div_euclid(1000), nanos))
    }

    /// The instant.
    pub fn timestamp(self) -> Timestamp {
        self.0
    }

    /// For people: the viewer's local time with its zone abbreviation,
    /// `2026-09-28 12:00:00 EDT`.
    pub fn local_display(self) -> String {
        self.0
            .to_zoned(TimeZone::system())
            .strftime("%Y-%m-%d %H:%M:%S %Z")
            .to_string()
    }
}

impl fmt::Display for MachineTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.strftime("%Y-%m-%dT%H:%M:%S%.3fZ"))
    }
}

/// A duration in whole milliseconds, saturating at `i64::MAX` (a duration
/// that long reaches past the end of time anyway).
fn whole_millis(duration: std::time::Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

/// Unix seconds that name no instant: not finite, or outside the range of
/// dates. Carries the value, so a report can say what was stored.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} Unix seconds names no instant")]
pub struct NotAnInstant(pub f64);

/// A text that is not an RFC 3339 instant.
#[derive(Debug, thiserror::Error)]
#[error("not an RFC 3339 time with an offset: {text:?}")]
pub struct MachineTimeError {
    text: String,
    #[source]
    source: jiff::Error,
}

impl FromStr for MachineTime {
    type Err = MachineTimeError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        text.parse::<Timestamp>()
            .map(Self::from_timestamp)
            .map_err(|source| MachineTimeError {
                text: text.to_owned(),
                source,
            })
    }
}

impl Serialize for MachineTime {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for MachineTime {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(MachineTimeVisitor)
    }
}

/// Reads the text in place where the format lends it, so a time costs no
/// allocation per field.
struct MachineTimeVisitor;

impl serde::de::Visitor<'_> for MachineTimeVisitor {
    type Value = MachineTime;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an RFC 3339 time with an offset")
    }

    fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<MachineTime, E> {
        text.parse().map_err(E::custom)
    }
}

/// The schema of what `MachineTime` writes: one RFC 3339 string.
impl utoipa::PartialSchema for MachineTime {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::Type::String)
            .format(Some(utoipa::openapi::SchemaFormat::KnownFormat(
                utoipa::openapi::KnownFormat::DateTime,
            )))
            .description(Some(
                "RFC 3339 in UTC with exactly three fractional digits, so string order is \
                 time order.",
            ))
            .examples(["2026-09-28T16:00:00.000Z"])
            .into()
    }
}

impl utoipa::ToSchema for MachineTime {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("MachineTime")
    }
}

/// A SQLite column of Unix seconds (`REAL`), as the job store keeps its
/// times. Decoding goes through [`MachineTime::from_unix_seconds`], so a row
/// whose time names no instant fails to decode, naming the column, instead
/// of becoming a time; encoding writes [`MachineTime::unix_seconds`].
#[cfg(feature = "sqlx")]
mod sqlite_column {
    use sqlx::encode::IsNull;
    use sqlx::error::BoxDynError;
    use sqlx::sqlite::{Sqlite, SqliteArgumentsBuffer, SqliteTypeInfo, SqliteValueRef};
    use sqlx::{Decode, Encode, Type};

    use super::MachineTime;

    impl Type<Sqlite> for MachineTime {
        fn type_info() -> SqliteTypeInfo {
            <f64 as Type<Sqlite>>::type_info()
        }

        fn compatible(ty: &SqliteTypeInfo) -> bool {
            <f64 as Type<Sqlite>>::compatible(ty)
        }
    }

    impl Encode<'_, Sqlite> for MachineTime {
        fn encode_by_ref(&self, args: &mut SqliteArgumentsBuffer) -> Result<IsNull, BoxDynError> {
            <f64 as Encode<'_, Sqlite>>::encode(self.unix_seconds(), args)
        }
    }

    impl<'r> Decode<'r, Sqlite> for MachineTime {
        fn decode(value: SqliteValueRef<'r>) -> Result<Self, BoxDynError> {
            let seconds = <f64 as Decode<'r, Sqlite>>::decode(value)?;
            Ok(MachineTime::from_unix_seconds(seconds)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UTC, `Z`, exactly three fractional digits, whatever the sub-second part.
    #[test]
    fn written_in_utc_with_exactly_three_fractional_digits()
    -> Result<(), Box<dyn std::error::Error>> {
        for (text, written) in [
            ("2026-09-28T16:00:00Z", "2026-09-28T16:00:00.000Z"),
            ("2026-09-28T12:00:00.5-04:00", "2026-09-28T16:00:00.500Z"),
            (
                "2026-09-28T16:00:00.123456789+00:00",
                "2026-09-28T16:00:00.123Z",
            ),
        ] {
            assert_eq!(text.parse::<MachineTime>()?.to_string(), written, "{text}");
        }
        Ok(())
    }

    /// Fixed width keeps string order equal to time order, including a
    /// whole second against a fraction of the same second.
    #[test]
    fn string_order_is_time_order() -> Result<(), Box<dyn std::error::Error>> {
        let whole: MachineTime = "2026-09-28T16:00:00Z".parse()?;
        let half: MachineTime = "2026-09-28T16:00:00.5Z".parse()?;
        assert!(whole < half);
        assert!(whole.to_string() < half.to_string());
        Ok(())
    }

    /// The job store keeps Unix seconds as `f64`; they convert at millisecond
    /// precision.
    #[test]
    fn unix_seconds_convert_at_millisecond_precision() -> Result<(), Box<dyn std::error::Error>> {
        let at = MachineTime::from_unix_seconds(1_790_870_400.123_4)?;
        assert_eq!(at.to_string(), "2026-10-01T16:00:00.123Z");
        Ok(())
    }

    /// Older output (`+00:00`, any fraction) still reads, through serde too.
    #[test]
    fn json_writes_the_fixed_form_and_reads_older_spellings()
    -> Result<(), Box<dyn std::error::Error>> {
        let at: MachineTime = serde_json::from_str("\"2026-09-28T16:00:00.5+00:00\"")?;
        assert_eq!(serde_json::to_string(&at)?, "\"2026-09-28T16:00:00.500Z\"");
        Ok(())
    }

    /// A time without an offset names no instant.
    #[test]
    fn a_time_without_an_offset_is_refused() {
        assert!("2026-09-28T16:00:00".parse::<MachineTime>().is_err());
    }

    /// The job store's seconds read back exactly, to the millisecond.
    #[test]
    fn unix_seconds_round_trip_to_the_millisecond() -> Result<(), Box<dyn std::error::Error>> {
        let at = MachineTime::from_unix_seconds(1_790_870_400.123)?;
        assert_eq!(at.unix_seconds(), 1_790_870_400.123);
        assert_eq!(MachineTime::from_unix_seconds(at.unix_seconds())?, at);
        Ok(())
    }

    /// Seconds that name no instant are an error that carries them.
    #[test]
    fn unix_seconds_that_name_no_instant_are_refused_with_the_value() {
        assert!(matches!(
            MachineTime::from_unix_seconds(f64::NAN),
            Err(error) if error.to_string() == "NaN Unix seconds names no instant"
        ));
        assert!(MachineTime::from_unix_seconds(f64::INFINITY).is_err());
        assert!(MachineTime::from_unix_seconds(1e300).is_err());
    }

    /// Adding and subtracting durations saturates at the ends of time
    /// instead of failing.
    #[test]
    fn arithmetic_saturates_at_the_ends_of_time() -> Result<(), Box<dyn std::error::Error>> {
        let at = MachineTime::from_unix_seconds(1_790_870_400.0)?;
        let later = at.plus(std::time::Duration::from_millis(1_500));
        assert_eq!(later.unix_seconds() - at.unix_seconds(), 1.5);
        assert_eq!(later.minus(std::time::Duration::from_millis(1_500)), at);
        let far = at.plus(std::time::Duration::from_secs(u64::MAX));
        assert_eq!(far.plus(std::time::Duration::from_secs(1)), far);
        let early = at.minus(std::time::Duration::from_secs(u64::MAX));
        assert_eq!(early.minus(std::time::Duration::from_secs(1)), early);
        Ok(())
    }

    /// A wait or an elapsed time is never negative: zero when the earlier
    /// instant is in fact later.
    #[test]
    fn saturating_duration_since_is_zero_when_backwards() -> Result<(), Box<dyn std::error::Error>>
    {
        let at = MachineTime::from_unix_seconds(1_790_870_400.0)?;
        let later = at.plus(std::time::Duration::from_millis(1_500));
        assert_eq!(
            later.saturating_duration_since(at),
            std::time::Duration::from_millis(1_500)
        );
        assert_eq!(
            at.saturating_duration_since(later),
            std::time::Duration::ZERO
        );
        Ok(())
    }
}
