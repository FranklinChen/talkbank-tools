//! Enrollment spans: the caller's claim that a stretch of audio holds one
//! known voice, alone.
//!
//! Everything a caller says about enrollment arrives as text on a command
//! line, and this module is the only place that text becomes a value. Past
//! this boundary an enrollment is a proven thing: its label is legal, its
//! window is non-empty, no two enrollments share a label, and no two overlap
//! in time.
//!
//! # Why "no two overlap" is refused here rather than reported later
//!
//! Two enrolled spans that overlap describe audio the caller has claimed
//! belongs to two different single speakers, which cannot both be true. There
//! is no useful thing to do with such a request, and the vectors it would
//! produce would each be contaminated by the other voice while still looking
//! like ordinary embeddings. Refusing at construction means no later stage has
//! to carry the possibility.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::media::window::{EmptyWindow, MediaWindow};
use crate::time::FileMs;

/// The name a caller gave one enrolled voice.
///
/// A newtype rather than a `String` because it is a closed vocabulary within a
/// run: the same set of names appears in the enrollment list, in every
/// verdict, and in the evidence file, and a stray free-form string in any of
/// those would be a label nobody enrolled.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EnrolledLabel(String);

/// A label a caller cannot have meant.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidLabel {
    /// The label was empty or only whitespace.
    #[error("an enrollment label cannot be empty")]
    Empty,
    /// The label held a character that would make the CLI form ambiguous.
    #[error(
        "the enrollment label {label:?} cannot contain a colon, a dash or whitespace, \
         because the enrollment form is <start_ms>-<end_ms>:<label>"
    )]
    Unparseable {
        /// The rejected label.
        label: String,
    },
}

impl EnrolledLabel {
    /// The single route from caller text to a label.
    ///
    /// The refused characters are exactly the ones that would make
    /// `<start_ms>-<end_ms>:<label>` ambiguous to read back. That is a real
    /// constraint rather than tidiness: an evidence file naming a label with a
    /// colon in it could not be re-issued as the enrollment that produced it.
    pub fn parse(value: &str) -> Result<Self, InvalidLabel> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(InvalidLabel::Empty);
        }
        if trimmed
            .chars()
            .any(|c| c == ':' || c == '-' || c.is_whitespace())
        {
            return Err(InvalidLabel::Unparseable {
                label: trimmed.to_owned(),
            });
        }
        Ok(Self(trimmed.to_owned()))
    }

    /// Borrow the label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EnrolledLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One stretch of the recording the caller says holds one known voice alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentSpec {
    label: EnrolledLabel,
    window: MediaWindow,
}

/// Why one enrollment argument could not be read.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InvalidEnrollment {
    /// The argument did not have the `<start_ms>-<end_ms>:<label>` shape.
    #[error(
        "{argument:?} is not an enrollment: write it as <start_ms>-<end_ms>:<label>, \
         for example 1500-9000:INV"
    )]
    Shape {
        /// The argument as given.
        argument: String,
    },
    /// One of the two millisecond bounds was not a whole number.
    #[error("{field} of the enrollment {argument:?} is not a whole number of milliseconds")]
    NotMilliseconds {
        /// Which bound.
        field: &'static str,
        /// The argument as given.
        argument: String,
    },
    /// The window held no audio.
    #[error("the enrollment {argument:?} spans no audio: {source}")]
    Empty {
        /// The argument as given.
        argument: String,
        /// The window's own complaint.
        #[source]
        source: EmptyWindow,
    },
    /// The label was not usable.
    #[error("the enrollment {argument:?} has an unusable label: {source}")]
    Label {
        /// The argument as given.
        argument: String,
        /// The label's own complaint.
        #[source]
        source: InvalidLabel,
    },
}

impl EnrollmentSpec {
    /// Read one `<start_ms>-<end_ms>:<label>` argument.
    ///
    /// The milliseconds are FILE milliseconds, the same space a CHAT bullet is
    /// written in, which is why they become [`FileMs`] here and never a bare
    /// integer that some later stage has to remember the origin of.
    pub fn parse(argument: &str) -> Result<Self, InvalidEnrollment> {
        let shape = || InvalidEnrollment::Shape {
            argument: argument.to_owned(),
        };
        let (span, label) = argument.split_once(':').ok_or_else(shape)?;
        let (start, end) = span.split_once('-').ok_or_else(shape)?;

        let start: u64 = start
            .trim()
            .parse()
            .map_err(|_| InvalidEnrollment::NotMilliseconds {
                field: "the start",
                argument: argument.to_owned(),
            })?;
        let end: u64 = end
            .trim()
            .parse()
            .map_err(|_| InvalidEnrollment::NotMilliseconds {
                field: "the end",
                argument: argument.to_owned(),
            })?;

        let window = MediaWindow::new(FileMs::new(start), FileMs::new(end)).map_err(|source| {
            InvalidEnrollment::Empty {
                argument: argument.to_owned(),
                source,
            }
        })?;
        let label = EnrolledLabel::parse(label).map_err(|source| InvalidEnrollment::Label {
            argument: argument.to_owned(),
            source,
        })?;
        Ok(Self { label, window })
    }

    /// The enrolled voice's name.
    #[must_use]
    pub const fn label(&self) -> &EnrolledLabel {
        &self.label
    }

    /// The stretch of recording it was enrolled from.
    #[must_use]
    pub const fn window(&self) -> MediaWindow {
        self.window
    }

    /// The canonical CLI spelling of this enrollment.
    ///
    /// Exact inverse of [`EnrollmentSpec::parse`], which is what lets a job
    /// row persist an enrollment as the argument that produced it and read it
    /// back through the SAME validation. Storing the parsed parts instead
    /// would give the persisted form its own route into the type, and a proof
    /// with two constructors is a label.
    #[must_use]
    pub fn as_argument(&self) -> String {
        format!(
            "{}-{}:{}",
            self.window.start().get(),
            self.window.end().get(),
            self.label
        )
    }
}

impl Serialize for EnrollmentSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.as_argument())
    }
}

impl<'de> Deserialize<'de> for EnrollmentSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let argument = String::deserialize(deserializer)?;
        Self::parse(&argument).map_err(serde::de::Error::custom)
    }
}

/// Every enrollment for one run: non-empty, uniquely labelled, non-overlapping.
///
/// Holding one is the proof of all three, so no later stage re-checks any of
/// them and no later stage can be handed a set that failed one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentSet {
    enrollments: Vec<EnrollmentSpec>,
}

/// Why a collection of enrollments is not usable as a set.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InvalidEnrollmentSet {
    /// Nothing was enrolled, so there is nothing to identify against.
    #[error("speaker identification needs at least one --enroll span")]
    Empty,
    /// Two enrollments claimed the same name.
    #[error("the enrollment label {label} was given twice; each label names one voice")]
    DuplicateLabel {
        /// The repeated label.
        label: EnrolledLabel,
    },
    /// Two enrollments claimed the same audio for different voices.
    #[error(
        "the enrollments {first} and {second} overlap between {overlap_start_ms}ms and \
         {overlap_end_ms}ms; the same audio cannot hold two different single speakers"
    )]
    Overlapping {
        /// The earlier enrollment's label.
        first: EnrolledLabel,
        /// The later enrollment's label.
        second: EnrolledLabel,
        /// Start of the overlap, in file milliseconds.
        overlap_start_ms: u64,
        /// End of the overlap, in file milliseconds.
        overlap_end_ms: u64,
    },
}

impl EnrollmentSet {
    /// The single route from a list of parsed enrollments to a usable set.
    pub fn new(mut enrollments: Vec<EnrollmentSpec>) -> Result<Self, InvalidEnrollmentSet> {
        if enrollments.is_empty() {
            return Err(InvalidEnrollmentSet::Empty);
        }

        let mut seen: BTreeSet<&EnrolledLabel> = BTreeSet::new();
        for enrollment in &enrollments {
            if !seen.insert(&enrollment.label) {
                return Err(InvalidEnrollmentSet::DuplicateLabel {
                    label: enrollment.label.clone(),
                });
            }
        }

        // Sorted by start so the overlap check is one linear pass and so the
        // evidence lists enrollments in recording order rather than in the
        // order the flags happened to be typed.
        enrollments.sort_by_key(|enrollment| enrollment.window.start());
        for pair in enrollments.windows(2) {
            let [earlier, later] = pair else { continue };
            if later.window.start() < earlier.window.end() {
                return Err(InvalidEnrollmentSet::Overlapping {
                    first: earlier.label.clone(),
                    second: later.label.clone(),
                    overlap_start_ms: later.window.start().get(),
                    overlap_end_ms: earlier.window.end().get().min(later.window.end().get()),
                });
            }
        }

        Ok(Self { enrollments })
    }

    /// The enrollments, in recording order.
    #[must_use]
    pub fn as_slice(&self) -> &[EnrollmentSpec] {
        &self.enrollments
    }

    /// The enrollment whose window overlaps `window`, if any (see below).
    ///
    /// Used to mark an utterance UNSCORED rather than scoring it: an utterance
    /// inside an enrollment span is the material the enrolled vector was
    /// computed from, so its similarity would measure the arithmetic and not
    /// the speaker.
    #[must_use]
    pub fn overlapping(&self, window: MediaWindow) -> Option<&EnrolledLabel> {
        self.enrollments
            .iter()
            .find(|enrollment| {
                window.start() < enrollment.window.end() && enrollment.window.start() < window.end()
            })
            .map(|enrollment| &enrollment.label)
    }
}

impl Serialize for EnrollmentSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.enrollments.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EnrollmentSet {
    /// Reading a persisted set goes through [`EnrollmentSet::new`], the same
    /// route a command line takes. A `#[derive(Deserialize)]` here would be a
    /// second constructor that skips the uniqueness and overlap checks, which
    /// is exactly how a proof type stops being one.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let enrollments = Vec::<EnrollmentSpec>::deserialize(deserializer)?;
        Self::new(enrollments).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    fn spec(argument: &str) -> EnrollmentSpec {
        match EnrollmentSpec::parse(argument) {
            Ok(spec) => spec,
            Err(error) => panic!("a legal enrollment: {error}"),
        }
    }

    /// The CLI form parses into file-coordinate values, once, here.
    #[test]
    fn the_enrollment_form_parses_into_a_window_and_a_label() {
        let parsed = spec("1500-9000:INV");
        assert_eq!(parsed.label().as_str(), "INV");
        assert_eq!(parsed.window().start().get(), 1500);
        assert_eq!(parsed.window().end().get(), 9000);
    }

    /// Every malformed shape is refused with a message naming the argument,
    /// never accepted with a guessed bound.
    #[test]
    fn malformed_enrollments_are_refused_rather_than_guessed() {
        assert!(EnrollmentSpec::parse("1500-9000").is_err());
        assert!(EnrollmentSpec::parse("1500:INV").is_err());
        assert!(EnrollmentSpec::parse("x-9000:INV").is_err());
        assert!(EnrollmentSpec::parse("1500-x:INV").is_err());
        assert!(EnrollmentSpec::parse("1500-9000:").is_err());
        assert!(EnrollmentSpec::parse("-1500-9000:INV").is_err());
    }

    /// A backwards or zero-length enrollment holds no audio and is refused by
    /// the window type, not by a check written here.
    #[test]
    fn an_enrollment_spanning_no_audio_is_refused() {
        assert!(EnrollmentSpec::parse("9000-1500:INV").is_err());
        assert!(EnrollmentSpec::parse("9000-9000:INV").is_err());
    }

    /// A label that would make the CLI form ambiguous is refused, so an
    /// evidence file can always be read back as the enrollment that made it.
    #[test]
    fn a_label_that_breaks_the_round_trip_is_refused() {
        assert!(EnrolledLabel::parse("").is_err());
        assert!(EnrolledLabel::parse("a:b").is_err());
        assert!(EnrolledLabel::parse("a-b").is_err());
        assert!(EnrolledLabel::parse("a b").is_err());
        assert!(EnrolledLabel::parse("INV").is_ok());
    }

    #[test]
    fn an_empty_enrollment_set_is_refused() {
        assert_eq!(
            EnrollmentSet::new(Vec::new()),
            Err(InvalidEnrollmentSet::Empty)
        );
    }

    #[test]
    fn two_enrollments_with_one_label_are_refused() {
        let set = EnrollmentSet::new(vec![spec("0-1000:INV"), spec("2000-3000:INV")]);
        assert!(matches!(
            set,
            Err(InvalidEnrollmentSet::DuplicateLabel { .. })
        ));
    }

    /// Overlapping enrollments describe audio claimed by two single speakers,
    /// which cannot both be true, so the set cannot be built.
    #[test]
    fn overlapping_enrollments_are_refused_with_the_overlap_named() {
        match EnrollmentSet::new(vec![spec("0-5000:INV"), spec("4000-9000:CHI")]) {
            Err(InvalidEnrollmentSet::Overlapping {
                overlap_start_ms,
                overlap_end_ms,
                ..
            }) => {
                assert_eq!(overlap_start_ms, 4000);
                assert_eq!(overlap_end_ms, 5000);
            }
            other => panic!("expected an overlap refusal, got {other:?}"),
        }
    }

    /// Abutting enrollments do not overlap: the window is half-open, so one
    /// ending exactly where the next begins shares no audio.
    #[test]
    fn abutting_enrollments_are_allowed() {
        assert!(EnrollmentSet::new(vec![spec("0-5000:INV"), spec("5000-9000:CHI")]).is_ok());
    }

    /// The set is held in recording order regardless of flag order, so two
    /// runs that enrolled the same spans produce comparable evidence.
    #[test]
    fn the_set_is_ordered_by_recording_position() {
        let set = match EnrollmentSet::new(vec![spec("5000-9000:CHI"), spec("0-4000:INV")]) {
            Ok(set) => set,
            Err(error) => panic!("a legal set: {error}"),
        };
        let labels: Vec<&str> = set
            .as_slice()
            .iter()
            .map(|enrollment| enrollment.label().as_str())
            .collect();
        assert_eq!(labels, ["INV", "CHI"]);
    }

    /// WIRE FORMAT: a persisted enrollment set reads back through the same
    /// construction a command line takes, so a job row cannot reconstitute a
    /// set that the CLI would have refused.
    #[test]
    fn a_persisted_set_roundtrips_through_its_own_validation() {
        let set = match EnrollmentSet::new(vec![spec("0-5000:INV"), spec("6000-9000:CHI")]) {
            Ok(set) => set,
            Err(error) => panic!("a legal set: {error}"),
        };
        let json = match serde_json::to_string(&set) {
            Ok(json) => json,
            Err(error) => panic!("a set serializes: {error}"),
        };
        assert_eq!(json, r#"["0-5000:INV","6000-9000:CHI"]"#);
        match serde_json::from_str::<EnrollmentSet>(&json) {
            Ok(read) => assert_eq!(read, set),
            Err(error) => panic!("a set deserializes: {error}"),
        }
        assert!(
            serde_json::from_str::<EnrollmentSet>(r#"["0-5000:INV","4000-9000:CHI"]"#).is_err(),
            "an overlapping persisted set must be refused on the way back in"
        );
        assert!(
            serde_json::from_str::<EnrollmentSet>("[]").is_err(),
            "an empty persisted set must be refused on the way back in"
        );
    }

    /// An utterance inside an enrolled span is found, so it can be reported
    /// unscored rather than scored against the vector it helped produce.
    #[test]
    fn an_utterance_inside_an_enrollment_is_found() {
        let set = match EnrollmentSet::new(vec![spec("0-5000:INV")]) {
            Ok(set) => set,
            Err(error) => panic!("a legal set: {error}"),
        };
        let inside = match MediaWindow::new(FileMs::new(1000), FileMs::new(2000)) {
            Ok(window) => window,
            Err(error) => panic!("a legal window: {error}"),
        };
        let after = match MediaWindow::new(FileMs::new(5000), FileMs::new(6000)) {
            Ok(window) => window,
            Err(error) => panic!("a legal window: {error}"),
        };
        assert_eq!(
            set.overlapping(inside).map(EnrolledLabel::as_str),
            Some("INV")
        );
        assert_eq!(set.overlapping(after), None);
    }
}
