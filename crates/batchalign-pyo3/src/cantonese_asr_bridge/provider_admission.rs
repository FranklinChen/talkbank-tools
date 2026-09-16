//! Admitting cloud-provider ASR payloads: which provider, where, and what was
//! wrong.
//!
//! Tencent and Aliyun both document every field of their result objects as
//! nullable ("注意：此字段可能返回 null，表示取不到有效值" on each property of
//! `SentenceDetail` and `SentenceWords`), and their Python SDKs initialize every
//! attribute to `None` before deserializing a response, so an absent field is
//! not hypothetical: it is the SDK's own resting state.
//!
//! Until 2026-09-16 this boundary read those fields with
//! `.ok().and_then(|value| value.extract().ok()).unwrap_or(0)`. Three different
//! facts, "the provider did not send this", "the provider sent something of the
//! wrong type", and "the provider sent zero", arrived as the same number, and
//! zero is a legal time and a legal speaker. A word whose offsets were missing
//! claimed to be spoken at the start of its segment; a segment whose `StartMs`
//! was missing claimed to start at the beginning of the recording. Nothing
//! downstream could tell those from measurements.
//!
//! The graph:
//!
//! ```text
//! Python attribute --FieldRead::attr--> Absent | Present(T) | WrongType
//!                  --required/optional-> T | Option<T> | ProviderAdmissionError
//! ProviderAdmissionError = { provider, locus, fault }
//! ```
//!
//! [`FieldRead`] is the ONLY extractor for these payloads. It has no `unwrap_or`
//! and no `Default`, so a caller must say what an absence means at the point
//! where it knows: an absent time becomes an untimed word (never a zero), while
//! an absent surface refuses the file.

use pyo3::prelude::*;

use crate::error::BatchalignBoundaryError;
use batchalign_transform::asr_postprocess::IntervalRefusal;

/// Which provider's payload failed admission.
///
/// A closed set rather than a string, because the message an operator reads
/// must name the service they have to go and look at, and because a new
/// provider should have to appear here before it can be refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderId {
    /// Tencent Cloud ASR.
    Tencent,
    /// Aliyun NLS ASR.
    Aliyun,
}

impl ProviderId {
    /// The provider's operator-facing name.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Tencent => "Tencent",
            Self::Aliyun => "Aliyun",
        }
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where in a provider payload a fault was found.
///
/// Positions, not contents: a refusal that says only "a word was malformed"
/// sends a reader through the whole response by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderLocus {
    /// One Tencent `ResultDetail` segment.
    Segment {
        /// Position of the segment in the response.
        index: usize,
    },
    /// One word inside a Tencent segment.
    SegmentWord {
        /// Position of the enclosing segment.
        segment: usize,
        /// Position of the word inside it.
        word: usize,
    },
    /// One word inside an Aliyun sentence.
    SentenceWord {
        /// Position of the enclosing sentence.
        sentence: usize,
        /// Position of the word inside it.
        word: usize,
    },
}

impl std::fmt::Display for ProviderLocus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Segment { index } => write!(f, "segment {index}"),
            Self::SegmentWord { segment, word } => write!(f, "segment {segment} word {word}"),
            Self::SentenceWord { sentence, word } => write!(f, "sentence {sentence} word {word}"),
        }
    }
}

/// What was wrong with the value at that position.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ProviderFault {
    /// The field was absent, or explicitly null, where a value is required.
    FieldAbsent {
        /// The provider's own name for the field.
        field: &'static str,
    },
    /// The field held a value of the wrong type.
    FieldWrongType {
        /// The provider's own name for the field.
        field: &'static str,
        /// What this boundary requires the field to be.
        expected: &'static str,
    },
    /// The field's value was present and well typed, but not an admissible
    /// time. Carries the interval owner's own refusal rather than restating it.
    Interval {
        /// The provider's own name for the field, or the pair of them.
        field: &'static str,
        /// Why the interval owner refused the value.
        refusal: IntervalRefusal,
    },
}

impl std::fmt::Display for ProviderFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldAbsent { field } => {
                write!(f, "{field} is absent, and this boundary requires a value")
            }
            Self::FieldWrongType { field, expected } => {
                write!(f, "{field} is not {expected}")
            }
            Self::Interval { field, refusal } => write!(f, "{field}: {refusal}"),
        }
    }
}

/// One provider payload refused, with everything needed to act on it.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("invalid {provider} ASR output at {locus}: {fault}")]
pub(super) struct ProviderAdmissionError {
    /// Whose payload it was.
    pub(super) provider: ProviderId,
    /// Where the fault was.
    pub(super) locus: ProviderLocus,
    /// What the fault was.
    pub(super) fault: ProviderFault,
}

impl ProviderAdmissionError {
    /// Build a refusal.
    pub(super) const fn new(
        provider: ProviderId,
        locus: ProviderLocus,
        fault: ProviderFault,
    ) -> Self {
        Self {
            provider,
            locus,
            fault,
        }
    }

    /// Raise this refusal across the FFI boundary.
    pub(super) fn into_py_err(self) -> PyErr {
        BatchalignBoundaryError::internal(self.to_string()).into_py_err()
    }
}

/// What reading one attribute of a provider object found.
///
/// Three states, because there are three facts. `Absent` covers both a missing
/// attribute and an explicit `None`: the SDKs use the latter for "this field
/// came back null", and no caller here needs to tell those apart. `WrongType`
/// is kept separate from `Absent` because they call for different actions: a
/// missing optional time is ordinary provider behaviour, while a time that is
/// a string is a contract change worth failing on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FieldRead<T> {
    /// The attribute is missing, or present and `None`.
    Absent,
    /// The attribute holds a value of the expected type.
    Present(T),
    /// The attribute holds a value of some other type.
    WrongType,
}

impl<T> FieldRead<T> {
    /// Read one attribute of a provider object.
    ///
    /// The ONLY route from a Python attribute to a Rust value on this path.
    /// Note what it does not have: no default, no `unwrap_or`, and no way to
    /// ask for "the value or zero".
    pub(super) fn attr<'py>(object: &Bound<'py, PyAny>, name: &str) -> Self
    where
        // Higher-ranked over the BORROW lifetime, not over `'py`. pyo3 0.29's
        // `FromPyObject<'a, 'py>` separates "how long the borrow of the source
        // object lives" from "how long the interpreter binding lives"; the
        // value read here is a local, so the borrow cannot be `'py`, while an
        // extracted `Bound<'py, _>` still belongs to the interpreter.
        T: for<'a> FromPyObject<'a, 'py>,
    {
        let Ok(value) = object.getattr(name) else {
            return Self::Absent;
        };
        if value.is_none() {
            return Self::Absent;
        }
        match value.extract::<T>() {
            Ok(value) => Self::Present(value),
            Err(_) => Self::WrongType,
        }
    }

    /// Require a value: absence and a wrong type are both refusals.
    pub(super) fn required(
        self,
        provider: ProviderId,
        locus: ProviderLocus,
        field: &'static str,
        expected: &'static str,
    ) -> Result<T, ProviderAdmissionError> {
        match self {
            Self::Present(value) => Ok(value),
            Self::Absent => Err(ProviderAdmissionError::new(
                provider,
                locus,
                ProviderFault::FieldAbsent { field },
            )),
            Self::WrongType => Err(ProviderAdmissionError::new(
                provider,
                locus,
                ProviderFault::FieldWrongType { field, expected },
            )),
        }
    }

    /// Admit an absence as a state, but still refuse a wrong type.
    ///
    /// This is the shape almost every provider TIME uses: the provider is
    /// entitled not to send one, and the caller turns that into an untimed
    /// word. It is not entitled to send a string where a number belongs.
    pub(super) fn optional(
        self,
        provider: ProviderId,
        locus: ProviderLocus,
        field: &'static str,
        expected: &'static str,
    ) -> Result<Option<T>, ProviderAdmissionError> {
        match self {
            Self::Present(value) => Ok(Some(value)),
            Self::Absent => Ok(None),
            Self::WrongType => Err(ProviderAdmissionError::new(
                provider,
                locus,
                ProviderFault::FieldWrongType { field, expected },
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::asr_postprocess::{AdmittedInterval, IntervalBound};

    /// A refusal names the provider, the position and the fault, because all
    /// three are what an operator needs to look at the right thing.
    #[test]
    fn a_refusal_names_provider_locus_and_fault() {
        let error = ProviderAdmissionError::new(
            ProviderId::Tencent,
            ProviderLocus::SegmentWord {
                segment: 3,
                word: 7,
            },
            ProviderFault::FieldAbsent { field: "Word" },
        );
        assert_eq!(
            error.to_string(),
            "invalid Tencent ASR output at segment 3 word 7: Word is absent, \
             and this boundary requires a value"
        );
    }

    /// An interval refusal is carried, not restated, so the reason a time was
    /// rejected has one owner.
    #[test]
    fn an_interval_refusal_is_reported_verbatim() {
        let refusal = AdmittedInterval::admit_millis(-1, 5).expect_err("negative start");
        let error = ProviderAdmissionError::new(
            ProviderId::Aliyun,
            ProviderLocus::SentenceWord {
                sentence: 0,
                word: 2,
            },
            ProviderFault::Interval {
                field: "startTime/endTime",
                refusal,
            },
        );
        let message = error.to_string();
        assert!(message.contains("Aliyun"), "{message}");
        assert!(message.contains("sentence 0 word 2"), "{message}");
        assert!(message.contains("negative"), "{message}");
        assert_eq!(
            refusal,
            IntervalRefusal::Negative {
                bound: IntervalBound::Start,
                value_ms: -1.0
            }
        );
    }
}
