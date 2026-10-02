//! The failure kind an `{"op": "error", ...}` line carries: the line a
//! worker writes when it cannot answer a request with that request's own
//! response.
//!
//! One wire vocabulary for every emitter and every reader: the Rust-owned
//! request dispatcher (`batchalign-pyo3`), the Python request loops (which
//! build their envelopes through the same PyO3 function), and the three Rust
//! readers (the sequential stdio handle, the sequential TCP handle and the
//! shared GPU reader).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// What kind of failure an error line reports, and therefore what a retry
/// could change.
///
/// Required on the wire: every reader refuses an envelope without it, so an
/// emitter cannot make a deterministic refusal look retryable by saying
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerErrorKind {
    /// The request was admitted and the work failed. Another attempt may
    /// succeed (a transient resource state, an external service hiccup).
    Runtime,
    /// A deterministic model-load, catalog-download or package-import failure:
    /// the same worker configuration fails the same way again.
    Bootstrap,
    /// The worker refused the request as sent: the line was not JSON, the op
    /// or its `request` mapping was missing, or the payload failed the
    /// worker's request model. Sending the same request again is refused
    /// again.
    InvalidRequest,
}

impl WorkerErrorKind {
    /// Every kind, so the spelling table below is the only one.
    pub const ALL: [Self; 3] = [Self::Runtime, Self::Bootstrap, Self::InvalidRequest];

    /// The wire spelling. Serialization and parsing both go through this one
    /// table, so the two directions cannot disagree.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Bootstrap => "bootstrap",
            Self::InvalidRequest => "invalid_request",
        }
    }

    /// Parse a wire spelling; `None` for anything this protocol does not name.
    pub fn from_wire(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.wire_name() == text)
    }
}

impl std::fmt::Display for WorkerErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_name())
    }
}

impl Serialize for WorkerErrorKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.wire_name())
    }
}

impl<'de> Deserialize<'de> for WorkerErrorKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Self::from_wire(&text)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown worker error kind {text:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind round-trips through its wire spelling, and the parser
    /// refuses a spelling the table does not hold.
    #[test]
    fn kinds_round_trip_and_unknown_spellings_are_refused() -> Result<(), Box<dyn std::error::Error>>
    {
        for kind in WorkerErrorKind::ALL {
            let json = serde_json::to_string(&kind)?;
            assert_eq!(json, format!("\"{}\"", kind.wire_name()));
            let back: WorkerErrorKind = serde_json::from_str(&json)?;
            assert_eq!(back, kind);
        }
        assert!(serde_json::from_str::<WorkerErrorKind>("\"fatal\"").is_err());
        assert_eq!(WorkerErrorKind::from_wire("Runtime"), None);
        Ok(())
    }
}
