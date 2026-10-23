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
    /// A deterministic model-load or package-import failure: the same worker
    /// configuration fails the same way again.
    Bootstrap,
    /// A model or catalog download failed in transit (the network, the
    /// remote host): the same worker configuration may succeed later. Not a
    /// bootstrap failure, so it neither withholds a fallback nor stops a
    /// retry that a deterministic failure would.
    Download,
    /// The accelerator ran out of memory (`torch.OutOfMemoryError`): the work
    /// is sound and may fit once other work releases memory.
    OutOfMemory,
    /// The worker refused the request as sent: the line was not JSON, the op
    /// or its `request` mapping was missing, or the payload failed the
    /// worker's request model. Sending the same request again is refused
    /// again.
    InvalidRequest,
}

impl WorkerErrorKind {
    /// Every kind, so the spelling table below is the only one.
    pub const ALL: [Self; 5] = [
        Self::Runtime,
        Self::Bootstrap,
        Self::Download,
        Self::OutOfMemory,
        Self::InvalidRequest,
    ];

    /// The wire spelling. Serialization and parsing both go through this one
    /// table, so the two directions cannot disagree.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Bootstrap => "bootstrap",
            Self::Download => "download",
            Self::OutOfMemory => "out_of_memory",
            Self::InvalidRequest => "invalid_request",
        }
    }

    /// What the worker does after writing an error line of this kind: the one
    /// owner of a decision both sides act on. The Python request loops read it
    /// (through the extension's `error_kind_ends_worker`) to decide whether to
    /// stop serving, and the Rust readers read it to decide whether the
    /// worker can be handed another request.
    ///
    /// A bootstrap or download failure may leave a model half loaded, so the
    /// worker stops serving: a stdio worker exits, and a TCP daemon closes the
    /// connection the error was written on.
    pub const fn worker_after(self) -> WorkerAfterError {
        match self {
            Self::Bootstrap | Self::Download => WorkerAfterError::Exits,
            Self::Runtime | Self::OutOfMemory | Self::InvalidRequest => WorkerAfterError::Serves,
        }
    }

    /// Parse a wire spelling; `None` for anything this protocol does not name.
    pub fn from_wire(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.wire_name() == text)
    }
}

/// Whether a worker that reported a failure is still serving requests on the
/// stream it reported it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerAfterError {
    /// The worker stopped serving after the report: a stdio worker exited, a
    /// TCP daemon closed the connection. A request sent to it again fails.
    Exits,
    /// The worker answered with a complete line and goes on serving.
    Serves,
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
