//! Binding an answer to the request it answers.
//!
//! Answers come from three places: the derived cache layer, replayed raw
//! worker evidence, and a live worker reply. Each is untrusted until it is
//! bound to a request: one timing slot per request word, every timing inside
//! the request window and carrying it. [`bind_timings`] is that check, and the
//! only one.
//!
//! A cache answer travels unbound, as a [`UnitAnswer`], until whoever holds
//! the request binds it: the reply ticket, against its own unit (which is the
//! only way to make the `UnitResolution` the ledger assembles, see
//! `replies.rs`), or the supplied-retry path, against its own request. So an
//! answer bound for one request cannot be delivered for another.

use crate::chat_ops::fa::{GroupUnit, WordTiming};
use crate::types::traces::FaEvidenceSourceTrace;

/// A cached answer for a request, not yet bound to one.
#[derive(Debug, Clone)]
pub(in crate::fa) struct UnitAnswer {
    timings: Vec<Option<WordTiming>>,
    source: FaEvidenceSourceTrace,
}

impl UnitAnswer {
    /// An answer read from admitted cache evidence (or a test fixture).
    pub(in crate::fa) fn new(
        timings: Vec<Option<WordTiming>>,
        source: FaEvidenceSourceTrace,
    ) -> Self {
        Self { timings, source }
    }

    /// The timings and where they came from, for binding.
    pub(super) fn into_parts(self) -> (Vec<Option<WordTiming>>, FaEvidenceSourceTrace) {
        (self.timings, self.source)
    }

    /// Bind the answer to the supplied-retry request it is offered for, and
    /// hand it to that path; an answer that does not fit the request is no
    /// answer.
    pub(in crate::fa) fn into_supplied_retry(
        self,
        request: GroupUnit<'_>,
        key: &str,
    ) -> crate::chat_ops::fa::supplied_retry::SuppliedRetryAnswer {
        use crate::chat_ops::fa::supplied_retry::SuppliedRetryAnswer;
        match bind_timings(request, self.timings) {
            Ok(timings) => SuppliedRetryAnswer::Timings {
                timings,
                source: format!("cache key {key}; {:?}", self.source),
            },
            Err(refusal) => SuppliedRetryAnswer::Unresolved {
                reason: format!("cache key {key} does not answer its request: {refusal}"),
            },
        }
    }
}

/// Why an answer could not be bound to its request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(in crate::fa) enum ReplyRefusal {
    /// A timing fell outside the request window.
    #[error(
        "timing {start_ms}..{end_ms} outside its request window {window_start_ms}..{window_end_ms}"
    )]
    OutsideWindow {
        start_ms: u64,
        end_ms: u64,
        window_start_ms: u64,
        window_end_ms: u64,
    },
    /// Timings for another number of words than the request holds.
    #[error("{answered} timings for a request of {expected} words")]
    WordCount { answered: usize, expected: usize },
}

/// Bind timings to the request they claim to answer: one slot per request
/// word, every timing inside the request window, each then carrying it.
pub(super) fn bind_timings(
    request: GroupUnit<'_>,
    timings: Vec<Option<WordTiming>>,
) -> Result<Vec<Option<WordTiming>>, ReplyRefusal> {
    let expected = request.words().len();
    if timings.len() != expected {
        return Err(ReplyRefusal::WordCount {
            answered: timings.len(),
            expected,
        });
    }
    let window = request.window();
    timings
        .into_iter()
        .map(|timing| match timing {
            None => Ok(None),
            Some(timing) => timing.bind_request(window).map(Some).map_err(|refused| {
                ReplyRefusal::OutsideWindow {
                    start_ms: refused.start_ms,
                    end_ms: refused.end_ms,
                    window_start_ms: window.audio_start().get(),
                    window_end_ms: window.end().get(),
                }
            }),
        })
        .collect()
}
