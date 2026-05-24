//! Generic helpers for Rust-owned text commands that can run either per-file or
//! as one cross-file batch.
//!
//! This keeps the command-facing request/output shape consistent across commands
//! like `utseg`, `translate`, and `coref` while still allowing each command to
//! keep its own orchestration internals.

use std::marker::PhantomData;

use async_trait::async_trait;

use crate::api::{ChatText, DisplayPath, LanguageCode3};
use crate::error::ServerError;

/// Owned serialized CHAT text produced by a text workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OwnedChatText(String);

impl OwnedChatText {
    /// Wrap one owned CHAT string.
    pub(crate) fn new(text: String) -> Self {
        Self(text)
    }

    /// Consume into the underlying `String`.
    pub(crate) fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for OwnedChatText {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for OwnedChatText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::ops::Deref for OwnedChatText {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for OwnedChatText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Maximum number of per-item failure samples retained inline on a
/// ``TextWorkflowFileError::ItemErrors`` value.
///
/// We keep the typed structure bounded so a file with thousands of
/// failing items produces a Display string that is still readable
/// (and a SQLite/JSON record that is still small). The full count
/// is preserved on the `total` field so users still see how many
/// items failed.
pub(crate) const MAX_ITEM_ERROR_SAMPLES: usize = 5;

/// One per-item failure from a worker batch.
///
/// `item_index` is the position within the originating file's payload
/// list (0-based). `message` is the engine's error string captured
/// verbatim from the Python worker (e.g. ``"Translation failed:
/// ConnectionResetError(...)"``); a typed split into
/// network/model/protocol classes is deferred until we have a
/// downstream consumer that distinguishes them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemError {
    /// Position in the originating file's payload list.
    pub item_index: usize,
    /// Verbatim engine error string.
    pub message: String,
}

/// Per-file error emitted by a text workflow after file identity is
/// already known.
///
/// Typed variants replace the previous single-string shape so the
/// dashboard, CLI, and downstream tooling can distinguish between
/// per-item engine/network/model failures and batch-level failures
/// (worker spawn, IPC, pre/post-validation). The ``ItemErrors``
/// variant in particular fixes the system-wide silent-empty-response
/// bug where per-item engine failures used to be logged as warnings
/// and silently dropped instead of failing the file.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub(crate) enum TextWorkflowFileError {
    /// Batch-level failure with no per-item attribution: worker spawn
    /// failure, IPC error, schema mismatch, pre- or post-validation
    /// rejection, serialization error. Preserves the legacy stringly
    /// shape so existing callsites that build a free-form message keep
    /// working unchanged.
    #[error("{0}")]
    Batch(String),

    /// One or more per-item inferences failed. Affects exactly the
    /// file these items came from (other files in the same cross-file
    /// batch are unaffected). The first ``MAX_ITEM_ERROR_SAMPLES``
    /// failures are retained inline; the rest are counted in `total`.
    #[error("{}", format_item_errors(.command, *.total, .samples))]
    ItemErrors {
        /// Command label used in the rendered message (e.g. ``"translate"``).
        command: &'static str,
        /// Total number of items that failed across this file.
        total: usize,
        /// First ``MAX_ITEM_ERROR_SAMPLES`` failures, ordered by item
        /// index. Display renders these inline with the total count
        /// so the user sees a representative slice without overflowing
        /// the log.
        samples: Vec<ItemError>,
    },
}

impl TextWorkflowFileError {
    /// Construct one batch-level workflow error from a message.
    ///
    /// Used by callsites that build a free-form message (pre- and
    /// post-validation reporters, infer-batch outer errors).
    pub(crate) fn batch(message: impl Into<String>) -> Self {
        Self::Batch(message.into())
    }

    /// Construct one per-item workflow error from a list of failing
    /// items.
    ///
    /// Caller passes the full list; this constructor caps the inline
    /// samples to ``MAX_ITEM_ERROR_SAMPLES`` while preserving the
    /// total count.
    pub(crate) fn item_errors(command: &'static str, failures: Vec<ItemError>) -> Self {
        let total = failures.len();
        let samples = failures.into_iter().take(MAX_ITEM_ERROR_SAMPLES).collect();
        Self::ItemErrors {
            command,
            total,
            samples,
        }
    }

    /// Consume into the rendered message string.
    ///
    /// Kept for backward compatibility with downstream runner code that
    /// treats per-file errors as opaque strings. Internal callers that
    /// can match the variant directly should do so.
    pub(crate) fn into_message(self) -> String {
        self.to_string()
    }
}

impl From<String> for TextWorkflowFileError {
    fn from(value: String) -> Self {
        Self::batch(value)
    }
}

impl From<&str> for TextWorkflowFileError {
    fn from(value: &str) -> Self {
        Self::batch(value)
    }
}

/// Collapse a flat ``Vec<Result<R, String>>`` into either the
/// successful responses or a typed ``ItemErrors`` failure.
///
/// Used at every text-pipeline seam that needs to surface per-item
/// engine/network/model failures up as a single typed error rather
/// than silently dropping the empties. Returns the typed
/// ``TextWorkflowFileError`` directly so callers can choose whether
/// to attribute it to one file (in single-file flows) or wrap it in
/// ``ServerError`` (in API-boundary flows).
pub(crate) fn unwrap_per_item_results<R>(
    command: &'static str,
    item_results: Vec<Result<R, String>>,
) -> Result<Vec<R>, TextWorkflowFileError> {
    let mut failures: Vec<ItemError> = Vec::new();
    let mut successes: Vec<R> = Vec::with_capacity(item_results.len());
    for (idx, r) in item_results.into_iter().enumerate() {
        match r {
            Ok(r) => successes.push(r),
            Err(message) => failures.push(ItemError {
                item_index: idx,
                message,
            }),
        }
    }
    if failures.is_empty() {
        Ok(successes)
    } else {
        Err(TextWorkflowFileError::item_errors(command, failures))
    }
}

/// Render the inline summary for ``ItemErrors``: command, total
/// failed, then up to ``MAX_ITEM_ERROR_SAMPLES`` samples.
fn format_item_errors(command: &str, total: usize, samples: &[ItemError]) -> String {
    let suffix = if total > samples.len() {
        format!(" (showing first {} of {})", samples.len(), total)
    } else {
        String::new()
    };
    let detail = samples
        .iter()
        .map(|e| format!("item {}: {}", e.item_index, e.message))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "{command} failed for {total} item(s){suffix}: {detail}",
        command = command,
        total = total,
        suffix = suffix,
        detail = detail,
    )
}

/// Named per-file outcome for one text workflow batch.
#[derive(Debug, Clone)]
pub(crate) struct TextBatchFileResult {
    /// Stable file identity for this output or error.
    pub filename: DisplayPath,
    /// File-local workflow outcome.
    pub result: Result<OwnedChatText, TextWorkflowFileError>,
}

/// Cross-file outputs for one text workflow family.
pub(crate) type TextBatchFileResults = Vec<TextBatchFileResult>;

impl TextBatchFileResult {
    /// Construct one successful named file result.
    pub(crate) fn ok(filename: impl Into<DisplayPath>, text: impl Into<OwnedChatText>) -> Self {
        Self {
            filename: filename.into(),
            result: Ok(text.into()),
        }
    }

    /// Construct one failed named file result.
    pub(crate) fn err(
        filename: impl Into<DisplayPath>,
        error: impl Into<TextWorkflowFileError>,
    ) -> Self {
        Self {
            filename: filename.into(),
            result: Err(error.into()),
        }
    }
}

/// Owned named input for one CHAT file in a batch workflow.
#[derive(Debug, Clone)]
pub(crate) struct TextBatchFileInput {
    /// Stable file identity for this input.
    pub filename: DisplayPath,
    /// Owned serialized CHAT document for this file.
    pub chat_text: OwnedChatText,
}

impl TextBatchFileInput {
    /// Construct one named batch input from a filename and CHAT text.
    pub(crate) fn new(
        filename: impl Into<DisplayPath>,
        chat_text: impl Into<OwnedChatText>,
    ) -> Self {
        Self {
            filename: filename.into(),
            chat_text: chat_text.into(),
        }
    }
}

/// Borrowed request bundle for one per-file text workflow execution.
pub(crate) struct TextPerFileWorkflowRequest<'a, Shared, Params> {
    /// CHAT text to process.
    pub chat_text: ChatText<'a>,
    /// Primary language shaping the text workflow.
    pub lang: &'a LanguageCode3,
    /// Shared context owned by the workflow family.
    pub shared: Shared,
    /// Command-specific parameters for this execution.
    pub params: Params,
}

/// Borrowed request bundle for one cross-file text workflow execution.
pub(crate) struct TextBatchWorkflowRequest<'a, Shared, Params> {
    /// Files and their CHAT text payloads.
    pub files: &'a [TextBatchFileInput],
    /// Primary language shaping the text workflow.
    pub lang: &'a LanguageCode3,
    /// Shared context owned by the workflow family.
    pub shared: Shared,
    /// Command-specific parameters shared across the batch.
    pub params: Params,
}

/// Command-specific behavior for a Rust-owned text workflow family.
#[async_trait]
pub(crate) trait TextBatchOperation {
    /// Shared context threaded through this workflow family.
    type Shared<'a>: Send
    where
        Self: 'a;

    /// Command-specific parameters threaded through the workflow.
    type Params<'a>: Send
    where
        Self: 'a;

    /// Run the command for one CHAT file.
    async fn run_single(
        chat_text: ChatText<'_>,
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> Result<String, ServerError>;

    /// Run the command over a batch of CHAT files.
    async fn run_batch(
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> TextBatchFileResults;
}

/// Generic wrapper around one [`TextBatchOperation`] implementation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TextBatchWorkflow<O>(PhantomData<O>);

impl<O> TextBatchWorkflow<O> {
    /// Construct the zero-sized workflow wrapper.
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O> TextBatchWorkflow<O>
where
    O: TextBatchOperation + Send + Sync + 'static,
{
    /// Run one per-file text workflow.
    pub(crate) async fn run_per_file<'a>(
        &self,
        request: TextPerFileWorkflowRequest<'a, O::Shared<'a>, O::Params<'a>>,
    ) -> Result<String, ServerError> {
        O::run_single(
            request.chat_text,
            request.lang,
            request.shared,
            request.params,
        )
        .await
    }

    /// Run one cross-file text workflow.
    pub(crate) async fn run_batch_files<'a>(
        &self,
        request: TextBatchWorkflowRequest<'a, O::Shared<'a>, O::Params<'a>>,
    ) -> TextBatchFileResults {
        O::run_batch(request.files, request.lang, request.shared, request.params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_error_renders_as_bare_message() {
        let e = TextWorkflowFileError::batch("worker spawn failed: oom");
        assert_eq!(e.to_string(), "worker spawn failed: oom");
    }

    #[test]
    fn batch_error_round_trips_via_from_string() {
        let e: TextWorkflowFileError = "boom".to_owned().into();
        match e {
            TextWorkflowFileError::Batch(s) => assert_eq!(s, "boom"),
            other => panic!("expected Batch variant, got: {other:?}"),
        }
    }

    #[test]
    fn item_errors_renders_command_and_total() {
        let e = TextWorkflowFileError::item_errors(
            "translate",
            vec![
                ItemError {
                    item_index: 0,
                    message: "Translation failed: ConnectionResetError".into(),
                },
                ItemError {
                    item_index: 3,
                    message: "Translation failed: 429 Too Many Requests".into(),
                },
            ],
        );
        let msg = e.to_string();
        assert!(
            msg.starts_with("translate failed for 2 item(s)"),
            "got {msg}"
        );
        assert!(msg.contains("item 0: Translation failed: ConnectionResetError"));
        assert!(msg.contains("item 3: Translation failed: 429 Too Many Requests"));
        // Two samples ≤ MAX_ITEM_ERROR_SAMPLES, so no truncation suffix.
        assert!(!msg.contains("showing first"), "got {msg}");
    }

    #[test]
    fn item_errors_caps_inline_samples_but_preserves_total() {
        let failures: Vec<ItemError> = (0..10)
            .map(|i| ItemError {
                item_index: i,
                message: format!("err {i}"),
            })
            .collect();
        let e = TextWorkflowFileError::item_errors("morphotag", failures);
        match &e {
            TextWorkflowFileError::ItemErrors { total, samples, .. } => {
                assert_eq!(*total, 10);
                assert_eq!(samples.len(), MAX_ITEM_ERROR_SAMPLES);
            }
            other => panic!("expected ItemErrors variant, got: {other:?}"),
        }
        let msg = e.to_string();
        assert!(
            msg.contains("showing first 5 of 10"),
            "expected truncation marker, got: {msg}"
        );
    }

    #[test]
    fn into_message_renders_via_display() {
        let e = TextWorkflowFileError::batch("plain message");
        assert_eq!(e.into_message(), "plain message");
    }
}
