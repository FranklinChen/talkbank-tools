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
use crate::pipeline::post_validate::{PostValidated, PostValidationFailure};
use crate::scheduling::FailureCategory;

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
    /// A failure the control plane has already classified, carried with its
    /// verdict.
    ///
    /// A refusal on validity grounds is one of these, built by
    /// [`TextWorkflowFileError::validation`], so the control plane hears "bad
    /// CHAT" rather than "the provider gave up". That used to be a
    /// `Validation(String)` variant of its own, which is exactly
    /// `Categorised(FailureCategory::Validation, _)` spelled a second time:
    /// two representations of one state, held in agreement by `category()`
    /// answering the same thing for both.
    ///
    /// The message is carried rendered rather than as a typed
    /// `ValidationError`, which is not `Eq`, and this enum is compared in
    /// tests.
    ///
    /// [`TextWorkflowFileError::from_server_error`] used to ask
    /// `classify_server_error` for a category and then throw all but one
    /// answer away, folding `MemoryPressure`, `ModelAccessDenied`, `System`
    /// and every retryable class into `Batch`, which reports
    /// `ProviderTerminal`. A retryable provider failure was therefore recorded
    /// as terminal and never retried, and an out-of-memory kill was reported
    /// as the provider giving up. The verdict now travels with the message
    /// instead of being recomputed or collapsed.
    #[error("{1}")]
    Categorised(FailureCategory, String),

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
    /// Construct one validity-grounds failure for this file from a message.
    ///
    /// Prefer the `From<PostValidationFailure>` conversion for an output-gate
    /// refusal; this constructor is for the pre-validation gates, which
    /// already build their own message.
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self::Categorised(FailureCategory::Validation, message.into())
    }

    /// Classify this failure for the control plane.
    ///
    /// Written as an exhaustive `match` with no catch-all so a new variant
    /// has to state its own category instead of inheriting a default. The
    /// call site that used to hardcode `ProviderTerminal` for every text
    /// failure is `execution::text_io::write_text_results`. There is
    /// deliberately no uncategorised variant left for a new call site to
    /// reach for.
    pub(crate) fn category(&self) -> FailureCategory {
        match self {
            // Already classified, by the control plane's own classifier or by
            // `Self::validation`; the verdict is carried, never re-derived
            // here. It replaced a `Batch(String)` variant that had no category
            // to carry and therefore answered `ProviderTerminal` for
            // everything, memory pressure and retryable provider failures
            // included.
            Self::Categorised(category, _) => *category,
            // Per-item engine failures come from the provider by definition.
            Self::ItemErrors { .. } => FailureCategory::ProviderTerminal,
        }
    }

    /// Attribute one orchestration error to this file, carrying the control
    /// plane's own classification verbatim.
    ///
    /// `classify_server_error` is the owner of "what kind of failure is this",
    /// so its answer is stored rather than matched on: the two-arm match that
    /// used to live here kept `Validation` and collapsed every other verdict
    /// into `Batch`, which reports `ProviderTerminal`. That silently retyped
    /// memory pressure, model-access denials, system errors and every
    /// retryable provider failure as a terminal provider failure, so the
    /// runner's retry policy could not see the cases it exists for.
    pub(crate) fn from_server_error(error: &ServerError) -> Self {
        Self::Categorised(
            crate::runner::util::classify_server_error(error),
            error.to_string(),
        )
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
}

impl ServerError {
    /// Carry one already-classified text-workflow failure, verdict included.
    ///
    /// Lives here, beside [`TextWorkflowFileError`], because the category must
    /// be that error's OWN answer: an `impl` in `error.rs` would take a
    /// category as a parameter and hand every call site a chance to pick the
    /// wrong one, which is precisely the defect this replaces.
    pub(crate) fn from_classified_failure(error: &TextWorkflowFileError) -> Self {
        Self::ClassifiedFailure {
            category: error.category(),
            message: error.to_string(),
        }
    }
}

impl From<PostValidationFailure> for TextWorkflowFileError {
    fn from(value: PostValidationFailure) -> Self {
        Self::validation(value.to_string())
    }
}

// There is deliberately no `From<String>` / `From<&str>`. A bare string cannot
// say what kind of failure it is, and while those conversions existed the
// natural thing to write at a call site produced `Batch`, hence
// `ProviderTerminal`, for failures that were nothing of the sort: a coref
// batch break and a translate language-resolution refusal both landed there.
// Every call site now names a constructor, which is where the category is
// stated.

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
    ///
    /// The success side is a [`PostValidated`] proof, not a bare string: the
    /// writer (`execution::text_io::write_text_results`) reads this field, so
    /// output that has not passed the post-validation gate has no route to
    /// disk.
    pub result: Result<PostValidated, TextWorkflowFileError>,
}

/// Cross-file outputs for one text workflow family.
pub(crate) type TextBatchFileResults = Vec<TextBatchFileResult>;

impl TextBatchFileResult {
    /// Construct one successful named file result from gate-proven output.
    ///
    /// Takes the proof by value, so the only way to report success for a file
    /// is to hold evidence that its output passed the post-validation gate
    /// (or that the command left the document untouched). See
    /// [`PostValidated`] for the full enumeration of routes to that proof.
    pub(crate) fn ok(filename: impl Into<DisplayPath>, output: PostValidated) -> Self {
        Self {
            filename: filename.into(),
            result: Ok(output),
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
    ///
    /// A plain `String`, not a newtype. It was an `OwnedChatText` wrapper with
    /// `new`, an infallible `From<String>`, `Display`, `Deref` and `AsRef` and
    /// no invariant to prove, so it forbade nothing and every borrow of it went
    /// through a `Deref` straight back to `str`. The BORROWED counterpart
    /// [`ChatText`] is a different matter: it is the parameter type at the
    /// workflow seams below, where it says which of several `&str` a caller is
    /// passing.
    pub chat_text: String,
}

impl TextBatchFileInput {
    /// Construct one named batch input from a filename and CHAT text.
    pub(crate) fn new(filename: impl Into<DisplayPath>, chat_text: impl Into<String>) -> Self {
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

    /// A file refused on validity grounds must reach the control plane as
    /// `Validation`, not as a provider failure. `write_text_results` used to
    /// hardcode `ProviderTerminal` for every text failure, which reported bad
    /// CHAT as "the provider gave up on this file".
    /// RED FIRST (review item 5): a per-item PROVIDER failure must reach the
    /// control plane as `ProviderTerminal`. `run_text_pipeline` rendered the
    /// typed `ItemErrors` into `ServerError::Validation(String)`, which
    /// `classify_server_error` answers with `Validation`: the retry policy
    /// never saw a provider failure, and the operator was told the CHAT was
    /// bad.
    #[test]
    fn a_per_item_provider_failure_stays_provider_terminal_through_server_error() {
        let items = TextWorkflowFileError::item_errors(
            "translate",
            vec![ItemError {
                item_index: 0,
                message: "Translation failed: ConnectionResetError(54)".to_string(),
            }],
        );
        assert_eq!(items.category(), FailureCategory::ProviderTerminal);

        let carried = ServerError::from_classified_failure(&items);
        assert_eq!(
            crate::runner::util::classify_server_error(&carried),
            FailureCategory::ProviderTerminal,
            "the provider verdict must survive the ServerError boundary"
        );
        assert_eq!(
            TextWorkflowFileError::from_server_error(&carried).category(),
            FailureCategory::ProviderTerminal,
            "and the round trip back to the per-file error must not lose it"
        );
        assert!(
            carried.to_string().contains("ConnectionResetError"),
            "the engine's own message must survive, got: {carried}"
        );
    }

    #[test]
    fn a_validation_failure_is_categorised_as_validation() {
        let e = TextWorkflowFileError::validation("utseg post-validation failed: ...");
        assert_eq!(e.category(), FailureCategory::Validation);
    }

    /// The converse, so the new variant cannot silently re-categorise the
    /// one that keeps its own answer: per-item engine failures stay
    /// `ProviderTerminal` because they come from the provider by definition.
    #[test]
    fn item_failures_stay_provider_terminal() {
        assert_eq!(
            TextWorkflowFileError::item_errors(
                "translate",
                vec![ItemError {
                    item_index: 0,
                    message: "boom".into(),
                }],
            )
            .category(),
            FailureCategory::ProviderTerminal
        );
    }

    /// RED FIRST: `from_server_error` must carry the control plane's OWN
    /// verdict for every class, not keep `Validation` and collapse the rest.
    ///
    /// The collapse is what this replaces: it reported memory pressure, a
    /// denied model download, a disk failure and a retryable provider error
    /// all as `ProviderTerminal`, so the runner's retry policy never saw the
    /// one case it exists for. The expectations here are read from
    /// `classify_server_error`, which is the owner.
    #[test]
    fn from_server_error_carries_every_category_verbatim() {
        let cases: Vec<(ServerError, FailureCategory)> = vec![
            (
                ServerError::Validation("morphotag post-validation failed".into()),
                FailureCategory::Validation,
            ),
            (
                ServerError::MemoryPressure("host is out of memory".into()),
                FailureCategory::MemoryPressure,
            ),
            (
                ServerError::ModelAccessDenied("gated repository".into()),
                FailureCategory::ModelAccessDenied,
            ),
            (
                ServerError::Persistence("disk full".into()),
                FailureCategory::System,
            ),
            (ServerError::Cancelled, FailureCategory::Cancelled),
        ];
        for (error, expected) in cases {
            let rendered = error.to_string();
            let attributed = TextWorkflowFileError::from_server_error(&error);
            assert_eq!(
                attributed.category(),
                expected,
                "wrong category for {rendered}"
            );
            assert_eq!(
                attributed.to_string(),
                rendered,
                "the message must survive the attribution"
            );
        }
    }

    /// A categorised failure renders as its bare message, exactly as the
    /// other variants do: the category is for the control plane, and the
    /// operator sees the engine's own words.
    #[test]
    fn categorised_error_renders_as_bare_message() {
        let e = TextWorkflowFileError::Categorised(
            FailureCategory::MemoryPressure,
            "worker killed: out of memory".into(),
        );
        assert_eq!(e.to_string(), "worker killed: out of memory");
        assert_eq!(e.category(), FailureCategory::MemoryPressure);
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
}
