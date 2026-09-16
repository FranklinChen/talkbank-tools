//! Generic helpers for Rust-owned text commands that can run either per-file or
//! as one cross-file batch.
//!
//! This keeps the command-facing request/output shape consistent across commands
//! like `utseg`, `translate`, and `coref` while still allowing each command to
//! keep its own orchestration internals.

use crate::api::{DisplayPath, FileStampOutcome};
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

/// Why one item of a worker batch failed.
///
/// Typed rather than a bare string, because the two cases are different news
/// for the control plane and a string can say neither: an engine that reported
/// a failure has given its verdict on that item, while a result with nothing
/// to apply is an answer a re-run can change. Each variant states its own
/// category, so a new one has to decide rather than inherit.
/// Generic in the command's own failure, so a command that has none cannot
/// represent one: `S` is [`std::convert::Infallible`] for utseg, coref and
/// morphotag, whose only per-item failure is the engine's own report, and a
/// command-specific type (translate's empty translation) where one exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ItemFailure<S> {
    /// The engine reported this item's failure; its message travels verbatim
    /// (for example `"Translation failed: ConnectionResetError(...)"`).
    EngineReported(String),
    /// A failure this command defines, which the command's own type describes.
    Command(S),
}

/// The per-item failure of a command whose only failure is the engine's own.
///
/// `Infallible` has no value, so `Command` is uninhabited here: utseg, coref
/// and morphotag cannot construct a translate-shaped failure, and a match on
/// one needs no arm for it.
pub(crate) type EngineItemFailure = ItemFailure<std::convert::Infallible>;

impl<S: std::fmt::Display> std::fmt::Display for ItemFailure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EngineReported(message) => f.write_str(message),
            Self::Command(failure) => failure.fmt(f),
        }
    }
}

/// One per-item failure from a worker batch.
///
/// `item_index` is the position within the originating file's payload
/// list (0-based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemError<S> {
    /// Position in the originating file's payload list.
    pub item_index: usize,
    /// What went wrong with this item.
    pub failure: ItemFailure<S>,
}

/// One failure as the file-level error carries it: rendered, because the
/// file-level error outlives the command's own failure type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemErrorSample {
    /// Position in the originating file's payload list.
    pub item_index: usize,
    /// What went wrong, rendered.
    pub message: String,
}

/// The failures of one file, at least one.
///
/// A file is reported as failed only when something actually failed, and that
/// is this type's invariant rather than a check each call site repeats: the
/// constructor is the only route in, and it refuses an empty list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemFailures<S>(Vec<ItemError<S>>);

impl<S: std::fmt::Display> ItemFailures<S> {
    /// The failures of one file, or `None` when nothing failed.
    pub(crate) fn new(failures: Vec<ItemError<S>>) -> Option<Self> {
        (!failures.is_empty()).then_some(Self(failures))
    }

    /// The single failure of a file whose whole document is one item (coref).
    pub(crate) fn of_one(failure: ItemError<S>) -> Self {
        Self(vec![failure])
    }

    /// How many items failed.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// The first `MAX_ITEM_ERROR_SAMPLES` failures, rendered.
    fn samples(self) -> Vec<ItemErrorSample> {
        self.0
            .into_iter()
            .take(MAX_ITEM_ERROR_SAMPLES)
            .map(|error| ItemErrorSample {
                item_index: error.item_index,
                message: error.failure.to_string(),
            })
            .collect()
    }
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
        samples: Vec<ItemErrorSample>,
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
            // Per-item failures come from the provider, and every class we
            // have is the provider's settled answer about that item: an engine
            // that reported a failure, or one that returned a result with
            // nothing to apply. The same request produces the same answer, so
            // telling the control plane to expect a different one would only
            // buy another full run. A class that genuinely could differ on a
            // retry would have to say so here.
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

    /// Construct one per-item workflow error from the failures of a file.
    ///
    /// Caller passes every failure; this constructor caps the inline samples
    /// to ``MAX_ITEM_ERROR_SAMPLES`` while preserving the total count.
    pub(crate) fn item_errors<S: std::fmt::Display>(
        command: &'static str,
        failures: ItemFailures<S>,
    ) -> Self {
        let total = failures.len();
        Self::ItemErrors {
            command,
            total,
            samples: failures.samples(),
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
pub(crate) fn unwrap_per_item_results<R, S: std::fmt::Display>(
    command: &'static str,
    item_results: Vec<Result<R, ItemFailure<S>>>,
) -> Result<Vec<R>, TextWorkflowFileError> {
    let mut failures: Vec<ItemError<S>> = Vec::new();
    let mut successes: Vec<R> = Vec::with_capacity(item_results.len());
    for (idx, r) in item_results.into_iter().enumerate() {
        match r {
            Ok(r) => successes.push(r),
            Err(failure) => failures.push(ItemError {
                item_index: idx,
                failure,
            }),
        }
    }
    // The constructor decides whether anything failed: there is no separate
    // emptiness check here to disagree with it.
    match ItemFailures::new(failures) {
        None => Ok(successes),
        Some(failures) => Err(TextWorkflowFileError::item_errors(command, failures)),
    }
}

/// Render the inline summary for ``ItemErrors``: command, total
/// failed, then up to ``MAX_ITEM_ERROR_SAMPLES`` samples.
fn format_item_errors(command: &str, total: usize, samples: &[ItemErrorSample]) -> String {
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
    /// What the command decided about stamping this file with provenance.
    /// Carried to the writer, which records it on the file's status.
    pub stamp: FileStampOutcome,
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
            stamp: FileStampOutcome::Unrecorded,
            filename: filename.into(),
            result: Ok(output),
        }
    }

    /// Construct one successful named file result, recording what the command
    /// decided about stamping it.
    pub(crate) fn ok_stamped(
        filename: impl Into<DisplayPath>,
        output: PostValidated,
        stamp: FileStampOutcome,
    ) -> Self {
        Self {
            stamp,
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
            // A failed file is never written, so no stamp decision exists.
            stamp: FileStampOutcome::Unrecorded,
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

// The per-file workflow request, the cross-file workflow request, the
// `TextBatchOperation` trait and the `TextBatchWorkflow` wrapper used to live
// here. Every text command runs as a batch (the dispatchers hand the gateway
// one file at a time where per-file language or durability requires it), so
// the per-file half had no caller at all, and what was left was a one-method
// trait plus a zero-sized wrapper that forwarded to it. Each command's batch
// entry point now calls its own implementation directly.

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
            ItemFailures::of_one(ItemError {
                item_index: 0,
                failure: EngineItemFailure::EngineReported(
                    "Translation failed: ConnectionResetError(54)".to_string(),
                ),
            }),
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

    /// A command-specific failure, standing in for a real one (translate's
    /// empty translation) so this module's tests need not reach into a
    /// command's own types.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestCommandFailure;

    impl std::fmt::Display for TestCommandFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the engine returned nothing usable")
        }
    }

    /// An engine's own failure is terminal for the file: the same request gets
    /// the same answer.
    #[test]
    fn item_failures_stay_provider_terminal() {
        assert_eq!(
            TextWorkflowFileError::item_errors(
                "translate",
                ItemFailures::of_one(ItemError {
                    item_index: 0,
                    failure: EngineItemFailure::EngineReported("boom".into()),
                }),
            )
            .category(),
            FailureCategory::ProviderTerminal
        );
    }

    /// A command's own failure is terminal too, and the operator reads what it
    /// said. Calling it retryable would tell the control plane to expect a
    /// different answer to an identical request.
    #[test]
    fn a_command_failure_is_terminal_and_visible() {
        let failures = ItemFailures::new(vec![
            ItemError {
                item_index: 0,
                failure: ItemFailure::Command(TestCommandFailure),
            },
            ItemError {
                item_index: 1,
                failure: ItemFailure::EngineReported("429 Too Many Requests".into()),
            },
        ])
        .expect("two failures");
        let error = TextWorkflowFileError::item_errors("translate", failures);
        assert_eq!(error.category(), FailureCategory::ProviderTerminal);
        let rendered = error.to_string();
        assert!(
            rendered.contains("the engine returned nothing usable"),
            "the command failure must be visible to the operator, got: {rendered}"
        );
        assert!(
            rendered.contains("429 Too Many Requests"),
            "the engine's own words must survive too, got: {rendered}"
        );
    }

    /// Nothing failed, so no failure value exists to report.
    #[test]
    fn no_failures_cannot_become_a_file_failure() {
        assert_eq!(
            ItemFailures::<std::convert::Infallible>::new(Vec::new()),
            None
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
            (
                ServerError::EmptyTranscription(crate::error::EmptyTranscription::Asr),
                FailureCategory::Validation,
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
            ItemFailures::new(vec![
                ItemError {
                    item_index: 0,
                    failure: EngineItemFailure::EngineReported(
                        "Translation failed: ConnectionResetError".into(),
                    ),
                },
                ItemError {
                    item_index: 3,
                    failure: EngineItemFailure::EngineReported(
                        "Translation failed: 429 Too Many Requests".into(),
                    ),
                },
            ])
            .expect("two failures"),
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
        let failures: Vec<ItemError<std::convert::Infallible>> = (0..10)
            .map(|i| ItemError {
                item_index: i,
                failure: EngineItemFailure::EngineReported(format!("err {i}")),
            })
            .collect();
        let e = TextWorkflowFileError::item_errors(
            "morphotag",
            ItemFailures::new(failures).expect("ten failures"),
        );
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
