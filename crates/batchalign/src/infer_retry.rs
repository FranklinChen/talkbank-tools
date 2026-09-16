//! Shared retry policy for worker inference calls.
//!
//! This helper keeps batch-oriented server orchestrators from immediately
//! degrading transient worker failures into terminal per-file errors. The retry
//! boundary is the actual worker request, not the higher-level per-file
//! orchestration that consumes its results.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::api::LanguageCode3;
use crate::scheduling::RetryPolicy;
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::pool::WorkerPool;
use tracing::warn;

use crate::error::ServerError;
use crate::runner::util::{classify_worker_error, is_retryable_worker_failure};

/// Whether a dispatch has a real job cancellation token to observe.
///
/// Not `Option<&CancellationToken>`: a `None` there reads identically
/// whether a call site genuinely has no job to cancel (a CLI direct path,
/// `compare`'s offline analysis) or simply never got wired up (the defect
/// this type replaces -- four call sites each passing a fresh
/// `CancellationToken::new()`, indistinguishable at every one of them from
/// a real absence). `NotWired` names the reason, so a caller admits which
/// case it is instead of fabricating a permanently-uncancellable stand-in
/// that looks, at the type level, exactly like a real one.
#[derive(Debug, Clone, Copy)]
pub enum Cancellation<'a> {
    /// A real job cancellation token. Cancelling it stops the dispatch.
    Token(&'a CancellationToken),
    /// This dispatch path has no job-level cancellation token to observe.
    /// `reason` names why, so a log line or panic message at this call
    /// site says which case it is rather than reading like a bug; also
    /// read by anything that wants to explain a stuck-looking dispatch
    /// (hence `pub`, not dead weight despite no reader yet in this crate).
    NotWired {
        /// Why this path has no token, e.g. `"compare-runs has no job"`.
        reason: &'static str,
    },
}

impl Cancellation<'_> {
    /// Wait for cancellation. [`Self::NotWired`] returns a future that
    /// never resolves (`std::future::pending`), so racing it in a
    /// `select!` simply means that branch never wins: the same runtime
    /// behavior a call site with no cancellation concept always had, now
    /// stated by the type rather than by a token nothing will ever cancel.
    async fn cancelled(&self) {
        match self {
            Self::Token(token) => token.cancelled().await,
            Self::NotWired { .. } => std::future::pending().await,
        }
    }
}

/// Dispatch one `execute_v2` request with automatic retries for transient worker
/// failures.
///
/// `cancellation` is mandatory: a retry loop with no cancellation signal in
/// its own signature is exactly how a stopped job kept dispatching for 43
/// minutes after cancellation, because nothing along its retry-and-backoff
/// path could ever observe the stop. See
/// [`dispatch_execute_v2_with_retry_and_progress`] for the full contract.
pub(crate) async fn dispatch_execute_v2_with_retry(
    pool: &WorkerPool,
    lang: &LanguageCode3,
    request: &ExecuteRequestV2,
    cancellation: Cancellation<'_>,
) -> Result<ExecuteResponseV2, ServerError> {
    dispatch_execute_v2_with_retry_and_progress(pool, lang, request, None, cancellation).await
}

/// Dispatch one `execute_v2` request with retries and progress forwarding.
///
/// `cancellation` is a required parameter, not an optional add-on: a retry
/// after cancellation is unconstructible by this signature. It is raced
/// against BOTH points where this loop can otherwise block indefinitely:
/// the attempt itself (`pool.dispatch_execute_v2_with_progress`, which can
/// be mid-decode on a long file) and the backoff sleep between attempts.
/// A cancellation observed at either point stops the loop immediately with
/// [`ServerError::Cancelled`] rather than completing one more attempt or
/// one more backoff first. [`Cancellation::NotWired`] can never produce
/// that outcome: its `cancelled()` future never resolves, so a `NotWired`
/// dispatch runs to its own success or failure exactly as it always did.
///
/// **Consequence of cancelling mid-attempt:** dropping the in-flight
/// `pool.dispatch_execute_v2_with_progress` future abandons a worker that
/// may have already written this request and not yet read its response.
/// The worker checkout carries that fact and is discarded rather than
/// returned to the idle pool when it is dropped in that state (see
/// `CheckedOutWorker`'s request-in-flight tracking); a cancelled attempt's
/// worker is never reused, so its eventual, unread response can never be
/// misread as the answer to a later, unrelated request.
pub(crate) async fn dispatch_execute_v2_with_retry_and_progress(
    pool: &WorkerPool,
    lang: &LanguageCode3,
    request: &ExecuteRequestV2,
    progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    cancellation: Cancellation<'_>,
) -> Result<ExecuteResponseV2, ServerError> {
    let retry_policy = RetryPolicy::default();

    for attempt_number in 1..=retry_policy.max_attempts {
        let attempt = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Err(ServerError::Cancelled),
            result = pool.dispatch_execute_v2_with_progress(lang, request, progress_tx) => result,
        };

        match attempt {
            Ok(response) => return Ok(response),
            Err(error) => {
                let category = classify_worker_error(&error);
                let has_retry_budget = attempt_number < retry_policy.max_attempts;

                if is_retryable_worker_failure(category) && has_retry_budget {
                    let backoff_ms = retry_policy.backoff_for_retry(attempt_number);
                    warn!(
                        task = ?request.task,
                        lang = %lang,
                        attempt_number,
                        max_attempts = retry_policy.max_attempts,
                        error = %error,
                        category = %category,
                        %backoff_ms,
                        "Retrying execute_v2 after transient worker failure"
                    );
                    tokio::select! {
                        biased;
                        () = cancellation.cancelled() => return Err(ServerError::Cancelled),
                        () = tokio::time::sleep(Duration::from_millis(backoff_ms.0)) => {}
                    }
                    continue;
                }

                return Err(ServerError::Worker(error));
            }
        }
    }

    // Loop invariant: every iteration either returns `Ok(...)` on
    // success, returns `Err(...)` on a non-retryable category, returns
    // `Err(Cancelled)` from either select above, or `continue`s after a
    // backoff. The terminal `continue` is gated by
    // `attempt_number < retry_policy.max_attempts`, so the for-loop bound
    // is exclusive. This `unreachable!` therefore covers the case where
    // the loop exits without taking any return path, which the bound
    // guarantees cannot happen.
    #[allow(clippy::unreachable)]
    {
        unreachable!("retry loop should return on success, cancellation, or terminal failure")
    }
}

#[cfg(test)]
// Test code: the panic-family lints are relaxed in source by house policy.
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Instant;

    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{
        AsrBackendV2, AsrInputV2, AsrRequestV2, ExecuteRequestV2, InferenceTaskV2,
        PreparedAudioInputV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    use crate::worker::pool::{PoolConfig, WorkerPool};

    use super::*;

    fn minimal_asr_request() -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("infer-retry-test"),
            task: InferenceTaskV2::Asr,
            payload: TaskRequestV2::Asr(AsrRequestV2 {
                lang: crate::api::WorkerLanguage::from(LanguageCode3::eng()),
                backend: AsrBackendV2::LocalWhisper,
                models: crate::types::worker_v2::AsrRequestedModelsV2::Whisper {
                    asr: crate::types::worker_v2::RequestedModelV2 {
                        id: crate::types::worker_v2::ModelIdV2::from_static(
                            "openai/whisper-large-v3",
                        ),
                        revision: crate::types::worker_v2::RequestedRevisionV2::Unpinned,
                    },
                },
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
            attachments: Vec::new(),
        }
    }

    /// A token cancelled before the call must stop the loop before it ever
    /// reaches the worker pool, and report the typed `Cancelled` outcome,
    /// not a worker failure or a silent hang.
    #[tokio::test]
    async fn cancelled_before_dispatch_returns_cancelled_without_attempting() {
        let pool = WorkerPool::new(PoolConfig::default());
        let lang = LanguageCode3::eng();
        let request = minimal_asr_request();
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();

        let started = Instant::now();
        let result = dispatch_execute_v2_with_retry(
            &pool,
            &lang,
            &request,
            Cancellation::Token(&cancel_token),
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            matches!(result, Err(ServerError::Cancelled)),
            "expected Err(ServerError::Cancelled), got {result:?}"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "a pre-cancelled call must return immediately, not after any real dispatch attempt \
             (took {elapsed:?})"
        );
    }

    /// [`Cancellation::NotWired`] can never produce `Err(Cancelled)`: its
    /// `cancelled()` future never resolves, so the dispatch runs to its
    /// own outcome (here, a worker failure, since `PoolConfig::default()`
    /// has no engine registered) rather than being interrupted. This is
    /// the outcome-type half of the reviewer's ask: a `NotWired` call
    /// cannot be cancelled, and the returned `Result` proves it by never
    /// being the `Cancelled` variant, however long the underlying dispatch
    /// takes or however it eventually fails.
    /// The direct, fast proof of the mechanism the retry loop's `select!`
    /// actually relies on: [`Cancellation::NotWired`]'s `cancelled()`
    /// future never resolves, so it can never win a race against the
    /// dispatch attempt. This is what makes `Err(ServerError::Cancelled)`
    /// unreachable for a `NotWired` dispatch, independent of how long or
    /// how the underlying attempt itself completes.
    #[tokio::test]
    async fn not_wired_cancelled_future_never_resolves() {
        let cancellation = Cancellation::NotWired {
            reason: "test: no job token in scope",
        };

        let outcome =
            tokio::time::timeout(Duration::from_millis(50), cancellation.cancelled()).await;

        assert!(
            outcome.is_err(),
            "Cancellation::NotWired.cancelled() must never resolve; a NotWired dispatch has \
             nothing that could fire it"
        );
    }

    /// Outcome-type-level companion to the test above: a full dispatch
    /// given `Cancellation::NotWired` must not resolve spuriously.
    /// Established by a DETERMINISTIC gate, not elapsed time: a future's
    /// first `poll` either returns the outcome or `Pending`, with no
    /// ambiguity and no dependence on how fast the runner schedules
    /// things. If `NotWired` could somehow still short-circuit the
    /// dispatch, this poll would already observe `Ready` instead of
    /// `Pending`.
    #[tokio::test]
    async fn not_wired_dispatch_is_not_spuriously_resolved() {
        let pool = WorkerPool::new(PoolConfig::default());
        let lang = LanguageCode3::eng();
        let request = minimal_asr_request();

        let mut fut = Box::pin(dispatch_execute_v2_with_retry(
            &pool,
            &lang,
            &request,
            Cancellation::NotWired {
                reason: "test: no job token in scope",
            },
        ));

        let first_poll = futures::poll!(&mut fut);
        assert!(
            matches!(first_poll, std::task::Poll::Pending),
            "a NotWired dispatch must not resolve (as Cancelled or otherwise) on its very \
             first poll; nothing here should be able to finish it early, got {first_poll:?}"
        );

        // The attempt genuinely never completes under this pool config (no
        // engine registered); drop rather than await so this test does
        // not itself hang.
        drop(fut);
    }

    /// A cancellation that fires while the FIRST attempt is still being
    /// awaited must win the race and stop the loop, not wait for that
    /// attempt to finish on its own. `PoolConfig::default()` with no
    /// engine registered makes the attempt itself take real, unbounded
    /// time (spawning/loading a worker that cannot succeed).
    ///
    /// "Still in flight" is established by a DETERMINISTIC gate the test
    /// controls, not by a sleep-then-check race: a single `poll` of the
    /// dispatch future either returns its outcome immediately or
    /// `Pending`. `Pending` after one poll proves the attempt is
    /// genuinely suspended (parked on the real dispatch, not merely
    /// not-yet-polled), independent of runner speed or scheduling, so the
    /// cancellation that follows is provably racing an IN-FLIGHT attempt
    /// rather than a call that had not started yet.
    #[tokio::test]
    async fn cancelled_during_the_first_attempt_stops_the_loop() {
        let pool = WorkerPool::new(PoolConfig::default());
        let lang = LanguageCode3::eng();
        let request = minimal_asr_request();
        let cancel_token = CancellationToken::new();

        let mut fut = Box::pin(dispatch_execute_v2_with_retry(
            &pool,
            &lang,
            &request,
            Cancellation::Token(&cancel_token),
        ));

        let first_poll = futures::poll!(&mut fut);
        assert!(
            matches!(first_poll, std::task::Poll::Pending),
            "the dispatch attempt must still be in flight (Pending on its first poll) before \
             cancellation for this test to prove anything, got {first_poll:?}"
        );

        cancel_token.cancel();

        let result = tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .expect("cancellation must stop the loop promptly, not hang");

        assert!(
            matches!(result, Err(ServerError::Cancelled)),
            "expected Err(ServerError::Cancelled), got {result:?}"
        );
    }
}
