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
use crate::worker::error::WorkerError;
use crate::worker::pool::WorkerPool;
use tracing::warn;

use crate::error::ServerError;
use crate::runner::util::classify_worker_error;
use crate::types::scheduling::NextAttempt;

/// The cancellation authority available locally to an operation.
///
/// Not `Option<&CancellationToken>`: a `None` there reads identically
/// whether a call is jobless, belongs to an enclosing supervised file task,
/// or lost its caller's token. `NotWired` states the absence of a LOCAL token,
/// not the absence of job cancellation: its reason must identify the actual
/// owner or the jobless caller. A caller that already carries an authority
/// forwards it instead of inventing a permanently-uncancellable stand-in.
#[derive(Debug, Clone, Copy)]
pub enum Cancellation<'a> {
    /// A real job cancellation token. Cancelling it stops the dispatch.
    Token(&'a CancellationToken),
    /// This operation has no local cancellation token to observe.
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
    /// Run work under this cancellation authority. A stopped operation cannot
    /// yield an output; its future is dropped before any later work can use it.
    pub(crate) async fn supervise<T>(
        &self,
        operation: crate::owned_future::OwnedFuture<'_, T>,
    ) -> Result<T, ServerError> {
        tokio::select! {
            biased;
            () = self.cancelled() => Err(ServerError::Cancelled),
            output = operation => Ok(output),
        }
    }

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

    /// Sleep for `duration`, unless cancellation wins first.
    ///
    /// Cancellation is polled first (`biased`), so a token cancelled before
    /// the call returns at once, and the sleep is dropped rather than
    /// finished. Every wait a dispatch takes on a job's behalf goes through
    /// here, so none of them can outlive a stop.
    pub(crate) async fn sleep(&self, duration: Duration) -> Result<(), ServerError> {
        self.supervise(Box::pin(tokio::time::sleep(duration))).await
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
/// `cancellation` is a required parameter, not an optional add-on: the caller
/// explicitly names its local cancellation authority. A supplied token is raced
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
    retry_worker_operation(cancellation, "execute_v2", || {
        Box::pin(pool.dispatch_execute_v2_with_progress(lang, request, progress_tx))
    })
    .await
    .map_err(ServerError::from)
}

/// Why [`retry_worker_operation`] stopped without a result.
#[derive(Debug)]
pub(crate) enum RetryEnd {
    /// The last attempt failed in a way the policy does not retry, or the
    /// attempts ran out.
    Failed(WorkerError),
    /// The cancellation fired, during an attempt or a backoff.
    Cancelled,
}

impl From<RetryEnd> for ServerError {
    fn from(end: RetryEnd) -> Self {
        match end {
            RetryEnd::Failed(error) => Self::Worker(error),
            RetryEnd::Cancelled => Self::Cancelled,
        }
    }
}

/// Run one worker operation, retrying transient failures under the default
/// [`RetryPolicy`]: the one retry loop for worker operations.
///
/// Each failure is classified ([`classify_worker_error`]); a retryable one
/// within the attempt budget is logged and retried after the policy's
/// backoff, anything else ends the loop. `cancellation` is raced against
/// both the attempt and the backoff, so a cancelled job stops at once rather
/// than after another attempt or sleep (the 43-minute incident this module's
/// contract records). `what` names the operation in the log line.
///
/// Each attempt is an [`crate::owned_future::OwnedFuture`], the form
/// [`Cancellation::supervise`] races, boxed by the caller, so the loop is not
/// generic over each attempt's own future type.
pub(crate) async fn retry_worker_operation<'a, T, F>(
    cancellation: Cancellation<'_>,
    what: &'static str,
    mut attempt: F,
) -> Result<T, RetryEnd>
where
    F: FnMut() -> crate::owned_future::OwnedFuture<'a, Result<T, WorkerError>>,
{
    let retry_policy = RetryPolicy::default();
    let mut attempt_number = 1;
    loop {
        let outcome = cancellation
            .supervise(attempt())
            .await
            .map_err(|_| RetryEnd::Cancelled)?;
        let error = match outcome {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        let category = classify_worker_error(&error);
        match retry_policy.after_failure(attempt_number, category) {
            NextAttempt::RetryAfter(backoff_ms) => {
                warn!(
                    what,
                    attempt_number,
                    max_attempts = retry_policy.max_attempts,
                    error = %error,
                    category = %category,
                    %backoff_ms,
                    "Retrying after a transient worker failure"
                );
                cancellation
                    .sleep(Duration::from_millis(backoff_ms.0))
                    .await
                    .map_err(|_| RetryEnd::Cancelled)?;
                attempt_number += 1;
            }
            NextAttempt::Stop(_) => return Err(RetryEnd::Failed(error)),
        }
    }
}

#[cfg(test)]
// Test code: the panic-family lints are relaxed in source by house policy.
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{
        AsrBackendV2, AsrInputV2, AsrRequestV2, ExecuteRequestV2, InferenceTaskV2,
        PreparedAudioInputV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    use crate::worker::pool::{PoolConfig, WorkerPool};

    use super::*;

    /// A text-task request, the kind this retry loop carries in production.
    ///
    /// The tests that need a dispatch still in flight use it: a GPU request is
    /// refused before any worker is involved (GPU requests travel on their
    /// job's route, never through request-keyed dispatch), so an ASR request
    /// would resolve on its first poll.
    fn minimal_text_request() -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("infer-retry-test"),
            task: InferenceTaskV2::Morphosyntax,
            payload: TaskRequestV2::Morphosyntax(crate::types::worker_v2::MorphosyntaxRequestV2 {
                lang: LanguageCode3::eng(),
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                item_count: 1,
                retokenize: false,
            }),
            attachments: Vec::new(),
        }
    }

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
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let lang = LanguageCode3::eng();
        let request = minimal_asr_request();
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();

        let mut dispatch = Box::pin(dispatch_execute_v2_with_retry(
            &pool,
            &lang,
            &request,
            Cancellation::Token(&cancel_token),
        ));
        let result = futures::poll!(&mut dispatch);

        assert!(
            matches!(result, std::task::Poll::Ready(Err(ServerError::Cancelled))),
            "expected Err(ServerError::Cancelled), got {result:?}"
        );
    }

    /// [`Cancellation::NotWired`] can never produce `Err(Cancelled)`: its
    /// `cancelled()` future never resolves, so the dispatch runs to its
    /// own outcome (here, a worker failure, since `PoolConfig::for_tests()`
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

        let mut waiting = Box::pin(cancellation.cancelled());
        let outcome = futures::poll!(&mut waiting);

        assert!(
            matches!(outcome, std::task::Poll::Pending),
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
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let lang = LanguageCode3::eng();
        let request = minimal_text_request();

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
    /// attempt to finish on its own. `PoolConfig::for_tests()` with no
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
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let lang = LanguageCode3::eng();
        let request = minimal_text_request();
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

        let result = futures::poll!(&mut fut);

        assert!(
            matches!(result, std::task::Poll::Ready(Err(ServerError::Cancelled))),
            "expected Err(ServerError::Cancelled), got {result:?}"
        );
    }

    #[tokio::test]
    async fn supervising_cancellation_drops_suspended_work_without_an_output() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let token = CancellationToken::new();
        let cancellation = Cancellation::Token(&token);
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = Dropped(Arc::clone(&dropped));
        let work = async move {
            let _receipt = receipt;
            std::future::pending::<()>().await;
        };
        let mut supervised = Box::pin(cancellation.supervise(Box::pin(work)));
        assert!(matches!(
            futures::poll!(&mut supervised),
            std::task::Poll::Pending
        ));
        assert!(!dropped.load(Ordering::SeqCst));
        token.cancel();
        assert!(matches!(
            futures::poll!(&mut supervised),
            std::task::Poll::Ready(Err(ServerError::Cancelled))
        ));
        assert!(dropped.load(Ordering::SeqCst));
    }

    /// A crash, the classic transient worker failure.
    fn crash() -> WorkerError {
        WorkerError::ProcessExited {
            code: None,
            stderr: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_transient_failure_is_retried_until_it_succeeds() {
        let mut attempts = 0;
        let result =
            retry_worker_operation(Cancellation::NotWired { reason: "test" }, "test op", || {
                attempts += 1;
                let attempt = attempts;
                Box::pin(async move {
                    if attempt < 3 {
                        Err(crash())
                    } else {
                        Ok(attempt)
                    }
                })
            })
            .await;
        assert!(matches!(result, Ok(3)), "{result:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_final_failure_stops_at_once() {
        let mut attempts = 0;
        let result: Result<(), RetryEnd> =
            retry_worker_operation(Cancellation::NotWired { reason: "test" }, "test op", || {
                attempts += 1;
                Box::pin(async { Err(WorkerError::SpawnFailed("no interpreter".into())) })
            })
            .await;
        assert!(matches!(
            result,
            Err(RetryEnd::Failed(WorkerError::SpawnFailed(_)))
        ));
        assert_eq!(attempts, 1, "a terminal failure is not retried");
    }

    /// A job cancelled while an attempt runs ends the loop as `Cancelled`,
    /// with no retry: the cancellation wins the supervised race (it is
    /// checked first), so neither the failed attempt nor a backoff follows.
    #[tokio::test(start_paused = true)]
    async fn a_cancelled_job_ends_the_loop_without_another_attempt() {
        let token = CancellationToken::new();
        let mut attempts = 0;
        let result: Result<(), RetryEnd> =
            retry_worker_operation(Cancellation::Token(&token), "test op", || {
                attempts += 1;
                token.cancel();
                Box::pin(async { Err(crash()) })
            })
            .await;
        assert!(matches!(result, Err(RetryEnd::Cancelled)), "{result:?}");
        assert_eq!(attempts, 1);
    }
}
