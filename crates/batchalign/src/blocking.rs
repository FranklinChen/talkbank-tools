//! Blocking work that keeps the caller's tracing context.
//!
//! `tokio::task::spawn_blocking` runs its closure on a pool thread with no
//! span, so anything logged there loses the file and job the calling task was
//! working on (see `runner::current_file_span`). Rev.AI's language-detection
//! warnings and the native Whisper engine's messages were written that way,
//! naming no file in a batch of hundreds. [`spawn_in_span`] is the one route to
//! a blocking thread: it carries the current span across. The workspace's
//! `clippy.toml` disallows calling `spawn_blocking` directly, so a new call
//! site cannot quietly drop the context again.

use tokio::task::JoinHandle;

/// Run `work` on tokio's blocking pool inside the caller's tracing context:
/// its subscriber and its current span.
///
/// Same contract as `tokio::task::spawn_blocking` otherwise: the handle
/// resolves to the closure's value, or a join error if it panicked.
pub fn spawn_in_span<F, R>(work: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    // Both halves of the context: the subscriber the caller logs to (a pool
    // thread otherwise uses the global one, which differs under a scoped
    // subscriber) and the span it is in.
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    let span = tracing::Span::current();
    // The one sanctioned direct call; every other site comes through here.
    #[allow(clippy::disallowed_methods)]
    tokio::task::spawn_blocking(move || {
        tracing::dispatcher::with_default(&dispatch, || span.in_scope(work))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocking closure runs inside the span that was current when it was
    /// spawned, which a bare `spawn_blocking` does not do.
    #[tokio::test]
    async fn blocking_work_runs_in_the_callers_span() {
        let subscriber = tracing_subscriber::registry();
        let _default = tracing::subscriber::set_default(subscriber);
        let span = tracing::info_span!("file", file = "lecture.cha");
        let expected = span.id();
        let seen = {
            let _entered = span.enter();
            spawn_in_span(|| tracing::Span::current().id())
        }
        .await
        .expect("the blocking task completes");
        assert!(expected.is_some(), "premise: the span is enabled");
        assert_eq!(seen, expected);
    }
}
