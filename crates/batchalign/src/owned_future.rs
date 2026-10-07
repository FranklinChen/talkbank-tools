//! The one representation of a future owned on the heap by its producer.
//!
//! A future passed BY VALUE into an `async fn` is stored in the callee's state
//! machine (twice, per rust-lang/rust#62958), so a generic wrapper that takes
//! `impl Future` grows by the wrapped future's whole size, and every nested
//! wrapper grows again. On 2026-10-06 that pattern made the transcribe
//! pipeline's state machine ~140 KB and overflowed a production tokio worker's
//! 2 MiB stack. Wrappers that observe, supervise or cancel other work take an
//! [`OwnedFuture`] instead: the wrapped work costs one pointer in the wrapper,
//! and the compiler refuses an inline future at every call site.

use std::future::Future;
use std::pin::Pin;

/// A `Send` future owned on the heap, as wrappers receive the work they wrap.
pub(crate) type OwnedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
