//! Runtime-independent cooperative checkpoints for CPU-bound alignment.

use std::convert::Infallible;

/// The caller owns interruption; algorithms never reinterpret it as evidence.
pub trait WorkControl {
    /// An interruption is separate from a search-budget refusal.
    type Error;
    /// Stop at a deterministic unit of algorithmic work.
    fn check(&self) -> Result<(), Self::Error>;
}

/// Synchronous callers cannot be interrupted. The error is uninhabited.
pub struct Uninterrupted;

impl WorkControl for Uninterrupted {
    type Error = Infallible;
    fn check(&self) -> Result<(), Infallible> {
        Ok(())
    }
}

/// Eliminate an impossible interruption without a panic or fallback result.
pub fn uninterrupted<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

#[cfg(test)]
mod tests;
