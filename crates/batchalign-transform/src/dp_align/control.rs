//! Runtime-independent cooperative checkpoints for CPU-bound alignment.

use std::convert::Infallible;

/// The caller owns interruption; algorithms never reinterpret it as evidence.
pub trait WorkControl {
    /// An interruption is separate from a search-budget refusal.
    type Error;
    /// Stop at a deterministic unit of algorithmic work.
    fn check(&self) -> Result<(), Self::Error>;
    /// Scalar alignment operations, reserved before their execution. Direct
    /// alignment callers remain unmetered; correspondence inspection owns a
    /// private meter and distinguishes its refusal from caller interruption.
    fn alignment_work(&self, _units: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Peak simultaneously live scalar score cells, checked before allocation.
    fn alignment_score_storage(&self, _cells: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Read-only diagnostic from an admitted viable packed search. This is
    /// not correspondence or timing authority; observers must not block.
    fn packed_search_cost(&self, _receipt: &super::interleaving::PackedSearchCost) {}
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
