//! Optional exact packed proof, admitted only after a work-refused literal band.
//! One input-bound operation owns both row representations and proof lifecycle.

use super::band::RefusedBand;
use super::*;
use crate::dp_align::packed_literal::{
    BIT_WORDS, LiteralReference, ReferenceOrder, policy, reverse_bits,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
};

mod geometry;
use geometry::{Geometry, Representation, work_refusal};
mod kernel;
mod profiles;
pub use kernel::cost::{PackedSearchCost, PackedSearchStage, PackedSearchStatus};

enum Stop<E> {
    Cancelled(E),
    Refused(LocalInterleavingRefusal),
}
trait PackedControl: WorkControl {
    fn charge(&self, units: usize) -> Result<(), Self::Error>;
    fn profile_width(&self, _words: usize) {}
    /// The stop a packed profile operation returns when its arithmetic
    /// contradicts the search's own invariants: checked in every build and
    /// refused, never only debug-asserted.
    fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error;
}
trait CandidateControl: PackedControl {
    fn candidate(&self) -> Result<(), Self::Error>;
}
#[cfg(test)]
impl PackedControl for Uninterrupted {
    fn charge(&self, _: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error {
        panic!("an uninterrupted test search contradicted itself: {inconsistency:?}")
    }
}

pub(super) struct Observation {
    pub(super) words: Vec<WordEvidence>,
    pub(super) selected: Vec<Selected>,
}
fn summary(bits: &[u64]) -> Vec<usize> {
    let mut total = 0;
    let mut result = Vec::with_capacity(bits.len() + 1);
    result.push(total);
    for bits in bits {
        total += bits.count_ones() as usize;
        result.push(total);
    }
    result
}
fn before(bits: &[u64], summary: &[usize], k: usize) -> usize {
    summary[k / 64]
        + if k.is_multiple_of(64) {
            0
        } else {
            (bits[k / 64] & ((1u64 << (k % 64)) - 1)).count_ones() as usize
        }
}

pub(super) fn observe<C: WorkControl + ?Sized>(
    refused: RefusedBand<'_, '_>,
    control: &C,
) -> Result<Result<Observation, LocalInterleavingRefusal>, C::Error> {
    kernel::observe(refused, control)
}

#[cfg(test)]
mod tests;
