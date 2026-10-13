use super::*;
use crate::dp_align::interleaving::{LocalInterleaving, SpeakerTurn};
use crate::dp_align::{CorrespondenceAdmission, CorrespondenceAnalysis, MatchMode};
use std::cell::Cell;

struct StopAt(Cell<usize>);
impl WorkControl for StopAt {
    type Error = ();
    fn check(&self) -> Result<(), ()> {
        match self.0.get() {
            0 => Err(()),
            n => {
                self.0.set(n - 1);
                Ok(())
            }
        }
    }
}

#[test]
fn every_interleaved_checkpoint_propagates_cancellation() {
    let a = vec!["one".into(), "two".into()];
    let b = vec!["yes".into(), "one".into()];
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let reference = vec!["one".into(), "yes".into(), "two".into(), "one".into()];
    let counter = StopAt(Cell::new(usize::MAX));
    let complete =
        LocalInterleaving::observe_controlled(&turns, &reference, MatchMode::Exact, &counter)
            .unwrap()
            .unwrap();
    assert_eq!(complete.matched_words(), 4);
    let checks = usize::MAX - counter.0.get();
    assert!(checks > reference.len() * 3);
    for checkpoint in 0..checks {
        assert!(
            LocalInterleaving::observe_controlled(
                &turns,
                &reference,
                MatchMode::Exact,
                &StopAt(Cell::new(checkpoint))
            )
            .is_err()
        );
    }
}

#[test]
fn every_correspondence_checkpoint_propagates_cancellation() {
    let payload = vec!["same".into(), "other".into(), "same".into()];
    let reference = vec!["other".into(), "same".into(), "same".into()];
    for mode in [
        MatchMode::Exact,
        MatchMode::CaseInsensitive,
        MatchMode::Fuzzy { threshold: 0.85 },
    ] {
        let counter = StopAt(Cell::new(usize::MAX));
        let complete =
            CorrespondenceAnalysis::observe_controlled(&payload, &reference, mode, &counter)
                .unwrap();
        assert_eq!(
            complete.selected(),
            CorrespondenceAnalysis::observe(&payload, &reference, mode).selected()
        );
        for checkpoint in 0..usize::MAX - counter.0.get() {
            assert!(
                CorrespondenceAnalysis::observe_controlled(
                    &payload,
                    &reference,
                    mode,
                    &StopAt(Cell::new(checkpoint))
                )
                .is_err()
            );
        }
    }
}

#[test]
fn every_proof_only_checkpoint_propagates_cancellation() {
    let payload = vec!["same".into(), "other".into(), "same".into()];
    let reference = vec!["other".into(), "same".into(), "same".into()];
    for mode in [
        MatchMode::Exact,
        MatchMode::CaseInsensitive,
        MatchMode::Fuzzy { threshold: 0.85 },
    ] {
        let counter = StopAt(Cell::new(usize::MAX));
        let complete =
            CorrespondenceAdmission::observe_controlled(&payload, &reference, mode, &counter)
                .unwrap();
        let CorrespondenceAdmission::Complete(common) = complete else {
            panic!("tiny proof must complete");
        };
        let observation = CorrespondenceAnalysis::observe(&payload, &reference, mode);
        let CorrespondenceAdmission::Complete(paired) = observation.admission() else {
            panic!("tiny paired proof must complete");
        };
        assert_eq!(
            common.pairs().collect::<Vec<_>>(),
            paired.pairs().collect::<Vec<_>>()
        );
        let checks = usize::MAX - counter.0.get();
        assert!(checks > 0);
        for checkpoint in 0..checks {
            assert!(
                CorrespondenceAdmission::observe_controlled(
                    &payload,
                    &reference,
                    mode,
                    &StopAt(Cell::new(checkpoint))
                )
                .is_err()
            );
        }
    }
}

#[test]
fn hirschberg_can_stop_inside_a_row_not_only_between_recursions() {
    let payload = vec!["a".into(); 4096];
    let reference = vec!["b".into(); 4];
    // Entry, generic, recursion, row, then the first row's cell blocks.
    assert!(
        crate::dp_align::align_controlled(
            &payload,
            &reference,
            MatchMode::Exact,
            &StopAt(Cell::new(6))
        )
        .is_err()
    );
}
