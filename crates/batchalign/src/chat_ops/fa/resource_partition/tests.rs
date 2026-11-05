//! Declared-model policy tests; no real model or acoustic timing producer.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::test_support::{DeclaredModel, DeclaredPreparation};
use std::num::NonZeroU64;

use super::*;
use crate::chat_ops::fa::{
    coordinates::{Ms, Recording},
    utr::{UtteranceAnchors, WordAnchor},
};
use crate::types::worker_v2::{FaConstraintUnitV2, FaPreparationRefusalV2};

fn source(count: usize, end: u64) -> ResourceRequest {
    let recording = Recording::of_duration(Ms(end)).unwrap();
    ResourceRequest {
        utterance: UtteranceIdx::new(0),
        words: NonEmptyWords::from_vec(
            (0..count)
                .map(|index| FaWord {
                    utterance_index: UtteranceIdx::new(0),
                    utterance_word_index: WordIdx::new(index),
                    text: format!("w{index}"),
                })
                .collect(),
        )
        .unwrap(),
        window: FaWindow::within(&recording, FileMs::new(0), FileMs::new(end)).unwrap(),
    }
}

fn anchors(count: usize, points: &[(usize, u64, u64)]) -> UtteranceAnchors {
    UtteranceAnchors::fixture(
        count,
        points
            .iter()
            .map(|&(word, start, end)| WordAnchor::fixture(word, start, end))
            .collect(),
    )
    .unwrap()
}

fn finish(plan: PreparedResourceSplit<DeclaredPreparation>) -> ResourceSplit {
    plan.finish(|piece, prepared| {
        prepared.verify_for(piece);
    })
}

fn audio_model(ms: u64) -> DeclaredModel {
    DeclaredModel::AudioSamples {
        maximum: NonZeroU64::new(ms * 16).unwrap(),
    }
}

#[tokio::test]
async fn a_complete_twenty_second_request_needs_no_anchor_when_it_fits() {
    let split = finish(
        plan(
            source(4, 20_000),
            AnchorLookup::NotRecovered,
            &mut audio_model(60_000),
        )
        .await
        .unwrap(),
    );
    let parts: Vec<_> = split.pieces().collect();
    let [ResourcePart::Request(piece)] = parts.as_slice() else {
        panic!("one unchanged request")
    };
    assert_eq!(piece.words().len(), 4);
    assert_eq!(piece.window().len(), Ms(20_000));
}

#[tokio::test]
async fn missing_anchors_cannot_invent_a_cut_or_erase_source_words() {
    let split = finish(
        plan(
            source(4, 20_000),
            AnchorLookup::NotRecovered,
            &mut audio_model(10_000),
        )
        .await
        .unwrap(),
    );
    let parts: Vec<_> = split.pieces().collect();
    let [ResourcePart::Unavailable(piece)] = parts.as_slice() else {
        panic!("one explicit unavailable population")
    };
    assert_eq!(piece.words().len(), 4);
    assert!(matches!(
        piece.constraint(),
        FaResourceConstraintV2::Model {
            cause: FaPreparationRefusalV2::ModelConstraint {
                unit: FaConstraintUnitV2::AudioSamples,
                provided: 320_000,
                maximum: 160_000,
                ..
            }
        }
    ));
}

#[tokio::test]
async fn real_label_constraints_are_probed_without_assuming_shorter_inputs_fit() {
    let anchors = anchors(
        4,
        &[
            (0, 0, 1_000),
            (1, 1_000, 2_000),
            (2, 2_000, 3_000),
            (3, 3_000, 4_000),
        ],
    );
    let mut model = DeclaredModel::Labels {
        minimum: NonZeroU64::new(2).unwrap(),
        maximum: NonZeroU64::new(2).unwrap(),
    };
    let split = finish(
        plan(
            source(4, 4_000),
            AnchorLookup::Anchored(&anchors),
            &mut model,
        )
        .await
        .unwrap(),
    );
    let lengths: Vec<_> = split
        .pieces()
        .map(|part| match part {
            ResourcePart::Request(piece) => piece.words().len(),
            ResourcePart::Unavailable(_) => panic!("both two-label requests fit"),
        })
        .collect();
    assert_eq!(lengths, [2, 2]);
}

#[tokio::test]
async fn a_supported_tail_joins_the_following_request_without_guessing_the_unknown_prefix() {
    let anchors = anchors(
        6,
        &[
            (0, 0, 5_000),
            (2, 50_000, 51_000),
            (3, 52_000, 53_000),
            (4, 54_000, 55_000),
            (5, 56_000, 57_000),
        ],
    );
    let split = finish(
        plan(
            source(6, 57_000),
            AnchorLookup::Anchored(&anchors),
            &mut audio_model(10_000),
        )
        .await
        .unwrap(),
    );
    let parts: Vec<_> = split.pieces().collect();
    let [
        ResourcePart::Request(before),
        ResourcePart::Unavailable(gap),
        ResourcePart::Request(after),
    ] = parts.as_slice()
    else {
        panic!("usable neighbours survive the unknown gap")
    };
    assert_eq!(before.words().len(), 1);
    assert_eq!(
        gap.words()
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(
        after
            .words()
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        [2, 3, 4, 5]
    );
    assert_eq!(after.window().audio_start().get(), 50_000);
    assert_eq!(
        split
            .pieces()
            .flat_map(ResourcePart::words)
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );
}

#[tokio::test]
async fn borrowed_context_is_prepared_but_only_owned_words_are_injected() {
    let anchors = anchors(
        6,
        &[
            (0, 0, 1_000),
            (1, 4_000, 5_000),
            (2, 8_000, 9_000),
            (3, 12_000, 13_000),
            (4, 16_000, 17_000),
            (5, 20_000, 21_000),
        ],
    );
    let split = finish(
        plan(
            source(6, 21_000),
            AnchorLookup::Anchored(&anchors),
            &mut audio_model(15_000),
        )
        .await
        .unwrap(),
    );
    let parts: Vec<_> = split.pieces().collect();
    let [ResourcePart::Request(before), ResourcePart::Request(after)] = parts.as_slice() else {
        panic!("two feasible requests")
    };
    assert_eq!(before.words().len(), 4);
    assert_eq!(
        after
            .words()
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        [4, 5]
    );
    assert_eq!(
        after
            .request_words()
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        [2, 3, 4, 5]
    );
    assert_eq!(after.owned_range(), 2..4);
    assert_eq!(after.window().audio_start().get(), 8_000);
    assert_eq!(after.ownership_window().audio_start().get(), 13_000);
}

#[tokio::test]
async fn a_short_final_owned_word_can_borrow_context_to_meet_the_model_minimum() {
    let anchors = anchors(3, &[(0, 0, 1_000), (1, 1_000, 2_000), (2, 2_000, 3_000)]);
    let mut model = DeclaredModel::Labels {
        minimum: NonZeroU64::new(2).unwrap(),
        maximum: NonZeroU64::new(2).unwrap(),
    };
    let split = finish(
        plan(
            source(3, 3_000),
            AnchorLookup::Anchored(&anchors),
            &mut model,
        )
        .await
        .unwrap(),
    );
    assert_eq!(
        split
            .pieces()
            .map(|part| match part {
                ResourcePart::Request(piece) => piece.words().len(),
                ResourcePart::Unavailable(_) =>
                    panic!("the final word can borrow its preceding context"),
            })
            .collect::<Vec<_>>(),
        [2, 1]
    );
}
