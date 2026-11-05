#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[tokio::test]
async fn source_retry_retains_unused_owner_and_cannot_repeat_selected_input() {
    let mut registry = PreparationRegistry::new();
    let mut selected = SelectedPreparations::new();
    for key in ["exact-source", "padded-primary"] {
        let ResourceProbeOutcome::Prepared(reference) = registry
            .prepare(scope(key), |_| async {
                Ok::<_, Infallible>(ResourceProbeOutcome::Prepared(key.to_owned()))
            })
            .await
            .unwrap()
        else {
            panic!("prepared control")
        };
        if key == "padded-primary" {
            selected.select(reference);
        }
    }
    let (primary, residual) = registry
        .freeze_retaining(
            selected,
            [
                (scope("exact-source"), "exact-source"),
                (scope("padded-primary"), "padded-primary"),
            ]
            .into_iter()
            .collect(),
        )
        .unwrap();
    assert_eq!(primary.len(), 1);
    let outcomes: HashMap<_, _> = residual.into_scoped().collect();
    assert!(
        matches!(outcomes.get("exact-source"), Some(RetainedPreparation::Ready(value)) if value == "exact-source")
    );
    assert!(matches!(
        outcomes.get("padded-primary"),
        Some(RetainedPreparation::Settled)
    ));
}

fn scope(key: &str) -> RequestScope {
    RequestScope {
        semantic_key: key.into(),
        word_ids: vec!["u0w0".into()],
    }
}

#[tokio::test]
async fn repeated_scope_constructs_and_runs_the_preparation_factory_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = PreparationRegistry::new();
    let mut selected = SelectedPreparations::new();
    let mut references = Vec::new();
    for _ in 0..3 {
        let calls = Arc::clone(&calls);
        let ResourceProbeOutcome::Prepared(reference) = registry
            .prepare(scope("same-input"), move |_permit| {
                calls.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok::<_, Infallible>(ResourceProbeOutcome::Prepared(
                        "owned-input-artifacts".to_owned(),
                    ))
                }
            })
            .await
            .unwrap()
        else {
            panic!("declared successful preparation")
        };
        selected.select(reference);
        references.push(reference);
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "even the expensive factory may only be constructed once"
    );
    assert_eq!(references[0], references[1]);
    assert_eq!(references[1], references[2]);
    assert_eq!(
        registry.freeze(selected).unwrap().len(),
        1,
        "one owner serves every projection of the same input"
    );
}

#[tokio::test]
async fn changed_input_is_a_new_preparation_and_unselected_inputs_are_dropped() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = PreparationRegistry::new();
    let mut selected = SelectedPreparations::new();
    for key in ["first-window", "changed-window"] {
        let calls = Arc::clone(&calls);
        let ResourceProbeOutcome::Prepared(reference) = registry
            .prepare(scope(key), move |_permit| {
                calls.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok::<_, Infallible>(ResourceProbeOutcome::Prepared("declared-input".to_owned()))
                }
            })
            .await
            .unwrap()
        else {
            panic!("successful control")
        };
        if key == "changed-window" {
            selected.select(reference);
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let frozen = registry.freeze(selected).unwrap();
    assert_eq!(frozen.len(), 1);
    assert!(frozen.contains_key(&scope("changed-window")));
}

#[tokio::test]
async fn dropping_a_claim_cannot_issue_a_second_preparation_capability() {
    let mut registry = PreparationRegistry::<String>::new();
    {
        let preparing = registry.prepare(scope("pending-input"), |_permit| {
            std::future::pending::<Result<ResourceProbeOutcome<String>, Infallible>>()
        });
        tokio::pin!(preparing);
        tokio::select! {
            biased;
            _ = &mut preparing => panic!("declared pending operation"),
            _ = tokio::task::yield_now() => {},
        }
    }
    let calls = AtomicUsize::new(0);
    let result = registry
        .prepare(scope("pending-input"), |_permit| {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Ok::<_, Infallible>(ResourceProbeOutcome::Prepared("must not run".into())) }
        })
        .await;
    assert!(matches!(
        result,
        Err(PreparationRegistryError::UnsettledClaim { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_unchanged_refused_input_cannot_repeat_materialization_or_model_preparation() {
    let calls = AtomicUsize::new(0);
    let mut registry = PreparationRegistry::<String>::new();
    for _ in 0..3 {
        let outcome = registry
            .prepare(scope("refused-input"), |_permit| {
                calls.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok::<_, Infallible>(ResourceProbeOutcome::Limited(
                        FaResourceConstraintV2::EmptyDecodedAudio {
                            shortfall: crate::types::worker_v2::DecodedAudioShortfallV2::NoBytes,
                        },
                    ))
                }
            })
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            ResourceProbeOutcome::Limited(FaResourceConstraintV2::EmptyDecodedAudio { .. })
        ));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        registry
            .freeze(SelectedPreparations::new())
            .unwrap()
            .is_empty()
    );
}
