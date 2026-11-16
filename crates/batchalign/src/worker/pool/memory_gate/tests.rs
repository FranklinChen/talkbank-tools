use super::*;
use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName, TranslateEngineName};
use crate::worker::WorkerProfile;

/// Guards: startup projections retain the profile baseline or the largest
/// selected engine footprint across every host tier, without summing models.
/// Kind: policy
/// Not a type: baseline/max aggregation and which engine sizes participate are
/// semantic choices. The completed result is retained privately on the actual group.
/// State: typed engine choices and declared tiers require no model, process or host probe.
/// Concurrency: none
#[test]
fn startup_projections_follow_the_selected_engines_and_host_tier() {
    for total in [16000, 32000, 64000, 256000] {
        let tier = crate::types::runtime::MemoryTier::from_total_mb(total);
        let cases = [
            (WorkerProfile::Io, EngineOverrides::default(), 0),
            (
                WorkerProfile::Io,
                EngineOverrides {
                    translate: Some(TranslateEngineName::Google),
                    ..Default::default()
                },
                TranslateEngineName::Google.resident_memory_mb(),
            ),
            (
                WorkerProfile::Io,
                EngineOverrides {
                    translate: Some(TranslateEngineName::Nllb),
                    ..Default::default()
                },
                TranslateEngineName::Nllb.resident_memory_mb(),
            ),
            (
                WorkerProfile::Io,
                EngineOverrides {
                    translate: Some(TranslateEngineName::Seamless),
                    ..Default::default()
                },
                TranslateEngineName::Seamless.resident_memory_mb(),
            ),
            (
                WorkerProfile::Gpu,
                EngineOverrides {
                    asr: Some(AsrEngineName::Whisper),
                    ..Default::default()
                },
                AsrEngineName::Whisper.resident_memory_mb(),
            ),
            (
                WorkerProfile::Gpu,
                EngineOverrides {
                    fa: Some(FaEngineName::Whisper),
                    ..Default::default()
                },
                FaEngineName::Whisper.resident_memory_mb(),
            ),
            (
                WorkerProfile::Gpu,
                EngineOverrides {
                    asr: Some(AsrEngineName::Whisper),
                    translate: Some(TranslateEngineName::Nllb),
                    ..Default::default()
                },
                AsrEngineName::Whisper
                    .resident_memory_mb()
                    .max(TranslateEngineName::Nllb.resident_memory_mb()),
            ),
            (
                WorkerProfile::Gpu,
                EngineOverrides {
                    asr: Some(AsrEngineName::HkTencent),
                    ..Default::default()
                },
                AsrEngineName::HkTencent.resident_memory_mb(),
            ),
        ];
        for (profile, engines, engine_floor) in cases {
            assert_eq!(
                engine_aware_startup_reservation_mb(profile, &engines, &tier).0,
                profile
                    .startup_reservation_mb_for_tier(&tier)
                    .0
                    .max(engine_floor)
            );
        }
    }
}

/// Guards: every admitted host tier uses its stated headroom floor, including
/// the canonical Medium default and the larger-host plateau.
/// Kind: policy
/// Not a type: the tier-to-headroom relationship is a chosen resource policy,
/// beyond associating each retained startup projection with its actual host.
/// State: immutable declared tiers require no OS or filesystem access.
/// Concurrency: none
#[test]
fn headroom_floors_follow_the_current_tier_policy() {
    for (total, expected) in [(16000, 1024), (32000, 2048), (64000, 4096), (256000, 4096)] {
        assert_eq!(
            host_min_free_mb_threshold_for_tier(&crate::types::runtime::MemoryTier::from_total_mb(
                total
            )),
            expected
        );
    }
}

/// Guards: cold-start bypass and warm-pool strict headroom preserve exact
/// floor/refusal facts for available, exhausted and oversized projections.
/// Kind: policy
/// Not a type: whether to admit a numeric projection at the strict floor is a
/// semantic decision. These values claim no actual reservation or live capacity.
/// State: one declared host supplies deterministic 14,000 MB availability.
/// Concurrency: none
#[test]
fn cold_and_warm_admission_preserve_the_projection_boundary() {
    let readings =
        crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
            .readings();
    for (state, floor, estimate, accepts) in [
        (PoolGateState::ColdStart, u64::MAX, 0, true),
        (PoolGateState::ColdStart, 0, u64::MAX, true),
        (PoolGateState::Warm, u64::MAX, 0, false),
        (PoolGateState::Warm, 0, u64::MAX, false),
        (PoolGateState::Warm, 1024, 12975, true),
        (PoolGateState::Warm, 1024, 12976, false),
        (PoolGateState::Warm, 0, 0, true),
    ] {
        let result = check_memory_saturation_with_state(state, &readings, floor, estimate);
        if accepts {
            assert!(result.is_ok(), "{result:?}");
        } else {
            let Err(GateRefusal::Constrained(refused)) = result else {
                panic!("expected constrained projection");
            };
            assert_eq!(
                (
                    refused.available_mb,
                    refused.reservation_mb,
                    refused.threshold_mb
                ),
                (14000, estimate, floor)
            );
        }
    }
}
