use super::*;
use crate::api::ReleasedCommand;
/// Guards: every task maps to its actual host family and label under each
/// bootstrap mode, and released command requirements choose their declared task.
/// Kind: wire
/// Not a type: The actual Python-visible labels and command to profile mappings are
/// explicit wire and semantic choices after `InferenceRequirement` admission.
/// State: immutable requirements and targets require no pool or process.
/// Concurrency: none
#[test]
fn admitted_targets_preserve_family_bootstrap_and_wire_labels() {
    for (task, profile) in [
        (InferTask::Asr, WorkerProfile::Gpu),
        (InferTask::Fa, WorkerProfile::Gpu),
        (InferTask::Speaker, WorkerProfile::Gpu),
        (InferTask::Morphosyntax, WorkerProfile::Stanza),
        (InferTask::Utseg, WorkerProfile::Stanza),
        (InferTask::Coref, WorkerProfile::Stanza),
        (InferTask::Translate, WorkerProfile::Io),
        (InferTask::Opensmile, WorkerProfile::Io),
        (InferTask::Avqi, WorkerProfile::Io),
        (InferTask::WordSegmentation, WorkerProfile::Io),
    ] {
        assert_eq!(WorkerProfile::for_task(task), profile);
        for mode in [
            WorkerBootstrapMode::Task,
            WorkerBootstrapMode::Profile,
            WorkerBootstrapMode::LazyProfile,
        ] {
            let target = WorkerTarget::from_infer_task(task, mode);
            match mode {
                WorkerBootstrapMode::Task => {
                    assert_eq!(target, WorkerTarget::InferTask(task));
                    assert_eq!(target.label(), format!("infer:{}", task.wire_name()));
                }
                WorkerBootstrapMode::Profile | WorkerBootstrapMode::LazyProfile => {
                    assert_eq!(target, WorkerTarget::Profile(profile));
                    assert_eq!(target.label(), format!("profile:{}", profile.name()));
                }
            }
        }
    }
    for (command, task, profile) in [
        (
            ReleasedCommand::Transcribe,
            InferTask::Asr,
            WorkerProfile::Gpu,
        ),
        (
            ReleasedCommand::Compare,
            InferTask::Morphosyntax,
            WorkerProfile::Stanza,
        ),
        (ReleasedCommand::Align, InferTask::Fa, WorkerProfile::Gpu),
        (
            ReleasedCommand::Morphotag,
            InferTask::Morphosyntax,
            WorkerProfile::Stanza,
        ),
    ] {
        let requirement = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
            .unwrap();
        assert_eq!(
            WorkerTarget::for_inference_with_mode(requirement, WorkerBootstrapMode::Task),
            WorkerTarget::InferTask(task)
        );
        assert_eq!(WorkerProfile::for_command(command), Some(profile));
    }
}
/// Guards: profile capacity defaults and canonical tier startup reservations
/// retain the exact established large-host values and reduced small-host costs.
/// Kind: policy
/// Not a type: concurrency defaults and per-profile resource sizes are chosen
/// policies, while the host-bound budget now carries their completed projection.
/// State: numeric tier facts and profile values perform no live sampling.
/// Concurrency: none
#[test]
fn profile_defaults_follow_the_admitted_tier() {
    assert_eq!(WorkerProfile::Gpu.default_max_workers(4), 1);
    assert_eq!(WorkerProfile::Stanza.default_max_workers(4), 4);
    assert_eq!(WorkerProfile::Io.default_max_workers(4), 1);
    let large = crate::types::runtime::MemoryTier::from_total_mb(64000);
    let small = crate::types::runtime::MemoryTier::from_total_mb(16000);
    assert_eq!(
        WorkerProfile::Gpu.startup_reservation_mb_for_tier(&large).0,
        16000
    );
    assert_eq!(
        WorkerProfile::Stanza
            .startup_reservation_mb_for_tier(&large)
            .0,
        12000
    );
    assert_eq!(
        WorkerProfile::Io.startup_reservation_mb_for_tier(&large).0,
        4000
    );
    let gpu = WorkerProfile::Gpu.startup_reservation_mb_for_tier(&small).0;
    let stanza = WorkerProfile::Stanza
        .startup_reservation_mb_for_tier(&small)
        .0;
    assert!(gpu < 16000 && stanza < 12000 && gpu > stanza);
}
