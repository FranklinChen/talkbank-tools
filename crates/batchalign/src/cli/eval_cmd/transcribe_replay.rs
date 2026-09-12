//! Offline replay of fingerprinted projected transcribe evidence.

use serde::Serialize;
use std::collections::BTreeSet;

use crate::api::{EngineVersion, LanguageCode3, LanguageSpec};
use crate::cache::UtteranceCache;
use crate::cli::args::{
    TranscribeReplayAction, TranscribeReplayArgs, TranscribeReplayManifestArgs,
    TranscribeReplayRunArgs, TranscribeReplayUtsegPasses, TranscribeReplayUtsegPolicy,
};
use crate::cli::dispatch::build_direct_pool_config;
use crate::cli::error::CliError;
use crate::config::{RuntimeLayout, load_validated_config_from_layout};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::transcribe::run_transcribe_pipeline_with_legacy_replay;
use crate::transcribe::replay::{
    AdmittedLegacyTranscribeReplay, LegacyProjectedAsrProducer, LegacyReplayManifestRequest,
    admit_legacy_replay_manifest, file_blake3_hex, write_legacy_replay_manifest,
};
use crate::transcribe::{AsrBackend, TranscribeCachePolicies, TranscribeOptions};
use crate::types::worker_v2::SpeakerBackendV2;
use crate::types::worker_v2::UtsegAdjacencyPolicyRevisionV2;
use crate::utseg::{TranscribeUtsegExecution, UtsegDecisionPolicy};
use crate::{RegistryDiscovery, prepare_workers};

/// Run one offline transcribe-replay action.
pub async fn run(args: &TranscribeReplayArgs) -> Result<(), CliError> {
    match &args.action {
        TranscribeReplayAction::Manifest(args) => author_manifest(args),
        TranscribeReplayAction::Run(args) => run_manifests(args).await,
    }
}

fn author_manifest(args: &TranscribeReplayManifestArgs) -> Result<(), CliError> {
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_legacy_replay_manifest(
        LegacyReplayManifestRequest {
            recording_id: &args.recording_id,
            media_path: &args.media,
            asr_response_path: &args.asr_response,
            speaker_turns_path: args.speaker_turns.as_deref(),
            producer: LegacyProjectedAsrProducer::RevAi,
        },
        &args.output,
    )
    .map_err(replay_error)?;
    eprintln!("Wrote replay manifest {}", args.output.display());
    Ok(())
}

async fn run_manifests(args: &TranscribeReplayRunArgs) -> Result<(), CliError> {
    let utseg_execution = admit_utseg_execution(args)?;
    // Admit the entire batch before creating output or loading a model. A
    // malformed or drifted artifact cannot leave a partial comparison run.
    let admitted = args
        .manifests
        .iter()
        .map(|path| admit_legacy_replay_manifest(path).map_err(replay_error))
        .collect::<Result<Vec<_>, _>>()?;
    validate_batch(&admitted, args.diarize)?;
    let lang = LanguageCode3::try_new(&args.lang)
        .map_err(|error| CliError::InvalidArgument(format!("invalid --lang: {error}")))?;

    if args.output.exists() {
        return Err(CliError::InvalidArgument(format!(
            "replay output must be a new immutable directory: {} already exists",
            args.output.display()
        )));
    }
    let output_parent = args
        .output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(output_parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".ba3-transcribe-replay-")
        .tempdir_in(output_parent)?;
    if let Some(debug_dir) = &args.debug_dir {
        std::fs::create_dir_all(debug_dir)?;
    }

    let layout = RuntimeLayout::from_env();
    let (config, warnings) = load_validated_config_from_layout(&layout, None)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
    let workers = prepare_workers(
        build_direct_pool_config(&config, false, false, layout.state_dir()),
        RegistryDiscovery::Ignore,
    )
    .await
    .map_err(CliError::from)?;
    let cache = UtteranceCache::tiered(None, None)
        .await
        .map_err(|error| CliError::InvalidArgument(format!("cache init failed: {error}")))?;
    let engine_version = EngineVersion::from(format!(
        "offline-replay:{}:{}",
        env!("CARGO_PKG_VERSION"),
        crate::cli::build_hash()
    ));
    let executable = std::env::current_exe()?;
    let executable_blake3 = file_blake3_hex(&executable).map_err(replay_error)?;

    let mut receipts = Vec::with_capacity(admitted.len());
    for replay in admitted {
        let recording_id = replay.recording_id().to_owned();
        let manifest_blake3 = replay.manifest_blake3().to_owned();
        let media_name = replay
            .media_path()
            .file_name()
            .and_then(|name| name.to_str())
            .map(ToOwned::to_owned);
        let opts = TranscribeOptions {
            auto_speakers: false,
            backend: AsrBackend::RustRevAi,
            diarize: args.diarize,
            speaker_backend: args.diarize.then_some(SpeakerBackendV2::PyannoteAi),
            lang: LanguageSpec::Resolved(lang.clone()),
            num_speakers: args.num_speakers,
            with_utseg: utseg_execution.pre_chat_policy().is_some(),
            with_morphosyntax: false,
            // These policies are unreachable in the replay typestate. Keeping
            // them fail-closed protects against a future routing regression.
            cache_policies: TranscribeCachePolicies::uniform(CachePolicy::RequireCache),
            allow_stanza_fallback_utseg: args.utseg_fallback_stanza,
            write_wor: args.wor,
            media_name,
            engine_extras: Default::default(),
        };
        let debug_dir = args.debug_dir.as_ref().map(|root| root.join(&recording_id));
        if let Some(path) = &debug_dir {
            std::fs::create_dir_all(path)?;
        }
        let chat = run_transcribe_pipeline_with_legacy_replay(
            replay,
            utseg_execution,
            PipelineServices::new(workers.pool(), &cache, &engine_version),
            &opts,
            None,
            debug_dir.as_deref(),
        )
        .await
        .map_err(CliError::from)?;
        let final_output_path = args.output.join(format!("{recording_id}.cha"));
        std::fs::write(
            staging.path().join(format!("{recording_id}.cha")),
            chat.as_bytes(),
        )?;
        receipts.push(ReplayReceipt {
            recording_id,
            manifest_blake3,
            output_path: final_output_path,
            output_blake3: blake3::hash(chat.as_bytes()).to_hex().to_string(),
        });
    }

    workers.pool().shutdown().await;
    write_run_receipt(
        staging.path(),
        args,
        utseg_execution,
        &engine_version,
        &executable_blake3,
        &receipts,
    )?;
    let staging_path = staging.keep();
    std::fs::rename(&staging_path, &args.output)?;
    eprintln!(
        "Replayed {} recording(s) without provider inference into {}",
        receipts.len(),
        args.output.display()
    );
    Ok(())
}

fn validate_batch(
    admitted: &[AdmittedLegacyTranscribeReplay],
    diarize: bool,
) -> Result<(), CliError> {
    let mut ids = BTreeSet::new();
    for replay in admitted {
        if !ids.insert(replay.recording_id()) {
            return Err(CliError::InvalidArgument(format!(
                "duplicate replay recording_id {:?}",
                replay.recording_id()
            )));
        }
        if diarize && replay.speaker_segments().is_none() {
            return Err(CliError::InvalidArgument(format!(
                "replay {} requested --diarize but has no speaker-turn artifact",
                replay.recording_id()
            )));
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct ReplayReceipt {
    recording_id: String,
    manifest_blake3: String,
    output_path: std::path::PathBuf,
    output_blake3: String,
}

#[derive(Serialize)]
struct ReplayRunReceipt<'a> {
    schema_version: u32,
    batchalign_version: &'static str,
    batchalign_build: &'static str,
    engine_version: &'a str,
    executable_blake3: &'a str,
    lang: &'a str,
    diarize: bool,
    utseg: bool,
    pre_chat_utseg_policy: Option<&'static str>,
    post_chat_utseg_policy: Option<&'static str>,
    wor: bool,
    outputs: &'a [ReplayReceipt],
}

fn write_run_receipt(
    output_root: &std::path::Path,
    args: &TranscribeReplayRunArgs,
    utseg_execution: TranscribeUtsegExecution,
    engine_version: &EngineVersion,
    executable_blake3: &str,
    receipts: &[ReplayReceipt],
) -> Result<(), CliError> {
    let receipt = ReplayRunReceipt {
        schema_version: 2,
        batchalign_version: env!("CARGO_PKG_VERSION"),
        batchalign_build: crate::cli::build_hash(),
        engine_version: engine_version.as_ref(),
        executable_blake3,
        lang: &args.lang,
        diarize: args.diarize,
        utseg: utseg_execution.pre_chat_policy().is_some(),
        pre_chat_utseg_policy: utseg_execution.pre_chat_policy().map(policy_receipt_name),
        post_chat_utseg_policy: utseg_execution.post_chat_policy().map(policy_receipt_name),
        wor: args.wor,
        outputs: receipts,
    };
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    std::fs::write(output_root.join("replay-run.json"), bytes)?;
    Ok(())
}

fn policy_receipt_name(policy: UtsegDecisionPolicy) -> &'static str {
    match policy {
        UtsegDecisionPolicy::WorkerDeclared => "worker-declared",
        UtsegDecisionPolicy::ReapplyBoundaryModel(
            UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
        ) => "suppress-earlier-adjacent-boundaries-v1",
        UtsegDecisionPolicy::ReapplyBoundaryModel(
            UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
        ) => "suppress-earlier-adjacent-nonordinary-v1",
        UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
            UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
        ) => "suppress-earlier-adjacent-boundaries-preserve-exact-retraces-v1",
        UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
            UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
        ) => "suppress-earlier-adjacent-nonordinary-preserve-exact-retraces-v1",
    }
}

fn selected_policy(args: &TranscribeReplayRunArgs) -> UtsegDecisionPolicy {
    match args.utseg_policy {
        TranscribeReplayUtsegPolicy::WorkerDeclared => UtsegDecisionPolicy::WorkerDeclared,
        TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesV1 => {
            UtsegDecisionPolicy::ReapplyBoundaryModel(
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
            )
        }
        TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesPreserveExactRetracesV1 => {
            UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
                UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
            )
        }
    }
}

fn admit_utseg_execution(
    args: &TranscribeReplayRunArgs,
) -> Result<TranscribeUtsegExecution, CliError> {
    if args.no_utseg {
        if args.utseg_policy != TranscribeReplayUtsegPolicy::WorkerDeclared
            || args.utseg_passes != TranscribeReplayUtsegPasses::Both
        {
            return Err(CliError::InvalidArgument(
                "--no-utseg cannot be combined with --utseg-policy or --utseg-passes".into(),
            ));
        }
        return Ok(TranscribeUtsegExecution::Disabled);
    }
    let selected = selected_policy(args);
    Ok(match args.utseg_passes {
        TranscribeReplayUtsegPasses::Both => TranscribeUtsegExecution::PreAndPostChat {
            pre_chat: selected,
            post_chat: selected,
        },
        TranscribeReplayUtsegPasses::PreChatOnly => {
            TranscribeUtsegExecution::PreChatOnly { pre_chat: selected }
        }
        TranscribeReplayUtsegPasses::PolicyOnPreChatOnly => {
            TranscribeUtsegExecution::PreAndPostChat {
                pre_chat: selected,
                post_chat: UtsegDecisionPolicy::WorkerDeclared,
            }
        }
        TranscribeReplayUtsegPasses::PolicyOnPostChatOnly => {
            TranscribeUtsegExecution::PreAndPostChat {
                pre_chat: UtsegDecisionPolicy::WorkerDeclared,
                post_chat: selected,
            }
        }
    })
}

fn replay_error(error: impl std::fmt::Display) -> CliError {
    CliError::InvalidArgument(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::DurationSeconds;
    use crate::transcribe::{AsrResponse, AsrToken};

    fn run_args(
        no_utseg: bool,
        utseg_policy: TranscribeReplayUtsegPolicy,
    ) -> TranscribeReplayRunArgs {
        TranscribeReplayRunArgs {
            manifests: vec!["recording.replay.json".into()],
            output: "output".into(),
            lang: "eng".into(),
            num_speakers: 2,
            diarize: false,
            no_utseg,
            utseg_policy,
            utseg_passes: TranscribeReplayUtsegPasses::Both,
            wor: false,
            utseg_fallback_stanza: false,
            debug_dir: None,
        }
    }

    fn admitted(id: &str, with_turns: bool) -> AdmittedLegacyTranscribeReplay {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        let media = dir.join(format!("{id}.wav"));
        let asr = dir.join(format!("{id}_asr_response.json"));
        let turns = dir.join(format!("{id}.turns.json"));
        let manifest = dir.join(format!("{id}.replay.json"));
        std::fs::write(&media, b"media").expect("media");
        std::fs::write(
            &asr,
            serde_json::to_vec(&AsrResponse {
                tokens: vec![AsrToken {
                    text: "hello".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("0".into()),
                    confidence: None,
                }],
                lang: LanguageCode3::eng(),
                source_monologues: None,
            })
            .expect("ASR JSON"),
        )
        .expect("ASR");
        if with_turns {
            std::fs::write(&turns, br#"{"source":"batchalign3:pyannote","turns":[]}"#)
                .expect("turns");
        }
        write_legacy_replay_manifest(
            LegacyReplayManifestRequest {
                recording_id: id,
                media_path: &media,
                asr_response_path: &asr,
                speaker_turns_path: with_turns.then_some(turns.as_path()),
                producer: LegacyProjectedAsrProducer::RevAi,
            },
            &manifest,
        )
        .expect("manifest");
        admit_legacy_replay_manifest(&manifest).expect("admitted")
    }

    #[test]
    fn batch_refuses_duplicate_output_identities() {
        let batch = [admitted("same", false), admitted("same", false)];
        assert!(validate_batch(&batch, false).is_err());
    }

    #[test]
    fn diarized_run_requires_turns_in_every_admitted_manifest() {
        let batch = [admitted("one", true), admitted("two", false)];
        assert!(validate_batch(&batch, true).is_err());
        assert!(validate_batch(&batch, false).is_ok());
    }

    #[test]
    fn candidate_policy_cannot_exist_in_a_segmentation_disabled_run() {
        let args = run_args(
            true,
            TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesV1,
        );
        assert!(admit_utseg_execution(&args).is_err());
    }

    #[test]
    fn admitted_candidate_policy_preserves_its_typed_revision() {
        let args = run_args(
            false,
            TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesV1,
        );
        let execution = admit_utseg_execution(&args).expect("candidate execution");
        assert_eq!(
            execution,
            TranscribeUtsegExecution::PreAndPostChat {
                pre_chat: UtsegDecisionPolicy::ReapplyBoundaryModel(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
                post_chat: UtsegDecisionPolicy::ReapplyBoundaryModel(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
            }
        );
    }

    #[test]
    fn post_only_candidate_scope_holds_the_pre_chat_policy_at_production() {
        let mut args = run_args(
            false,
            TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesPreserveExactRetracesV1,
        );
        args.utseg_passes = TranscribeReplayUtsegPasses::PolicyOnPostChatOnly;
        assert_eq!(
            admit_utseg_execution(&args).expect("phase-specific execution"),
            TranscribeUtsegExecution::PreAndPostChat {
                pre_chat: UtsegDecisionPolicy::WorkerDeclared,
                post_chat: UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
            }
        );
    }

    #[test]
    fn pre_only_candidate_scope_holds_the_post_chat_policy_at_production() {
        let mut args = run_args(
            false,
            TranscribeReplayUtsegPolicy::SuppressEarlierAdjacentBoundariesPreserveExactRetracesV1,
        );
        args.utseg_passes = TranscribeReplayUtsegPasses::PolicyOnPreChatOnly;
        assert_eq!(
            admit_utseg_execution(&args).expect("phase-specific execution"),
            TranscribeUtsegExecution::PreAndPostChat {
                pre_chat: UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
                post_chat: UtsegDecisionPolicy::WorkerDeclared,
            }
        );
    }

    #[test]
    fn disabled_segmentation_cannot_carry_a_pass_scope() {
        let mut args = run_args(true, TranscribeReplayUtsegPolicy::WorkerDeclared);
        args.utseg_passes = TranscribeReplayUtsegPasses::PreChatOnly;
        assert!(admit_utseg_execution(&args).is_err());
    }
}
