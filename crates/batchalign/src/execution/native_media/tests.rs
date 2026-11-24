//! Admission controls use placeholder bytes: they do not certify audio.
//! Actual encoding and dual publication are covered by `media::export` controls.

use super::*;
use crate::api::{ContentType, JobId, LanguageSpec, NumSpeakers, ReleasedCommand};
use crate::options::{CommonOptions, ConvertOptions};
use crate::store::{RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity};

fn snapshot(
    root: &std::path::Path,
    names: &[&str],
    format: AudioExportFormat,
) -> RunnerJobSnapshot {
    crate::store::test_support::with_pending(RunnerJobSnapshot {
        identity: RunnerJobIdentity {
            job_id: JobId::from("native-admission"),
            correlation_id: "test".into(),
        },
        dispatch: RunnerDispatchConfig {
            command: ReleasedCommand::Convert,
            lang: LanguageSpec::PerFile,
            num_speakers: NumSpeakers(1),
            options: CommandOptions::Convert(ConvertOptions {
                common: CommonOptions::default(),
                format,
            }),
            runtime_state: Default::default(),
            debug_traces: false,
        },
        filesystem: RunnerFilesystemConfig {
            inputs: crate::store::test_support::paths_inputs(
                names
                    .iter()
                    .map(|name| crate::store::PathsInput {
                        display: (*name).into(),
                        kind: crate::store::InputKind::Media,
                        source: crate::store::test_support::admitted(
                            root.join(name).to_str().unwrap(),
                        ),
                        output: crate::store::test_support::admitted(
                            root.join("out").join(name).to_str().unwrap(),
                        ),
                        before: None,
                    })
                    .collect(),
            ),
            staging_dir: batchalign_types::paths::ServerPath::new(root.join("job")),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: batchalign_types::paths::ClientPath::new(root.to_str().unwrap()),
        },
        cancel_token: tokio_util::sync::CancellationToken::new(),
        pending_files: Vec::new(),
    })
}

/// Guards: Selected encoding binds requested and staged filenames without loading a worker profile.
/// Kind: contract
/// Not a type: Per-file input records cannot establish codec-specific output naming or whether native admission creates directories.
/// State: An isolated root supplies admitted source/destination identities and absent destination entries for native admission to inspect. No source audio bytes, encoder, model or registry runs.
/// Concurrency: The test owns its root and immutable declarations, then checks the admitted paths and directory absence before cleanup.
#[test]
fn selected_encoding_binds_both_destination_roles_and_needs_no_model_profile() {
    let root = tempfile::tempdir().unwrap();
    for (format, kind) in [
        (AudioExportFormat::Wav, ContentType::Wav),
        (AudioExportFormat::Mp3, ContentType::Mp3),
    ] {
        let job = snapshot(root.path(), &["nested/résumé.source.wav"], format);
        let work = admit_work(&job, &job.pending_files).unwrap();
        assert_eq!(work.len(), 1);
        let expected = format!("nested/résumé.source.converted.{}", format.extension());
        assert_eq!(work[0].artifact.display_path.as_ref(), expected);
        assert_eq!(work[0].artifact.content_type, kind);
        assert_eq!(work[0].primary, root.path().join("out").join(&expected));
        assert_eq!(
            work[0].staged,
            root.path().join("job/output").join(expected)
        );
        assert_eq!(work[0].source, root.path().join("nested/résumé.source.wav"));
        assert!(crate::worker::WorkerProfile::for_command(ReleasedCommand::Convert).is_none());
        assert!(!root.path().join("out").exists());
    }
}

/// Guards: Colliding output stems and existing destinations refuse the entire export before any write.
/// Kind: contract
/// Not a type: Path-set admission cannot determine later codec-rewritten name collisions or current filesystem destination occupancy.
/// State: The root supplies admitted identities, colliding rewritten stems and a retained destination containing known bytes. Real metadata checks refuse admission and the retained bytes are read back. No encoder, model or registry runs.
/// Concurrency: The test owns every path and establishes destination occupancy before admission. All absence and byte checks finish before root cleanup.
#[test]
fn source_stem_collision_and_existing_output_refuse_the_whole_admission() {
    let root = tempfile::tempdir().unwrap();
    let job = snapshot(
        root.path(),
        &["same.wav", "same.mp3"],
        AudioExportFormat::Wav,
    );
    assert!(matches!(
        admit_work(&job, &job.pending_files),
        Err(NativeExportAdmissionError::Collision(_))
    ));
    assert!(!root.path().join("out").exists());
    let job = snapshot(root.path(), &["one.wav", "two.wav"], AudioExportFormat::Wav);
    std::fs::create_dir(root.path().join("out")).unwrap();
    let retained = root.path().join("out/two.converted.wav");
    std::fs::write(&retained, b"retain").unwrap();
    assert!(matches!(
        admit_work(&job, &job.pending_files),
        Err(NativeExportAdmissionError::Existing(_))
    ));
    assert!(!root.path().join("out/one.converted.wav").exists());
    assert!(!root.path().join("job").exists());
    assert_eq!(std::fs::read(retained).unwrap(), b"retain");
}

/// Guards: A persisted command and encoding-options mismatch refuses native export before side effects.
/// Kind: contract
/// Not a type: Complete input records cannot establish semantic consistency of separately persisted command and typed encoding options.
/// State: The root supplies admitted input identities and an absent output directory. The independently supplied command/options mismatch is refused during planning. No publication, encoder, model or registry runs.
/// Concurrency: The test owns its declarations and root. Planning refusal finishes before checking directory absence and cleanup.
#[test]
fn persisted_command_and_options_cannot_choose_different_execution_lanes() {
    let root = tempfile::tempdir().unwrap();
    let mut job = snapshot(root.path(), &["one.wav"], AudioExportFormat::Wav);
    job.dispatch.command = ReleasedCommand::Transcribe;
    assert!(matches!(
        admit_work(&job, &job.pending_files),
        Err(NativeExportAdmissionError::Planning(_))
    ));
    assert!(!root.path().join("out").exists());
}

/// Guards: The wire requires an explicit native encoding and preserves each supported format on round-trip.
/// Kind: wire
/// Not a type: Typed format variants exist only after decoding, so missing external JSON fields and exact serialized format strings need boundary observations.
/// State: None
/// Concurrency: None
#[test]
fn encoding_is_required_on_the_wire_and_round_trips_without_guessing() {
    assert!(
        serde_json::from_value::<CommandOptions>(serde_json::json!({"command":"convert"})).is_err()
    );
    for format in ["wav", "mp3"] {
        let options: CommandOptions =
            serde_json::from_value(serde_json::json!({"command":"convert", "format":format}))
                .unwrap();
        assert_eq!(options.command(), ReleasedCommand::Convert);
        assert_eq!(serde_json::to_value(options).unwrap()["format"], format);
    }
}
