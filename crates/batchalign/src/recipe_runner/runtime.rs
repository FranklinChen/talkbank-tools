//! Shared runner adapters for input discovery, planning and output publication.

use std::path::Path;

use crate::api::{DisplayPath, ReleasedCommand};
use crate::command_model;
use crate::store::{PendingJobFile, RunnerFilesystemConfig, RunnerJobSnapshot};

use super::materialize::{PlannedMaterializedFile, plan_materialized_files};
use super::planner::{PlanningError, plan_work_units};
use super::work_unit::{DiscoveredInput, PlannedWorkUnit};

mod targets;
#[cfg(test)]
mod targets_tests;
pub(crate) use targets::{ChatMetricsOutputs, OutputSourceIdentity, OutputTarget, SourceOutputs};

/// Options used by legacy naming controls only, not production admission.
/// Commands with mandatory user choices must supply those explicitly.
#[cfg(test)]
pub(crate) fn test_options(command: ReleasedCommand) -> crate::options::CommandOptions {
    use crate::chat_ops::speaker_identity::{
        EnrollmentSet, EnrollmentSpec, MatchThreshold, documented_permutation_plan,
    };
    use crate::options::{CommandOptions, CommonOptions, SpeakerIdentifyOptions};
    match command {
        ReleasedCommand::Convert => panic!("export controls must choose their format explicitly"),
        ReleasedCommand::SpeakerIdentify => {
            CommandOptions::SpeakerIdentify(SpeakerIdentifyOptions {
                common: CommonOptions::default(),
                enrollments: EnrollmentSet::new(vec![EnrollmentSpec::parse("0-1000:INV").unwrap()])
                    .unwrap(),
                threshold: MatchThreshold::try_from(0.7).unwrap(),
                tiers: vec![],
                permutation: documented_permutation_plan(),
            })
        }
        _ => serde_json::from_value(serde_json::json!({"command": command.as_str()})).unwrap(),
    }
}

/// Derive one discovered input from the runner's current filesystem layout.
pub(crate) fn discover_input_for_pending_file(
    filesystem: &RunnerFilesystemConfig,
    file: &PendingJobFile,
) -> DiscoveredInput {
    let source_path = file
        .input_path(&filesystem.staging_dir)
        .as_path()
        .to_owned();
    let before_path = file.before_path().map(|path| path.as_path().to_owned());
    DiscoveredInput {
        display_path: file.filename().clone(),
        source_path,
        before_path,
    }
}

/// Derive all discovered inputs for one runner job.
pub(crate) fn discover_inputs_for_job(job: &RunnerJobSnapshot) -> Vec<DiscoveredInput> {
    job.pending_files
        .iter()
        .map(|file| discover_input_for_pending_file(&job.filesystem, file))
        .collect()
}

/// Plan command-family-specific work units from the runner snapshot.
pub(crate) fn plan_work_units_for_job(
    command: ReleasedCommand,
    job: &RunnerJobSnapshot,
) -> Result<Vec<PlannedWorkUnit>, PlanningError> {
    let inputs = discover_inputs_for_job(job);
    plan_work_units(command_model::command_spec(command).planner, &inputs)
}

/// Plan all materialized artifacts for one released command and source path.
pub(crate) fn planned_output_artifacts(
    command: ReleasedCommand,
    options: &crate::options::CommandOptions,
    source_display_path: &DisplayPath,
) -> Result<super::materialize::PlannedArtifactSet, PlanningError> {
    Ok(plan_materialized_files(
        source_display_path,
        command_model::command_spec(command).selected_output_policy(options)?,
    ))
}

/// Return the primary result artifact for one command/input pair.
pub(crate) fn primary_output_artifact(
    command: ReleasedCommand,
    options: &crate::options::CommandOptions,
    source_display_path: &DisplayPath,
) -> Result<PlannedMaterializedFile, PlanningError> {
    Ok(
        planned_output_artifacts(command, options, source_display_path)?
            .into_primary()
            .into_materialized(),
    )
}

/// Derive only the primary result display path for one command/input pair.
pub(crate) fn result_display_path_for_command(
    command: ReleasedCommand,
    options: &crate::options::CommandOptions,
    filename: &str,
) -> Result<DisplayPath, PlanningError> {
    Ok(primary_output_artifact(command, options, &DisplayPath::from(filename))?.display_path)
}

/// The daemon's one route from an output path to bytes on disk:
/// `crate::atomic_file::write_text_artifact`, on a blocking thread because it
/// syncs. Every per-file target here and the kernel's consolidated
/// `compare.csv` come through it, so no runner path truncates a user's file
/// before refilling it. The path and content are copied for the blocking
/// task, as `tokio::fs::write` copied the content.
pub(crate) async fn publish_text(path: &Path, content: &str) -> std::io::Result<()> {
    let path = path.to_path_buf();
    let content = content.to_owned();
    crate::blocking::spawn_in_span(move || crate::atomic_file::write_text_artifact(&path, &content))
        .await
        .map_err(std::io::Error::other)?
}

/// Write a text result to the command's primary target and to staged output.
///
/// Paths-mode jobs still write to the execution host's requested output path,
/// but they also preserve a staged copy so the submitting CLI can download and
/// write the same result back to its own local filesystem.
pub(crate) async fn write_text_output_artifact(
    target: OutputTarget<'_, impl Sized>,
    content: &str,
) -> std::io::Result<()> {
    for path in target.into_paths() {
        publish_text(&path, content).await?;
    }

    Ok(())
}

/// Write a CHAT result to the command's primary target and staged output,
/// suppressing the write at any path where the only difference vs the
/// existing on-disk text is inside the provenance stamps the command's recipe
/// writes (its own and those of every stage it composes, from the catalog),
/// under either stamp name (`[fc-ba3 <command> | ...]` or the legacy
/// `[ba3 <command> | ...]`), in name and timestamp only.
///
/// This is the seam that touches disk, so it takes a judged proof
/// ([`WritableChat`]: an admitted `PostValidated` or a producer's
/// `ProducedOutput`) rather than the bytes: a caller cannot gate one document and write
/// another, and there is no signature here for an ungated `String` to travel
/// through. The command comes off the proof for the same reason, so the
/// provenance line this suppresses is the one the gated command wrote.
///
/// Use this for any pipeline that injects a `ProvenanceComment` for a
/// known [`ReleasedCommand`] before serializing CHAT. The gate eliminates
/// the "re-running the same command produced a 1-line timestamp/version
/// diff per file" failure mode that produces large, semantically empty
/// commits in corpus repos.
///
/// Behavior per write target:
///
/// - Target file does not exist → write (first run).
/// - Existing bytes are byte-equal to `content` → skip (already correct).
/// - Existing bytes differ from `content` only in the stamps' names and
///   timestamps (and our ASR warning's build identity) → skip.
/// - Existing bytes differ in any other content (a stamp field such as an
///   engine, `%mor`, `%gra`, `%wor`, the stamp of a command the recipe does
///   not compose, anything else) → write.
///
/// A skipped write touches nothing; a performed one goes through
/// [`publish_text`].
///
/// The primary write_path and the staged output path are evaluated
/// independently. In the common paths_mode layout the staged copy lives
/// in `staging_dir/output/...` and is fresh per job, so it almost always
/// triggers a write; the primary path is the one the gate effectively
/// guards. We still apply the same logic to both for symmetry, there
/// is no scenario where it is correct to update one and not the other.
///
/// [`WritableChat`]: crate::pipeline::post_validate::WritableChat
pub(crate) async fn write_chat_output_artifact_with_provenance_gate(
    target: OutputTarget<'_, super::materialize::PrimaryArtifact>,
    document: &impl crate::pipeline::post_validate::WritableChat,
) -> std::io::Result<()> {
    let content = document.chat_text();
    let command = document.chat_command();
    for path in target.into_paths() {
        write_chat_if_meaningful_diff(&path, content, command).await?;
    }

    Ok(())
}

/// Read the existing file at `path` (if any), compare it to `candidate`
/// for the specific `command`'s provenance gate, and write only if a
/// real (non-provenance-only) difference exists.
///
/// The read is best-effort: any read error other than NotFound is
/// surfaced. NotFound is treated as "first run" and triggers a write.
///
/// Cheap pre-check: the gate only fires when the existing-vs-candidate
/// difference is bounded to a single provenance line. If the candidate's
/// length differs from the existing file's length by more than a
/// provenance line could plausibly contribute, the gate cannot fire and
/// we skip the read entirely. The bound is generous (one full
/// provenance line is well under 200 bytes; doubling that absorbs
/// engine-string / language-tag drift); we want false positives that
/// fall through to the existing read+compare path, not false negatives
/// that wrongly suppress a write.
async fn write_chat_if_meaningful_diff(
    path: &Path,
    candidate: &str,
    command: ReleasedCommand,
) -> std::io::Result<()> {
    match tokio::fs::metadata(path).await {
        Ok(meta) => {
            let existing_len = meta.len();
            let candidate_len = candidate.len() as u64;
            let diff = existing_len.abs_diff(candidate_len);
            if diff > PROVENANCE_LINE_DIFF_BUDGET_BYTES {
                // The two files differ by more than a provenance line
                // could contribute; the gate cannot possibly fire and
                // there is no point reading the existing file just to
                // compare. Write through.
                return publish_text(path, candidate).await;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // First run: file does not exist yet. Write through.
            return publish_text(path, candidate).await;
        }
        Err(error) => return Err(error),
    }

    match tokio::fs::read_to_string(path).await {
        Ok(existing) => {
            if existing == candidate {
                // Bytes already match. Skip, avoid mtime churn.
                return Ok(());
            }
            if crate::provenance::is_provenance_only_difference(&existing, candidate, command) {
                // The only would-be change is inside the command's
                // provenance line. Per the no-spurious-update policy,
                // do not write.
                return Ok(());
            }
            publish_text(path, candidate).await
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Race: file existed at metadata time but is gone now.
            // Treat as first-run and write through.
            publish_text(path, candidate).await
        }
        Err(error) => Err(error),
    }
}

/// Bound on how much the candidate output's byte length may differ from
/// the on-disk byte length while still being eligible for the
/// provenance-only-diff gate. The gate fires only when the stamps' fields are
/// unchanged, so the length can drift only by stamp names (`ba3` to `fc-ba3`
/// is 3 bytes per stamp), timezone offsets and the warning's build identity:
/// a few tens of bytes even for a transcribe job's three stamps. A typical
/// `[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-05-08T02:52:17-04:00]`
/// line is ~100 bytes, so two lines' worth is a generous bound.
const PROVENANCE_LINE_DIFF_BUDGET_BYTES: u64 = 200;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{CorrelationId, JobId, LanguageSpec, NumSpeakers};
    use crate::options::CommandOptions;
    use crate::pipeline::post_validate::PostValidated;
    use crate::store::{
        RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity, RunnerJobSnapshot,
    };
    use std::path::PathBuf;

    /// A paths-mode job whose one input is `<root>/input/sample.cha`, whose
    /// requested output is `host_out`, and whose staging tree is
    /// `<root>/jobs/job-1`: the shape every writer test here uses.
    fn paths_mode_filesystem(
        root: &Path,
        host_out: &Path,
        display: &str,
    ) -> RunnerFilesystemConfig {
        let client = |path: PathBuf| {
            batchalign_types::paths::ClientPath::new(path.to_string_lossy().to_string())
        };
        RunnerFilesystemConfig {
            inputs: crate::store::test_support::paths_inputs(vec![crate::store::PathsInput {
                display: display.into(),
                kind: crate::store::InputKind::Chat,
                source: crate::store::test_support::admitted(
                    root.join("input/sample.cha").to_str().unwrap(),
                ),
                output: crate::store::test_support::admitted(host_out.to_str().unwrap()),
                before: None,
            }]),
            staging_dir: batchalign_types::paths::ServerPath::from(root.join("jobs/job-1")),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: client(root.join("input")),
        }
    }

    fn runner_snapshot(
        options: CommandOptions,
        inputs: crate::store::JobInputs,
    ) -> RunnerJobSnapshot {
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-runtime"),
                correlation_id: CorrelationId::from("corr-runtime"),
            },
            dispatch: RunnerDispatchConfig {
                command: options.command(),
                // Compare is a per-file command; no job-level language.
                lang: LanguageSpec::PerFile,
                num_speakers: NumSpeakers(1),
                options,
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: RunnerFilesystemConfig {
                inputs,
                staging_dir: batchalign_types::paths::ServerPath::new("/tmp/staging"),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/source"),
            },
            cancel_token: CancellationToken::new(),
            pending_files: Vec::new(),
        })
    }

    fn admitted_morphotag_output(text: &str) -> PostValidated {
        let parser = crate::chat_parser();
        let admitted = crate::pipeline::text_infer::admit_retained_text(
            &parser,
            text,
            talkbank_model::model::TranscriptName::for_path(Path::new("writer-fixture.cha")),
        )
        .expect("complete authored CHAT source");
        PostValidated::gate_owned(
            admitted.into_valid_file().into_unchecked(),
            ReleasedCommand::Morphotag,
        )
        .expect("complete morphotag output")
    }

    /// A transcribe re-run on a new build over identical content changes only
    /// the stamps its recipe writes (transcribe, utseg, morphotag) and the
    /// warning's build identity, so the file on disk is not rewritten. A
    /// changed engine in a composed stage's stamp is written.
    #[tokio::test]
    async fn a_transcribe_rerun_on_a_new_build_leaves_identical_content_unwritten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.cha");
        let on_disk = "\
@UTF8
@Begin
@Comment:\t[ba3 transcribe | asr=rev ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:\t[ba3 utseg | engine=talkbank/utterance-boundary@revision-1 ; lang=eng | 2026-03-29T18:31:00-04:00]
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-03-29T18:32:00-04:00]
@Comment:\tBatchalign 0.8.5, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
%mor:\tintj|hello .
@End
";
        let rebuilt = "\
@UTF8
@Begin
@Comment:\t[fc-ba3 transcribe | asr=rev ; lang=eng | 2026-09-15T18:30:00-04:00]
@Comment:\t[fc-ba3 utseg | engine=talkbank/utterance-boundary@revision-1 ; lang=eng | 2026-09-15T18:31:00-04:00]
@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-09-15T18:32:00-04:00]
@Comment:\tfc-ba3 new-build-2, ASR engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
%mor:\tintj|hello .
@End
";
        tokio::fs::write(&path, on_disk)
            .await
            .expect("write existing file");

        write_chat_if_meaningful_diff(&path, rebuilt, ReleasedCommand::Transcribe)
            .await
            .expect("gate");
        assert_eq!(
            tokio::fs::read_to_string(&path).await.expect("read back"),
            on_disk
        );

        let new_model = rebuilt.replace("stanza-1.11.1:eng:standard", "stanza-1.12.0:eng:standard");
        write_chat_if_meaningful_diff(&path, &new_model, ReleasedCommand::Transcribe)
            .await
            .expect("gate");
        assert_eq!(
            tokio::fs::read_to_string(&path).await.expect("read back"),
            new_model
        );
    }

    /// Guards: Discovered nested sources retain each optional prior on its own file.
    /// Kind: contract
    /// Not a type: Per-file admitted inputs prevent mismatched lists but cannot prove the adapter copies the correct source and optional prior paths.
    /// State: Real same-user path admission observes synthetic absolute paths. The contract creates no file or directory and performs no content IO.
    /// Concurrency: The test owns its immutable input values. It mutates no filesystem path or process state.
    #[test]
    fn optional_priors_keep_recipe_and_text_consumers_on_the_same_file() {
        // A prior belongs to its own file's input, so the second file's prior
        // can reach neither neighbour. (A prior for a file past the end, or
        // in a content-mode job, has no input to live on.)
        let names = ["nested/a.cha", "nested/b.cha", "nested/c.cha"];
        let inputs = crate::store::test_support::paths_inputs(
            names
                .iter()
                .map(|name| crate::store::PathsInput {
                    display: (*name).into(),
                    kind: crate::store::InputKind::Chat,
                    source: crate::store::test_support::admitted(&format!("/src/{name}")),
                    output: crate::store::test_support::admitted(&format!("/out/{name}")),
                    before: (*name == "nested/b.cha")
                        .then(|| crate::store::test_support::admitted("/before/nested/b.cha")),
                })
                .collect(),
        );
        let job = runner_snapshot(CommandOptions::Morphotag(Default::default()), inputs);
        for (index, file) in job.pending_files.iter().enumerate() {
            let expected = (index == 1).then(|| std::path::PathBuf::from("/before/nested/b.cha"));
            let observed = discover_input_for_pending_file(&job.filesystem, file);
            assert_eq!(
                observed.source_path,
                PathBuf::from(format!("/src/{}", names[index]))
            );
            assert_eq!(observed.before_path, expected);
        }
    }

    /// Guards: Text publication preserves the requested output and nested staged copy byte for byte.
    /// Kind: contract
    /// Not a type: Admitted per-file paths cannot establish successful writes or nested directory creation at both actual destinations.
    /// State: An isolated root supplies admitted paths and actual requested/staged text writes and reads. No document gate, event recorder, registry or model is used.
    /// Concurrency: The test owns both destinations and awaits publication before byte comparison and root cleanup.
    #[tokio::test]
    async fn write_text_output_artifact_keeps_staged_copy_for_paths_mode() {
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        let filesystem = paths_mode_filesystem(root.path(), &host_out, "nested/sample.cha");
        let file = crate::store::test_support::only_file(&filesystem.inputs);

        let outputs = SourceOutputs::plan(
            &filesystem,
            &file,
            ReleasedCommand::Morphotag,
            &test_options(ReleasedCommand::Morphotag),
        )
        .expect("source-bound output plan");
        let target = outputs.primary_target().expect("primary target");
        write_text_output_artifact(target, "@Begin\n@End\n")
            .await
            .expect("write output");

        assert_eq!(
            std::fs::read_to_string(&host_out).expect("host output"),
            "@Begin\n@End\n"
        );
        assert_eq!(
            std::fs::read_to_string(
                filesystem
                    .staging_dir
                    .join("output")
                    .join("nested/sample.cha")
            )
            .expect("staged output"),
            "@Begin\n@End\n"
        );
    }

    /// Re-running morphotag over an unchanged corpus must not modify the
    /// on-disk file just to update the provenance comment.
    /// Pre-condition: a CHAT file exists at the primary output path with
    /// an old, legacy-named `[ba3 morphotag | ...]` provenance line. We then
    /// call the gated writer with new text that differs only in that line
    /// (now written as `[fc-ba3 morphotag | ...]`, with a new timestamp and
    /// engine slot). Post-condition: the file's bytes are unchanged, so the
    /// stamp rename alone never rewrites a file.
    /// Guards: A stamp-only rerun preserves the original primary bytes and modification time.
    /// Kind: contract
    /// Not a type: Native source and output admission prove document validity but cannot establish the no-op write decision or actual file metadata.
    /// State: The isolated root owns an existing primary and its filesystem metadata. Native admission retains the candidate source/output proof and serialization. The test observes preserved bytes and modification time plus staged publication. No event recorder, registry or model runs.
    /// Concurrency: The test owns all paths and awaits the gated write before inspecting bytes and metadata or dropping its root.
    #[tokio::test]
    async fn write_chat_output_artifact_with_provenance_gate_skips_when_only_provenance_differs() {
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        std::fs::create_dir_all(host_out.parent().unwrap()).unwrap();

        let original = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
*PAR:\thello .
%mor:\tintj|hello .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
";
        std::fs::write(&host_out, original).unwrap();
        let before = std::fs::metadata(&host_out).expect("metadata before");

        let filesystem = paths_mode_filesystem(root.path(), &host_out, "nested/sample.cha");
        let file = crate::store::test_support::only_file(&filesystem.inputs);

        // Same content as `original` except for the stamp's name (legacy
        // `ba3` to `fc-ba3`) and its timestamp: the fields are identical.
        // This is the canonical pointless-rerun shape we want to suppress. A
        // stamp whose fields differ (a new `engine=` spelling, say) is a
        // meaningful difference and is written; see `provenance`.
        let candidate = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tintj|hello .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
";

        let outputs = SourceOutputs::plan(
            &filesystem,
            &file,
            ReleasedCommand::Morphotag,
            &test_options(ReleasedCommand::Morphotag),
        )
        .expect("source-bound output plan");
        let target = outputs.primary_target().expect("primary target");
        let output = admitted_morphotag_output(candidate);
        write_chat_output_artifact_with_provenance_gate(target, &output)
            .await
            .expect("gated write");

        let after = std::fs::read_to_string(&host_out).expect("read after");
        assert_eq!(
            after, original,
            "primary output bytes must be unchanged when only the provenance differs"
        );
        // A skip is a skip: the atomic writer was not reached, so the file was
        // not replaced by a staged copy of the same bytes (its modification
        // time would churn and corpus tooling would see a touched file).
        let after = std::fs::metadata(&host_out).expect("metadata after");
        assert_eq!(
            after.modified().expect("mtime after"),
            before.modified().expect("mtime before"),
            "a stamp-only rerun must not touch the file"
        );
    }

    /// The in-place output path is the user's only copy of the file, and the
    /// writer reaches the atomic primitive rather than rewriting it: the path
    /// names a new inode afterwards (a truncating write keeps the inode).
    /// `atomic_file`'s own tests prove what a new inode implies.
    #[cfg(unix)]
    /// Guards: Replacing text uses a new inode and writes the requested bytes.
    /// Kind: contract
    /// Not a type: Typed destinations cannot establish the operating system inode identity after an actual atomic replacement.
    /// State: One isolated root owns an existing primary inode. The atomic writer publishes replacement text and its staged copy. Filesystem bytes and inode identity are the observations. No document gate, event recorder, registry or model runs.
    /// Concurrency: The test owns both destinations and compares the inode only after the replacement future completes.
    #[tokio::test]
    async fn write_text_output_artifact_replaces_the_previous_file_rather_than_rewriting_it() {
        use std::os::unix::fs::MetadataExt;
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        std::fs::create_dir_all(host_out.parent().unwrap()).unwrap();
        std::fs::write(&host_out, "@UTF8\n@Begin\n*CHI:\told .\n@End\n").unwrap();
        let old_inode = std::fs::metadata(&host_out).expect("metadata").ino();
        let filesystem = paths_mode_filesystem(root.path(), &host_out, "nested/sample.cha");
        let file = crate::store::test_support::only_file(&filesystem.inputs);

        let outputs = SourceOutputs::plan(
            &filesystem,
            &file,
            ReleasedCommand::Morphotag,
            &test_options(ReleasedCommand::Morphotag),
        )
        .expect("source-bound output plan");
        let target = outputs.primary_target().expect("primary target");
        write_text_output_artifact(target, "@UTF8\n@Begin\n*CHI:\tnew .\n@End\n")
            .await
            .expect("write output");

        assert_eq!(
            std::fs::read_to_string(&host_out).expect("host output"),
            "@UTF8\n@Begin\n*CHI:\tnew .\n@End\n"
        );
        assert_ne!(
            std::fs::metadata(&host_out).expect("metadata").ino(),
            old_inode
        );
    }

    /// Real `%mor` content change must still write through. The gate is
    /// strictly a noise-suppressor; any real difference must fall back
    /// to the existing write semantics.
    /// Guards: Changed morphology publishes the native admitted output through the real writer.
    /// Kind: contract
    /// Not a type: A native output proof cannot establish the disk replacement chosen when semantic morphology differs from the retained file.
    /// State: The root owns an existing CHAT primary. Native admission supplies the candidate proof with changed morphology. Actual publication writes requested/staged text, and the primary bytes are read back. No event recorder, registry or model runs.
    /// Concurrency: The test owns its root and immutable candidate proof. Publication completes before byte comparison and cleanup.
    #[tokio::test]
    async fn write_chat_output_artifact_with_provenance_gate_writes_when_real_content_differs() {
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        std::fs::create_dir_all(host_out.parent().unwrap()).unwrap();

        let original = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
*PAR:\thello .
%mor:\tintj|hello .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
";
        std::fs::write(&host_out, original).unwrap();

        let filesystem = paths_mode_filesystem(root.path(), &host_out, "nested/sample.cha");
        let file = crate::store::test_support::only_file(&filesystem.inputs);

        // Different `%mor` content: a legitimate result update.
        let candidate = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1:eng ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tintj|hi .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
";

        let outputs = SourceOutputs::plan(
            &filesystem,
            &file,
            ReleasedCommand::Morphotag,
            &test_options(ReleasedCommand::Morphotag),
        )
        .expect("source-bound output plan");
        let target = outputs.primary_target().expect("primary target");
        let output = admitted_morphotag_output(candidate);
        write_chat_output_artifact_with_provenance_gate(target, &output)
            .await
            .expect("gated write");

        let after = std::fs::read_to_string(&host_out).expect("read after");
        assert_eq!(
            after,
            output.as_str(),
            "primary output must be replaced when the new %mor content differs"
        );
    }

    /// First run (no existing file at the primary output path): the gate
    /// has nothing to compare against and must write through.
    /// Guards: An absent target receives the exact serialized native admitted output.
    /// Kind: contract
    /// Not a type: A native output proof cannot establish whether a previously absent filesystem destination is actually created and written.
    /// State: The root starts without the requested primary. Native admission supplies the candidate output proof. The real writer creates both requested and staged text copies. No event recorder, registry or model runs.
    /// Concurrency: The test owns the missing destination and candidate proof, awaits publication, then reads the resulting primary before root cleanup.
    #[tokio::test]
    async fn write_chat_output_artifact_with_provenance_gate_writes_when_target_missing() {
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        // Note: target file does NOT exist yet.

        let filesystem = paths_mode_filesystem(root.path(), &host_out, "nested/sample.cha");
        let file = crate::store::test_support::only_file(&filesystem.inputs);

        let candidate = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1:eng ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tintj|hello .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
";

        let outputs = SourceOutputs::plan(
            &filesystem,
            &file,
            ReleasedCommand::Morphotag,
            &test_options(ReleasedCommand::Morphotag),
        )
        .expect("source-bound output plan");
        let target = outputs.primary_target().expect("primary target");
        let output = admitted_morphotag_output(candidate);
        write_chat_output_artifact_with_provenance_gate(target, &output)
            .await
            .expect("gated write");

        let after = std::fs::read_to_string(&host_out).expect("read after");
        assert_eq!(after, output.as_str());
    }
}
