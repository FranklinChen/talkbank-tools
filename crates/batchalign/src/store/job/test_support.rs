//! Test fixtures for job lifecycle unit tests.
//!
//! Centralised so that every lifecycle test starts from the same
//! well-known in-memory state rather than re-inventing the fixture
//! in each test module.

use std::collections::{BTreeMap, HashMap};

use crate::api::{
    DisplayPath, JobStatus, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand,
};
use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
use crate::store::{
    FileStatus,
    job::{
        Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobRuntimeControl, JobScheduleState, JobSourceContext,
    },
};

/// Build a minimal `Job` whose `execution.status = Running`.
///
/// Used by lifecycle unit tests that need an in-flight job to
/// transition via `interrupt_for_shutdown` and similar operations.
/// All fields not relevant to lifecycle are set to inert defaults;
/// the fixture contains one file (`job-file.cha`) in `FileStatusKind::Queued`
/// so that `pending_files()` / `all_terminal_files_failed()` have a
/// non-empty domain to act on.
pub(crate) fn running_job_fixture() -> Job {
    let mut file_statuses = HashMap::new();
    let filename = DisplayPath::from("job-file.cha");
    file_statuses.insert(
        String::from(filename.clone()),
        FileStatus::new(filename.clone()),
    );

    Job {
        identity: JobIdentity {
            job_id: "test-job-running".into(),
            correlation_id: "corr-test-running".into(),
        },
        dispatch: JobDispatchConfig {
            command: ReleasedCommand::Morphotag,
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),
                ..Default::default()
            }),
            runtime_state: BTreeMap::new(),
            debug_traces: false,
        },
        source: JobSourceContext {
            submitter: Some(crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                String::new(),
            )),
            source_dir: Default::default(),
        },
        filesystem: JobFilesystemConfig {
            staging_dir: Default::default(),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
        },
        execution: JobExecutionState {
            lifecycle: crate::store::JobLifecycle::for_test(
                JobStatus::Running,
                crate::store::job::test_support::inputs_from_parts(
                    vec![filename],
                    vec![true],
                    false,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
            ),
            file_statuses,
            results: Vec::new(),
            error: None,
        },
        schedule: JobScheduleState {
            submitted_at: crate::unix_time(1_000_000.0),
            completed_at: None,
            next_eligible_at: None,
            num_workers: None,
            lease: None,
            last_cancel: None,
        },
        runtime: JobRuntimeControl {
            runner_active: true,
            ..JobRuntimeControl::unclaimed()
        },
        execution_plan: None,
    }
}

/// Build a `Job` with three named files in the `file_statuses` map,
/// all still `Queued`. Mirrors the 2026-05-11 morphotag job
/// (`150c824a-48e`) shape: 3 files submitted, none completed yet.
/// Tests then transition individual files via the standard
/// `mark_file_done` / `mark_file_error` methods.
pub(crate) fn three_file_job_fixture() -> Job {
    let mut job = running_job_fixture();
    let names = ["65-3.cha", "60home-3.cha", "65home-3.cha"];
    job.execution.file_statuses.clear();
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let path = DisplayPath::from(name);
        job.execution
            .file_statuses
            .insert(String::from(path.clone()), FileStatus::new(path.clone()));
        files.push(crate::store::ContentInput {
            display: path,
            kind: crate::store::InputKind::Chat,
        });
    }
    job.execution
        .lifecycle
        .replace_inputs_for_test(crate::store::JobInputs::Content(files));
    job
}

/// Fixture inputs from the old parallel parts, every paths-mode path admitted
/// through real admission as the user's own process (fixtures name paths
/// like `/src/a.cha` that need not exist). Display names come from
/// `filenames`, or, when it is empty, from each source's file name.
pub(crate) fn inputs_from_parts(
    filenames: Vec<DisplayPath>,
    has_chat: Vec<bool>,
    paths_mode: bool,
    source_paths: Vec<batchalign_types::paths::ClientPath>,
    output_paths: Vec<batchalign_types::paths::ClientPath>,
    before_paths: Vec<Option<batchalign_types::paths::ClientPath>>,
) -> crate::store::JobInputs {
    use crate::store::{ContentInput, InputKind, JobInputs, PathsInput};
    let kind = |index: usize, name: &str| match has_chat.get(index) {
        Some(true) => InputKind::Chat,
        Some(false) => InputKind::Media,
        None => InputKind::of_source_name(name),
    };
    match paths_mode {
        true => JobInputs::Paths {
            files: crate::store::PathsFiles::admit(
                source_paths
                    .iter()
                    .zip(&output_paths)
                    .enumerate()
                    .map(|(index, (source, output))| {
                        let display = filenames.get(index).cloned().unwrap_or_else(|| {
                            DisplayPath::from(
                                std::path::Path::new(source.as_str())
                                    .file_name()
                                    .map(|name| name.to_string_lossy().into_owned())
                                    .unwrap_or_default(),
                            )
                        });
                        PathsInput {
                            kind: kind(index, display.as_ref()),
                            display,
                            source: admitted(source.as_str()),
                            output: admitted(output.as_str()),
                            before: before_paths
                                .get(index)
                                .cloned()
                                .flatten()
                                .map(|before| admitted(before.as_str())),
                        }
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("fixture files that do not collide"),
            source_dir: None,
        },
        false => JobInputs::Content(
            filenames
                .iter()
                .enumerate()
                .map(|(index, display)| ContentInput {
                    kind: kind(index, display.as_ref()),
                    display: display.clone(),
                })
                .collect(),
        ),
    }
}

/// A fixture path admitted as the user's own process admits it.
pub(crate) fn admitted(path: &str) -> batchalign_types::paths::AdmittedPath {
    batchalign_types::paths::AllowedRoots::same_user()
        .admit(&batchalign_types::paths::ClientPath::new(path))
        .expect("an absolute fixture path")
}

/// Every file of `inputs` as pending runner work: what a fresh job's
/// snapshot carries.
pub(crate) fn pending_files(inputs: &crate::store::JobInputs) -> Vec<crate::store::PendingJobFile> {
    inputs.entries(|_| true).collect()
}

/// A snapshot whose pending files are every file of its own inputs, as a
/// fresh job's projection gives.
pub(crate) fn with_pending(
    mut snapshot: crate::store::RunnerJobSnapshot,
) -> crate::store::RunnerJobSnapshot {
    snapshot.pending_files = pending_files(&snapshot.filesystem.inputs);
    snapshot
}

/// The location of a one-file job's only input.
pub(crate) fn only_location(inputs: &crate::store::JobInputs) -> crate::store::InputLocation {
    let [file] = pending_files(inputs).try_into().expect("a one-file job");
    file.location().clone()
}
