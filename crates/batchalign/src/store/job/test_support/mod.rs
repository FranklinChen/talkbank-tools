//! Test fixtures for job lifecycle unit tests.
//!
//! Centralised so that every lifecycle test starts from the same
//! well-known in-memory state rather than re-inventing the fixture
//! in each test module.

mod clock;
mod inputs;
mod scenario;

pub use clock::ManualClockStore;
pub use inputs::{PathsFixture, admitted, only_file, paths_inputs, pending_files, with_pending};
pub use scenario::{ChatFileFixture, JobFixture};

use std::collections::BTreeMap;

use crate::api::{LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand};
use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
use crate::store::{
    FileStatus,
    job::{
        Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobRuntimeControl, JobScheduleState, JobSourceContext,
    },
};

/// One admitted content input, queued and unclaimed. The same construction
/// serves runner tests and the consuming transition to the running fixture.
pub fn queued_morphotag_job_fixture(input: crate::store::ContentInput) -> Job {
    morphotag_job_fixture(crate::store::JobInputs::Content(vec![input]))
}

/// Every fresh model begins queued, on its own inputs and without a claim.
fn morphotag_job_fixture(inputs: crate::store::JobInputs) -> Job {
    let file_statuses = inputs
        .displays()
        .map(|display| (display.to_string(), FileStatus::new(display.clone())))
        .collect();

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
            lifecycle: crate::store::JobLifecycle::Active {
                status: crate::api::ActiveStatus::Queued,
                inputs,
            },
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
        runtime: JobRuntimeControl::unclaimed(),
        execution_plan: None,
    }
}

/// Admit a fixture through the real actor and return its exclusive run owner.
pub(in crate::store) async fn claim_registry_runner(
    store: &crate::store::JobStore,
    id: &crate::api::JobId,
) -> crate::store::RunnerClaim {
    match store.begin_runner(id).await.expect("job exists") {
        crate::store::BeginRunnerOutcome::Started { claim, .. } => claim,
        crate::store::BeginRunnerOutcome::RunnerStillLive => panic!("test job already claimed"),
        crate::store::BeginRunnerOutcome::Stopped { .. } => panic!("test job already stopped"),
        crate::store::BeginRunnerOutcome::NotRunnable { status } => {
            panic!("test job is not runnable: {status}")
        }
    }
}

/// A queued paths job uses the same admitted input set and lifecycle factory.
pub fn queued_paths_morphotag_job_fixture(files: crate::store::PathsFiles) -> Job {
    morphotag_job_fixture(crate::store::JobInputs::Paths {
        files,
        source_dir: None,
    })
}

/// Admit through the production composition edge and retain its actual store.
pub async fn claim_runner(
    store: &std::sync::Arc<crate::store::JobStore>,
    id: &crate::api::JobId,
) -> crate::store::AdmittedRun {
    match store.admit_runner(id).await.expect("job exists") {
        crate::store::RunnerAdmission::Started(claim) => claim,
        crate::store::RunnerAdmission::Stopped(_) => panic!("fixture job is stopped"),
        crate::store::RunnerAdmission::RunnerStillLive => panic!("fixture runner already live"),
        crate::store::RunnerAdmission::NotRunnable { status } => {
            panic!("fixture job cannot run: {status}")
        }
    }
}
