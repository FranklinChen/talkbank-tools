//! In-memory job model and lifecycle methods.
//!
//! Split into focused submodules by responsibility:
//!
//! - `model`: job metadata and runner snapshots
//! - `inputs`: admitted inputs and their database representation
//! - `state`: lifecycle states and transitions
//! - `run`: run generations, claims and completion authority
//! - `pending`: source-bound pending files and preflight classification
//! - `completion`: file effect records
//! - `submitter`: submitter identity and database admission
//! - [`lease`]: queue lease management (claim, renew, release, expiry)
//! - [`file_status`]: per-file status mutations (processing, done, error, retry)
//! - [`lifecycle`]: job-level state transitions (running, failed, cancelled, restart, recovery)
//! - [`projections`]: API response projections (`JobInfo`, `JobListItem`, `RunnerJobSnapshot`)
//! - [`conflict`]: file-level conflict detection between incoming and active jobs

mod completion;
mod conflict;
mod deferral;
mod file_status;
mod inputs;
mod lease;
mod lifecycle;
mod model;
mod pending;
mod projections;
mod run;
mod state;
mod submitter;

#[cfg(test)]
pub(crate) mod test_support;

pub(crate) use completion::*;
pub use conflict::*;
pub(crate) use deferral::{DeferredRun, MemoryRequeueOutcome};
pub(crate) use file_status::FileWriteRefusal;
pub use inputs::*;
pub use model::*;
pub use pending::*;
pub(crate) use projections::{JobStatusColumns, Stop};
pub use run::*;
pub use state::*;
pub use submitter::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        ContentType, DisplayPath, FileProgressStage, FileStatusKind, JobStatus, NodeId,
    };
    use crate::scheduling::LeaseRecord;
    use crate::store::test_support::{ChatFileFixture, JobFixture};
    use std::collections::HashMap;

    /// One admitted input is sufficient for file-operation and lease contracts.
    fn sample_job(job_id: &str, filename: &str) -> Job {
        JobFixture::content(job_id, ChatFileFixture::new(filename).pending(), [])
            .expect("one input")
            .into_subject()
    }

    /// Guards: Pending projection excludes written files and retains unfinished names.
    /// Kind: policy
    /// Not a type: The fixture binds completion to its input but cannot establish which files the pending projection selects.
    /// State: Pure in-memory job values exercise model transitions. No filesystem, database or worker is required.
    /// Concurrency: Each case owns its inputs and job. No shared mutable state or asynchronous task is involved.
    #[test]
    fn pending_files_skip_terminal_entries() {
        let fixture = JobFixture::content(
            "pending",
            ChatFileFixture::new("a.cha")
                .processing(crate::unix_time(1.0))
                .written(crate::unix_time(2.0)),
            [ChatFileFixture::new("b.cha").pending()],
        )
        .expect("distinct inputs")
        .start(
            &NodeId::from("fixture-node"),
            crate::unix_time(0.0),
            crate::config::LeaseTtl::from_secs(90).unwrap(),
        );
        let pending = fixture.subject().pending_files().expect("admitted content");
        assert_eq!(
            pending
                .iter()
                .map(|file| file.filename().as_ref())
                .collect::<Vec<_>>(),
            ["b.cha"]
        );
    }

    /// Guards: Conflicts require matching submitter, source directory and filename.
    /// Kind: policy
    /// Not a type: Distinct admitted jobs cannot encode whether independently submitted scopes overlap under the conflict policy.
    /// State: Pure in-memory job values exercise model transitions. No filesystem, database or worker is required.
    /// Concurrency: Each case owns its inputs and job. No shared mutable state or asynchronous task is involved.
    #[test]
    fn find_conflicts_uses_submitter_and_source_scope() {
        enum Scope {
            Same,
            OtherSubmitter,
            OtherDirectory,
        }
        for scope in [Scope::Same, Scope::OtherSubmitter, Scope::OtherDirectory] {
            let active = JobFixture::content("active", ChatFileFixture::new("a.cha").pending(), [])
                .expect("one input")
                .start(
                    &NodeId::from("fixture-node"),
                    crate::unix_time(0.0),
                    crate::config::LeaseTtl::from_secs(90).unwrap(),
                )
                .into_subject();
            let source = match scope {
                Scope::Same => active.source.clone(),
                Scope::OtherSubmitter => JobSourceContext {
                    submitter: Some(Submitter::client(
                        std::net::Ipv4Addr::new(192, 0, 2, 1).into(),
                        String::new(),
                    )),
                    source_dir: active.source.source_dir.clone(),
                },
                Scope::OtherDirectory => JobSourceContext {
                    submitter: active.source.submitter.clone(),
                    source_dir: "/other".into(),
                },
            };
            let incoming =
                JobFixture::content("incoming", ChatFileFixture::new("a.cha").pending(), [])
                    .expect("one input")
                    .with_source(source);
            let jobs = HashMap::from([(active.identity.job_id.clone(), active)]);
            let conflicts = find_conflicts(&jobs, incoming.subject());
            let actual: Vec<(&str, &str)> = conflicts
                .iter()
                .map(|conflict| (conflict.filename.as_ref(), conflict.job_id.as_ref()))
                .collect::<Vec<_>>();
            let expected = match scope {
                Scope::Same => vec![("a.cha", "active")],
                Scope::OtherSubmitter | Scope::OtherDirectory => Vec::new(),
            };
            assert_eq!(actual, expected);
        }
    }

    /// Guards: Conflict lookup preserves a queued owner regardless of submission age.
    /// Kind: policy
    /// Not a type: The admitted queued state cannot establish the lookup policy for stale timestamps in the registry.
    /// State: Pure in-memory job values exercise model transitions. No filesystem, database or worker is required.
    /// Concurrency: Each case owns its inputs and job. No shared mutable state or asynchronous task is involved.
    #[test]
    fn find_conflicts_trusts_store_state_and_does_not_reconcile() {
        let abandoned =
            JobFixture::content("abandoned", ChatFileFixture::new("stale.cha").pending(), [])
                .expect("one input")
                .submitted_at(crate::unix_time(0.0))
                .into_subject();
        let incoming =
            JobFixture::content("incoming", ChatFileFixture::new("stale.cha").pending(), [])
                .expect("one input");
        let jobs = HashMap::from([(abandoned.identity.job_id.clone(), abandoned)]);
        let conflicts = find_conflicts(&jobs, incoming.subject());
        assert_eq!(
            conflicts
                .iter()
                .map(|conflict| conflict.job_id.as_ref())
                .collect::<Vec<_>>(),
            ["abandoned"]
        );
    }

    /// Guards: Claim, heartbeat and release expose the ordered lease values on the owned job.
    /// Kind: policy
    /// Not a type: The admitted job cannot encode the lease arithmetic and owner policy for caller-supplied clock observations.
    /// State: Pure owned job values and fixed instants exercise the model. No filesystem, database or worker is used.
    /// Concurrency: Each test owns its job and input declarations. No shared state or background task is involved.
    #[test]
    fn local_dispatch_claim_and_renew_roundtrip() {
        let mut job = sample_job("job-1", "a.cha");
        let node_id = NodeId::from("node-a");
        let claimed = job
            .claim_for_local_dispatch(
                &node_id,
                crate::unix_time(10.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            )
            .expect("claim");

        assert_eq!(claimed.leased_by_node(), &node_id);
        assert_eq!(claimed.heartbeat_at(), crate::unix_time(10.0));
        assert_eq!(claimed.expires_at(), crate::unix_time(100.0));
        assert!(job.runtime.runner_active);

        let renewed = job
            .renew_local_dispatch_lease(
                &node_id,
                crate::unix_time(20.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            )
            .expect("renew");
        assert_eq!(renewed.heartbeat_at(), crate::unix_time(20.0));
        assert_eq!(renewed.expires_at(), crate::unix_time(110.0));

        job.release_local_dispatch_claim();
        assert!(!job.runtime.runner_active);
        assert!(job.schedule.lease.is_none());
    }

    /// Guards: Dispatch readiness and wake time reflect live lease expiry and memory deferral.
    /// Kind: policy
    /// Not a type: The queued fixture cannot encode scheduling decisions for changing clock observations and independent persisted deadline values.
    /// State: Pure owned job values and fixed instants exercise the model. No filesystem, database or worker is used.
    /// Concurrency: Each test owns its job and input declarations. No shared state or background task is involved.
    #[test]
    fn ready_for_local_dispatch_respects_leases_and_deferrals() {
        let mut job = sample_job("job-1", "a.cha");
        let now = crate::unix_time(10.0);
        assert!(job.ready_for_local_dispatch(now));

        job.schedule.next_eligible_at = Some(crate::unix_time(20.0));
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(crate::unix_time(20.0))
        );

        job.schedule.next_eligible_at = None;
        job.schedule.lease = Some(
            LeaseRecord::new(
                NodeId::from("node-a"),
                crate::unix_time(10.0),
                crate::unix_time(30.0),
            )
            .expect("an ordered fixture lease"),
        );
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(crate::unix_time(30.0))
        );
    }

    /// Guards: Restart preserves written results, requeues failures, clears deadlines and waits for the old claim release without replaying preparation.
    /// Kind: contract
    /// Not a type: Consuming setup forbids restarting before failure or starting before release but cannot establish the production restart policy for retained outputs.
    /// State: Pure in-memory job values exercise model transitions. No filesystem, database or worker is required.
    /// Concurrency: Each case owns its inputs and job. No shared mutable state or asynchronous task is involved.
    #[test]
    fn restart_keeps_written_files_and_requeues_failure() {
        enum Written {
            Clean,
            Diagnosed,
        }
        for written in [Written::Clean, Written::Diagnosed] {
            let t = crate::unix_time;
            let file = ChatFileFixture::new("a.cha").processing(t(1.0));
            let (file, expected) = match written {
                Written::Clean => (file.written(t(2.0)), FileStatusKind::Done),
                Written::Diagnosed => (
                    file.diagnosed(
                        t(2.0),
                        crate::api::FileOutputDiagnostics::of_findings(
                            vec![crate::api::FileOutputDiagnostics::coded_finding(
                                "E241",
                                "reserved marker",
                            )],
                            Vec::new(),
                        ),
                    ),
                    FileStatusKind::Diagnosed,
                ),
            };
            let fixture = JobFixture::content(
                "restart",
                file,
                [ChatFileFixture::new("b.cha")
                    .processing(t(1.0))
                    .failed(FileFailureRecord {
                        message: "worker crashed".into(),
                        category: crate::scheduling::FailureCategory::WorkerCrash,
                        finished_at: crate::store::EventTime::fixed(t(3.0)),
                    })],
            )
            .expect("distinct inputs")
            .start(
                &NodeId::from("fixture-node"),
                t(0.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            )
            .defer(t(25.0))
            .fail("worker crashed", t(20.0));
            assert!(fixture.subject().any_terminal_files_failed());
            assert_eq!(fixture.subject().schedule.completed_at, Some(t(20.0)));
            assert_eq!(fixture.subject().schedule.next_eligible_at, Some(t(25.0)));
            assert!(fixture.subject().schedule.lease.is_some());
            let fixture = fixture
                .restart(&batchalign_types::paths::AllowedRoots::same_user())
                .expect("admitted restart");
            let job = fixture.subject();
            assert_eq!(job.status(), JobStatus::Queued);
            assert!(job.execution.error.is_none());
            assert_eq!(job.execution.completed_files(), 1);
            assert_eq!(job.execution.results.len(), 1);
            assert_eq!(job.execution.file_statuses["a.cha"].status(), expected);
            assert_eq!(
                job.execution.file_statuses["b.cha"].status(),
                FileStatusKind::Queued
            );
            assert!(job.schedule.completed_at.is_none());
            assert!(job.schedule.next_eligible_at.is_none());
            assert!(job.schedule.lease.is_none());
            assert!(job.runtime.runner_active);
            assert_eq!(
                job.runtime.run_generation,
                crate::store::RunGeneration::FIRST.next()
            );
            let fixture = fixture.release();
            assert!(!fixture.subject().runtime.runner_active);
            let fixture = fixture.start(
                &NodeId::from("next-node"),
                t(30.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            );
            let job = fixture.subject();
            assert_eq!(
                job.execution.results.len(),
                1,
                "completed work is never replayed"
            );
            assert_eq!(job.execution.file_statuses["a.cha"].status(), expected);
            assert_eq!(
                job.execution.file_statuses["b.cha"].status(),
                FileStatusKind::Queued
            );
        }
    }

    /// Guards: Written output removes retry errors and preserves exclusions in the file projection and success result.
    /// Kind: contract
    /// Not a type: Input ownership cannot establish the completion operation's semantic projection of authored exclusions and prior retry data.
    /// State: Pure owned job values and fixed instants exercise the model. No filesystem, database or worker is used.
    /// Concurrency: Each test owns its job and input declarations. No shared state or background task is involved.
    #[test]
    fn mark_file_done_updates_file_state() {
        let mut job = sample_job("job-1", "a.cha");
        assert_eq!(
            job.mark_file_retry_pending(
                "a.cha",
                &FileRetryRecord {
                    message: "stale".into(),
                    category: crate::scheduling::FailureCategory::WorkerTimeout,
                    finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                    retry_at: crate::unix_time(11.5),
                }
            ),
            Ok(())
        );

        assert_eq!(
            job.mark_file_done(
                "a.cha",
                crate::unix_time(12.0),
                FileCompletion::Clean(CompletedFileOutput {
                    exclusions: vec![crate::api::OutputExclusionRecord::NotInRecording {
                        excluded_utterances: 1,
                        excluded_words: 3,
                        first_excluded: vec![crate::api::ExcludedUtteranceRecord {
                            location: crate::api::UtteranceLocator::for_test(
                                2,
                                "*PAR:\tdiary note . [+ diary]"
                            )
                            .into(),
                            words: 3,
                            postcode: crate::api::OffRecordPostcode::Diary,
                        }],
                    }],
                    filename: DisplayPath::from("a.cha"),
                    content_type: ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                })
            ),
            Ok(())
        );

        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert_eq!(entry.status, FileStatusKind::Done);
        // A clean file's exclusions move onto its phase and reach the wire.
        assert_eq!(entry.exclusions.len(), 1, "{:?}", entry.exclusions);
        assert!(
            entry.diagnostics.is_none(),
            "an exclusion is not a diagnosis"
        );
        assert_eq!(entry.finished_at, Some(crate::unix_time(12.0)));
        assert!(entry.error.is_none(), "the retry's error is gone");
        assert!(entry.error_category.is_none());
        assert_eq!(job.execution.completed_files(), 1);
        assert_eq!(job.execution.results.len(), 1);
    }

    /// Guards: A retry deadline exposes processing and retry progress without a terminal duration.
    /// Kind: contract
    /// Not a type: Admitted input ownership cannot establish the displayed retry timestamp, progress stage and absence of a terminal duration.
    /// State: Pure owned job values and fixed instants exercise the model. No filesystem, database or worker is used.
    /// Concurrency: Each test owns its job and input declarations. No shared state or background task is involved.
    #[test]
    fn mark_file_retry_pending_sets_retry_metadata() {
        let mut job = sample_job("job-1", "a.cha");

        assert_eq!(
            job.mark_file_retry_pending(
                "a.cha",
                &FileRetryRecord {
                    message: "retry".into(),
                    category: crate::scheduling::FailureCategory::WorkerTimeout,
                    finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                    retry_at: crate::unix_time(20.0),
                }
            ),
            Ok(())
        );

        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert_eq!(entry.status, FileStatusKind::Processing);
        assert_eq!(entry.next_eligible_at, Some(crate::unix_time(20.0)));
        assert_eq!(
            entry.progress_stage,
            Some(FileProgressStage::RetryScheduled)
        );
        // Still in flight: the failed attempt's end is not the file's finish.
        assert_eq!(entry.finished_at, None);
        assert_eq!(entry.duration_s, None);
    }

    /// Guards: Clearing a prior retry removes its error, finish and next-attempt columns.
    /// Kind: contract
    /// Not a type: The admitted job cannot establish how the real retry clearing operation projects previously recorded failure columns.
    /// State: Pure owned job values and fixed instants exercise the model. No filesystem, database or worker is used.
    /// Concurrency: Each test owns its job and input declarations. No shared state or background task is involved.
    #[test]
    fn clear_file_retry_state_clears_retry_error_metadata() {
        let mut job = sample_job("job-1", "a.cha");
        assert_eq!(
            job.mark_file_retry_pending(
                "a.cha",
                &FileRetryRecord {
                    message: "retry".into(),
                    category: crate::scheduling::FailureCategory::WorkerTimeout,
                    finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                    retry_at: crate::unix_time(20.0),
                }
            ),
            Ok(())
        );

        assert_eq!(job.clear_file_retry_state("a.cha"), Ok(()));
        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert!(entry.error.is_none());
        assert!(entry.error_category.is_none());
        assert!(entry.finished_at.is_none());
        assert!(entry.next_eligible_at.is_none());
    }

    /// A row image: the columns a phase owns, as `from_row` reads them.
    fn row(
        status: FileStatusKind,
        error: Option<&str>,
        started_at: Option<crate::api::MachineTime>,
        finished_at: Option<crate::api::MachineTime>,
        next_eligible_at: Option<crate::api::MachineTime>,
    ) -> crate::store::FilePhaseColumns<'_> {
        crate::store::FilePhaseColumns {
            exclusions: None,
            status,
            error,
            error_category: error.map(|_| crate::scheduling::FailureCategory::WorkerTimeout),
            diagnostics: None,
            started_at,
            finished_at,
            next_eligible_at,
        }
    }

    /// The database boundary rebuilds the one phase a row describes: a
    /// processing row with a deadline is a pending retry whose failed-attempt
    /// end is not a finish time, and an error column a phase cannot hold is
    /// reported rather than kept.
    #[test]
    fn rows_become_the_phase_they_describe() {
        use crate::store::FilePhase;
        let t = crate::unix_time;

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Processing,
            Some("timeout"),
            Some(t(10.0)),
            Some(t(11.0)),
            Some(t(20.0)),
        ));
        assert_eq!(dropped, None);
        assert_eq!(phase.kind(), FileStatusKind::Processing);
        assert_eq!(phase.finished_at(), None);
        assert_eq!(phase.next_eligible_at(), Some(t(20.0)));
        assert_eq!(phase.last_activity_at(), Some(t(11.0)));

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Queued,
            Some("timeout"),
            Some(t(10.0)),
            Some(t(11.0)),
            None,
        ));
        assert_eq!(phase, FilePhase::Queued);
        assert!(dropped.is_some(), "the stray error is reported");
    }

    /// `FilePhase::columns` and `FilePhase::from_row` are inverses: every
    /// phase written as its column image reads back as itself, with nothing
    /// reported dropped. Two functions that must agree, which no type pins.
    #[test]
    fn every_phase_roundtrips_through_its_columns() {
        use crate::store::{FileFailure, FilePhase};
        let failure = || {
            FileFailure::recorded(
                "timeout".into(),
                crate::scheduling::FailureCategory::WorkerTimeout,
            )
        };
        let t = crate::unix_time;
        let phases = [
            FilePhase::Queued,
            FilePhase::Processing {
                started_at: Some(t(1.0)),
            },
            FilePhase::RetryPending {
                started_at: Some(t(1.0)),
                failed_at: Some(t(2.0)),
                retry_at: t(3.0),
                failure: failure(),
            },
            FilePhase::Done {
                exclusions: Vec::new(),
                started_at: Some(t(1.0)),
                finished_at: Some(t(2.0)),
            },
            FilePhase::Diagnosed {
                exclusions: Vec::new(),
                started_at: Some(t(1.0)),
                finished_at: Some(t(2.0)),
                diagnostics: Some(crate::api::FileOutputDiagnostics::of_findings(
                    vec![crate::api::FileOutputDiagnostics::coded_finding(
                        "E220",
                        "digits inside a word",
                    )],
                    vec![crate::api::OutputShortfallRecord::StageSkipped {
                        stage: crate::api::OptionalStage::Morphosyntax,
                    }],
                )),
            },
            FilePhase::Diagnosed {
                exclusions: Vec::new(),
                started_at: None,
                finished_at: None,
                diagnostics: None,
            },
            FilePhase::Error {
                started_at: None,
                finished_at: Some(t(2.0)),
                failure: failure(),
            },
            FilePhase::Error {
                started_at: None,
                finished_at: None,
                failure: FileFailure::of_failed_row(None, None),
            },
            FilePhase::Interrupted {
                started_at: Some(t(1.0)),
                last_failure: Some(failure()),
            },
            FilePhase::Interrupted {
                started_at: None,
                last_failure: None,
            },
        ];
        for phase in phases {
            let (back, dropped) = FilePhase::from_row(phase.columns());
            assert_eq!(back, phase);
            assert_eq!(dropped, None, "{phase:?} wrote a column it does not own");
        }
    }

    /// A failure is never empty: stored columns holding neither a message nor
    /// a category are no failure, so an interrupted file cannot carry a
    /// failure that writes nothing and reads back as none.
    #[test]
    fn an_empty_failure_is_no_failure() {
        use crate::store::FileFailure;
        assert_eq!(FileFailure::from_columns(None, None), None);
        let unrecorded = FileFailure::of_failed_row(None, None);
        assert_eq!((unrecorded.message(), unrecorded.category()), (None, None));
        let category_only =
            FileFailure::from_columns(None, Some(crate::scheduling::FailureCategory::WorkerCrash))
                .expect("a category alone is a failure");
        assert_eq!(category_only.message(), None);
    }

    /// Interrupting moves only in-flight phases, keeping the start and the
    /// failure a pending retry was retrying.
    #[test]
    fn interrupting_keeps_the_start_and_the_retried_failure() {
        use crate::store::{FileFailure, FilePhase};
        let t = crate::unix_time;
        let failure = FileFailure::recorded(
            "crash".into(),
            crate::scheduling::FailureCategory::WorkerCrash,
        );
        let retry = FilePhase::RetryPending {
            started_at: Some(t(1.0)),
            failed_at: Some(t(2.0)),
            retry_at: t(3.0),
            failure: failure.clone(),
        };
        assert_eq!(
            retry.interrupted(),
            FilePhase::Interrupted {
                started_at: Some(t(1.0)),
                last_failure: Some(failure),
            }
        );
        let done = FilePhase::Done {
            exclusions: Vec::new(),
            started_at: Some(t(1.0)),
            finished_at: Some(t(2.0)),
        };
        assert_eq!(done.clone().interrupted(), done);
    }

    /// Every phase reports the columns it cannot hold, by one rule.
    #[test]
    fn a_column_a_phase_does_not_own_is_reported_for_every_phase() {
        use crate::store::FilePhase;
        let t = crate::unix_time;

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Processing,
            None,
            Some(t(1.0)),
            Some(t(2.0)),
            None,
        ));
        assert_eq!(
            phase,
            FilePhase::Processing {
                started_at: Some(t(1.0))
            }
        );
        assert_eq!(
            dropped.as_deref(),
            Some("the processing row held finished_at, which was not kept")
        );

        let (_, dropped) = FilePhase::from_row(row(
            FileStatusKind::Error,
            None,
            None,
            Some(t(2.0)),
            Some(t(3.0)),
        ));
        assert_eq!(
            dropped.as_deref(),
            Some("the error row held next_eligible_at, which was not kept")
        );

        let (_, dropped) = FilePhase::from_row(row(
            FileStatusKind::Interrupted,
            None,
            Some(t(1.0)),
            Some(t(2.0)),
            Some(t(3.0)),
        ));
        assert_eq!(
            dropped.as_deref(),
            Some("the interrupted row held finished_at, next_eligible_at, which were not kept")
        );
    }
}
