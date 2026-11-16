//! Consuming model-fixture preparation through real job operations.

use std::collections::HashSet;

use crate::api::{ContentType, DisplayPath, MachineTime};
use crate::store::{
    CompletedFileOutput, ContentInput, FileCompletion, FileFailureRecord, InputKind, Job,
};

pub(crate) struct PendingFile;
pub(crate) struct ProcessingFile {
    started_at: MachineTime,
}

/// A declared CHAT file whose operations depend on its preparation phase.
pub(crate) struct ChatFileFixture<State> {
    input: ContentInput,
    state: State,
}

pub(crate) struct PlannedFile {
    declaration: FileDeclaration,
}

enum FileDeclaration {
    Pending(ContentInput),
    Processing { input: ContentInput, work: FileWork },
}

struct FileWork {
    started_at: MachineTime,
    finish: FileFinish,
}

enum FileFinish {
    Pending,
    Written {
        finished_at: MachineTime,
        completion: Box<FileCompletion>,
    },
    Failed(FileFailureRecord),
}

impl ChatFileFixture<PendingFile> {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            input: ContentInput {
                display: name.into(),
                kind: InputKind::Chat,
            },
            state: PendingFile,
        }
    }

    pub(crate) fn pending(self) -> PlannedFile {
        PlannedFile {
            declaration: FileDeclaration::Pending(self.input),
        }
    }

    pub(crate) fn processing(self, started_at: MachineTime) -> ChatFileFixture<ProcessingFile> {
        ChatFileFixture {
            input: self.input,
            state: ProcessingFile { started_at },
        }
    }
}

impl ChatFileFixture<ProcessingFile> {
    pub(crate) fn in_flight(self) -> PlannedFile {
        self.finish(FileFinish::Pending)
    }

    pub(crate) fn written(self, finished_at: MachineTime) -> PlannedFile {
        let completion = FileCompletion::Clean(self.output());
        self.finish(FileFinish::Written {
            finished_at,
            completion: Box::new(completion),
        })
    }

    pub(crate) fn diagnosed(
        self,
        finished_at: MachineTime,
        diagnostics: crate::api::FileOutputDiagnostics,
    ) -> PlannedFile {
        let completion = FileCompletion::Diagnosed {
            result: self.output(),
            diagnostics,
        };
        self.finish(FileFinish::Written {
            finished_at,
            completion: Box::new(completion),
        })
    }

    pub(crate) fn failed(self, failure: FileFailureRecord) -> PlannedFile {
        self.finish(FileFinish::Failed(failure))
    }

    fn output(&self) -> CompletedFileOutput {
        CompletedFileOutput {
            exclusions: Vec::new(),
            filename: self.input.display.clone(),
            content_type: ContentType::Chat,
            stamp: crate::api::FileStampOutcome::Unrecorded,
        }
    }

    fn finish(self, finish: FileFinish) -> PlannedFile {
        PlannedFile {
            declaration: FileDeclaration::Processing {
                input: self.input,
                work: FileWork {
                    started_at: self.state.started_at,
                    finish,
                },
            },
        }
    }
}

struct NamedWork {
    display: DisplayPath,
    work: FileWork,
}

pub(crate) struct Queued {
    work: Vec<NamedWork>,
}
pub(crate) struct Running;
pub(crate) struct Failed;
pub(crate) struct Deferred;

pub(crate) trait OwnsClaim {}
impl OwnsClaim for Running {}
impl OwnsClaim for Deferred {}
pub(crate) struct AwaitingRelease;

/// Fields stay private throughout preparation. Raw release consumes the proof.
pub(crate) struct JobFixture<State> {
    job: Job,
    state: State,
}

#[derive(Debug, thiserror::Error)]
#[error("duplicate fixture input {0}")]
pub(crate) struct DuplicateFixtureInput(DisplayPath);

impl JobFixture<Queued> {
    /// The first file is required. Every name is admitted once before setup.
    pub(crate) fn content(
        id: &str,
        first: PlannedFile,
        rest: impl IntoIterator<Item = PlannedFile>,
    ) -> Result<Self, DuplicateFixtureInput> {
        let mut names = HashSet::new();
        let mut inputs = Vec::new();
        let mut work = Vec::new();
        for file in std::iter::once(first).chain(rest) {
            let input = match file.declaration {
                FileDeclaration::Pending(input) => input,
                FileDeclaration::Processing {
                    input,
                    work: effect,
                } => {
                    work.push(NamedWork {
                        display: input.display.clone(),
                        work: effect,
                    });
                    input
                }
            };
            inputs.push(input);
        }
        for input in &inputs {
            if !names.insert(&input.display) {
                return Err(DuplicateFixtureInput(input.display.clone()));
            }
        }
        let mut job = super::morphotag_job_fixture(crate::store::JobInputs::Content(inputs));
        job.identity.job_id = id.into();
        job.identity.correlation_id = format!("fixture-{id}").into();
        Ok(Self {
            job,
            state: Queued { work },
        })
    }

    pub(crate) fn with_source(mut self, source: crate::store::JobSourceContext) -> Self {
        self.job.source = source;
        self
    }

    pub(crate) fn submitted_at(mut self, at: MachineTime) -> Self {
        self.job.schedule.submitted_at = at;
        self
    }

    pub(crate) fn start(
        mut self,
        node: &crate::api::NodeId,
        at: MachineTime,
        ttl: crate::config::LeaseTtl,
    ) -> JobFixture<Running> {
        self.job
            .claim_for_local_dispatch(node, at, ttl)
            .expect("queued fixture admits its real lease");
        assert!(
            self.job.mark_running(),
            "claimed fixture starts through dispatch admission"
        );
        for NamedWork { display, work } in self.state.work {
            self.job
                .mark_file_processing(display.as_ref(), work.started_at)
                .expect("declared file begins processing");
            match work.finish {
                FileFinish::Pending => {}
                FileFinish::Written {
                    finished_at,
                    completion,
                } => self
                    .job
                    .mark_file_done(display.as_ref(), finished_at, *completion)
                    .expect("declared file records its actual written result"),
                FileFinish::Failed(failure) => self
                    .job
                    .mark_file_error(display.as_ref(), &failure)
                    .expect("declared file records its actual failure"),
            }
        }
        JobFixture {
            job: self.job,
            state: Running,
        }
    }
}

impl JobFixture<Running> {
    pub(crate) fn defer(mut self, until: MachineTime) -> JobFixture<Deferred> {
        assert!(
            self.job.requeue_after_memory_gate(until),
            "claimed fixture accepts memory deferral"
        );
        JobFixture {
            job: self.job,
            state: Deferred,
        }
    }
}

impl<State: OwnsClaim> JobFixture<State> {
    pub(crate) fn fail(mut self, reason: &str, at: MachineTime) -> JobFixture<Failed> {
        assert!(self.job.fail(reason, at), "running fixture accepts failure");
        JobFixture {
            job: self.job,
            state: Failed,
        }
    }
}

impl JobFixture<Failed> {
    pub(crate) fn restart(
        mut self,
        authority: &batchalign_types::paths::AllowedRoots,
    ) -> Result<JobFixture<AwaitingRelease>, crate::error::ServerError> {
        self.job.prepare_for_restart(authority)?;
        Ok(JobFixture {
            job: self.job,
            state: AwaitingRelease,
        })
    }
}

impl JobFixture<AwaitingRelease> {
    pub(crate) fn release(mut self) -> JobFixture<Queued> {
        self.job.release_local_dispatch_claim();
        JobFixture {
            job: self.job,
            state: Queued { work: Vec::new() },
        }
    }
}

impl<State> JobFixture<State> {
    pub(crate) fn subject(&self) -> &Job {
        &self.job
    }
    pub(crate) fn into_subject(self) -> Job {
        self.job
    }
}
