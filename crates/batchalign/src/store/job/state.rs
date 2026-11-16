//! Closed job lifecycle states and their admitted transitions.
use super::{JobInputs, StoredInputs, StoredInputsRef};
use crate::api::{ActiveStatus, FinishedStatus, JobStatus, StatusKind};

/// Where a job is in its life, with the inputs that state needs: an active
/// job (queued or running) holds inputs this process admitted, so it can
/// run; a finished one holds its inputs as stored, admitted or recorded.
/// "Active with unadmitted inputs" is not a value this type has, so a job
/// that cannot run can never be queued.
///
/// The transitions are one method per event, each legal only from the
/// states that event leaves, so the state machine is these signatures and
/// their matches rather than a table beside them. Leaving finished for
/// active ([`Self::restart`], [`Self::resume`]) admits the inputs first,
/// under the rules in force now. A refused event changes nothing and says
/// which status refused it.
#[derive(Debug, Clone)]
pub enum JobLifecycle {
    /// Queued or running, on admitted inputs.
    Active {
        /// Which active status.
        status: ActiveStatus,
        /// Inputs this process admitted.
        inputs: JobInputs,
    },
    /// Completed, failed, cancelled, interrupted or failed at writeback.
    Finished {
        /// Which finished status.
        status: FinishedStatus,
        /// Its inputs as stored.
        inputs: StoredInputs,
    },
}

/// An event a job's lifecycle refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobEvent {
    /// A runner picked a queued job up.
    Dispatch,
    /// The memory gate parked an active job.
    MemoryRequeue,
    /// A user cancel or a shutdown stopped an active job.
    Stop,
    /// An active job ended under its own power.
    Finish,
    /// An operator restarted a cancelled or failed job.
    Restart,
    /// Startup recovery resumed an interrupted job.
    Resume,
    /// Startup recovery settled an interrupted job whose files all ended.
    Settle,
}

impl std::fmt::Display for JobEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dispatch => "be dispatched",
            Self::MemoryRequeue => "be requeued by the memory gate",
            Self::Stop => "be stopped",
            Self::Finish => "finish",
            Self::Restart => "be restarted",
            Self::Resume => "be resumed",
            Self::Settle => "be settled by recovery",
        })
    }
}

/// A lifecycle event refused in the job's current status. Most refusals are
/// ordinary races (a user cancels while a runner is mid-flight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a {from} job cannot {event}")]
pub(crate) struct Refused {
    /// The status the job was in, and stays in.
    pub(crate) from: JobStatus,
    /// What was refused.
    pub(crate) event: JobEvent,
}

/// Why a finished job did not become active again.
#[derive(Debug)]
pub(crate) enum Reactivation<E> {
    /// Its status does not allow the event; nothing was admitted.
    Refused(Refused),
    /// Its inputs were refused under the rules in force now.
    Admission(E),
}

/// How an active job finished under its own power.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Every file ended, none failed.
    Completed,
    /// It failed, or a file did.
    Failed,
    /// Its cancellation took effect as it finished.
    Cancelled,
    /// Its outputs could not be written back.
    WritebackFailed,
}

impl From<Outcome> for FinishedStatus {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Completed => Self::Completed,
            Outcome::Failed => Self::Failed,
            Outcome::Cancelled => Self::Cancelled,
            Outcome::WritebackFailed => Self::WritebackFailed,
        }
    }
}

/// What startup recovery concluded for an interrupted job whose files had
/// all ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Settlement {
    /// None failed.
    Completed,
    /// One failed, or its inputs were refused.
    Failed,
}

impl JobLifecycle {
    /// A job read back from the database. Nothing runs it in this process,
    /// so a stored active status means the server stopped while it ran: it
    /// is interrupted (startup marks such rows so first).
    pub(crate) fn recovered(status: JobStatus, inputs: StoredInputs) -> Self {
        let status = match status.kind() {
            StatusKind::Active(_) => FinishedStatus::Interrupted,
            StatusKind::Finished(status) => status,
        };
        Self::Finished { status, inputs }
    }

    /// The status.
    pub fn status(&self) -> JobStatus {
        match self {
            Self::Active { status, .. } => (*status).into(),
            Self::Finished { status, .. } => (*status).into(),
        }
    }

    /// The admitted inputs a runner may use: an active job's, or a
    /// finished one's that this process admitted (a cancelled run closing
    /// its files). `None` for recorded history.
    pub(crate) fn admitted(&self) -> Option<&JobInputs> {
        self.stored().admitted()
    }

    /// The inputs as stored, borrowed whichever state the job is in.
    pub(crate) fn stored(&self) -> StoredInputsRef<'_> {
        match self {
            Self::Active { inputs, .. } => StoredInputsRef::Admitted(inputs),
            Self::Finished { inputs, .. } => inputs.as_ref(),
        }
    }

    /// A refusal of `event` in the current status.
    fn refused(&self, event: JobEvent) -> Refused {
        Refused {
            from: self.status(),
            event,
        }
    }

    /// Queued to running.
    pub(crate) fn dispatch(&mut self) -> Result<(), Refused> {
        match self {
            Self::Active {
                status: status @ ActiveStatus::Queued,
                ..
            } => {
                *status = ActiveStatus::Running;
                Ok(())
            }
            Self::Active {
                status: ActiveStatus::Running,
                ..
            }
            | Self::Finished { .. } => Err(self.refused(JobEvent::Dispatch)),
        }
    }

    /// Active to queued, for the memory gate. A job that has left the
    /// active set (cancelled while waiting) stays gone.
    pub(crate) fn memory_requeue(&mut self) -> Result<(), Refused> {
        match self {
            Self::Active { status, .. } => {
                *status = ActiveStatus::Queued;
                Ok(())
            }
            Self::Finished { .. } => Err(self.refused(JobEvent::MemoryRequeue)),
        }
    }

    /// Active to cancelled or interrupted.
    pub(crate) fn stop(&mut self, stop: super::Stop) -> Result<(), Refused> {
        self.finish_active(stop.status(), JobEvent::Stop)
    }

    /// Active to its own `outcome`.
    pub(crate) fn finish(&mut self, outcome: Outcome) -> Result<(), Refused> {
        self.finish_active(outcome.into(), JobEvent::Finish)
    }

    /// Active to `next`, its admitted inputs becoming its stored ones.
    fn finish_active(&mut self, next: FinishedStatus, event: JobEvent) -> Result<(), Refused> {
        match self {
            Self::Active { inputs, .. } => {
                // Moved out from behind `&mut`; `Content(Vec::new())`
                // allocates nothing and is dropped as `self` is replaced on
                // the next line.
                let inputs = std::mem::replace(inputs, JobInputs::Content(Vec::new()));
                *self = Self::Finished {
                    status: next,
                    inputs: StoredInputs::Admitted(inputs),
                };
                Ok(())
            }
            Self::Finished { .. } => Err(self.refused(event)),
        }
    }

    /// Cancelled, failed or failed at writeback, back to queued on inputs
    /// `admit` admits now.
    pub(crate) fn restart<E>(
        &mut self,
        admit: impl FnOnce(StoredInputsRef<'_>) -> Result<JobInputs, E>,
    ) -> Result<(), Reactivation<E>> {
        let admitted = match self {
            Self::Finished {
                status:
                    FinishedStatus::Cancelled | FinishedStatus::Failed | FinishedStatus::WritebackFailed,
                inputs,
            } => admit(inputs.as_ref()),
            Self::Finished {
                status: FinishedStatus::Completed | FinishedStatus::Interrupted,
                ..
            }
            | Self::Active { .. } => {
                return Err(Reactivation::Refused(self.refused(JobEvent::Restart)));
            }
        };
        self.requeue_on(admitted)
    }

    /// Interrupted, back to queued on inputs `admit` admits now.
    pub(crate) fn resume<E>(
        &mut self,
        admit: impl FnOnce(StoredInputsRef<'_>) -> Result<JobInputs, E>,
    ) -> Result<(), Reactivation<E>> {
        let admitted = match self {
            Self::Finished {
                status: FinishedStatus::Interrupted,
                inputs,
            } => admit(inputs.as_ref()),
            Self::Finished {
                status:
                    FinishedStatus::Completed
                    | FinishedStatus::Failed
                    | FinishedStatus::Cancelled
                    | FinishedStatus::WritebackFailed,
                ..
            }
            | Self::Active { .. } => {
                return Err(Reactivation::Refused(self.refused(JobEvent::Resume)));
            }
        };
        self.requeue_on(admitted)
    }

    /// Queued on `admitted`, or unchanged when admission was refused.
    fn requeue_on<E>(&mut self, admitted: Result<JobInputs, E>) -> Result<(), Reactivation<E>> {
        let inputs = admitted.map_err(Reactivation::Admission)?;
        *self = Self::Active {
            status: ActiveStatus::Queued,
            inputs,
        };
        Ok(())
    }

    /// Interrupted to `settlement`, for a recovered job that will not run.
    pub(crate) fn settle(&mut self, settlement: Settlement) -> Result<(), Refused> {
        match self {
            Self::Finished {
                status: status @ FinishedStatus::Interrupted,
                ..
            } => {
                *status = match settlement {
                    Settlement::Completed => FinishedStatus::Completed,
                    Settlement::Failed => FinishedStatus::Failed,
                };
                Ok(())
            }
            Self::Finished {
                status:
                    FinishedStatus::Completed
                    | FinishedStatus::Failed
                    | FinishedStatus::Cancelled
                    | FinishedStatus::WritebackFailed,
                ..
            }
            | Self::Active { .. } => Err(self.refused(JobEvent::Settle)),
        }
    }
}
