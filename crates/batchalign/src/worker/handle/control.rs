//! A control request owns its wait policy and the only reply type it admits.

use super::protocol::{
    CapabilitiesRequest, ControlRequestId, EnsureTaskRequest, ReportedFailure, WorkerRequest,
    WorkerResponse,
};
use crate::types::worker_v2::{ExecuteResponseV2, ProgressEventV2, batched_items_timeout};
use crate::worker::error::{WorkerError, WorkerWait};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, EnsureTaskResponse, InferRequest, InferResponse,
    InferTask, LoadingStall, ResponseWait, WorkerCapabilities, WorkerHealthResponse,
};

mod sealed {
    pub(crate) trait Sealed {}
    pub(crate) trait Operation {}
}

pub(crate) trait ExpectedReply: sealed::Sealed + Send {
    type Reply;
    fn wait(&self) -> ResponseWait;
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError>;
}

/// Only real request producers implement this effect capability. ExpectedLoad
/// is an observation descriptor, so it can inspect bytes without loading a model.
pub(crate) trait ControlOperation: ExpectedReply + sealed::Operation {
    fn request(&self) -> WorkerRequest<'_>;
}

pub(crate) struct ControlCompletion<Reply> {
    reply: Result<Reply, WorkerError>,
}
impl<Reply> ControlCompletion<Reply> {
    fn answered(reply: Result<Reply, WorkerError>) -> Self {
        Self { reply }
    }
    fn map<Next>(self, map: impl FnOnce(Reply) -> Next) -> ControlCompletion<Next> {
        ControlCompletion {
            reply: self.reply.map(map),
        }
    }
    pub(crate) fn into_result(self) -> Result<Reply, WorkerError> {
        self.reply
    }
}

/// Progress has no representation in an operation's terminal input.
pub(crate) enum ControlTerminal {
    Infer(InferResponse),
    BatchInfer(BatchInferResponse),
    Execute(ExecuteResponseV2),
    TaskLoaded(EnsureTaskResponse),
    Health(WorkerHealthResponse),
    Capabilities(WorkerCapabilities),
    Shutdown,
    Error(ReportedFailure),
}

impl std::fmt::Debug for ControlTerminal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infer(value) => formatter.debug_tuple("Infer").field(value).finish(),
            Self::BatchInfer(value) => formatter.debug_tuple("BatchInfer").field(value).finish(),
            Self::Execute(value) => formatter.debug_tuple("Execute").field(value).finish(),
            Self::TaskLoaded(value) => formatter.debug_tuple("TaskLoaded").field(value).finish(),
            Self::Health(value) => formatter.debug_tuple("Health").field(value).finish(),
            Self::Capabilities(value) => {
                formatter.debug_tuple("Capabilities").field(value).finish()
            }
            Self::Error(value) => formatter.debug_tuple("Error").field(value).finish(),
            Self::Shutdown => formatter.write_str("Shutdown"),
        }
    }
}

pub(super) enum ControlFrame {
    Progress(ProgressEventV2),
    Terminal(ControlTerminal),
}
impl From<WorkerResponse> for ControlFrame {
    fn from(response: WorkerResponse) -> Self {
        match response {
            WorkerResponse::ProgressV2 { event } => Self::Progress(event),
            WorkerResponse::Infer { response } => Self::Terminal(ControlTerminal::Infer(response)),
            WorkerResponse::BatchInfer { response } => {
                Self::Terminal(ControlTerminal::BatchInfer(response))
            }
            WorkerResponse::ExecuteV2 { response } => {
                Self::Terminal(ControlTerminal::Execute(response))
            }
            WorkerResponse::EnsureTask { response } => {
                Self::Terminal(ControlTerminal::TaskLoaded(response))
            }
            WorkerResponse::Health { response } => {
                Self::Terminal(ControlTerminal::Health(response))
            }
            WorkerResponse::Capabilities { response } => {
                Self::Terminal(ControlTerminal::Capabilities(response))
            }
            WorkerResponse::Shutdown => Self::Terminal(ControlTerminal::Shutdown),
            WorkerResponse::Error(failure) => Self::Terminal(ControlTerminal::Error(failure)),
        }
    }
}

pub(crate) struct Health;
pub(crate) struct Capabilities {
    request_id: ControlRequestId,
}
pub(crate) struct Infer<'a> {
    request: &'a InferRequest,
}
pub(crate) struct ExpectedInfer;
pub(crate) struct BatchInfer<'a> {
    request: &'a BatchInferRequest,
    expected: ExpectedBatch,
}
pub(crate) struct ExpectedBatch {
    items: usize,
}
impl ExpectedBatch {
    pub(crate) fn new(items: usize) -> Self {
        Self { items }
    }
}

impl Capabilities {
    pub(crate) fn new() -> Self {
        Self {
            request_id: ControlRequestId::next(),
        }
    }
}
impl<'a> Infer<'a> {
    pub(crate) fn new(request: &'a InferRequest) -> Self {
        Self { request }
    }
}
impl<'a> BatchInfer<'a> {
    pub(crate) fn new(request: &'a BatchInferRequest) -> Self {
        Self {
            request,
            expected: ExpectedBatch::new(request.items.len()),
        }
    }
}

/// This describes a desired observation and confers no request-sending capability.
pub(crate) struct ExpectedLoad {
    task: InferTask,
    stall: LoadingStall,
}
impl ExpectedLoad {
    pub(crate) fn new(task: InferTask, stall: LoadingStall) -> Self {
        Self { task, stall }
    }
}

/// The actual load owns its grant through matching terminal admission.
pub(crate) struct Loading {
    admitted: crate::worker::pool::AdmittedLoad,
    expected: ExpectedLoad,
    request_id: ControlRequestId,
}
impl Loading {
    pub(crate) fn new(admitted: crate::worker::pool::AdmittedLoad, stall: LoadingStall) -> Self {
        let expected = ExpectedLoad::new(admitted.load().task(), stall);
        Self {
            admitted,
            expected,
            request_id: ControlRequestId::next(),
        }
    }
}

pub(crate) struct CompletedLoad {
    admitted: crate::worker::pool::AdmittedLoad,
    response: EnsureTaskResponse,
}
impl CompletedLoad {
    pub(crate) fn response(&self) -> &EnsureTaskResponse {
        &self.response
    }
    pub(crate) async fn confirm(self) -> crate::worker::pool::ConfirmedLoad {
        self.admitted.confirm().await
    }
}

impl sealed::Sealed for Loading {}
impl ExpectedReply for Loading {
    type Reply = CompletedLoad;
    fn wait(&self) -> ResponseWait {
        self.expected.wait()
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        let Self {
            admitted, expected, ..
        } = self;
        expected
            .admit(terminal)
            .map(|complete| complete.map(|response| CompletedLoad { admitted, response }))
    }
}
impl ControlOperation for Loading {
    fn request(&self) -> WorkerRequest<'_> {
        WorkerRequest::EnsureTask {
            request: EnsureTaskRequest {
                request_id: &self.request_id,
                task: self.admitted.load().task(),
                engine_overrides: self.admitted.load().wire_overrides(),
            },
        }
    }
}

impl sealed::Sealed for Health {}
impl ExpectedReply for Health {
    type Reply = WorkerHealthResponse;
    fn wait(&self) -> ResponseWait {
        ResponseWait::Budget {
            waited_for: WorkerWait::Health,
            limit: crate::worker::HEALTH_TIMEOUT,
        }
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        match terminal {
            ControlTerminal::Health(response) => {
                Ok(ControlCompletion::answered(if response.status.is_ok() {
                    Ok(response)
                } else {
                    Err(WorkerError::HealthCheckFailed(format!(
                        "status={}",
                        response.status
                    )))
                }))
            }
            ControlTerminal::Error(failure) => Ok(ControlCompletion::answered(Err(
                failure.into_health_error()
            ))),
            other @ (ControlTerminal::Infer(_)
            | ControlTerminal::BatchInfer(_)
            | ControlTerminal::Execute(_)
            | ControlTerminal::TaskLoaded(_)
            | ControlTerminal::Capabilities(_)
            | ControlTerminal::Shutdown) => Err(WorkerError::HealthCheckFailed(format!(
                "unexpected response for health: {other:?}"
            ))),
        }
    }
}

impl sealed::Sealed for Capabilities {}
impl ExpectedReply for Capabilities {
    type Reply = WorkerCapabilities;
    fn wait(&self) -> ResponseWait {
        ResponseWait::Budget {
            waited_for: WorkerWait::Capabilities,
            limit: crate::worker::CAPABILITY_TIMEOUT,
        }
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        match terminal {
            ControlTerminal::Capabilities(response) => {
                Ok(ControlCompletion::answered(Ok(response)))
            }
            ControlTerminal::Error(failure) => Ok(ControlCompletion::answered(Err(
                failure.into_worker_error()
            ))),
            other @ (ControlTerminal::Infer(_)
            | ControlTerminal::BatchInfer(_)
            | ControlTerminal::Execute(_)
            | ControlTerminal::TaskLoaded(_)
            | ControlTerminal::Health(_)
            | ControlTerminal::Shutdown) => Err(WorkerError::Protocol(format!(
                "unexpected response for capabilities: {other:?}"
            ))),
        }
    }
}

impl sealed::Sealed for ExpectedInfer {}
impl ExpectedReply for ExpectedInfer {
    type Reply = InferResponse;
    fn wait(&self) -> ResponseWait {
        ResponseWait::Budget {
            waited_for: WorkerWait::Infer,
            limit: crate::worker::INFER_TIMEOUT,
        }
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        match terminal {
            ControlTerminal::Infer(response) => Ok(ControlCompletion::answered(Ok(response))),
            ControlTerminal::Error(failure) => Ok(ControlCompletion::answered(Err(
                failure.into_worker_error()
            ))),
            other @ (ControlTerminal::BatchInfer(_)
            | ControlTerminal::Execute(_)
            | ControlTerminal::TaskLoaded(_)
            | ControlTerminal::Health(_)
            | ControlTerminal::Capabilities(_)
            | ControlTerminal::Shutdown) => Err(WorkerError::Protocol(format!(
                "unexpected response for infer: {other:?}"
            ))),
        }
    }
}

impl sealed::Sealed for ExpectedBatch {}
impl ExpectedReply for ExpectedBatch {
    type Reply = BatchInferResponse;
    fn wait(&self) -> ResponseWait {
        ResponseWait::Budget {
            waited_for: WorkerWait::BatchInfer { items: self.items },
            limit: batched_items_timeout(self.items as u64),
        }
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        match terminal {
            ControlTerminal::BatchInfer(response) => Ok(ControlCompletion::answered(Ok(response))),
            ControlTerminal::Error(failure) => Ok(ControlCompletion::answered(Err(
                failure.into_worker_error()
            ))),
            other @ (ControlTerminal::Infer(_)
            | ControlTerminal::Execute(_)
            | ControlTerminal::TaskLoaded(_)
            | ControlTerminal::Health(_)
            | ControlTerminal::Capabilities(_)
            | ControlTerminal::Shutdown) => Err(WorkerError::Protocol(format!(
                "unexpected response for batch_infer: {other:?}"
            ))),
        }
    }
}

impl sealed::Sealed for ExpectedLoad {}
impl ExpectedReply for ExpectedLoad {
    type Reply = EnsureTaskResponse;
    fn wait(&self) -> ResponseWait {
        ResponseWait::Loading {
            waited_for: crate::worker::LoadingWait::EnsureTask { task: self.task },
            stall: self.stall,
        }
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        match terminal {
            ControlTerminal::TaskLoaded(response) => {
                Ok(ControlCompletion::answered(Ok(response.about(self.task)?)))
            }
            ControlTerminal::Error(failure) => Ok(ControlCompletion::answered(Err(
                failure.into_ensure_task_error()
            ))),
            other @ (ControlTerminal::Infer(_)
            | ControlTerminal::BatchInfer(_)
            | ControlTerminal::Execute(_)
            | ControlTerminal::Health(_)
            | ControlTerminal::Capabilities(_)
            | ControlTerminal::Shutdown) => Err(WorkerError::Protocol(format!(
                "unexpected response for ensure_task: {other:?}"
            ))),
        }
    }
}

impl ControlOperation for Health {
    fn request(&self) -> WorkerRequest<'_> {
        WorkerRequest::Health
    }
}
impl ControlOperation for Capabilities {
    fn request(&self) -> WorkerRequest<'_> {
        WorkerRequest::Capabilities {
            request: CapabilitiesRequest {
                request_id: &self.request_id,
            },
        }
    }
}
impl ControlOperation for Infer<'_> {
    fn request(&self) -> WorkerRequest<'_> {
        WorkerRequest::Infer {
            request: self.request,
        }
    }
}
impl ControlOperation for BatchInfer<'_> {
    fn request(&self) -> WorkerRequest<'_> {
        WorkerRequest::BatchInfer {
            request: self.request,
        }
    }
}

impl sealed::Sealed for Infer<'_> {}
impl ExpectedReply for Infer<'_> {
    type Reply = InferResponse;
    fn wait(&self) -> ResponseWait {
        ExpectedInfer.wait()
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        ExpectedInfer.admit(terminal)
    }
}
impl sealed::Sealed for BatchInfer<'_> {}
impl ExpectedReply for BatchInfer<'_> {
    type Reply = BatchInferResponse;
    fn wait(&self) -> ResponseWait {
        self.expected.wait()
    }
    fn admit(
        self,
        terminal: ControlTerminal,
    ) -> Result<ControlCompletion<Self::Reply>, WorkerError> {
        self.expected.admit(terminal)
    }
}

impl sealed::Operation for Health {}
impl sealed::Operation for Capabilities {}
impl sealed::Operation for Infer<'_> {}
impl sealed::Operation for BatchInfer<'_> {}
impl sealed::Operation for Loading {}
