//! Option-family admission for consuming dispatch planning.

use crate::api::{LanguageSpec, NumSpeakers, ReleasedCommand};
use crate::options::{AlignOptions, BenchmarkOptions, CommandOptions, TranscribeOptions};
use crate::store::RunnerDispatchConfig;

use super::{CommandKernelPlan, DispatchPlanRefusal, HostExecutionPolicy};

/// A selected option family, bound to the same configuration and workload.
/// Fields are private so planning cannot receive independently paired facts.
pub struct PlanRequest<'a, Options> {
    options: &'a Options,
    context: PlanContext<'a>,
}

impl<'a, Options> PlanRequest<'a, Options> {
    pub(super) const fn options(&self) -> &'a Options {
        self.options
    }

    pub(super) const fn context(&self) -> &PlanContext<'a> {
        &self.context
    }
}

pub(super) struct PlanContext<'a> {
    command: ReleasedCommand,
    pub(super) language: &'a LanguageSpec,
    pub(super) num_speakers: NumSpeakers,
    pub(super) runtime_state: &'a std::collections::BTreeMap<String, serde_json::Value>,
    file_count: usize,
}

impl<'a> PlanContext<'a> {
    const fn from_dispatch(dispatch: &'a RunnerDispatchConfig, file_count: usize) -> Self {
        Self {
            command: dispatch.command,
            language: &dispatch.lang,
            num_speakers: dispatch.num_speakers,
            runtime_state: &dispatch.runtime_state,
            file_count,
        }
    }

    pub(super) fn kernel_plan(&self, policy: &HostExecutionPolicy) -> CommandKernelPlan {
        CommandKernelPlan::for_command_with_policy(self.command, self.file_count, policy)
    }
}

/// The selected family is established at configuration admission once.
pub trait PlanOptions: Sized {
    fn select(command: ReleasedCommand, options: &CommandOptions) -> Option<&Self>;
}

impl PlanOptions for AlignOptions {
    fn select(command: ReleasedCommand, options: &CommandOptions) -> Option<&Self> {
        match (command, options) {
            (ReleasedCommand::Align, CommandOptions::Align(options)) => Some(options),
            _ => None,
        }
    }
}

impl PlanOptions for TranscribeOptions {
    fn select(command: ReleasedCommand, options: &CommandOptions) -> Option<&Self> {
        match (command, options) {
            (ReleasedCommand::Transcribe, CommandOptions::Transcribe(options))
            | (ReleasedCommand::TranscribeS, CommandOptions::TranscribeS(options)) => Some(options),
            _ => None,
        }
    }
}

impl PlanOptions for BenchmarkOptions {
    fn select(command: ReleasedCommand, options: &CommandOptions) -> Option<&Self> {
        match (command, options) {
            (ReleasedCommand::Benchmark, CommandOptions::Benchmark(options)) => Some(options),
            _ => None,
        }
    }
}

impl<'a, Options: PlanOptions> PlanRequest<'a, Options> {
    pub(crate) fn from_dispatch(
        dispatch: &'a RunnerDispatchConfig,
        file_count: usize,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = Options::select(dispatch.command, &dispatch.options)
            .ok_or(DispatchPlanRefusal::Options)?;
        Ok(Self {
            options,
            context: PlanContext::from_dispatch(dispatch, file_count),
        })
    }
}

pub(super) enum MediaOptions<'a> {
    Opensmile(&'a crate::options::OpensmileOptions),
    Avqi,
    Diarize(&'a crate::options::DiarizeOptions),
}

pub struct MediaAnalysisRequest<'a> {
    options: MediaOptions<'a>,
    context: PlanContext<'a>,
}

impl<'a> MediaAnalysisRequest<'a> {
    pub(super) const fn options(&self) -> &MediaOptions<'a> {
        &self.options
    }

    pub(super) const fn context(&self) -> &PlanContext<'a> {
        &self.context
    }

    pub(crate) const fn from_dispatch(
        dispatch: &'a RunnerDispatchConfig,
        file_count: usize,
    ) -> Result<Self, DispatchPlanRefusal> {
        let options = match (dispatch.command, &dispatch.options) {
            (ReleasedCommand::Opensmile, CommandOptions::Opensmile(options)) => {
                MediaOptions::Opensmile(options)
            }
            (ReleasedCommand::Avqi, CommandOptions::Avqi(_)) => MediaOptions::Avqi,
            (ReleasedCommand::Diarize, CommandOptions::Diarize(options)) => {
                MediaOptions::Diarize(options)
            }
            _ => return Err(DispatchPlanRefusal::Options),
        };
        Ok(Self {
            options,
            context: PlanContext::from_dispatch(dispatch, file_count),
        })
    }
}
