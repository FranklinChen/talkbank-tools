use crate::runner::DispatchHostContext;
use crate::store::RunnerJobSnapshot;

pub(super) use crate::execution::text_io::LoadedTextInputs as LoadedMorphotagInputs;

pub(super) async fn load_morphotag_inputs(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
) -> LoadedMorphotagInputs {
    crate::execution::text_io::load_text_inputs(job, host, true).await
}

#[cfg(test)]
pub(super) fn resolve_input_path(
    job: &RunnerJobSnapshot,
    file: &crate::store::PendingJobFile,
) -> std::path::PathBuf {
    crate::execution::text_io::resolve_input_path(job, file)
}

#[cfg(test)]
pub(super) fn resolve_before_path(
    job: &RunnerJobSnapshot,
    file: &crate::store::PendingJobFile,
) -> Option<std::path::PathBuf> {
    crate::execution::text_io::resolve_before_path(job, file)
}
