use crate::planning;
use crate::runner::DispatchHostContext;
use crate::store::RunnerJobSnapshot;
use crate::text_batch::TextBatchFileResults;

pub(super) async fn write_morphotag_results(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    plan: &planning::JobPlan,
    results: TextBatchFileResults,
) {
    crate::execution::text_io::write_text_results(job, host, plan, results, "Morphotag").await;
}
