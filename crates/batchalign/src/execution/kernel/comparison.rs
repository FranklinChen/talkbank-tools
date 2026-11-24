//! Consuming compare phases retain admitted inputs and their publication owner.

use crate::compare::{AdmittedComparisonReference, CompareMaterializedOutputs};
use crate::error::ServerError;
use crate::infer_retry::Cancellation;
use crate::pipeline::post_validate::PostValidated;
use crate::planning::{self, JobPlan};
use crate::recipe_runner::runtime::{ChatMetricsOutputs, SourceOutputs};
use crate::recipe_runner::work_unit::CompareWorkUnit;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::store::PendingJobFile;

use super::ExecutionContext;
use super::outputs::{ConsolidatedCompareMetricsRow, write_outputs};

/// The output plan and execution context remain bound through every phase.
/// Only source admission can construct this owner. Publication consumes it.
pub(super) struct ComparisonFile<'a> {
    artifacts: ChatMetricsOutputs<'a>,
    unit: &'a CompareWorkUnit,
    context: &'a ExecutionContext<'a>,
}

impl<'a> ComparisonFile<'a> {
    pub(super) const fn artifacts(&self) -> &ChatMetricsOutputs<'a> {
        &self.artifacts
    }

    pub(super) fn lifecycle(&self) -> FileRunTracker<'_> {
        FileRunTracker::new(
            self.context.host.sink().as_ref(),
            self.artifacts.source_identity().filename().as_ref(),
        )
    }
}

/// Output authority is established before reading or inference.
pub(super) struct PreparedComparison<'a> {
    file: ComparisonFile<'a>,
}

/// Both external inputs have been read and the reference is fully admitted.
struct ReadComparison<'a> {
    file: ComparisonFile<'a>,
    main: String,
    reference: AdmittedComparisonReference,
}

/// One morphology call admitted by the comparison that owns the main input.
pub struct CompareMorphologySubmission<'a> {
    main: String,
    mwt: &'a crate::chat_ops::morphosyntax_ops::MwtDict,
    cancellation: Cancellation<'a>,
}

impl<'a> CompareMorphologySubmission<'a> {
    pub(in crate::execution) fn into_parts(
        self,
    ) -> (
        String,
        &'a crate::chat_ops::morphosyntax_ops::MwtDict,
        Cancellation<'a>,
    ) {
        (self.main, self.mwt, self.cancellation)
    }
}

/// The admitted reference remains associated with completed morphology.
struct MorphotaggedComparison<'a> {
    file: ComparisonFile<'a>,
    main: PostValidated,
    reference: AdmittedComparisonReference,
}

/// Judged CHAT and computed metrics are ready for their required writes.
struct ComparedComparison<'a> {
    file: ComparisonFile<'a>,
    outputs: CompareMaterializedOutputs,
}

impl<'a> PreparedComparison<'a> {
    pub(super) fn admit(
        unit: &'a CompareWorkUnit,
        file: &'a PendingJobFile,
        plan: &'a JobPlan,
        context: &'a ExecutionContext<'a>,
    ) -> Result<Self, ServerError> {
        let artifacts = planning::artifact_set_for_source(plan, &unit.main.display_path)
            .ok_or_else(|| {
                ServerError::Validation(format!(
                    "compare job plan has no CHAT and metrics outputs for {}",
                    unit.main.display_path
                ))
            })?;
        let artifacts = SourceOutputs::admit_planned(&context.job.filesystem, file, artifacts)?
            .into_chat_metrics()?;
        Ok(Self {
            file: ComparisonFile {
                artifacts,
                unit,
                context,
            },
        })
    }

    pub(super) async fn run(self) -> Result<ConsolidatedCompareMetricsRow, ServerError> {
        self.read()
            .await?
            .morphotag()
            .await?
            .compare()
            .await?
            .publish()
            .await
    }

    async fn read(self) -> Result<ReadComparison<'a>, ServerError> {
        self.file.lifecycle().stage(FileStage::Reading).await;
        let unit = self.file.unit;
        let main = tokio::fs::read_to_string(&unit.main.source_path)
            .await
            .map_err(|error| {
                ServerError::Validation(format!(
                    "failed to read compare input {}: {error}",
                    unit.main.display_path
                ))
            })?;
        let template = unit.main.source_path.with_file_name("template.gold.cha");
        let reference = match tokio::fs::read_to_string(&unit.gold.source_path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tokio::fs::read_to_string(&template).await.map_err(|_| {
                    ServerError::Validation(format!(
                        "No gold .cha file found for comparison. main: {}, expected: {} or {}",
                        unit.main.display_path,
                        unit.gold.display_path,
                        crate::compare::template_gold_path_for(unit.main.display_path.as_ref()),
                    ))
                })?
            }
            Err(error) => {
                return Err(ServerError::Validation(format!(
                    "failed to read compare reference {}: {error}",
                    unit.gold.display_path
                )));
            }
        };
        let reference = AdmittedComparisonReference::admit(&reference)?;
        Ok(ReadComparison {
            file: self.file,
            main,
            reference,
        })
    }
}

impl<'a> ReadComparison<'a> {
    async fn morphotag(self) -> Result<MorphotaggedComparison<'a>, ServerError> {
        self.file
            .lifecycle()
            .stage(FileStage::AnalyzingMorphosyntax)
            .await;
        let context = self.file.context;
        // The transcript's own language header selects morphology.
        let main = context
            .gateway
            .morphotag_for_compare(CompareMorphologySubmission {
                main: self.main,
                mwt: context.mwt,
                cancellation: Cancellation::Token(&context.job.cancel_token),
            })
            .await?;
        Ok(MorphotaggedComparison {
            file: self.file,
            main,
            reference: self.reference,
        })
    }
}

impl<'a> MorphotaggedComparison<'a> {
    async fn compare(self) -> Result<ComparedComparison<'a>, ServerError> {
        self.file.lifecycle().stage(FileStage::Comparing).await;
        let outputs = crate::compare::process_compare_morphotagged_main(self.main, self.reference)?;
        self.file.lifecycle().stage(FileStage::Finalizing).await;
        Ok(ComparedComparison {
            file: self.file,
            outputs,
        })
    }
}

impl ComparedComparison<'_> {
    async fn publish(self) -> Result<ConsolidatedCompareMetricsRow, ServerError> {
        self.file.lifecycle().stage(FileStage::Writing).await;
        Ok(write_outputs(self.file, self.outputs)
            .await?
            .complete()
            .await)
    }
}
