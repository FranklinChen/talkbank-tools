//! Compare output transitions: typed metrics remain typed until CSV encoding,
//! and a completion receipt is produced only after all required file writes.

use std::collections::BTreeMap;
use std::path::Path;

use batchalign_transform::compare::{CompareMetricName, CompareMetricsCsvTable};

use crate::api::{ContentType, DisplayPath};
use crate::compare::CompareMaterializedOutputs;
use crate::error::ServerError;
use crate::planning::PlannedArtifactSet;
use crate::recipe_runner::materialize::MaterializedArtifactRole;
use crate::recipe_runner::runtime::{ChatOutputTarget, write_text_output_artifact};
use crate::runner::util::FileRunTracker;
use crate::store::RunnerFilesystemConfig;

/// One source-bound row, never reconstructed from serialized CSV.
pub(super) struct ConsolidatedCompareMetricsRow {
    file: String,
    metrics: CompareMetricsCsvTable,
}

impl ConsolidatedCompareMetricsRow {
    fn from_source(
        source: &DisplayPath,
        metrics: CompareMetricsCsvTable,
    ) -> Result<Self, ServerError> {
        let file = Path::new(source.as_ref())
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                ServerError::Validation(format!("compare source has no UTF-8 filename: {source}"))
            })?;
        Ok(Self {
            file: file.to_owned(),
            metrics,
        })
    }
}

/// Only the successful writer can construct this receipt. Failed writes cannot
/// enter the completion method or supply a row to the consolidated report.
pub(super) struct WrittenComparison {
    primary_path: DisplayPath,
    primary_content_type: ContentType,
    metrics: ConsolidatedCompareMetricsRow,
}

impl WrittenComparison {
    pub(super) async fn complete(
        self,
        lifecycle: FileRunTracker<'_>,
    ) -> ConsolidatedCompareMetricsRow {
        lifecycle
            .complete_with_result(self.primary_path, self.primary_content_type)
            .await;
        self.metrics
    }
}

pub(super) async fn write_outputs(
    filesystem: &RunnerFilesystemConfig,
    file_index: usize,
    artifacts: &PlannedArtifactSet,
    outputs: CompareMaterializedOutputs,
) -> Result<WrittenComparison, ServerError> {
    let primary: Vec<_> = artifacts
        .files
        .iter()
        .filter(|artifact| artifact.role == MaterializedArtifactRole::Primary)
        .collect();
    let sidecars: Vec<_> = artifacts
        .files
        .iter()
        .filter(|artifact| artifact.role == MaterializedArtifactRole::Sidecar)
        .collect();
    let ([primary], [sidecar]) = (primary.as_slice(), sidecars.as_slice()) else {
        return Err(ServerError::Validation(
            "compare requires one CHAT output and one metrics sidecar".into(),
        ));
    };
    let CompareMaterializedOutputs {
        chat_output,
        metrics,
    } = outputs;
    let metrics_csv = metrics.to_csv_string().map_err(|error| {
        ServerError::Persistence(format!("compare CSV serialization failed: {error}"))
    })?;
    let row = ConsolidatedCompareMetricsRow::from_source(&artifacts.source_display_path, metrics)?;
    let target = ChatOutputTarget::new(filesystem, file_index, &primary.display_path);
    write_text_output_artifact(&target, chat_output.as_str())
        .await
        .map_err(|error| {
            ServerError::Persistence(format!(
                "failed to write compare CHAT output {}: {error}",
                primary.display_path
            ))
        })?;
    let csv_target = ChatOutputTarget::new(filesystem, file_index, &sidecar.display_path);
    write_text_output_artifact(&csv_target, &metrics_csv)
        .await
        .map_err(|error| {
            ServerError::Persistence(format!(
                "failed to write compare CSV {}: {error}",
                sidecar.display_path
            ))
        })?;
    Ok(WrittenComparison {
        primary_path: primary.display_path.clone(),
        primary_content_type: primary.content_type,
        metrics: row,
    })
}

pub(super) fn format_consolidated_csv(
    rows: &[ConsolidatedCompareMetricsRow],
) -> Result<String, ServerError> {
    let mut headers = Vec::new();
    for row in rows {
        for metric in &row.metrics.rows {
            if !headers.contains(&metric.metric) {
                headers.push(metric.metric.clone());
            }
        }
    }
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    let mut header = vec!["file".to_owned()];
    header.extend(headers.iter().map(CompareMetricName::to_csv_field));
    writer.write_record(header).map_err(csv_error)?;
    for row in rows {
        let values: BTreeMap<_, _> = row
            .metrics
            .rows
            .iter()
            .map(|metric| {
                let mut value = metric.value.to_csv_field();
                // Preserve the consolidated wire format without parsing our own CSV.
                if matches!(
                    metric.metric,
                    CompareMetricName::Wer | CompareMetricName::Accuracy
                ) {
                    while value.contains('.') && value.ends_with('0') {
                        value.pop();
                    }
                    if value.ends_with('.') {
                        value.push('0');
                    }
                }
                (metric.metric.to_csv_field(), value)
            })
            .collect();
        let mut record = vec![row.file.clone()];
        record.extend(
            headers
                .iter()
                .map(|key| values.get(&key.to_csv_field()).cloned().unwrap_or_default()),
        );
        writer.write_record(record).map_err(csv_error)?;
    }
    let bytes = writer
        .into_inner()
        .map_err(|error| csv_error(error.into_error().into()))?;
    String::from_utf8(bytes)
        .map_err(|error| ServerError::Persistence(format!("compare CSV encoding failed: {error}")))
}

fn csv_error(error: csv::Error) -> ServerError {
    ServerError::Persistence(format!("compare CSV serialization failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::compare::{CompareMetricValue, CompareMetricsCsvRow};

    fn metrics() -> CompareMetricsCsvTable {
        CompareMetricsCsvTable {
            rows: vec![
                CompareMetricsCsvRow {
                    metric: CompareMetricName::Wer,
                    value: CompareMetricValue::Decimal(0.25),
                },
                CompareMetricsCsvRow {
                    metric: CompareMetricName::Accuracy,
                    value: CompareMetricValue::Decimal(0.75),
                },
            ],
        }
    }

    #[test]
    fn compare_csv_roundtrips_source_names_without_rewriting_or_splitting_them() {
        let names = [
            "comma,name.cha",
            "quote\"name.cha",
            "line\nbreak.cha",
            "résumé.cha",
            "name.compare.csvx.cha",
        ];
        let rows: Vec<_> = names
            .iter()
            .map(|name| {
                ConsolidatedCompareMetricsRow::from_source(&DisplayPath::from(*name), metrics())
                    .expect("row")
            })
            .collect();
        let csv = format_consolidated_csv(&rows).expect("CSV");
        let mut reader = csv::Reader::from_reader(csv.as_bytes());
        assert_eq!(
            reader.headers().unwrap(),
            &csv::StringRecord::from(vec!["file", "wer", "accuracy"])
        );
        let records: Vec<_> = reader.records().map(Result::unwrap).collect();
        for (record, name) in records.iter().zip(names) {
            assert_eq!(record.get(0), Some(name));
            assert_eq!(record.get(1), Some("0.25"));
            assert_eq!(record.get(2), Some("0.75"));
            assert_eq!(record.len(), 3);
        }
        assert_eq!(records.len(), names.len());
    }

    #[test]
    fn compare_absent_metric_cells_stay_blank_and_existing_decimal_format_is_preserved() {
        let mut sparse = metrics();
        sparse.rows.remove(1);
        let rows = [
            ConsolidatedCompareMetricsRow::from_source(&DisplayPath::from("a.cha"), metrics())
                .unwrap(),
            ConsolidatedCompareMetricsRow::from_source(&DisplayPath::from("b.cha"), sparse)
                .unwrap(),
        ];
        insta::assert_snapshot!(format_consolidated_csv(&rows).unwrap(), @"
        file,wer,accuracy
        a.cha,0.25,0.75
        b.cha,0.25,
        ");
    }
}
