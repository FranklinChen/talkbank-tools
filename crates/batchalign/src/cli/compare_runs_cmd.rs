//! Offline, manifest-driven cross-run comparison CLI.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use batchalign_transform::compare::{
    AggregatePolicy, ComparisonPlanDocument, ComparisonSubject, HumanIdentity, MachineIdentity,
    RunIdentity, RunManifest, ValidatedAlignmentPlan, ValidatedComparisonPlan,
    ValidatedMorphotagPlan, ValidatedTranscriptionPlan, compare_validated_alignment_plan,
    compare_validated_morphotag_plan, compare_validated_transcription_pairs,
};
use serde::Serialize;
use serde_json::{Value, json};

use super::args::{
    CompareRunsAction, CompareRunsArgs, CompareRunsCachePolicy, CompareRunsExecuteArgs,
    CompareRunsManifestIdentity, HumanManifestArgs, MachineManifestArgs, ManifestCommonArgs,
};
use super::error::CliError;

/// The report envelope's shape, including the serialized rows it carries.
///
/// Bumped to 2 on 2026-09-16: an alignment row's `left_timing` / `right_timing`
/// changed from a nullable timing object to a tagged timing STATE that names
/// why an untimed token has no timing.
const REPORT_SCHEMA: u32 = 2;
/// What the comparison COMPUTED, which is part of a comparison's identity.
///
/// Bumped alongside the schema above, and this is the load-bearing half. Pair
/// results are cached under a `comparison_id` derived from this constant, so
/// leaving it at 1 would serve rows in the old shape out of an earlier run's
/// cache, and the columns added beside them would read as empty cells rather
/// than as a mismatch. A cache hit is not a correctness certificate.
const ALGORITHM_VERSION: u32 = 2;

pub(super) fn run(args: &CompareRunsArgs) -> Result<(), CliError> {
    match &args.action {
        CompareRunsAction::Manifest(args) => match &args.identity {
            CompareRunsManifestIdentity::Machine(args) => author_machine(args),
            CompareRunsManifestIdentity::Human(args) => author_human(args),
        },
        CompareRunsAction::Transcribe(args) => execute(args, ComparisonSubject::Transcription),
        CompareRunsAction::Morphotag(args) => execute(args, ComparisonSubject::Morphotag),
        CompareRunsAction::Align(args) => execute(args, ComparisonSubject::Alignment),
    }
}

fn author_machine(args: &MachineManifestArgs) -> Result<(), CliError> {
    let identity = MachineIdentity::new(
        args.implementation.clone(),
        args.command.clone(),
        args.build.clone(),
    )
    .map_err(invalid)?;
    author_manifest(&args.common, RunIdentity::Machine(identity))
}

fn author_human(args: &HumanManifestArgs) -> Result<(), CliError> {
    let identity =
        HumanIdentity::new(args.protocol.clone(), args.cohort.clone()).map_err(invalid)?;
    author_manifest(&args.common, RunIdentity::Human(identity))
}

fn author_manifest(args: &ManifestCommonArgs, identity: RunIdentity) -> Result<(), CliError> {
    let root = fs::canonicalize(&args.artifacts)?;
    let output = canonical_destination(&args.output)?;
    if output.starts_with(&root) {
        return Err(CliError::InvalidArgument(
            "manifest output must be outside its artifact root".to_string(),
        ));
    }
    let arguments = parse_arguments(&args.arguments)?;
    let manifest = RunManifest::from_artifact_root(
        &args.artifacts,
        args.run_id.clone(),
        identity,
        args.source_id.clone(),
        arguments,
    )
    .map_err(invalid)?;
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    if args.output.exists() {
        if fs::read(&args.output)? == bytes {
            return Ok(());
        }
        return Err(CliError::InvalidArgument(format!(
            "refusing to replace conflicting manifest {}",
            args.output.display()
        )));
    }
    atomic_write(&args.output, &bytes)?;
    Ok(())
}

fn parse_arguments(values: &[String]) -> Result<BTreeMap<String, String>, CliError> {
    let mut result = BTreeMap::new();
    for item in values {
        let Some((key, value)) = item.split_once('=') else {
            return Err(CliError::InvalidArgument(format!(
                "manifest argument must be KEY=VALUE: {item:?}"
            )));
        };
        if key.trim().is_empty() || result.insert(key.to_string(), value.to_string()).is_some() {
            return Err(CliError::InvalidArgument(format!(
                "manifest argument key must be unique: {key:?}"
            )));
        }
    }
    Ok(result)
}

fn execute(args: &CompareRunsExecuteArgs, subject: ComparisonSubject) -> Result<(), CliError> {
    let resolved =
        ComparisonPlanDocument::read_and_resolve(&args.plan, subject).map_err(invalid)?;
    let validated = resolved.validate().map_err(invalid)?;
    match validated {
        ValidatedComparisonPlan::Transcription(plan) => run_plan(
            &plan,
            args.cache_policy(),
            "transcription",
            compare_validated_transcription_pairs,
        ),
        ValidatedComparisonPlan::Morphotag(plan) => run_plan(
            &plan,
            args.cache_policy(),
            "morphotag",
            compare_validated_morphotag_plan,
        ),
        ValidatedComparisonPlan::Alignment(plan) => run_plan(
            &plan,
            args.cache_policy(),
            "alignment",
            compare_validated_alignment_plan,
        ),
    }
}

#[derive(Debug, Clone, Serialize)]
struct PairMeta {
    left: String,
    right: String,
    aggregate: AggregatePolicy,
    speaker_map: Option<BTreeMap<String, String>>,
    left_hash: String,
    right_hash: String,
}

trait PlanMeta {
    fn output_path(&self) -> &Path;
    fn pairing_value(&self) -> Value;
    fn manifests_value(&self) -> Value;
    fn pairs_meta(&self) -> Vec<PairMeta>;
}

macro_rules! impl_plan_meta {
    ($type:ty) => {
        impl PlanMeta for $type {
            fn output_path(&self) -> &Path {
                self.output()
            }
            fn pairing_value(&self) -> Value {
                serde_json::to_value(self.pairing()).unwrap_or(Value::Null)
            }
            fn manifests_value(&self) -> Value {
                serde_json::to_value([self.runs()[0].manifest(), self.runs()[1].manifest()])
                    .unwrap_or(Value::Null)
            }
            fn pairs_meta(&self) -> Vec<PairMeta> {
                self.artifact_pairs()
                    .iter()
                    .map(|pair| PairMeta {
                        left: pair.left().as_str().to_string(),
                        right: pair.right().as_str().to_string(),
                        aggregate: pair.aggregate().clone(),
                        speaker_map: pair.speaker_map().cloned(),
                        left_hash: hash_for(self.runs()[0].manifest(), pair.left().as_str()),
                        right_hash: hash_for(self.runs()[1].manifest(), pair.right().as_str()),
                    })
                    .collect()
            }
        }
    };
}
impl_plan_meta!(ValidatedTranscriptionPlan);
impl_plan_meta!(ValidatedMorphotagPlan);
impl_plan_meta!(ValidatedAlignmentPlan);

fn run_plan<P, T>(
    plan: &P,
    cache_policy: CompareRunsCachePolicy,
    subject: &str,
    execute: fn(&P) -> Vec<T>,
) -> Result<(), CliError>
where
    P: PlanMeta,
    T: Serialize,
{
    let metadata = plan.pairs_meta();
    let identity_value = json!({"algorithm_version": ALGORITHM_VERSION, "subject": subject, "pairing": plan.pairing_value(), "manifests": plan.manifests_value(), "pairs": metadata});
    let comparison_id = digest(&identity_value)?;
    let root = plan.output_path().join("runs").join(&comparison_id);
    let pairs_root = root.join("pairs");
    let review_root = root.join("review");
    fs::create_dir_all(&pairs_root)?;
    fs::create_dir_all(&review_root)?;
    let pair_ids: Vec<String> = metadata
        .iter()
        .map(|pair| digest(&json!({"comparison_id": comparison_id, "pair": pair})))
        .collect::<Result<_, _>>()?;
    let mut cached = Vec::new();
    let mut need_compute = false;
    for id in &pair_ids {
        let path = pairs_root.join(format!("{id}.json"));
        if matches!(cache_policy, CompareRunsCachePolicy::Recompute) || !path.exists() {
            cached.push(None);
            need_compute = true;
        } else {
            cached.push(Some(serde_json::from_slice(&fs::read(path)?)?));
        }
    }
    let generated = if need_compute {
        Some(
            execute(plan)
                .into_iter()
                .map(|item| serde_json::to_value(item))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let mut records = Vec::new();
    let mut failures = 0;
    let mut computed = 0;
    let mut reused = 0;
    for (index, ((id, meta), old)) in pair_ids.iter().zip(&metadata).zip(cached).enumerate() {
        let record = if let Some(value) = old {
            reused += 1;
            value
        } else {
            computed += 1;
            let outcome = generated
                .as_ref()
                .and_then(|items| items.get(index))
                .cloned()
                .ok_or_else(|| {
                    CliError::InvalidArgument(
                        "comparison executor returned an incorrect pair count".to_string(),
                    )
                })?;
            let value = json!({"schema_version": REPORT_SCHEMA, "pair_id": id, "left_artifact": meta.left, "right_artifact": meta.right, "aggregate": meta.aggregate, "outcome": outcome});
            atomic_json(&pairs_root.join(format!("{id}.json")), &value)?;
            value
        };
        if record.pointer("/outcome/outcome").and_then(Value::as_str) == Some("unpairable") {
            failures += 1;
        }
        atomic_json(
            &review_root.join(format!("{id}.json")),
            &json!({"schema_version": 1, "pair_id": id, "evidence_only": true, "outcome": record["outcome"]}),
        )?;
        records.push(record);
    }
    let report = json!({"schema_version": REPORT_SCHEMA, "algorithm_version": ALGORITHM_VERSION, "comparison_id": comparison_id, "subject": subject, "pairing": plan.pairing_value(), "manifests": plan.manifests_value(), "pairs": records});
    atomic_json(&root.join("report.json"), &report)?;
    atomic_csv(&root.join("summary.csv"), &records, subject)?;
    eprintln!(
        "compare-runs {subject}: {computed} computed, {reused} reused, {failures} unpairable"
    );
    if failures > 0 {
        return Err(CliError::InvalidArgument(format!(
            "{failures} pair(s) were unpairable; evidence written to {}",
            root.display()
        )));
    }
    Ok(())
}

fn hash_for(manifest: &RunManifest, path: &str) -> String {
    manifest
        .artifacts()
        .iter()
        .find(|item| item.relative_path() == path)
        .map(|item| item.blake3().to_string())
        .unwrap_or_default()
}
fn digest(value: &Value) -> Result<String, CliError> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
fn invalid(error: impl std::fmt::Display) -> CliError {
    CliError::InvalidArgument(error.to_string())
}
/// Resolve a not-yet-existing destination to a symlink-free absolute path.
///
/// Containment checks must compare like with like. `fs::canonicalize` resolves
/// symlinks; [`absolute`] does not, so testing an `absolute` output against a
/// `canonicalize`d root is a false negative wherever either path crosses a
/// symlink. On macOS that is the normal case, not an edge case: `/tmp` and
/// `/var` are symlinks into `/private`, so a manifest written INSIDE its own
/// artifact root compared clean and the containment guard never fired. The
/// destination itself may not exist yet, so its parent is canonicalized and the
/// file name re-attached.
fn canonical_destination(path: &Path) -> Result<PathBuf, CliError> {
    let file_name = path.file_name().ok_or_else(|| {
        CliError::InvalidArgument(format!("{} does not name a file", path.display()))
    })?;
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        // A bare file name is relative to the working directory.
        _ => Path::new("."),
    };
    Ok(fs::canonicalize(parent)?.join(file_name))
}

fn atomic_json(path: &Path, value: &Value) -> Result<(), CliError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}
fn atomic_csv(path: &Path, records: &[Value], subject: &str) -> Result<(), CliError> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    match subject {
        "transcription" => {
            writer
                .write_record([
                    "pair_id",
                    "left_artifact",
                    "right_artifact",
                    "left_speaker",
                    "right_speaker",
                    "left_words",
                    "right_words",
                    "matches",
                    "insertions",
                    "deletions",
                    "wer_numerator",
                    "wer_rate",
                    "cwer_numerator",
                    "cwer_rate",
                    "excluded_left_tokens",
                    "excluded_right_tokens",
                ])
                .map_err(invalid)?;
            for record in records {
                let rows = record
                    .pointer("/outcome/result/speaker_agreements")
                    .and_then(Value::as_array);
                if let Some(rows) = rows {
                    for row in rows {
                        let metrics = &row["metrics"];
                        writer
                            .write_record([
                                text(record, "pair_id"),
                                text(record, "left_artifact"),
                                text(record, "right_artifact"),
                                text(row, "left_speaker"),
                                text(row, "right_speaker"),
                                pointer(metrics, "/words/left"),
                                pointer(metrics, "/words/right"),
                                pointer(metrics, "/tally/matches"),
                                pointer(metrics, "/tally/insertions"),
                                pointer(metrics, "/tally/deletions"),
                                wer_numerator(metrics),
                                rate(metrics, wer_numerator(metrics)),
                                num(metrics, "cwer_numerator"),
                                rate(metrics, num(metrics, "cwer_numerator")),
                                pointer(metrics, "/excluded_tokens/left"),
                                pointer(metrics, "/excluded_tokens/right"),
                            ])
                            .map_err(invalid)?;
                    }
                } else {
                    writer
                        .write_record([
                            text(record, "pair_id"),
                            text(record, "left_artifact"),
                            text(record, "right_artifact"),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                            String::new(),
                        ])
                        .map_err(invalid)?;
                }
            }
        }
        "morphotag" => {
            writer
                .write_record([
                    "pair_id",
                    "left_artifact",
                    "right_artifact",
                    "left_speaker",
                    "right_speaker",
                    "utterance",
                    "token",
                    "left_text",
                    "right_text",
                    "tokenization",
                    "lemma",
                    "pos",
                    "feature_set",
                    "clitic_chunk",
                    "dependency_head",
                    "relation",
                ])
                .map_err(invalid)?;
            for record in records {
                for row in record
                    .pointer("/outcome/result/tokens")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    writer
                        .write_record([
                            text(record, "pair_id"),
                            text(record, "left_artifact"),
                            text(record, "right_artifact"),
                            text(row, "left_speaker"),
                            text(row, "right_speaker"),
                            num(row, "utterance"),
                            num(row, "token"),
                            text(row, "left_text"),
                            text(row, "right_text"),
                            differs(row, "tokenization"),
                            differs(row, "lemma"),
                            differs(row, "pos"),
                            differs(row, "feature_set"),
                            differs(row, "clitic_chunk"),
                            differs(row, "dependency_head"),
                            differs(row, "relation"),
                        ])
                        .map_err(invalid)?;
                }
            }
        }
        _ => {
            writer
                .write_record([
                    "pair_id",
                    "left_artifact",
                    "right_artifact",
                    "left_speaker",
                    "right_speaker",
                    "utterance",
                    "token",
                    "text",
                    "left_timing_state",
                    "left_start_ms",
                    "left_end_ms",
                    "right_timing_state",
                    "right_start_ms",
                    "right_end_ms",
                    "start_delta_ms",
                    "end_delta_ms",
                ])
                .map_err(invalid)?;
            for record in records {
                for row in record
                    .pointer("/outcome/result/tokens")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    writer
                        .write_record([
                            text(record, "pair_id"),
                            text(record, "left_artifact"),
                            text(record, "right_artifact"),
                            text(row, "left_speaker"),
                            text(row, "right_speaker"),
                            num(row, "utterance"),
                            num(row, "token"),
                            text(row, "text"),
                            pointer_text(row, "/left_timing/state"),
                            pointer(row, "/left_timing/start_ms"),
                            pointer(row, "/left_timing/end_ms"),
                            pointer_text(row, "/right_timing/state"),
                            pointer(row, "/right_timing/start_ms"),
                            pointer(row, "/right_timing/end_ms"),
                            scalar(row, "start_delta_ms"),
                            scalar(row, "end_delta_ms"),
                        ])
                        .map_err(invalid)?;
                }
            }
        }
    }
    let bytes = writer
        .into_inner()
        .map_err(|error| CliError::Io(error.into_error()))?;
    atomic_write(path, &bytes)
}

fn text(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Whether a morphotag row reports a difference on `axis`.
///
/// The producer emits one `differences` array rather than seven `bool` fields
/// (see `MorphotagDifference`), but the CSV keeps a column per axis, because
/// that is what a spreadsheet wants. The reader bridges the two.
fn differs(row: &Value, axis: &str) -> String {
    let present = row
        .get("differences")
        .and_then(Value::as_array)
        .is_some_and(|axes| axes.iter().any(|a| a.as_str() == Some(axis)));
    present.to_string()
}

/// `insertions + deletions`, which the producer derives rather than stores.
fn wer_numerator(metrics: &Value) -> String {
    let at = |path: &str| metrics.pointer(path).and_then(Value::as_u64).unwrap_or(0);
    (at("/tally/insertions") + at("/tally/deletions")).to_string()
}

/// A numerator over the left-side word count, absent when there is nothing to
/// divide by. Mirrors `AgreementMetrics::rate`, which is not on the wire.
fn rate(metrics: &Value, numerator: String) -> String {
    let left = metrics
        .pointer("/words/left")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    match (left, numerator.parse::<u64>()) {
        (0, _) | (_, Err(_)) => String::new(),
        (left, Ok(n)) => (n as f64 / left as f64).to_string(),
    }
}

fn num(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_default()
}
fn scalar(value: &Value, key: &str) -> String {
    value
        .get(key)
        .filter(|v| !v.is_null())
        .map(Value::to_string)
        .unwrap_or_default()
}
fn pointer(value: &Value, path: &str) -> String {
    value
        .pointer(path)
        .filter(|v| !v.is_null())
        .map(Value::to_string)
        .unwrap_or_default()
}
/// A STRING at a JSON pointer, unquoted.
///
/// [`pointer`] renders through `Value::to_string`, which is right for a number
/// and wrong for a string: it would write `"timed"` into the cell, quotes
/// included.
fn pointer_text(value: &Value, path: &str) -> String {
    value
        .pointer(path)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let parent = path.parent().ok_or_else(|| {
        CliError::InvalidArgument(format!("output path has no parent: {}", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .map_err(|error| CliError::Io(error.error))?;
    Ok(())
}

#[cfg(test)]
mod csv_shape_tests {
    use super::{differs, pointer, pointer_text, rate, wer_numerator};
    use serde_json::json;

    /// The CSV readers must track the producer's serialized shape.
    ///
    /// These columns are read by JSON pointer, and a pointer that misses
    /// yields an EMPTY CELL rather than an error, so nothing in the type
    /// system connects the two sides. That silence is not hypothetical: one
    /// refactor of the producer emptied eleven of sixteen transcription
    /// columns and seven of seventeen morphotag columns, and both shipped
    /// green because every gate passed. This is the connection.
    #[test]
    fn morphotag_axes_come_from_the_differences_array() {
        let row = json!({ "differences": ["lemma", "pos"] });
        assert_eq!(differs(&row, "lemma"), "true");
        assert_eq!(differs(&row, "pos"), "true");
        assert_eq!(differs(&row, "relation"), "false");
        // A row with no differences is "false", not empty.
        let identical = json!({ "differences": [] });
        assert_eq!(differs(&identical, "lemma"), "false");
    }

    #[test]
    fn transcription_metrics_are_read_and_derived_from_the_nested_shape() {
        let metrics = json!({
            "cwer_numerator": 1,
            "words": { "left": 2, "right": 1 },
            "tally": { "matches": 1, "insertions": 1, "deletions": 0 },
            "excluded_tokens": { "left": 3, "right": 4 },
        });

        // Derived, because the producer no longer stores them.
        assert_eq!(wer_numerator(&metrics), "1");
        assert_eq!(rate(&metrics, "1".to_string()), "0.5");

        // Absent rather than zero when there is nothing to divide by, matching
        // the producer: a rate of 0 would read as perfect agreement.
        let empty_left = json!({ "words": { "left": 0, "right": 1 } });
        assert_eq!(rate(&empty_left, "1".to_string()), "");
    }

    /// An alignment row's timing columns are read out of the TAGGED state.
    ///
    /// The millisecond columns keep resolving for a timed token, because the
    /// state is internally tagged and `start_ms` / `end_ms` sit beside the tag
    /// rather than under a further level. The state column is what an untimed
    /// token now carries in place of three empty cells and no reason.
    #[test]
    fn alignment_timing_columns_are_read_from_the_tagged_state() {
        let timed = json!({ "left_timing": { "state": "timed", "start_ms": 10, "end_ms": 20 } });
        assert_eq!(pointer_text(&timed, "/left_timing/state"), "timed");
        assert_eq!(pointer(&timed, "/left_timing/start_ms"), "10");
        assert_eq!(pointer(&timed, "/left_timing/end_ms"), "20");

        let drifted = json!({
            "left_timing": { "state": "wor_tier_drifted", "wor_slots": 3, "main_words": 4 }
        });
        assert_eq!(
            pointer_text(&drifted, "/left_timing/state"),
            "wor_tier_drifted"
        );
        assert_eq!(
            pointer(&drifted, "/left_timing/start_ms"),
            "",
            "an untimed token has no start; the state column is where it says why"
        );

        // `pointer` would have quoted it, which is why the state has a reader
        // of its own.
        assert_eq!(pointer(&timed, "/left_timing/state"), "\"timed\"");
    }
}
