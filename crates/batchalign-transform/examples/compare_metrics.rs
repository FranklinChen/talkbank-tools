//! Score validated hypothesis/gold CHAT pairs without running inference.
//!
//! Usage: compare_metrics main1.cha gold1.cha [main2.cha gold2.cha ...]
//! Output is `pair,metric,value` CSV, followed by `ALL` count totals. Rates
//! are not summed. Every input must pass parsing and model validation before
//! any comparison runs; the complete report is prepared before bytes are written.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use batchalign_transform::compare::{
    CompareMetricValue, CompareMetricsCsvTable, CompareSerializationError, GoldCoverage, compare,
};
use talkbank_model::{ChatFile, ErrorCollector};
use talkbank_parser::TreeSitterParser;

#[derive(Debug, Clone, Copy)]
enum Role {
    Hypothesis,
    Gold,
}

#[derive(Debug, Clone, Copy)]
enum AdmissionStage {
    Parsing,
    ModelValidation,
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("give one or more hypothesis/gold pairs: main1.cha gold1.cha ...")]
    Arguments,
    #[error("parser initialization failed: {0}")]
    Parser(String),
    #[error("cannot read {role:?} transcript {}: {source}", path.display())]
    Read {
        role: Role,
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{role:?} transcript {} failed {stage:?}: {codes}", path.display())]
    Invalid {
        role: Role,
        path: PathBuf,
        stage: AdmissionStage,
        codes: String,
    },
    #[error(transparent)]
    Metrics(#[from] CompareSerializationError),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Write(#[from] std::io::Error),
    #[error("aggregate count overflow for {0}")]
    CountOverflow(String),
}

/// No caller can score the parser's recovered model without admission.
struct ValidatedTranscript(ChatFile);

impl ValidatedTranscript {
    fn read(parser: &TreeSitterParser, path: &Path, role: Role) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            role,
            path: path.to_owned(),
            source,
        })?;
        Self::admit(parser, path, role, &text)
    }

    fn admit(
        parser: &TreeSitterParser,
        path: &Path,
        role: Role,
        text: &str,
    ) -> Result<Self, Error> {
        let errors = ErrorCollector::new();
        let chat = parser.parse_chat_file_streaming(text, &errors);
        let diagnostics = errors.into_vec();
        if !diagnostics.is_empty() {
            return Err(Error::Invalid {
                role,
                path: path.to_owned(),
                stage: AdmissionStage::Parsing,
                codes: diagnostics
                    .iter()
                    .map(|error| error.code.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        let errors = ErrorCollector::new();
        chat.clone()
            .validate_into(
                &errors,
                talkbank_model::model::TranscriptName::for_path(path),
            )
            .map_err(|_| Error::Invalid {
                role,
                path: path.to_owned(),
                stage: AdmissionStage::ModelValidation,
                codes: errors
                    .to_vec()
                    .iter()
                    .map(|error| error.code.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            })?;
        Ok(Self(chat))
    }
}

struct AdmittedPair {
    label: String,
    hypothesis: ValidatedTranscript,
    gold: ValidatedTranscript,
}

/// All pairs are admitted before the comparison engine receives any of them.
struct AdmittedBenchmark(Vec<AdmittedPair>);

impl AdmittedBenchmark {
    fn read(arguments: &[String]) -> Result<Self, Error> {
        let (pairs, []) = arguments.as_chunks::<2>() else {
            return Err(Error::Arguments);
        };
        if pairs.is_empty() {
            return Err(Error::Arguments);
        }
        let parser = TreeSitterParser::new().map_err(|error| Error::Parser(error.to_string()))?;
        let mut admitted = Vec::with_capacity(pairs.len());
        for [hypothesis, gold] in pairs {
            let label = Path::new(hypothesis)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| hypothesis.clone());
            admitted.push(AdmittedPair {
                label,
                hypothesis: ValidatedTranscript::read(
                    &parser,
                    Path::new(hypothesis),
                    Role::Hypothesis,
                )?,
                gold: ValidatedTranscript::read(&parser, Path::new(gold), Role::Gold)?,
            });
        }
        Ok(Self(admitted))
    }

    fn score(self) -> Result<PreparedReport, Error> {
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        let mut bytes = Vec::new();
        {
            let mut output = csv::Writer::from_writer(&mut bytes);
            output.write_record(["pair", "metric", "value"])?;
            for pair in self.0 {
                let metrics =
                    compare(&pair.hypothesis.0, &pair.gold.0, GoldCoverage::Complete).metrics;
                let table = CompareMetricsCsvTable::from_metrics(&metrics)?;
                for row in &table.rows {
                    let key = row.metric.to_csv_field();
                    let value = row.value.to_csv_field();
                    if let CompareMetricValue::Count(count) = row.value {
                        let total = totals.entry(key.clone()).or_default();
                        *total = total
                            .checked_add(count)
                            .ok_or_else(|| Error::CountOverflow(key.clone()))?;
                    }
                    output.write_record([pair.label.as_str(), key.as_str(), value.as_str()])?;
                }
            }
            for (key, total) in &totals {
                output.write_record(["ALL", key.as_str(), total.to_string().as_str()])?;
            }
            output.flush()?;
        }
        Ok(PreparedReport(bytes))
    }
}

/// Only a fully scored report can cross the output boundary.
struct PreparedReport(Vec<u8>);

impl PreparedReport {
    fn write(self, mut output: impl Write) -> Result<(), Error> {
        output.write_all(&self.0)?;
        output.flush()?;
        Ok(())
    }
}

fn run(arguments: &[String], output: impl Write) -> Result<(), Error> {
    AdmittedBenchmark::read(arguments)?.score()?.write(output)
}

fn main() -> Result<(), Error> {
    run(
        &std::env::args().skip(1).collect::<Vec<_>>(),
        std::io::stdout().lock(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(words: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\t{words}\n@End\n"
        )
    }

    #[test]
    fn recovered_parse_is_not_admitted_as_gold() {
        let parser = TreeSitterParser::new().unwrap();
        let malformed = chat("hello [ .");
        assert!(matches!(
            ValidatedTranscript::admit(&parser, Path::new("gold.cha"), Role::Gold, &malformed),
            Err(Error::Invalid {
                role: Role::Gold,
                stage: AdmissionStage::Parsing,
                ..
            })
        ));
    }

    #[test]
    fn parsed_but_invalid_model_is_not_admitted() {
        let parser = TreeSitterParser::new().unwrap();
        let invalid = chat("hello .").replace("*CHI:", "*MOT:");
        assert!(matches!(
            ValidatedTranscript::admit(&parser, Path::new("gold.cha"), Role::Gold, &invalid),
            Err(Error::Invalid {
                stage: AdmissionStage::ModelValidation,
                ..
            })
        ));
    }

    #[test]
    fn later_invalid_gold_publishes_no_partial_report() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("valid.cha");
        let invalid = dir.path().join("invalid.cha");
        std::fs::write(&valid, chat("hello .")).unwrap();
        std::fs::write(&invalid, chat("hello [ .")).unwrap();
        let v = valid.to_str().unwrap().to_owned();
        let i = invalid.to_str().unwrap().to_owned();
        let mut output = Vec::new();
        assert!(run(&[v.clone(), v.clone(), v, i], &mut output).is_err());
        assert!(output.is_empty());
    }

    #[test]
    fn valid_pair_retains_counts_and_csv_contract() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("valid.cha");
        std::fs::write(&valid, chat("hello .")).unwrap();
        let v = valid.to_str().unwrap().to_owned();
        let mut output = Vec::new();
        run(&[v.clone(), v], &mut output).unwrap();
        let csv = String::from_utf8(output).unwrap();
        assert!(csv.starts_with("pair,metric,value\n"));
        assert!(csv.contains("valid,matches,1\n"));
        assert!(csv.contains("ALL,matches,1\n"));
    }
}
