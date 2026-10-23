//! Analysis and provenance outcomes travel beside CHAT admission. Only their
//! producers record outcomes; valid CHAT does not certify complete analysis
//! or complete attribution.

use crate::api::{
    OutputExclusionRecord, OutputShortfallRecord, SecondaryMorphologyFailure,
    SecondaryMorphologyFailureKind, SecondaryMorphologySpan,
};
use crate::pipeline::post_validate::PostValidated;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SecondaryReport {
    failures: Vec<SecondaryMorphologyFailure>,
    exclusions: Vec<SecondaryMorphologySpan>,
    unplaced: Vec<crate::api::SecondaryMorphologyUnplacedUtterance>,
}

impl SecondaryReport {
    #[cfg(test)]
    pub(crate) fn for_test(
        failures: Vec<SecondaryMorphologyFailure>,
        exclusions: Vec<SecondaryMorphologySpan>,
    ) -> Self {
        Self {
            failures,
            exclusions,
            unplaced: Vec::new(),
        }
    }
    pub(super) fn failed(
        &mut self,
        span: SecondaryMorphologySpan,
        cause: SecondaryMorphologyFailureKind,
        detail: impl std::fmt::Display,
    ) {
        self.failures.push(SecondaryMorphologyFailure {
            span,
            cause: cause.into(),
            detail: detail.to_string().chars().take(500).collect(),
        });
    }

    pub(super) fn excluded(&mut self, span: SecondaryMorphologySpan) {
        self.exclusions.push(span);
    }

    pub(super) fn extend(&mut self, other: Self) {
        self.failures.extend(other.failures);
        self.exclusions.extend(other.exclusions);
        self.unplaced.extend(other.unplaced);
    }

    pub(super) fn capture_extraction(
        &mut self,
        chat: &crate::chat_ops::ChatFile,
        extraction: &batchalign_transform::morphosyntax::l2::L2Extraction,
    ) {
        use crate::api::SecondaryExtractionFailure as Cause;
        use batchalign_transform::morphosyntax::l2::L2ExtractError;
        for unplaced in extraction.unaligned() {
            let cause = match &unplaced.error {
                L2ExtractError::Alignment(_) => Cause::Alignment,
                L2ExtractError::TokenizationDisagrees { .. } => Cause::TokenizationDisagrees,
                L2ExtractError::SplitWord { .. } => Cause::SplitWord,
                L2ExtractError::UnmappedWord { .. } => Cause::UnmappedWord,
            };
            self.unplaced
                .push(crate::api::SecondaryMorphologyUnplacedUtterance {
                    utterance: chat
                        .lines
                        .iter()
                        .take(unplaced.line_idx + 1)
                        .filter(|line| matches!(line, talkbank_model::model::Line::Utterance(_)))
                        .count() as u64,
                    words: unplaced.at_s_words as u64,
                    cause,
                    detail: unplaced.error.to_string().chars().take(500).collect(),
                });
        }
    }

    /// Counts include every span; inline examples are bounded for job polling.
    pub(crate) fn into_records(
        mut self,
    ) -> (Vec<OutputShortfallRecord>, Vec<OutputExclusionRecord>) {
        self.failures
            .sort_by_key(|s| (s.span.utterance, s.span.first_word));
        self.exclusions.sort_by_key(|s| (s.utterance, s.first_word));
        let limit = crate::api::FileOutputDiagnostics::FIRST_FINDINGS;
        let mut shortfalls = if self.failures.is_empty() {
            Vec::new()
        } else {
            vec![OutputShortfallRecord::SecondaryMorphologyIncomplete {
                spans: self.failures.len() as u64,
                words: self.failures.iter().map(|s| s.span.words).sum(),
                first_spans: self.failures.into_iter().take(limit).collect(),
            }]
        };
        if !self.unplaced.is_empty() {
            self.unplaced.sort_by_key(|u| u.utterance);
            shortfalls.push(OutputShortfallRecord::SecondaryMorphologyUnplaced {
                utterances: self.unplaced.len() as u64,
                words: self.unplaced.iter().map(|u| u.words).sum(),
                first_utterances: self.unplaced.into_iter().take(limit).collect(),
            });
        }
        let exclusions = if self.exclusions.is_empty() {
            Vec::new()
        } else {
            vec![OutputExclusionRecord::UnsupportedSecondaryMorphology {
                spans: self.exclusions.len() as u64,
                words: self.exclusions.iter().map(|s| s.words).sum(),
                first_spans: self.exclusions.into_iter().take(limit).collect(),
            }]
        };
        (shortfalls, exclusions)
    }
}

/// A judged document and the producer's analysis outcomes. Extracting the
/// document requires handling its report; CHAT validity says nothing about
/// whether secondary morphology was available.
#[derive(Debug, Clone)]
pub(crate) struct MorphosyntaxOutput<D = PostValidated> {
    document: D,
    report: MorphologyReport,
}

/// Independent report domains share writeback, not each other's semantics.
#[derive(Debug, Clone)]
pub(crate) struct MorphologyReport {
    secondary: SecondaryReport,
    provenance: crate::provenance::MorphologyProvenanceReport,
}

impl MorphologyReport {
    pub(crate) fn into_records(self) -> (Vec<OutputShortfallRecord>, Vec<OutputExclusionRecord>) {
        let (mut shortfalls, exclusions) = self.secondary.into_records();
        shortfalls.extend(self.provenance.into_record());
        (shortfalls, exclusions)
    }
}

impl<D> MorphosyntaxOutput<D> {
    pub(crate) fn new(document: D, secondary: SecondaryReport) -> Self {
        Self {
            document,
            report: MorphologyReport {
                secondary,
                provenance: crate::provenance::MorphologyProvenanceReport::Complete,
            },
        }
    }

    pub(crate) fn with_provenance(
        mut self,
        provenance: crate::provenance::MorphologyProvenanceReport,
    ) -> Self {
        self.report.provenance = provenance;
        self
    }

    pub(crate) fn into_parts(self) -> (D, MorphologyReport) {
        (self.document, self.report)
    }

    /// Consumers without L2 or prior-reuse reporting cannot silently discard
    /// either report if their policy changes later.
    pub(crate) fn without_reports(self) -> Result<D, crate::error::ServerError> {
        if self.report.secondary.failures.is_empty()
            && self.report.secondary.exclusions.is_empty()
            && self.report.secondary.unplaced.is_empty()
            && matches!(
                self.report.provenance,
                crate::provenance::MorphologyProvenanceReport::Complete
            )
        {
            Ok(self.document)
        } else {
            Err(crate::error::ServerError::UnexpectedMorphologyReport)
        }
    }
}

impl From<PostValidated> for MorphosyntaxOutput {
    fn from(document: PostValidated) -> Self {
        Self::new(document, SecondaryReport::default())
    }
}

impl MorphosyntaxOutput {
    #[cfg(test)]
    pub(crate) fn as_str(&self) -> &str {
        self.document.as_str()
    }

    #[cfg(test)]
    pub(crate) fn into_text(self) -> String {
        self.without_reports()
            .expect("fixture expects no morphology report")
            .into_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(utterance: u64) -> SecondaryMorphologySpan {
        SecondaryMorphologySpan {
            utterance,
            first_word: 2,
            words: 3,
            language: "spa".into(),
        }
    }

    #[test]
    fn reports_sort_bound_and_count_every_span_without_conflating_exclusions() {
        let mut report = SecondaryReport::default();
        for utterance in (1..=100).rev() {
            report.failed(
                span(utterance),
                SecondaryMorphologyFailureKind::Merge,
                "é".repeat(1000),
            );
        }
        report.excluded(span(101));
        let (shortfalls, exclusions) = report.into_records();
        let OutputShortfallRecord::SecondaryMorphologyIncomplete {
            spans,
            words,
            first_spans,
        } = &shortfalls[0]
        else {
            panic!("expected secondary shortfall");
        };
        assert_eq!((*spans, *words), (100, 300));
        assert_eq!(
            first_spans.len(),
            crate::api::FileOutputDiagnostics::FIRST_FINDINGS
        );
        assert_eq!(first_spans[0].span.utterance, 1);
        assert_eq!(first_spans[0].detail.chars().count(), 500);
        insta::assert_json_snapshot!(exclusions, @r#"
        [
          {
            "kind": "unsupported_secondary_morphology",
            "spans": 1,
            "words": 3,
            "first_spans": [
              {
                "utterance": 101,
                "first_word": 2,
                "words": 3,
                "language": "spa"
              }
            ]
          }
        ]
        "#);
    }

    #[test]
    fn opting_out_consumer_cannot_silently_drop_a_secondary_report() {
        let mut report = SecondaryReport::default();
        report.excluded(span(1));
        let error = MorphosyntaxOutput::new((), report)
            .without_reports()
            .unwrap_err();
        assert!(matches!(
            error,
            crate::error::ServerError::UnexpectedMorphologyReport
        ));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
        assert!(
            MorphosyntaxOutput::new((), SecondaryReport::default())
                .without_reports()
                .is_ok()
        );
        let reported = MorphosyntaxOutput::new((), SecondaryReport::default()).with_provenance(
            crate::provenance::MorphologyProvenanceReport::Incomplete(
                crate::provenance::IncrementalMorphotagStampError::PriorIncomplete,
            ),
        );
        assert!(reported.clone().without_reports().is_err());
        let (_, report) = reported.into_parts();
        let (shortfalls, exclusions) = report.into_records();
        assert!(exclusions.is_empty());
        let wire = serde_json::to_value(&shortfalls).unwrap();
        assert_eq!(
            wire,
            serde_json::json!([{
                "kind": "morphology_provenance_incomplete",
                "detail": "the prior morphology provenance is explicitly incomplete"
            }])
        );
        assert_eq!(
            serde_json::from_value::<Vec<OutputShortfallRecord>>(wire).unwrap(),
            shortfalls
        );
        let diagnostics = crate::api::FileOutputDiagnostics::of_findings(Vec::new(), shortfalls);
        assert!(
            crate::cli::progress::diagnosed_summary(Some(&diagnostics))
                .contains("provenance incomplete")
        );
    }
}
