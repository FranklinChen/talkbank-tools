//! Analysis and provenance outcomes travel beside CHAT admission. Only their
//! producers record outcomes; valid CHAT does not certify complete analysis
//! or complete attribution.

use crate::api::{
    OutputExclusionRecord, OutputShortfallRecord, PrimaryMorphologyDeclineCause,
    PrimaryMorphologyDeclinedUtterance, SecondaryExtractionFailure, SecondaryMorphologyFailure,
    SecondaryMorphologyFailureKind, SecondaryMorphologySpan, SecondaryMorphologyUnplacedUtterance,
    UtteranceLocator, UtteranceNotInDocument,
};
use crate::pipeline::post_validate::PostValidated;
use talkbank_model::UtteranceIdx;

/// Where a secondary-language span is, by utterance index in the document
/// the report describes: the report's own address. It becomes a located
/// [`SecondaryMorphologySpan`] only when the report is recorded against that
/// document, for the spans the record names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpanAddress {
    utterance: UtteranceIdx,
    /// One-based morphology-item position within that utterance.
    first_word: u64,
    /// Morphology items in the span, not expanded clitic chunks.
    words: u64,
    language: String,
}

impl SpanAddress {
    /// The address of a planned span: the plan carries its utterance's
    /// position from the walk that collected it.
    pub(super) fn of(span: &batchalign_transform::morphosyntax::l2::L2SpanPlan) -> Self {
        Self {
            utterance: span.address().utterance(),
            first_word: span.first_word().as_usize() as u64 + 1,
            words: span.len() as u64,
            language: span.target_lang().as_str().to_owned(),
        }
    }

    /// An address from its parts, for tests of the report.
    #[cfg(test)]
    pub(crate) fn for_test(utterance: usize, first_word: u64, words: u64, language: &str) -> Self {
        Self {
            utterance: UtteranceIdx::new(utterance),
            first_word,
            words,
            language: language.to_owned(),
        }
    }

    /// Transcript order: by utterance, then by first word.
    pub(super) fn order(&self) -> (UtteranceIdx, u64) {
        (self.utterance, self.first_word)
    }

    fn locate(
        &self,
        chat: &crate::chat_ops::ChatFile,
    ) -> Result<SecondaryMorphologySpan, UtteranceNotInDocument> {
        Ok(SecondaryMorphologySpan {
            location: UtteranceLocator::of(chat, self.utterance)?.into(),
            first_word: self.first_word,
            words: self.words,
            language: self.language.clone(),
        })
    }
}

/// Utterances whose primary morphology was declined, by index in the document
/// the report describes. Each is left exactly as it was; the report is what
/// says so.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PrimaryReport {
    declined: Vec<DeclinedPrimary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclinedPrimary {
    utterance: UtteranceIdx,
    cause: PrimaryMorphologyDeclineCause,
    detail: String,
}

impl PrimaryReport {
    /// No utterance was declined.
    pub(crate) fn none() -> Self {
        Self {
            declined: Vec::new(),
        }
    }

    /// Record the utterances one injection declined.
    pub(crate) fn declined(
        &mut self,
        declined: Vec<batchalign_transform::morphosyntax::DeclinedUtterance>,
    ) {
        self.declined
            .extend(declined.into_iter().map(|utterance| DeclinedPrimary {
                utterance: utterance.address.utterance(),
                cause: utterance.cause.into(),
                detail: utterance.reason.chars().take(500).collect(),
            }));
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.declined.extend(other.declined);
    }

    /// Write a `%xbatchalign` note on every declined utterance of `chat`,
    /// replacing morphotag's notes from earlier runs: the notes and the
    /// job record's shortfall are read from this one list, so they cannot
    /// disagree. Runs on every morphotag output, so a rerun with nothing
    /// declined clears the old notes.
    pub(crate) fn annotate(
        &self,
        chat: &mut crate::chat_ops::ChatFile,
    ) -> Result<(), batchalign_transform::notes::NoteError> {
        use batchalign_transform::notes::{NoteCommand, UtteranceNote, replace_notes};
        // By utterance position, which the provenance stamps written before
        // this do not move; `replace_notes` finds the line.
        let notes = self.declined.iter().map(|declined| {
            UtteranceNote::new(
                declined.utterance,
                format!(
                    "not analysed; {} ({})",
                    declined.cause.describe(),
                    declined.detail
                ),
            )
        });
        replace_notes(chat, NoteCommand::Morphotag, notes)
    }

    /// The shortfall, located against `chat`, when any utterance was declined.
    fn into_record(
        mut self,
        chat: &crate::chat_ops::ChatFile,
    ) -> Result<Option<OutputShortfallRecord>, UtteranceNotInDocument> {
        if self.declined.is_empty() {
            return Ok(None);
        }
        self.declined.sort_by_key(|declined| declined.utterance);
        let limit = crate::api::FileOutputDiagnostics::FIRST_FINDINGS;
        Ok(Some(OutputShortfallRecord::PrimaryMorphologyIncomplete {
            utterances: self.declined.len() as u64,
            first_utterances: self
                .declined
                .iter()
                .take(limit)
                .map(|declined| {
                    Ok(PrimaryMorphologyDeclinedUtterance {
                        location: UtteranceLocator::of(chat, declined.utterance)?.into(),
                        cause: declined.cause,
                        detail: declined.detail.clone(),
                    })
                })
                .collect::<Result<_, UtteranceNotInDocument>>()?,
        }))
    }
}

/// The analysis reports an output carries: primary declines and secondary
/// outcomes, each its own domain. One value, required by
/// [`MorphosyntaxOutput::new`], so no path can drop one by forgetting it.
#[derive(Debug, Clone)]
pub(crate) struct AnalysisReports {
    pub(crate) primary: PrimaryReport,
    pub(crate) secondary: SecondaryReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpanFailure {
    span: SpanAddress,
    cause: SecondaryMorphologyFailureKind,
    detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Unplaced {
    utterance: UtteranceIdx,
    words: u64,
    cause: SecondaryExtractionFailure,
    detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SecondaryReport {
    failures: Vec<SpanFailure>,
    exclusions: Vec<SpanAddress>,
    unplaced: Vec<Unplaced>,
}

impl SecondaryReport {
    #[cfg(test)]
    pub(crate) fn for_test(
        failures: Vec<(SpanAddress, SecondaryMorphologyFailureKind, &str)>,
        exclusions: Vec<SpanAddress>,
    ) -> Self {
        Self {
            failures: failures
                .into_iter()
                .map(|(span, cause, detail)| SpanFailure {
                    span,
                    cause,
                    detail: detail.to_owned(),
                })
                .collect(),
            exclusions,
            unplaced: Vec::new(),
        }
    }
    pub(super) fn failed(
        &mut self,
        span: SpanAddress,
        cause: SecondaryMorphologyFailureKind,
        detail: impl std::fmt::Display,
    ) {
        self.failures.push(SpanFailure {
            span,
            cause,
            detail: detail.to_string().chars().take(500).collect(),
        });
    }

    pub(super) fn excluded(&mut self, span: SpanAddress) {
        self.exclusions.push(span);
    }

    pub(super) fn extend(&mut self, other: Self) {
        self.failures.extend(other.failures);
        self.exclusions.extend(other.exclusions);
        self.unplaced.extend(other.unplaced);
    }

    pub(super) fn capture_extraction(
        &mut self,
        extraction: &batchalign_transform::morphosyntax::l2::L2Extraction,
    ) {
        use crate::api::SecondaryExtractionFailure as Cause;
        use batchalign_transform::morphosyntax::l2::L2ExtractError;
        for unplaced in extraction.unaligned() {
            let cause = match &unplaced.error {
                L2ExtractError::LanguageUnresolved => Cause::LanguageUnresolved,
                L2ExtractError::Alignment(_) => Cause::Alignment,
                L2ExtractError::TokenizationDisagrees { .. } => Cause::TokenizationDisagrees,
                L2ExtractError::SplitWord { .. } => Cause::SplitWord,
                L2ExtractError::UnmappedWord { .. } => Cause::UnmappedWord,
            };
            self.unplaced.push(Unplaced {
                utterance: unplaced.address.utterance(),
                words: unplaced.at_s_words as u64,
                cause,
                detail: unplaced.error.to_string().chars().take(500).collect(),
            });
        }
    }

    /// Put the same producer-owned availability facts in the saved transcript.
    /// Replacing only this named comment preserves contributor comments.
    pub(crate) fn annotate(
        &self,
        chat: &mut crate::chat_ops::ChatFile,
    ) -> Result<(), talkbank_model::model::BlankContent> {
        use talkbank_model::{Header, Line, Location};
        const PREFIX: &str = "[batchalign morphology availability]";
        let preserve = |line: &Line| {
            !matches!(line,
            Line::Header { header, .. } if matches!(header.as_ref(),
                Header::Comment { content } if content.as_bullet_content().to_chat_string().starts_with(PREFIX)))
        };
        let mut notes = Vec::new();
        if !self.exclusions.is_empty() {
            let languages: std::collections::BTreeSet<_> = self
                .exclusions
                .iter()
                .map(|span| span.language.as_str())
                .collect();
            let words: u64 = self.exclusions.iter().map(|span| span.words).sum();
            notes.push(format!(
                "L2 morphology unsupported for {}: {words} word(s) in {} span(s)",
                languages.into_iter().collect::<Vec<_>>().join(", "),
                self.exclusions.len()
            ));
        }
        if !self.failures.is_empty() {
            notes.push(format!(
                "L2 analysis incomplete for {} span(s); see job diagnostics for causes",
                self.failures.len()
            ));
        }
        if !self.unplaced.is_empty() {
            notes.push(format!("L2 analysis could not be dispatched for {} word(s); see job diagnostics for language or alignment causes", self.unplaced.iter().map(|item| item.words).sum::<u64>()));
        }
        if notes.is_empty() {
            chat.lines.retain(preserve);
            return Ok(());
        }
        let text = format!(
            "{PREFIX} {}. L2|xxx marks unavailable analysis; surrounding morphology is retained.",
            notes.join("; ")
        );
        let content = talkbank_model::model::CommentContent::try_from_text(text)?;
        chat.lines.retain(preserve);
        let position = crate::provenance::insert_pos_after_constant_headers(chat);
        chat.lines.insert(
            position,
            Line::header(Location::Unlocated, Header::Comment { content }),
        );
        Ok(())
    }

    /// Counts include every span; inline examples are bounded for job
    /// polling, and only those are located, against `chat`, the document
    /// this report describes.
    pub(crate) fn into_records(
        mut self,
        chat: &crate::chat_ops::ChatFile,
    ) -> Result<(Vec<OutputShortfallRecord>, Vec<OutputExclusionRecord>), UtteranceNotInDocument>
    {
        self.failures.sort_by_key(|failure| failure.span.order());
        self.exclusions.sort_by_key(SpanAddress::order);
        self.unplaced.sort_by_key(|unplaced| unplaced.utterance);
        let limit = crate::api::FileOutputDiagnostics::FIRST_FINDINGS;
        let mut shortfalls = Vec::new();
        if !self.failures.is_empty() {
            shortfalls.push(OutputShortfallRecord::SecondaryMorphologyIncomplete {
                spans: self.failures.len() as u64,
                words: self.failures.iter().map(|failure| failure.span.words).sum(),
                first_spans: self
                    .failures
                    .iter()
                    .take(limit)
                    .map(|failure| {
                        Ok(SecondaryMorphologyFailure {
                            span: failure.span.locate(chat)?,
                            cause: failure.cause.into(),
                            detail: failure.detail.clone(),
                        })
                    })
                    .collect::<Result<_, UtteranceNotInDocument>>()?,
            });
        }
        if !self.unplaced.is_empty() {
            shortfalls.push(OutputShortfallRecord::SecondaryMorphologyUnplaced {
                utterances: self.unplaced.len() as u64,
                words: self.unplaced.iter().map(|unplaced| unplaced.words).sum(),
                first_utterances: self
                    .unplaced
                    .iter()
                    .take(limit)
                    .map(|unplaced| {
                        Ok(SecondaryMorphologyUnplacedUtterance {
                            location: UtteranceLocator::of(chat, unplaced.utterance)?.into(),
                            words: unplaced.words,
                            cause: unplaced.cause,
                            detail: unplaced.detail.clone(),
                        })
                    })
                    .collect::<Result<_, UtteranceNotInDocument>>()?,
            });
        }
        let exclusions = if self.exclusions.is_empty() {
            Vec::new()
        } else {
            vec![OutputExclusionRecord::UnsupportedSecondaryMorphology {
                spans: self.exclusions.len() as u64,
                words: self.exclusions.iter().map(|span| span.words).sum(),
                first_spans: self
                    .exclusions
                    .iter()
                    .take(limit)
                    .map(|span| span.locate(chat))
                    .collect::<Result<_, UtteranceNotInDocument>>()?,
            }]
        };
        Ok((shortfalls, exclusions))
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
    primary: PrimaryReport,
    secondary: SecondaryReport,
    provenance: crate::provenance::MorphologyProvenanceReport,
}

impl MorphologyReport {
    /// The records of both reports, located against `chat`, the document
    /// they describe.
    pub(crate) fn into_records(
        self,
        chat: &crate::chat_ops::ChatFile,
    ) -> Result<(Vec<OutputShortfallRecord>, Vec<OutputExclusionRecord>), UtteranceNotInDocument>
    {
        let mut shortfalls: Vec<OutputShortfallRecord> =
            self.primary.into_record(chat)?.into_iter().collect();
        let (secondary, exclusions) = self.secondary.into_records(chat)?;
        shortfalls.extend(secondary);
        shortfalls.extend(self.provenance.into_record());
        Ok((shortfalls, exclusions))
    }
}

/// A text command's output as it will be written, with its records located
/// against that same document. Made only by [`MorphosyntaxOutput::finish`].
#[derive(Debug)]
pub(crate) struct FinishedMorphosyntax<D = PostValidated> {
    pub(crate) document: D,
    pub(crate) shortfalls: Vec<OutputShortfallRecord>,
    pub(crate) exclusions: Vec<OutputExclusionRecord>,
}

/// A judged document the reports can be located against: the one written.
pub(crate) trait WrittenDocument {
    fn written(&self) -> &crate::chat_ops::ChatFile;
}

impl WrittenDocument for PostValidated {
    fn written(&self) -> &crate::chat_ops::ChatFile {
        self.document()
    }
}

impl WrittenDocument for crate::pipeline::post_validate::ProducedOutput {
    fn written(&self) -> &crate::chat_ops::ChatFile {
        self.document()
    }
}

impl<D: WrittenDocument> MorphosyntaxOutput<D> {
    /// Locate the reports against this output's own document, the one
    /// written, leaving no way to pass a different one. Refused only when a
    /// record cannot name its utterance there: an internal fault.
    pub(crate) fn finish(self) -> Result<FinishedMorphosyntax<D>, UtteranceNotInDocument> {
        let (shortfalls, exclusions) = self.report.into_records(self.document.written())?;
        Ok(FinishedMorphosyntax {
            document: self.document,
            shortfalls,
            exclusions,
        })
    }
}

impl<D> MorphosyntaxOutput<D> {
    pub(crate) fn new(document: D, reports: AnalysisReports) -> Self {
        let AnalysisReports { primary, secondary } = reports;
        Self {
            document,
            report: MorphologyReport {
                primary,
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

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (D, MorphologyReport) {
        (self.document, self.report)
    }

    /// Consumers without L2 or prior-reuse reporting cannot silently discard
    /// either report if their policy changes later.
    pub(crate) fn without_reports(self) -> Result<D, crate::error::ServerError> {
        if self.report.primary.declined.is_empty()
            && self.report.secondary.failures.is_empty()
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
        Self::new(
            document,
            AnalysisReports {
                primary: PrimaryReport::none(),
                secondary: SecondaryReport::default(),
            },
        )
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

    /// The span at `utterance` (counting from 1) of [`document`].
    fn span(utterance: usize) -> SpanAddress {
        SpanAddress::for_test(utterance - 1, 2, 3, "spa")
    }

    /// A document of `utterances` utterances, `*PAR:\tutterance N .`.
    fn document(utterances: usize) -> crate::chat_ops::ChatFile {
        use talkbank_parser_tests::test_error::ParseProductTestExt;
        let mut text = String::from(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
             @ID:\teng|test|PAR|||||Participant|||\n",
        );
        for n in 1..=utterances {
            text.push_str(&format!("*PAR:\tutterance {n} .\n"));
        }
        text.push_str("@End\n");
        talkbank_parser::TreeSitterParser::new()
            .expect("parser init")
            .parse_chat_file(&text)
            .expect_built()
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
        let (shortfalls, exclusions) = report.into_records(&document(101)).unwrap();
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
        assert_eq!(first_spans[0].span.location.utterance(), 1);
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
                "main_tier": "*PAR:\tutterance 101 .",
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
        let error = MorphosyntaxOutput::new(
            (),
            AnalysisReports {
                primary: PrimaryReport::none(),
                secondary: report,
            },
        )
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
            MorphosyntaxOutput::new(
                (),
                AnalysisReports {
                    primary: PrimaryReport::none(),
                    secondary: SecondaryReport::default()
                }
            )
            .without_reports()
            .is_ok()
        );
        let reported = MorphosyntaxOutput::new(
            (),
            AnalysisReports {
                primary: PrimaryReport::none(),
                secondary: SecondaryReport::default(),
            },
        )
        .with_provenance(crate::provenance::MorphologyProvenanceReport::Incomplete(
            crate::provenance::IncrementalMorphotagStampError::PriorIncomplete,
        ));
        assert!(reported.clone().without_reports().is_err());
        let (_, report) = reported.into_parts();
        let (shortfalls, exclusions) = report.into_records(&document(1)).unwrap();
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
        assert!(diagnostics.summary().contains("provenance incomplete"));
    }
}
