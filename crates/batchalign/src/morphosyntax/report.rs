//! Analysis and provenance outcomes travel beside CHAT admission. Only their
//! producers record outcomes; valid CHAT does not certify complete analysis
//! or complete attribution.

use crate::api::{
    OutputExclusionRecord, OutputShortfallRecord, SecondaryExtractionFailure,
    SecondaryMorphologyFailure, SecondaryMorphologyFailureKind, SecondaryMorphologySpan,
    SecondaryMorphologyUnplacedUtterance, UtteranceLocator, UtteranceNotInDocument,
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
    pub(crate) fn annotate(&self, chat: &mut crate::chat_ops::ChatFile) {
        use talkbank_model::{Header, Line, Location};
        const PREFIX: &str = "[batchalign morphology availability]";
        chat.lines.retain(|line| {
            !matches!(line,
            Line::Header { header, .. } if matches!(header.as_ref(),
                Header::Comment { content } if content.to_chat_string().starts_with(PREFIX)))
        });
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
            return;
        }
        let text = format!(
            "{PREFIX} {}. L2|xxx marks unavailable analysis; surrounding morphology is retained.",
            notes.join("; ")
        );
        let position = crate::provenance::insert_pos_after_constant_headers(chat);
        chat.lines.insert(
            position,
            Line::header(
                Location::Unlocated,
                Header::Comment {
                    content: crate::chat_ops::BulletContent::from_text(text),
                },
            ),
        );
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
        let (mut shortfalls, exclusions) = self.secondary.into_records(chat)?;
        shortfalls.extend(self.provenance.into_record());
        Ok((shortfalls, exclusions))
    }
}

/// A text command's output as it will be written, with its records located
/// against that same document. Made only by [`MorphosyntaxOutput::finish`].
#[derive(Debug)]
pub(crate) struct FinishedMorphosyntax {
    pub(crate) document: PostValidated,
    pub(crate) shortfalls: Vec<OutputShortfallRecord>,
    pub(crate) exclusions: Vec<OutputExclusionRecord>,
}

/// Why a text command's output could not be finished.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FinishRefused {
    /// The abbreviation merge broke its own gate: a transform defect. The
    /// unmerged output is not written in its place.
    #[error(transparent)]
    Merge(Box<crate::pipeline::post_validate::AbbreviationMergeRefused>),
    /// A record could not name its utterance in the written document: an
    /// internal fault.
    #[error(transparent)]
    Location(#[from] UtteranceNotInDocument),
}

impl MorphosyntaxOutput<PostValidated> {
    /// Merge abbreviations when asked, then locate the reports against the
    /// merged document, the one written. Merging rewrites main tiers (`F B I`
    /// becomes `FBI`), so a record located before it could quote text the
    /// written file does not hold; locating here, against this output's own
    /// document, leaves no way to pass a different one.
    pub(crate) fn finish(
        self,
        merge: crate::pipeline::post_validate::MergeAbbreviations,
    ) -> Result<FinishedMorphosyntax, FinishRefused> {
        use crate::pipeline::post_validate::MergeAbbreviations;
        let document = match merge {
            MergeAbbreviations::Merge => self
                .document
                .with_abbreviations_merged()
                .map_err(FinishRefused::Merge)?,
            MergeAbbreviations::Leave => self.document,
        };
        let (shortfalls, exclusions) = self.report.into_records(document.document())?;
        Ok(FinishedMorphosyntax {
            document,
            shortfalls,
            exclusions,
        })
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

    #[cfg(test)]
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
