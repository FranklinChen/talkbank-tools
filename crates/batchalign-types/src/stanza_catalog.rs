//! What a worker reports about its installed Stanza catalog.
//!
//! The report is one of three outcomes, by type: the catalog the worker read
//! (every ISO 639-3 language it can serve and that entry's processors), Stanza
//! not installed in the worker, or a failure to build the report, with its
//! reason. A worker whose table build failed therefore cannot be mistaken for
//! one whose catalog serves no language, and the server can say why it has no
//! registry.
//!
//! Language keys are [`LanguageCode3`] and processors are [`StanzaProcessor`],
//! both parsed when the report is deserialized: a report naming an unknown
//! processor or a key that is not an ISO 639-3 code fails as a malformed
//! capabilities response, so nothing downstream compares processor names as
//! strings.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::domain::LanguageCode3;

/// One processor a Stanza catalog entry can ship.
///
/// The worker reports exactly these names (`StanzaProcessorName` in
/// `batchalign/worker/_types.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StanzaProcessor {
    /// Tokenization.
    Tokenize,
    /// Multi-word token expansion.
    Mwt,
    /// Part-of-speech and morphological features.
    Pos,
    /// Lemmatization.
    Lemma,
    /// Dependency parsing.
    Depparse,
    /// Constituency parsing.
    Constituency,
    /// Coreference.
    Coref,
}

impl StanzaProcessor {
    /// Every processor, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Tokenize,
        Self::Mwt,
        Self::Pos,
        Self::Lemma,
        Self::Depparse,
        Self::Constituency,
        Self::Coref,
    ];

    /// The name the worker reports and Stanza's own processor name.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Tokenize => "tokenize",
            Self::Mwt => "mwt",
            Self::Pos => "pos",
            Self::Lemma => "lemma",
            Self::Depparse => "depparse",
            Self::Constituency => "constituency",
            Self::Coref => "coref",
        }
    }
}

/// What one command's Stanza pipeline needs from a language's catalog entry:
/// the processors its loader requests unconditionally.
///
/// Closed, and one per Stanza-backed command, so a command is admitted for a
/// language by what IT loads, never by another command's needs. The worker
/// reads these sets from Rust (`batchalign_core.STANZA_REQUIREMENTS`) to build
/// its pipelines, so the admitted set and the loaded set are one list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StanzaRequirement {
    /// `morphotag` (and `compare`, which morphotags its main transcript):
    /// tokenize, pos, lemma, depparse; `mwt` is added where shipped.
    Morphosyntax,
    /// `utseg`'s Stanza constituency route: tokenize, pos (which the
    /// constituency parser requires) and constituency itself, whose trees are
    /// the only thing the route reads; `mwt` is added where shipped. Lemmas
    /// are never read, so a lemmatizer is not required. A catalog entry
    /// without a constituency parser cannot segment at all, so it is not
    /// admitted: before 2026-10-24 it was, and every multi-word item failed.
    UtteranceSegmentation,
    /// `coref`: tokenize and coref.
    Coreference,
}

impl StanzaRequirement {
    /// Every requirement, in declaration order.
    pub const ALL: [Self; 3] = [
        Self::Morphosyntax,
        Self::UtteranceSegmentation,
        Self::Coreference,
    ];

    /// The processors a catalog entry must ship.
    pub const fn processors(self) -> &'static [StanzaProcessor] {
        match self {
            Self::Morphosyntax => &[
                StanzaProcessor::Tokenize,
                StanzaProcessor::Pos,
                StanzaProcessor::Lemma,
                StanzaProcessor::Depparse,
            ],
            Self::UtteranceSegmentation => &[
                StanzaProcessor::Tokenize,
                StanzaProcessor::Pos,
                StanzaProcessor::Constituency,
            ],
            Self::Coreference => &[StanzaProcessor::Tokenize, StanzaProcessor::Coref],
        }
    }

    /// The name the worker reads the set under.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Morphosyntax => "morphosyntax",
            Self::UtteranceSegmentation => "utterance_segmentation",
            Self::Coreference => "coreference",
        }
    }

    /// What the requirement serves, for a refusal message.
    pub const fn purpose(self) -> &'static str {
        match self {
            Self::Morphosyntax => "morphology",
            Self::UtteranceSegmentation => "utterance segmentation",
            Self::Coreference => "coreference",
        }
    }

    /// The processors, comma-separated, for a refusal message.
    pub fn processor_list(self) -> String {
        self.processors()
            .iter()
            .map(|processor| processor.wire_name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// A worker's statement of why it reports no catalog: never empty, so a
/// report always says something an operator can act on.
///
/// The only constructor is [`TryFrom<String>`], which serde deserializes
/// through, so an empty or blank statement is refused at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct StanzaReportReason(String);

/// A blank statement where a worker owed one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a Stanza catalog report's reason must not be blank")]
pub struct BlankStanzaReportReason;

impl TryFrom<String> for StanzaReportReason {
    type Error = BlankStanzaReportReason;

    fn try_from(reason: String) -> Result<Self, Self::Error> {
        if reason.trim().is_empty() {
            Err(BlankStanzaReportReason)
        } else {
            Ok(Self(reason))
        }
    }
}

impl From<StanzaReportReason> for String {
    fn from(reason: StanzaReportReason) -> Self {
        reason.0
    }
}

impl StanzaReportReason {
    /// The statement.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StanzaReportReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A worker's account of its installed Stanza catalog.
///
/// On the wire `{"kind": "catalog", "languages": {"eng": ["tokenize", ...]}}`,
/// `{"kind": "not_installed", "import_error": "..."}` or
/// `{"kind": "failed", "reason": "..."}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StanzaCatalogReport {
    /// The catalog the worker read: each language it can serve, with the
    /// processors of the catalog entry serving it.
    Catalog {
        /// Keyed by ISO 639-3 code.
        languages: BTreeMap<LanguageCode3, BTreeSet<StanzaProcessor>>,
    },
    /// The `stanza` package does not import in this worker: a worker that
    /// serves no Stanza task (a test echo worker reports this too, saying
    /// it imports none).
    NotInstalled {
        /// Why it does not import, as the worker saw it.
        import_error: StanzaReportReason,
    },
    /// Stanza imported, but the worker could not build its report.
    Failed {
        /// The failure as the worker saw it (exception type and message).
        reason: StanzaReportReason,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures; the parse is the assertion
mod tests {
    use super::*;

    #[test]
    fn every_processor_round_trips_its_wire_name() {
        for processor in StanzaProcessor::ALL {
            let json = serde_json::to_string(&processor).expect("serializes");
            assert_eq!(json, format!("\"{}\"", processor.wire_name()));
            assert_eq!(
                serde_json::from_str::<StanzaProcessor>(&json).expect("parses"),
                processor
            );
        }
    }

    /// Each outcome keeps its own shape; a failure is never an empty catalog.
    #[test]
    fn each_outcome_has_its_own_wire_shape() {
        let failed: StanzaCatalogReport =
            serde_json::from_str(r#"{"kind":"failed","reason":"ImportError: no ISO table"}"#)
                .expect("a failure parses");
        assert_eq!(
            failed,
            StanzaCatalogReport::Failed {
                reason: StanzaReportReason::try_from("ImportError: no ISO table".to_owned())
                    .expect("not blank")
            }
        );
        let not_installed: StanzaCatalogReport = serde_json::from_str(
            r#"{"kind":"not_installed","import_error":"No module named 'stanza'"}"#,
        )
        .expect("parses");
        assert_eq!(
            not_installed,
            StanzaCatalogReport::NotInstalled {
                import_error: StanzaReportReason::try_from("No module named 'stanza'".to_owned())
                    .expect("not blank")
            }
        );
        let catalog: StanzaCatalogReport =
            serde_json::from_str(r#"{"kind":"catalog","languages":{"kaz":["tokenize","lemma"]}}"#)
                .expect("a catalog parses");
        assert_eq!(
            catalog,
            StanzaCatalogReport::Catalog {
                languages: BTreeMap::from([(
                    LanguageCode3::try_new("kaz").expect("valid"),
                    BTreeSet::from([StanzaProcessor::Tokenize, StanzaProcessor::Lemma]),
                )]),
            }
        );
    }

    /// Each command's set is what its loader requests: utterance
    /// segmentation needs no lemmatizer, so a catalog entry without one (the
    /// Vietnamese entry ships constituency but no lemma) still serves it, and
    /// it needs the constituency parser whose trees it reads.
    #[test]
    fn each_requirement_names_what_its_loader_requests() {
        use StanzaProcessor::{Constituency, Coref, Depparse, Lemma, Pos, Tokenize};
        assert_eq!(
            StanzaRequirement::Morphosyntax.processors(),
            [Tokenize, Pos, Lemma, Depparse]
        );
        assert_eq!(
            StanzaRequirement::UtteranceSegmentation.processors(),
            [Tokenize, Pos, Constituency]
        );
        assert_eq!(
            StanzaRequirement::Coreference.processors(),
            [Tokenize, Coref]
        );
        assert_eq!(
            StanzaRequirement::Morphosyntax.processor_list(),
            "tokenize, pos, lemma, depparse"
        );
    }

    /// A report the server cannot parse is a malformed response, not an
    /// empty catalog.
    #[test]
    fn a_malformed_catalog_is_refused_at_the_boundary() {
        for raw in [
            r#"{"kind":"catalog","languages":{"en":["tokenize"]}}"#,
            r#"{"kind":"catalog","languages":{"eng":["sentiment"]}}"#,
            r#"{"kind":"catalog"}"#,
            r#"{"kind":"failed"}"#,
            // RED FIRST: a blank reason, and an import failure with none,
            // parsed, so a report could say nothing about why.
            r#"{"kind":"failed","reason":""}"#,
            r#"{"kind":"failed","reason":"  "}"#,
            r#"{"kind":"not_installed"}"#,
            r#"{"kind":"not_installed","import_error":""}"#,
        ] {
            assert!(
                serde_json::from_str::<StanzaCatalogReport>(raw).is_err(),
                "{raw} must not parse"
            );
        }
    }
}
