//! What the workers reported about each morphosyntax response a run applies:
//! which Stanza model produced it, and which relations it repaired inside it.
//!
//! The worker reports both on every analyzed item
//! (`MorphosyntaxModelIdentityV2`: Stanza version, language, and the pipeline
//! variant that ran; `UdRelationRepairV2`: one per relation it rewrote). The
//! server admits them per item into an [`AdmittedMorphosyntaxResponse`], and
//! the provenance comment names the distinct models behind the responses that
//! were actually applied and counts the repairs inside them
//! ([`AppliedAnalyses`]).
//!
//! An item with no words carries neither: the worker ran no model for it, so
//! there is nothing to name, nothing could have been repaired, and neither is
//! invented.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use crate::api::{StampJoiner, StampSafeText};
use crate::chat_ops::nlp::UdResponse;
use crate::provenance::EngineNames;
use crate::types::worker_v2::{MorphosyntaxModelIdentityV2, UdRelationRepairV2};

/// Where one admitted response came from.
///
/// Deliberately visible no wider than this module tree, which is the one that
/// receives worker results. An enum's variant fields are public to whoever can
/// name the enum, so a wider visibility would let any module in the crate mint
/// a `Worker` source carrying a model that never ran and repairs nobody made,
/// and hand it to [`AppliedAnalyses::record`]. Narrowing it is what makes
/// "these repairs came from a worker" a property of the graph rather than of
/// the care taken by callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum MorphosyntaxResponseSource {
    /// A worker analyzed the item with this model, repairing these relations
    /// on the way. The two travel together because a repair is something a
    /// model's output needed: there is no repair without a model that ran.
    Worker {
        /// The model that analyzed the item.
        model: MorphosyntaxModelIdentityV2,
        /// Every relation the worker rewrote in this item's analysis, in the
        /// order it made them. Empty means it rewrote nothing.
        repairs: Vec<UdRelationRepairV2>,
    },
    /// The worker had no words to analyze for this item, so no model ran.
    NoWordsToAnalyze,
    /// No model ran: the item's language has no Stanza morphosyntax support,
    /// so the server filled an empty response that leaves the `L2|xxx`
    /// placeholder in place.
    UnsupportedLanguagePlaceholder,
}

/// One morphosyntax response admitted at the worker boundary, paired with
/// where it came from. Constructed only by the dispatch code that receives the
/// worker result, and by the unsupported-language fill.
#[derive(Debug, Clone)]
pub(crate) struct AdmittedMorphosyntaxResponse {
    response: UdResponse,
    source: MorphosyntaxResponseSource,
}

impl AdmittedMorphosyntaxResponse {
    /// A response a worker produced, with the model it reported and the
    /// relations it repaired. All three are taken by value: they move out of
    /// the wire result, uncloned.
    pub(super) fn from_worker(
        response: UdResponse,
        model: MorphosyntaxModelIdentityV2,
        repairs: Vec<UdRelationRepairV2>,
    ) -> Self {
        Self {
            response,
            source: MorphosyntaxResponseSource::Worker { model, repairs },
        }
    }

    /// The empty response for an item the worker had no words to analyze.
    pub(super) fn no_words() -> Self {
        Self {
            response: UdResponse {
                sentences: Vec::new(),
            },
            source: MorphosyntaxResponseSource::NoWordsToAnalyze,
        }
    }

    /// The empty response filled for an item in an unsupported language.
    pub(super) fn unsupported_language_placeholder() -> Self {
        Self {
            response: UdResponse {
                sentences: Vec::new(),
            },
            source: MorphosyntaxResponseSource::UnsupportedLanguagePlaceholder,
        }
    }

    /// The UD analysis.
    pub(crate) fn response(&self) -> &UdResponse {
        &self.response
    }

    /// Where the analysis came from. Scoped to this module tree with the type
    /// it returns.
    pub(super) fn source(&self) -> &MorphosyntaxResponseSource {
        &self.source
    }

    /// A worker-produced response from the standard pipeline that repaired
    /// nothing, for tests that stub the worker.
    #[cfg(test)]
    pub(crate) fn for_test(response: UdResponse, stanza_version: &str, lang: &str) -> Self {
        Self::for_test_with_repairs(response, stanza_version, lang, Vec::new())
    }

    /// The same, carrying the repairs a worker reported, for tests about them.
    #[cfg(test)]
    pub(crate) fn for_test_with_repairs(
        response: UdResponse,
        stanza_version: &str,
        lang: &str,
        repairs: Vec<UdRelationRepairV2>,
    ) -> Self {
        Self {
            response,
            source: MorphosyntaxResponseSource::Worker {
                model: MorphosyntaxModelIdentityV2 {
                    stanza_version: crate::api::ReportedEngineName::try_from(stanza_version)
                        .expect("test Stanza version is a valid engine name"),
                    lang: crate::api::LanguageCode3::try_new(lang).expect("test language is ISO 639-3"),
                    pipeline: crate::types::worker_v2::MorphosyntaxPipelineV2::Standard,
                },
                repairs,
            },
        }
    }
}

/// Every distinct Stanza model behind the responses one run applied.
///
/// Empty when no model analyzed anything (no items, wordless items, or only
/// unsupported-language placeholders), in which case provenance names no
/// engine rather than inventing one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MorphosyntaxEngines(BTreeSet<MorphosyntaxModelIdentityV2>);

impl MorphosyntaxEngines {
    /// The `engine=` names: one per distinct model,
    /// `stanza-<version>:<lang>:<pipeline>`.
    ///
    /// Total. The version is a reported engine name, the language a language
    /// code and the pipeline its wire name, each already stamp-safe text, and
    /// the joins between them add only `stanza-` and `:`. The pipeline's name
    /// is the one its wire form uses, so there is no second table to drift.
    fn engine_names(&self) -> EngineNames {
        self.0
            .iter()
            .map(|model| {
                let versioned = StampSafeText::join(
                    &const { StampSafeText::from_static("stanza-") },
                    [model.stanza_version.as_stamp_text()],
                    StampJoiner::Concat,
                );
                StampSafeText::join(
                    &versioned,
                    [
                        &StampSafeText::from(&model.lang),
                        &model.pipeline.stamp_name(),
                    ],
                    StampJoiner::Colon,
                )
            })
            .collect()
    }
}

/// Every relation the workers repaired inside the responses one run applied.
///
/// Kept whole rather than as a running total, because a repair is meant to be
/// attributable as well as countable: the word and both relations are what
/// makes a count actionable, and they exist nowhere else once the analysis is
/// injected. The count is what provenance writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MorphosyntaxRepairs(Vec<UdRelationRepairV2>);

impl MorphosyntaxRepairs {
    /// How many repairs were made, or `None` when none were.
    ///
    /// `NonZeroUsize` rather than `usize`, so "some repairs" and "no repairs"
    /// are different values rather than a number a caller must remember to
    /// compare against zero. The stamp writes a field only in the first case.
    fn count(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.0.len())
    }

    /// The repairs tallied by kind, for one operator-facing line: for example
    /// `relation_alias=2 unknown_relation=1`.
    fn tally(&self) -> String {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for repair in &self.0 {
            *counts.entry(repair.kind.wire_name()).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|(kind, count)| format!("{kind}={count}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// What the workers reported about the responses one file applied: the models
/// behind them, and the relations repaired inside them.
///
/// One value rather than two travelling side by side. They are collected in
/// the same walk, from the same source, and read by the same provenance stamp,
/// so a phase that carried one and not the other would be a phase that can
/// stamp a file whose repairs nobody counted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AppliedAnalyses {
    engines: MorphosyntaxEngines,
    repairs: MorphosyntaxRepairs,
}

impl AppliedAnalyses {
    /// Nothing has been applied yet.
    pub(crate) fn none() -> Self {
        Self::default()
    }

    /// Split a batch that is applied as a whole into its UD responses and what
    /// the workers reported about them. By value: each model and each repair
    /// moves into this value (a repeated model is simply dropped), so the
    /// common path clones nothing.
    pub(crate) fn take_applied(
        responses: Vec<AdmittedMorphosyntaxResponse>,
    ) -> (Vec<UdResponse>, Self) {
        let mut applied = Self::none();
        let responses = responses
            .into_iter()
            .map(|admitted| {
                match admitted.source {
                    MorphosyntaxResponseSource::Worker { model, repairs } => {
                        applied.engines.0.insert(model);
                        applied.repairs.0.extend(repairs);
                    }
                    MorphosyntaxResponseSource::NoWordsToAnalyze
                    | MorphosyntaxResponseSource::UnsupportedLanguagePlaceholder => {}
                }
                admitted.response
            })
            .collect();
        (responses, applied)
    }

    /// Record what one applied response reported, while it is still borrowed.
    /// The model is cloned only when the set does not already hold it.
    ///
    /// Scoped with the source it takes: outside this module tree the only way
    /// to add to an `AppliedAnalyses` is [`Self::take_applied`] or
    /// [`Self::extend`], both of which consume values a worker produced.
    pub(super) fn record(&mut self, source: &MorphosyntaxResponseSource) {
        match source {
            MorphosyntaxResponseSource::Worker { model, repairs } => {
                if !self.engines.0.contains(model) {
                    self.engines.0.insert(model.clone());
                }
                self.repairs.0.extend(repairs.iter().cloned());
            }
            MorphosyntaxResponseSource::NoWordsToAnalyze
            | MorphosyntaxResponseSource::UnsupportedLanguagePlaceholder => {}
        }
    }

    /// Merge in what another applied batch reported.
    pub(crate) fn extend(&mut self, other: Self) {
        self.engines.0.extend(other.engines.0);
        self.repairs.0.extend(other.repairs.0);
    }

    /// Whether no model analyzed anything, in which case nothing was repaired
    /// either: a repair is something a model's output needed.
    pub(crate) fn ran_no_model(&self) -> bool {
        self.engines.0.is_empty()
    }

    /// The `engine=` names of the distinct models behind the applied
    /// responses.
    pub(crate) fn engine_names(&self) -> EngineNames {
        self.engines.engine_names()
    }

    /// How many relations were repaired, or `None` when none were.
    pub(crate) fn repair_count(&self) -> Option<NonZeroUsize> {
        self.repairs.count()
    }

    /// The repairs tallied by kind, for one operator-facing line.
    pub(crate) fn repair_tally(&self) -> String {
        self.repairs.tally()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{MorphosyntaxPipelineV2, UdRelationRepairKindV2};

    fn empty() -> UdResponse {
        UdResponse {
            sentences: Vec::new(),
        }
    }

    fn repair(kind: UdRelationRepairKindV2, from: &str, to: &str) -> UdRelationRepairV2 {
        UdRelationRepairV2 {
            kind,
            word: "ne".to_owned(),
            from_relation: from.to_owned(),
            to_relation: to.to_owned(),
        }
    }

    /// The `engine=` value the models join into.
    fn joined(applied: &AppliedAnalyses) -> Option<String> {
        applied
            .engine_names()
            .joined()
            .map(|joined| joined.as_str().to_owned())
    }

    #[test]
    fn engines_name_distinct_worker_models_and_skip_items_no_model_ran_for() {
        let applied = vec![
            AdmittedMorphosyntaxResponse::for_test(empty(), "1.11.1", "eng"),
            AdmittedMorphosyntaxResponse::unsupported_language_placeholder(),
            AdmittedMorphosyntaxResponse::no_words(),
            AdmittedMorphosyntaxResponse::for_test(empty(), "1.11.1", "eng"),
            AdmittedMorphosyntaxResponse::for_test(empty(), "1.11.1", "deu"),
        ];
        let (responses, applied) = AppliedAnalyses::take_applied(applied);
        assert_eq!(responses.len(), 5);
        assert_eq!(
            joined(&applied).as_deref(),
            Some("stanza-1.11.1:deu:standard+stanza-1.11.1:eng:standard")
        );
    }

    /// The pipeline variant is part of the identity: the same Stanza version
    /// and language through a different pipeline is a different model. Names
    /// are joined in text order, like every engine list.
    #[test]
    fn the_pipeline_variant_is_named() {
        let mut applied = AppliedAnalyses::none();
        for pipeline in [
            MorphosyntaxPipelineV2::MandarinRetokenize,
            MorphosyntaxPipelineV2::Standard,
            MorphosyntaxPipelineV2::MandarinRetokenize,
        ] {
            applied.record(&MorphosyntaxResponseSource::Worker {
                model: MorphosyntaxModelIdentityV2 {
                    stanza_version: crate::api::ReportedEngineName::try_from("1.11.1")
                        .expect("valid"),
                    lang: crate::api::LanguageCode3::try_new("cmn").expect("valid"),
                    pipeline,
                },
                repairs: Vec::new(),
            });
        }
        assert_eq!(
            joined(&applied).as_deref(),
            Some("stanza-1.11.1:cmn:mandarin_retokenize+stanza-1.11.1:cmn:standard")
        );
    }

    #[test]
    fn no_analysis_names_no_engine_and_repairs_nothing() {
        let (_, applied) = AppliedAnalyses::take_applied(vec![
            AdmittedMorphosyntaxResponse::unsupported_language_placeholder(),
            AdmittedMorphosyntaxResponse::no_words(),
        ]);
        assert!(applied.ran_no_model());
        assert_eq!(applied.repair_count(), None);
        assert!(AppliedAnalyses::none().ran_no_model());
        assert_eq!(AppliedAnalyses::none().repair_count(), None);
    }

    /// Repairs accumulate across the items of a batch and across batches, and
    /// they are counted, not deduplicated: the same rewrite on two words is
    /// two repairs.
    #[test]
    fn repairs_are_counted_across_items_and_batches() {
        let (_, mut applied) = AppliedAnalyses::take_applied(vec![
            AdmittedMorphosyntaxResponse::for_test_with_repairs(
                empty(),
                "1.11.1",
                "ita",
                vec![
                    repair(UdRelationRepairKindV2::RelationAlias, "iob", "iobj"),
                    repair(UdRelationRepairKindV2::RelationAlias, "iob", "iobj"),
                ],
            ),
            AdmittedMorphosyntaxResponse::no_words(),
        ]);
        let (_, secondary) = AppliedAnalyses::take_applied(vec![
            AdmittedMorphosyntaxResponse::for_test_with_repairs(
                empty(),
                "1.11.1",
                "ita",
                vec![repair(UdRelationRepairKindV2::UnknownRelation, "wat", "dep")],
            ),
        ]);
        applied.extend(secondary);
        assert_eq!(applied.repair_count().map(NonZeroUsize::get), Some(3));
        assert_eq!(applied.repair_tally(), "relation_alias=2 unknown_relation=1");
    }

    /// A response no model produced cannot carry repairs: the source that
    /// holds them is the one that names a model.
    #[test]
    fn a_response_no_model_produced_carries_no_repairs() {
        assert_eq!(
            AdmittedMorphosyntaxResponse::no_words().source(),
            &MorphosyntaxResponseSource::NoWordsToAnalyze
        );
        assert_eq!(
            AdmittedMorphosyntaxResponse::unsupported_language_placeholder().source(),
            &MorphosyntaxResponseSource::UnsupportedLanguagePlaceholder
        );
    }
}
