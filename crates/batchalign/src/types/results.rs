//! Structured result types for server-side orchestrators.
//!
//! Each orchestrator returns a rich result type that includes both the
//! serialized CHAT output and any intermediate data produced during
//! processing.  The dispatch layer decides what to write to disk vs.
//! what to store in the trace cache.

use crate::chat_ops::fa::WordGapHealing;
use crate::chat_ops::morphosyntax_ops::RetokenizationInfo;
use batchalign_transform::asr_postprocess::AsrPipelineSnapshot;
use talkbank_model::ChatFile;

use super::traces::{
    AsrPipelineTrace, AsrTokenTrace, FaDecisionTrace, FaFallbackEventTrace, FaGroupTrace,
    FaTimelineTrace, FaTimingDecisionTrace, RetokenizationTrace, TimedWordTrace, TimingTrace,
    UtteranceTrace, WordTrace,
};

// ---------------------------------------------------------------------------
// Forced alignment
// ---------------------------------------------------------------------------

/// Structured result from the internal `crate::fa::run_fa_from_ast` pipeline.
///
/// Two phases, one type: `FaResult<ChatFile>` is the DRAFT an FA path builds,
/// whose document has not yet taken the media/timing transition, and
/// `FaResult<FaOutput>` (the default) is the result after it. Every FA path
/// builds a draft and hands it to `FaAdmission::finish`, which owns the
/// source's admitted `@Media` declaration and so is the one place the
/// transition runs, against a declaration known to admit it. The paths used
/// to reconcile the document themselves, each with a `?` that turned an input
/// condition (no `@Media`) into an internal error after all of the work.
/// The fields stay crate-visible for the pass-through path and the trace
/// conversions, but `FaOutput::Processed` holds a
/// [`crate::fa::ReconciledOutput`], which only that transition constructs.
pub(crate) struct FaResult<O = FaOutput> {
    /// The CHAT document: a draft, or reconciled output.
    pub(crate) output: O,
    /// Per-group evidence whose fields cannot become cardinality-misaligned.
    pub(crate) group_evidence: Vec<FaGroupEvidence>,
    /// Selected forced-alignment engine.
    pub(crate) engine: String,
    /// The FA engine the selected worker reported: the namespace every cache
    /// row and evidence envelope of this run was read and written under.
    pub(crate) cache_namespace: crate::engine_reports::FaCacheNamespace,
    /// Decisions made while injecting and enforcing final timing invariants.
    pub(crate) decisions: Vec<FaDecisionTrace>,
    /// Numeric monotonicity effects corresponding to the generic decisions.
    pub(crate) timing_decisions: Vec<FaTimingDecisionTrace>,
    /// Gap-healing policy used for this run.
    pub(crate) gap_healing: WordGapHealing,
    /// Engine fallback events captured during worker inference.
    pub(crate) fallback_events: Vec<FaFallbackEventTrace>,
}

/// CHAT output whose provenance determines whether media timing was allowed to
/// transition.
pub(crate) enum FaOutput {
    /// A declared dummy/NoAlign path that must preserve the input document.
    PassThrough(ChatFile),
    /// A path that performed timing work and reconciled the document state
    /// against its admitted `@Media` declaration.
    Processed(crate::fa::ReconciledOutput),
}

impl FaOutput {
    /// Borrow the AST without erasing whether it was processed or passed through.
    pub(crate) fn as_chat_file(&self) -> &ChatFile {
        match self {
            Self::PassThrough(file) => file,
            Self::Processed(output) => output.as_chat_file(),
        }
    }
}

/// One FA group inseparably paired with the evidence that produced its timing.
///
/// Each request's source and cache key live inside `group.span`, one per
/// dispatch unit, so a group's evidence is one value rather than three
/// parallel fields kept the same length by a cardinality check.
#[derive(Debug)]
pub(crate) struct FaGroupEvidence {
    pub(crate) group: FaGroupTrace,
    pub(crate) pre_injection_timings: Vec<Option<TimingTrace>>,
}

impl FaResult<ChatFile> {
    /// Construct a draft for a legitimate no-group path such as complete
    /// `%wor` reuse or a file with no alignable words.
    ///
    /// Infallible: the media/timing transition belongs to `FaAdmission::finish`.
    pub(crate) fn without_groups(
        chat_file: ChatFile,
        gap_healing: WordGapHealing,
        engine: &str,
        cache_namespace: &crate::engine_reports::FaCacheNamespace,
    ) -> Self {
        Self {
            output: chat_file,
            group_evidence: Vec::new(),
            engine: engine.to_owned(),
            cache_namespace: cache_namespace.clone(),
            decisions: Vec::new(),
            timing_decisions: Vec::new(),
            gap_healing,
            fallback_events: Vec::new(),
        }
    }
}

impl<O> FaResult<O> {
    /// Replace the document, keeping every piece of evidence with it. The
    /// phase transition `FaAdmission::finish` is written with this, so no
    /// evidence field can be dropped on the way from draft to output.
    pub(crate) fn try_map_output<P, E>(
        self,
        transition: impl FnOnce(O) -> Result<P, E>,
    ) -> Result<FaResult<P>, E> {
        Ok(FaResult {
            output: transition(self.output)?,
            group_evidence: self.group_evidence,
            engine: self.engine,
            cache_namespace: self.cache_namespace,
            decisions: self.decisions,
            timing_decisions: self.timing_decisions,
            gap_healing: self.gap_healing,
            fallback_events: self.fallback_events,
        })
    }

    /// Attach the exact decision set after it has passed through CHAT
    /// projection, for a legitimate path with no fresh FA groups.
    pub(crate) fn with_written_decisions(
        mut self,
        written: crate::chat_ops::fa::WrittenFaDecisions,
    ) -> Self {
        let (records, effects) = written.into_evidence();
        self.decisions = records.into_iter().map(Into::into).collect();
        self.timing_decisions = effects.into_iter().map(Into::into).collect();
        self
    }

    /// Convert into a [`FaTimelineTrace`] for dashboard visualization.
    ///
    /// This consumes the whole result, its CHAT output included. The document
    /// itself no longer leaves through here: it leaves as the
    /// [`crate::pipeline::post_validate::PostValidated`] proof that
    /// `FaAdmission` produced, so a caller cannot obtain align's bytes without
    /// also holding the evidence that they may be written.
    pub(crate) fn into_timeline_trace(self) -> FaTimelineTrace {
        let mut groups = Vec::with_capacity(self.group_evidence.len());
        let mut pre_injection_timings = Vec::with_capacity(self.group_evidence.len());
        for evidence in self.group_evidence {
            groups.push(evidence.group);
            pre_injection_timings.push(evidence.pre_injection_timings);
        }
        FaTimelineTrace {
            evidence_schema_version: crate::types::traces::CURRENT_FA_EVIDENCE_SCHEMA_VERSION,
            engine: self.engine,
            engine_version: self.cache_namespace.name().to_string(),
            groups,
            pre_injection_timings,
            post_injection_timings: Vec::new(), // TODO Phase 4
            decisions: self.decisions,
            // Derived from the effects immediately above, never carried
            // beside them: one producer, so the flat section and the tagged
            // union cannot drift, and no drop can reach one without the
            // other. Present and empty when this run discarded nothing.
            dropped_word_timings: self
                .timing_decisions
                .iter()
                .flat_map(FaTimingDecisionTrace::dropped_word_timings)
                .collect(),
            timing_decisions: self.timing_decisions,
            gap_healing: format!("{:?}", self.gap_healing),
            // Post-validation is a fail-closed gate: a violating file fails
            // the command and no trace is produced for it, so there are never
            // violations to report. The wire format still declares the field,
            // and this is the one place that has to say so.
            violations: Vec::new(),
            fallback_events: self.fallback_events,
        }
    }
}

// ---------------------------------------------------------------------------
// ASR pipeline trace conversion
// ---------------------------------------------------------------------------

/// Lossy conversion from the chat-ops-side per-stage snapshot to the
/// dashboard-facing `AsrPipelineTrace`.
///
/// Drops timing and structural detail not surfaced in the trace shape
/// (e.g. `AsrWord::kind`). The trace shape is the dashboard contract;
/// the snapshot is the wire-protocol-free internal capture.
pub fn snapshot_into_pipeline_trace(snapshot: AsrPipelineSnapshot) -> AsrPipelineTrace {
    AsrPipelineTrace {
        raw_tokens: snapshot
            .raw_elements
            .iter()
            .map(|e| AsrTokenTrace {
                value: e.value.as_str().to_owned(),
                ts: e.ts,
                end_ts: e.end_ts,
                token_type: format!("{:?}", e.kind).to_lowercase(),
            })
            .collect(),
        after_compound_merge: snapshot
            .after_compound_merge
            .iter()
            .map(|e| WordTrace {
                text: e.value.as_str().to_owned(),
            })
            .collect(),
        after_timing_extract: snapshot
            .after_timing_extract
            .iter()
            .map(asr_word_to_timed_trace)
            .collect(),
        after_multiword_split: snapshot
            .after_multiword_split
            .iter()
            .map(asr_word_to_timed_trace)
            .collect(),
        after_number_expand: snapshot
            .after_number_expand
            .iter()
            .map(asr_word_to_timed_trace)
            .collect(),
        after_cantonese_norm: snapshot
            .after_cantonese_norm
            .map(|words| words.iter().map(asr_word_to_timed_trace).collect()),
        after_long_turn_split: snapshot
            .after_long_turn_split
            .iter()
            .map(|chunk| chunk.iter().map(asr_word_to_timed_trace).collect())
            .collect(),
        final_utterances: snapshot
            .final_utterances
            .iter()
            .map(|u| UtteranceTrace {
                speaker: u.speaker.as_usize(),
                words: u.words.iter().map(asr_word_to_timed_trace).collect(),
            })
            .collect(),
    }
}

fn asr_word_to_timed_trace(w: &batchalign_transform::asr_postprocess::AsrWord) -> TimedWordTrace {
    TimedWordTrace {
        text: w.text.as_str().to_owned(),
        start_ms: w.start_ms,
        end_ms: w.end_ms,
    }
}

// ---------------------------------------------------------------------------
// Morphosyntax
// ---------------------------------------------------------------------------

/// Structured result from a single-file morphosyntax run.
pub struct MorphosyntaxResult {
    /// Serialized CHAT text with %mor/%gra injected.
    pub chat_text: String,
    /// Retokenization mappings (empty when retokenization is off).
    pub retokenizations: Vec<RetokenizationInfo>,
}

impl MorphosyntaxResult {
    /// Convert retokenization info into dashboard trace format.
    pub fn into_retokenization_traces(self) -> Vec<RetokenizationTrace> {
        self.retokenizations
            .into_iter()
            .map(|info| RetokenizationTrace {
                utterance_index: info.utterance_ordinal,
                original_words: info.original_words,
                stanza_tokens: info.stanza_tokens,
                normalized_original: String::new(), // not captured at this level
                normalized_tokens: String::new(),
                mapping: info.mapping,
                used_fallback: info.used_fallback,
            })
            .collect()
    }
}

#[cfg(test)]
mod fa_result_tests {
    use super::*;

    #[test]
    fn asr_trace_preserves_missing_and_zero_endpoints() {
        use crate::api::AudioPositionSeconds;
        use batchalign_transform::asr_postprocess::{AsrElement, AsrElementKind, AsrRawText};
        let zero = Some(AudioPositionSeconds::try_from(0.0).expect("zero is a position"));
        let snapshot = AsrPipelineSnapshot {
            raw_elements: vec![AsrElement {
                value: AsrRawText::new("hello"),
                ts: None,
                end_ts: zero,
                kind: AsrElementKind::Text,
            }],
            ..Default::default()
        };
        let trace = snapshot_into_pipeline_trace(snapshot);
        let json = serde_json::to_value(&trace).unwrap();
        assert!(json["raw_tokens"][0]["ts"].is_null());
        assert_eq!(json["raw_tokens"][0]["end_ts"], 0.0);
        let decoded: AsrPipelineTrace = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.raw_tokens[0].ts, None);
        assert_eq!(decoded.raw_tokens[0].end_ts, zero);
    }
}
