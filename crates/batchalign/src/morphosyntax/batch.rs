//! Secondary L2 dispatch for @s words.

use std::collections::HashMap;

use super::identity::{AdmittedMorphosyntaxResponse, AppliedAnalyses, MorphosyntaxResponseSource};
use super::worker::infer_batch;
use crate::api::{SecondaryMorphologyFailureKind as Failure, SecondaryMorphologySpan};
use crate::chat_ops::morphosyntax_ops::l2;
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::error::ServerError;
use crate::infer_retry::Cancellation;
use crate::pipeline::PipelineServices;
use crate::stanza_registry::{Admitted, Morphosyntax};

/// Owns the unchanged CHAT spans throughout secondary inference. There is no
/// retokenizing transition: secondary analysis has no authority to rewrite CHAT.
struct FixedSpanBatch {
    spans: Vec<l2::L2SpanPlan>,
}

struct AnalyzedSpan {
    plan: l2::L2SpanPlan,
    analysis: AdmittedMorphosyntaxResponse,
}

enum SpanBatchOutcome {
    Analyzed(Vec<AnalyzedSpan>),
    Refused {
        batch: FixedSpanBatch,
        cause: Failure,
        detail: String,
    },
}

impl FixedSpanBatch {
    fn items(&self) -> Vec<crate::chat_ops::morphosyntax_ops::MorphosyntaxBatchItem> {
        use crate::chat_ops::morphosyntax_ops::{BatchWord, MorphosyntaxBatchItem};
        self.spans
            .iter()
            .map(|span| {
                MorphosyntaxBatchItem::new(
                    span.words().cloned().map(BatchWord::analysed).collect(),
                    span.terminator().clone(),
                    span.target_lang().clone(),
                )
            })
            .collect()
    }

    async fn infer(
        self,
        services: PipelineServices<'_>,
        lang: &Admitted<Morphosyntax>,
        cancellation: Cancellation<'_>,
    ) -> Result<SpanBatchOutcome, ServerError> {
        let items = self.items();
        // Boundary preservation still admits model MWT analyses inside each
        // CHAT word; it forbids merging or splitting CHAT word positions.
        match infer_batch(
            services.pool,
            &items,
            lang,
            &Default::default(),
            false,
            None,
            cancellation,
        )
        .await
        {
            Ok(responses) => Ok(self.admit(responses)),
            Err(ServerError::Cancelled) => Err(ServerError::Cancelled),
            Err(error) => Ok(SpanBatchOutcome::Refused {
                batch: self,
                cause: Failure::Worker,
                detail: error.to_string(),
            }),
        }
    }

    fn admit(self, responses: Vec<AdmittedMorphosyntaxResponse>) -> SpanBatchOutcome {
        // Positional worker output cannot establish ownership if its count
        // differs. Never expose two independent arrays to the merge consumer.
        if self.spans.len() != responses.len() {
            let detail = format!(
                "expected {} span responses, received {}",
                self.spans.len(),
                responses.len()
            );
            return SpanBatchOutcome::Refused {
                batch: self,
                cause: Failure::ResponseCount,
                detail,
            };
        }
        SpanBatchOutcome::Analyzed(
            self.spans
                .into_iter()
                .zip(responses)
                .map(|(plan, analysis)| AnalyzedSpan { plan, analysis })
                .collect(),
        )
    }
}

/// Retains identity and source until installation, not merely merge success.
struct PendingSplice {
    span: l2::MergedL2Span,
    address: SecondaryMorphologySpan,
    source: MorphosyntaxResponseSource,
}

fn span_address(chat: &ChatFile, span: &l2::L2SpanPlan) -> SecondaryMorphologySpan {
    SecondaryMorphologySpan {
        utterance: chat
            .lines
            .iter()
            .take(span.line_idx() + 1)
            .filter(|line| matches!(line, talkbank_model::model::Line::Utterance(_)))
            .count() as u64,
        first_word: span.first_word().as_usize() as u64 + 1,
        words: span.len() as u64,
        language: span.target_lang().as_str().to_owned(),
    }
}

fn merge_responses(
    chat: &ChatFile,
    outcome: SpanBatchOutcome,
    applied: &mut AppliedAnalyses,
    merged: &mut Vec<PendingSplice>,
) {
    let analyzed = match outcome {
        SpanBatchOutcome::Analyzed(analyzed) => analyzed,
        SpanBatchOutcome::Refused {
            batch,
            cause,
            detail,
        } => {
            for span in batch.spans {
                applied
                    .secondary_mut()
                    .failed(span_address(chat, &span), cause, &detail);
            }
            return;
        }
    };
    for AnalyzedSpan {
        plan: span,
        analysis: admitted,
    } in analyzed
    {
        let address = span_address(chat, &span);
        let Some(sentence) = admitted.response().sentence() else {
            applied.secondary_mut().failed(
                address,
                Failure::EmptyAnalysis,
                "worker returned no sentence",
            );
            continue;
        };
        match l2::merge_planned_secondary_span(span, sentence) {
            Ok(span) => merged.push(PendingSplice {
                span,
                address,
                source: admitted.source().clone(),
            }),
            Err(error) => applied
                .secondary_mut()
                .failed(address, Failure::Merge, error),
        }
    }
}

fn install_spans(
    chat: &mut ChatFile,
    mut merged: Vec<PendingSplice>,
    applied: &mut AppliedAnalyses,
) {
    merged.sort_by_key(|pending| (pending.address.utterance, pending.address.first_word));
    for pending in merged {
        let outcome = l2::splice_l2_into_chat(chat, vec![pending.span]);
        for receipt in outcome.spans() {
            match receipt.disposition() {
                l2::SpanSpliceDisposition::Applied { .. } => applied.record(&pending.source),
                l2::SpanSpliceDisposition::Retained(reason) => applied.secondary_mut().failed(
                    pending.address.clone(),
                    Failure::Splice,
                    format!("{reason:?}"),
                ),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Experimental: secondary L2 dispatch for @s words
// ---------------------------------------------------------------------------

/// Dispatch @s words to secondary language workers and splice results back.
///
/// This function:
/// 1. Plans contiguous spans by target language, consuming the positions
/// 2. For each supported target language, builds one `MorphosyntaxBatchItem`
///    per span and dispatches them to a secondary Stanza worker via
///    `infer_batch`
/// 3. Merges each span with its secondary analysis (the secondary owns the
///    words' morphology, the primary the span's attachment)
/// 4. Splices the merged spans into the ChatFile, replacing `L2|xxx`
///
/// Words of unsupported languages, failed dispatches and spans whose
/// secondary analysis does not merge keep `L2|xxx`, each reported.
///
/// Returns what the secondary workers reported about the analysis merged in,
/// so a caller that stamps provenance names those models beside the
/// primary-language ones and counts their repairs with the rest.
/// Cancellation is not a successful placeholder fallback: it aborts this
/// transition, so callers cannot post-validate or write a stopped analysis.
pub(crate) async fn dispatch_secondary_l2(
    chat_file: &mut ChatFile,
    deferred: Vec<l2::L2DeferredPosition>,
    primary: &Admitted<Morphosyntax>,
    services: PipelineServices<'_>,
    filename: &str,
    cancellation: Cancellation<'_>,
) -> Result<AppliedAnalyses, ServerError> {
    cancellation
        .supervise(Box::pin(dispatch_secondary_l2_active(
            chat_file,
            deferred,
            primary,
            services,
            filename,
            cancellation,
        )))
        .await?
}

async fn dispatch_secondary_l2_active(
    chat_file: &mut ChatFile,
    deferred: Vec<l2::L2DeferredPosition>,
    primary: &Admitted<Morphosyntax>,
    services: PipelineServices<'_>,
    filename: &str,
    cancellation: Cancellation<'_>,
) -> Result<AppliedAnalyses, ServerError> {
    let deferred_words = deferred.len();
    let dispatch_plan = l2::plan_dispatch_spans(deferred);
    let planned_spans = dispatch_plan.spans.len();

    // Group spans by target language for batched dispatch.
    let mut by_lang: HashMap<LanguageCode, Vec<l2::L2SpanPlan>> = HashMap::new();
    for span in dispatch_plan.spans {
        by_lang
            .entry(span.target_lang().clone())
            .or_default()
            .push(span);
    }

    tracing::info!(
        filename = %filename,
        deferred = deferred_words,
        spans = planned_spans,
        languages = by_lang.len(),
        "L2 morphotag: dispatching @s words to secondary workers"
    );

    let mut merged = Vec::new();
    let mut applied = AppliedAnalyses::none();

    for (target_lang, lang_spans) in by_lang {
        let total_words: usize = lang_spans.iter().map(l2::L2SpanPlan::len).sum();
        let lang3 = match crate::api::LanguageCode3::try_new(target_lang.as_ref()) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    lang = %target_lang,
                    words = total_words,
                    "L2 morphotag: invalid language code; words stay L2|xxx"
                );
                for span in lang_spans {
                    applied.secondary_mut().failed(
                        span_address(chat_file, &span),
                        Failure::InvalidLanguage,
                        "language code cannot be dispatched",
                    );
                }
                continue;
            }
        };

        // The registry that admitted the file's primary language decides,
        // under the same requirement. A language it has no morphotag
        // pipeline for is an exclusion (the words stay `L2|xxx`, reported as
        // unsupported). There is no "no registry" case: the primary's proof
        // holds the registry.
        let secondary = match primary.sibling(&lang3) {
            Ok(secondary) => secondary,
            Err(gap) => {
                tracing::info!(
                    lang = %lang3,
                    words = total_words,
                    %gap,
                    "L2 morphotag: no Stanza pipeline for this language; words stay L2|xxx"
                );
                for span in lang_spans {
                    applied
                        .secondary_mut()
                        .excluded(span_address(chat_file, &span));
                }
                continue;
            }
        };

        let outcome = FixedSpanBatch { spans: lang_spans }
            .infer(services, &secondary, cancellation)
            .await?;
        merge_responses(chat_file, outcome, &mut applied, &mut merged);
    }

    install_spans(chat_file, merged, &mut applied);
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::dispatch_secondary_l2;

    /// The file's admitted primary language, from a registry double.
    fn english() -> crate::stanza_registry::Admitted<crate::stanza_registry::Morphosyntax> {
        crate::stanza_registry::StanzaRegistry::admitting(&["eng"])
            .shared()
            .admit(&crate::api::LanguageCode3::eng())
            .expect("the double admits English")
    }

    fn planned_fixture() -> (crate::chat_ops::ChatFile, Vec<super::l2::L2SpanPlan>) {
        use crate::chat_ops::morphosyntax_ops::*;
        let parser = crate::chat_parser();
        let (mut chat, _) = batchalign_transform::parse::parse_lenient(
            &parser,
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello mundo@s:spa .\n@End\n",
        );
        let lang = crate::chat_ops::LanguageCode::new("eng").unwrap();
        let langs = declared_languages(&chat, &lang).expect("admitted fixture language context");
        let items = collect_payloads(&chat, &lang, langs.as_ref(), MultilingualPolicy::ProcessAll)
            .batch_items;
        let response = serde_json::from_str(
            r#"{"sentences":[{"words":[
            {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
            {"id":2,"text":"xbxxx","lemma":"xbxxx","upos":"NOUN","head":1,"deprel":"obj"},
            {"id":3,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
        ]}]}"#,
        )
        .unwrap();
        let injection = inject_results(
            &parser,
            &mut chat,
            items,
            vec![response],
            &lang,
            TokenizationMode::Preserve,
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let (_, deferred) = injection.into_parts();
        (
            chat,
            super::l2::plan_dispatch_spans(deferred.into_reported_positions()).spans,
        )
    }

    #[test]
    fn unanswered_and_empty_secondary_spans_are_reported_without_provenance_or_mutation() {
        for response_count in [0, 1, 2] {
            let (chat, spans) = planned_fixture();
            let before = batchalign_transform::serialize::to_chat_string(&chat);
            let responses = (0..response_count)
                .map(|_| super::AdmittedMorphosyntaxResponse::no_words())
                .collect();
            let mut applied = super::AppliedAnalyses::none();
            let mut merged = Vec::new();
            let outcome = super::FixedSpanBatch { spans }.admit(responses);
            super::merge_responses(&chat, outcome, &mut applied, &mut merged);
            assert!(merged.is_empty());
            assert!(applied.ran_no_model());
            let (shortfalls, exclusions) = applied.into_secondary().into_records();
            assert!(exclusions.is_empty());
            let crate::api::OutputShortfallRecord::SecondaryMorphologyIncomplete {
                spans,
                words,
                first_spans,
            } = &shortfalls[0]
            else {
                panic!("secondary failure");
            };
            assert_eq!((*spans, *words), (1, 1));
            assert_eq!(
                first_spans[0].cause,
                if response_count == 1 {
                    super::Failure::EmptyAnalysis
                } else {
                    super::Failure::ResponseCount
                }
                .into()
            );
            assert_eq!(
                batchalign_transform::serialize::to_chat_string(&chat),
                before
            );
        }
    }

    #[test]
    fn secondary_provenance_is_recorded_only_after_actual_installation() {
        let (mut chat, spans) = planned_fixture();
        let response = serde_json::from_str(
            r#"{"sentences":[{"words":[
            {"id":1,"text":"mundo","lemma":"mundo","upos":"NOUN","head":0,"deprel":"root"}
        ]}]}"#,
        )
        .unwrap();
        let mut applied = super::AppliedAnalyses::none();
        let mut merged = Vec::new();
        super::merge_responses(
            &chat,
            super::FixedSpanBatch { spans }.admit(vec![
                super::AdmittedMorphosyntaxResponse::for_test(response, "9.9.9", "spa"),
            ]),
            &mut applied,
            &mut merged,
        );
        assert_eq!(merged.len(), 1);
        assert!(applied.ran_no_model(), "merge success is not installation");
        super::install_spans(&mut chat, merged, &mut applied);
        assert!(!applied.ran_no_model());
        let (shortfalls, exclusions) = applied.into_secondary().into_records();
        assert!(shortfalls.is_empty());
        assert!(exclusions.is_empty());
        assert!(batchalign_transform::serialize::to_chat_string(&chat).contains("noun|mundo"));
    }

    #[tokio::test]
    async fn cancelled_secondary_transition_cannot_admit_applied_analysis() {
        use crate::cache::UtteranceCache;
        use crate::infer_retry::Cancellation;
        use crate::pipeline::PipelineServices;
        use crate::worker::pool::{PoolConfig, WorkerPool};

        let input = include_str!("../../../../test-fixtures/eng_spa_at_s_contiguous.cha");
        let (mut chat, _) =
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), input);
        let before = batchalign_transform::serialize::to_chat_string(&chat);
        let scratch = tempfile::tempdir().expect("test cache directory");
        let cache = UtteranceCache::sqlite(Some(scratch.path().join("cache")))
            .await
            .expect("test cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let result = dispatch_secondary_l2(
            &mut chat,
            Vec::new(),
            &english(),
            PipelineServices::new(&pool, &cache),
            "test.cha",
            Cancellation::Token(&token),
        )
        .await;
        assert!(matches!(result, Err(crate::error::ServerError::Cancelled)));
        assert_eq!(
            batchalign_transform::serialize::to_chat_string(&chat),
            before
        );
    }

    #[tokio::test]
    async fn secondary_transition_with_no_work_and_no_job_remains_successful() {
        use crate::cache::UtteranceCache;
        use crate::infer_retry::Cancellation;
        use crate::pipeline::PipelineServices;
        use crate::worker::pool::{PoolConfig, WorkerPool};

        let input = include_str!("../../../../test-fixtures/eng_spa_at_s_contiguous.cha");
        let (mut chat, _) =
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), input);
        let before = batchalign_transform::serialize::to_chat_string(&chat);
        let scratch = tempfile::tempdir().expect("test cache directory");
        let cache = UtteranceCache::sqlite(Some(scratch.path().join("cache")))
            .await
            .expect("test cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let result = dispatch_secondary_l2(
            &mut chat,
            Vec::new(),
            &english(),
            PipelineServices::new(&pool, &cache),
            "test.cha",
            Cancellation::NotWired {
                reason: "offline test has no job",
            },
        )
        .await;
        assert!(result.is_ok());
        assert_eq!(
            batchalign_transform::serialize::to_chat_string(&chat),
            before
        );
    }
}
