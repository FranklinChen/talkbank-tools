//! Server-side translation orchestrator.
//!
//! Owns the full CHAT lifecycle for translate jobs:
//! parse → collect payloads → infer → inject %xtra → serialize.
//!
//! Python workers receive only the rendered source text of each item and
//! return the engine's raw translation: pure inference, with zero CHAT
//! awareness.
//!
//! # What is sent
//!
//! What was spoken: a [`TranslationSource`] per utterance, rendered once at
//! the worker boundary (see `batchalign_transform::translate`). Every word the
//! speaker produced travels, retraced words and filled pauses included, and so
//! does the terminator, so a question reaches the engine as a question.
//!
//! # Engine identity
//!
//! Every translated item names the engine that translated it, and the
//! provenance comment names the engines behind the translations that were
//! applied. Nothing is read from the worker's capability report, so a
//! translate job never fails for want of a pre-dispatch identity.
//!
//! # Empty translations
//!
//! A translation with nothing to apply is refused where it is admitted, as a
//! typed per-item failure that names the engine, so the file fails visibly
//! instead of being written with a `%xtra` tier silently missing. The verdict
//! is terminal: the same request gets the same answer from the same engine, so
//! the remedy is a different engine or different options, not a retry.
//!
//! # One request per utterance
//!
//! Items go to the worker one at a time ([`items::translate_items`]), and the
//! control plane keeps the gap between requests and waits out a provider's
//! transient answers by the engine's [`provider::ProviderPolicy`]. The worker
//! reports and never sleeps: a wait inside a worker request was invisible to
//! the job's deadline and cancellation. A failed item fails its file, so the
//! items after it are not sent.

pub(crate) mod items;
pub(crate) mod provider;

use std::collections::HashMap;

use crate::api::{InvalidStampSafeText, LanguageCode3, ReportedEngineName};
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::provenance::{ResultNamedCommand, TextStamp};
use crate::types::engines::TranslateEngineName;
use crate::types::worker_v2::TranslationItemResultV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_translate_request_v2};
use crate::worker::text_result_v2::parse_translate_result_v2;
use batchalign_transform::translate::{
    TranslateBatchItem, TranslationSource, TranslationText, WritingSystem, apply_translate_results,
    chat_punct_chars, collect_translate_payloads,
};
use batchalign_transform::validate::ValidityLevel;
use tracing::{info, warn};

use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::pipeline::text_infer::{TextBatchHooks, run_text_batch_pipeline};
use crate::text_batch::{ItemFailure, TextBatchFileInput, TextBatchFileResults};

use self::items::{ItemTransport, Wait, translate_items};
use self::provider::ProviderGiveUp;

/// The failures translate itself defines for one item.
///
/// Translate's own type, so utseg, coref and morphotag cannot represent them:
/// their per-item failure is [`crate::text_batch::EngineItemFailure`], whose
/// command-specific variant is uninhabited. Every variant is final for the
/// file: the control plane's retry (`ProviderTerminal`) is not asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TranslateFailure {
    /// The engine answered, with nothing that can be written.
    EmptyTranslation {
        /// The engine that produced the empty translation, as the worker
        /// REPORTED it on the result (the identity provenance records), which
        /// is not the SELECTED engine `Provider` names below: one is a fact
        /// about the answer, the other about the job.
        engine: ReportedEngineName,
    },
    /// The provider answered something other than a translation, and the
    /// engine's policy gave up on it (at once, or after its cooldowns).
    Provider {
        /// The engine whose provider answered.
        engine: TranslateEngineName,
        /// Why the policy stopped.
        give_up: ProviderGiveUp,
        /// The provider library's own message.
        error: String,
    },
    /// Not sent: the batch stopped at an earlier item's failure.
    NotAttempted {
        /// The item whose failure stopped the batch.
        after_item: usize,
    },
}

impl std::fmt::Display for TranslateFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTranslation { engine } => write!(
                f,
                "{engine} returned an empty translation; try a different --translate-engine \
                 or options and run the file again"
            ),
            Self::Provider {
                engine,
                give_up,
                error,
            } => write!(
                f,
                "translate engine {} {give_up}; the item was not translated (use \
                 --translate-engine nllb for a local model, or try later): {error}",
                engine.as_wire_name()
            ),
            Self::NotAttempted { after_item } => write!(
                f,
                "not attempted: the file stopped at item {after_item}, which failed"
            ),
        }
    }
}

/// What one translate item can fail with.
pub(crate) type TranslateItemFailure = ItemFailure<TranslateFailure>;

/// One translation item admitted at the worker boundary.
///
/// A translated item holds a [`TranslationText`], which exists only when there
/// is something to apply, so no later stage has to ask whether a translation
/// is usable.
#[derive(Debug, Clone)]
pub(crate) enum AdmittedTranslation {
    /// Translated, by the engine the worker named on the result.
    Translated {
        /// The postprocessed translation.
        text: TranslationText,
        /// The engine that produced it.
        engine: ReportedEngineName,
    },
    /// The input was blank, so nothing was translated and no engine ran.
    BlankInput,
}

impl AdmittedTranslation {
    /// The translation to apply, if there is one.
    fn translation(&self) -> Option<&TranslationText> {
        match self {
            Self::Translated { text, .. } => Some(text),
            Self::BlankInput => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-file batch translate processing
// ---------------------------------------------------------------------------

/// Translate one CHAT file through the text pipeline.
///
/// One file, by signature: the per-item loop stops a file at its first failed
/// utterance and reports the rest as not attempted by their index in that
/// file, which is only true of a file that was sent alone. The pipeline
/// underneath still takes a slice; this is the only translate caller, and it
/// hands it exactly one element.
///
/// Returns `(filename, Ok(output) | Err(error))` for the file.
pub(crate) async fn process_translate_file(
    file: &TextBatchFileInput,
    lang: &LanguageCode3,
    engine: &TranslateEngineName,
    pool: &WorkerPool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    run_text_batch_pipeline(
        std::slice::from_ref(file),
        lang,
        pool,
        TextBatchHooks {
            command: crate::api::ReleasedCommand::Translate,
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_translate_payloads,
            apply: apply_translate_file,
            provenance: translation_provenance,
        },
        async move |pool, items, lang| infer_batch(pool, items, lang, engine, cancellation).await,
    )
    .await
}

/// The translate stamp, named by the engines of the translations applied.
/// Never `Err`: each engine is a reported engine name, already stamp-safe; the
/// `Result` is the shape every text command's provenance hook shares.
fn translation_provenance(
    lang: &LanguageCode3,
    responses: &[AdmittedTranslation],
) -> Result<TextStamp, InvalidStampSafeText> {
    Ok(crate::provenance::result_named_provenance(
        ResultNamedCommand::Translate,
        lang,
        responses.iter().filter_map(|response| match response {
            AdmittedTranslation::Translated { engine, .. } => Some(engine),
            AdmittedTranslation::BlankInput => None,
        }),
    ))
}

/// Apply translate responses for one file. A blank input has nothing to apply;
/// a translation that had nothing to apply never reached this point, because
/// admission refused it.
fn apply_translate_file(
    chat_file: &mut ChatFile,
    items: &[(usize, TranslationSource)],
    responses: &[AdmittedTranslation],
) {
    let translation_map: HashMap<usize, TranslationText> = items
        .iter()
        .zip(responses)
        .filter_map(|((line_idx, _item), response)| {
            response
                .translation()
                .map(|translation| (*line_idx, translation.clone()))
        })
        .collect();
    if !translation_map.is_empty() {
        apply_translate_results(chat_file, &translation_map);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Translate one batch of items, one worker request per item.
///
/// Renders each [`TranslationSource`] for the source language's script (the
/// one place source text is built), then hands the rendered items to
/// [`translate_items`], which paces, retries and stops by the engine's
/// provider policy. Each request carries one item, so its transport deadline
/// is the per-request floor and never includes a wait, and each wait here is
/// raced against the job's cancellation.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, TranslationSource)],
    lang: &LanguageCode3,
    engine: &TranslateEngineName,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedTranslation, TranslateItemFailure>>, ServerError> {
    // Fallible in chatter 0.3.0; stringified error (`LanguageCodeError` not
    // re-exported upstream).
    let src_lang_code = LanguageCode::new(lang.as_ref()).map_err(|e| {
        ServerError::Validation(format!(
            "translate: invalid source language code {:?}: {e}",
            lang.as_ref()
        ))
    })?;

    // Render what was spoken, in the writing system this language uses.
    let script = WritingSystem::of_language(&src_lang_code);
    let rendered: Vec<TranslateBatchItem> = items
        .iter()
        .map(|(_, source)| source.render(script))
        .collect();
    let punct_strings = chat_punct_chars();
    let punct_refs: Vec<&str> = punct_strings.iter().map(|s| s.as_str()).collect();

    info!(
        num_items = items.len(),
        lang = %lang,
        engine = %engine.as_wire_name(),
        "Dispatching translate execute_v2 requests, one per item"
    );

    let mut transport = WorkerTransport {
        pool,
        lang,
        engine,
        cancellation,
    };
    let outcome = translate_items(&rendered, engine, &punct_refs, &mut transport).await?;
    if outcome.echoed > 0 {
        warn!(
            echoed = outcome.echoed,
            total = rendered.len(),
            engine = %engine.as_wire_name(),
            "translations identical to their source text; an engine that echoes its input \
             is not refused, but this many is worth a look"
        );
    }
    Ok(outcome.results)
}

/// The production [`ItemTransport`]: one request per item to the worker
/// pool, with the item carried inline in the request envelope (no prepared
/// artifact touches the filesystem), and waits raced against the job's
/// cancellation.
struct WorkerTransport<'a> {
    pool: &'a WorkerPool,
    lang: &'a LanguageCode3,
    engine: &'a TranslateEngineName,
    cancellation: Cancellation<'a>,
}

impl ItemTransport for WorkerTransport<'_> {
    async fn send(
        &mut self,
        item: &TranslateBatchItem,
    ) -> Result<TranslationItemResultV2, ServerError> {
        let request = build_translate_request_v2(
            &PreparedTextRequestIdsV2::for_task("translate"),
            self.lang,
            &LanguageCode3::eng(),
            self.engine.worker_backend(),
            item,
        )
        .map_err(|error| {
            ServerError::Validation(format!(
                "failed to build translate V2 worker request: {error}"
            ))
        })?;
        let response =
            dispatch_execute_v2_with_retry(self.pool, self.lang, &request, self.cancellation)
                .await?;
        let result = parse_translate_result_v2(response).map_err(|error| {
            ServerError::Validation(format!("invalid translate V2 result: {error}"))
        })?;
        single_item(result.items)
    }

    async fn pause(&mut self, wait: Wait) -> Result<(), ServerError> {
        self.cancellation.sleep(wait.duration()).await
    }
}

/// The one item result of a one-item request.
///
/// A count other than one is a protocol bug in the batch, not a per-item
/// failure, so it is the outer `Err(ServerError)`.
fn single_item(
    items: Vec<TranslationItemResultV2>,
) -> Result<TranslationItemResultV2, ServerError> {
    <[TranslationItemResultV2; 1]>::try_from(items)
        .map(|[item]| item)
        .map_err(|items| {
            ServerError::Validation(format!(
                "translate V2 returned {} items for 1 request",
                items.len()
            ))
        })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn engine() -> ReportedEngineName {
        ReportedEngineName::try_from("googletrans-v1").expect("valid engine name")
    }

    fn admitted(text: &str) -> AdmittedTranslation {
        AdmittedTranslation::Translated {
            text: TranslationText::admit(text).expect("test translation has content"),
            engine: engine(),
        }
    }

    #[test]
    fn a_one_item_request_yields_its_one_item() {
        let item = single_item(vec![TranslationItemResultV2::BlankInput]).expect("one item");
        assert!(matches!(item, TranslationItemResultV2::BlankInput));
    }

    #[test]
    fn a_count_other_than_one_is_a_protocol_error_not_an_item_failure() {
        for items in [
            Vec::new(),
            vec![
                TranslationItemResultV2::BlankInput,
                TranslationItemResultV2::BlankInput,
            ],
        ] {
            let count = items.len();
            let err = single_item(items).expect_err("wrong count");
            assert!(
                err.to_string()
                    .contains(&format!("returned {count} items for 1 request")),
                "{err}"
            );
        }
    }

    /// The empty-translation failure names the engine and the remedy, and the
    /// file's verdict is terminal: the same request gets the same answer, so
    /// the control plane must not be told to expect a different one.
    #[test]
    fn an_empty_translation_fails_the_file_terminally() {
        let failure = crate::text_batch::TextWorkflowFileError::item_errors(
            "translate",
            crate::text_batch::ItemFailures::of_one(crate::text_batch::ItemError {
                item_index: 0,
                failure: ItemFailure::Command(TranslateFailure::EmptyTranslation {
                    engine: engine(),
                }),
            }),
        );
        assert_eq!(
            failure.category(),
            crate::scheduling::FailureCategory::ProviderTerminal
        );
        let rendered = failure.to_string();
        assert!(rendered.contains("googletrans-v1"), "{rendered}");
        assert!(rendered.contains("--translate-engine"), "{rendered}");
    }

    #[test]
    fn a_not_attempted_item_names_the_item_that_stopped_the_file() {
        let rendered = TranslateFailure::NotAttempted { after_item: 3 }.to_string();
        assert_eq!(
            rendered,
            "not attempted: the file stopped at item 3, which failed"
        );
    }

    /// A blank input is admitted as its own outcome: nothing to apply, and no
    /// engine to name.
    #[test]
    fn blank_input_translates_nothing_and_names_no_engine() {
        let applied = vec![AdmittedTranslation::BlankInput];
        assert!(applied[0].translation().is_none());
        assert!(matches!(
            translation_provenance(&LanguageCode3::eng(), &applied).expect("valid fields"),
            TextStamp::NotStamped(crate::provenance::NoStampReason::NothingApplied)
        ));
    }

    /// The translate stamp names the engine the translations named.
    #[test]
    fn translation_provenance_names_the_engine_of_the_applied_results() {
        let applied = vec![admitted("Hello .")];
        let TextStamp::Stamped(comment) =
            translation_provenance(&LanguageCode3::eng(), &applied).expect("valid fields")
        else {
            panic!("a translation names its engine");
        };
        let stamp = comment.format();
        assert!(
            stamp.starts_with("[fc-ba3 translate | engine=googletrans-v1 ; lang=eng | "),
            "{stamp}"
        );
    }
}
