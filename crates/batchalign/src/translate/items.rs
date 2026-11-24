//! The per-item translate loop: one worker request per utterance, paced and
//! retried by the engine's [`ProviderPolicy`], stopped at the first failure.
//!
//! One request per item, rather than one batch per file, is what lets the
//! control plane own the waits. The worker translates one utterance and
//! reports what happened; this loop decides whether to wait, to send the same
//! item again, to move on, or to stop. Its only effects go through an
//! [`ItemTransport`]: the production one sends worker requests and races each
//! wait against the job's cancellation; a test one answers from a script and
//! records the waits instead of sleeping.
//!
//! A failed item fails its file (one bad utterance abandons the file, the
//! established per-file verdict), so the items after it are not sent: a
//! persistent refusal is paid for once, not once per remaining utterance,
//! and the loop reports them as [`TranslateFailure::NotAttempted`].

use std::future::Future;
use std::time::Duration;

use batchalign_transform::translate::{
    EmptyTranslation, TranslateBatchItem, TranslationText, postprocess_translation,
};
use tracing::warn;

use super::provider::{ProviderAnswer, Verdict};
use super::{AdmittedTranslation, TranslateFailure, TranslateItemFailure};
use crate::error::ServerError;
use crate::text_batch::ItemFailure;
use crate::types::engines::TranslateEngineName;
use crate::types::worker_v2::TranslationItemResultV2;

/// A wait the loop asks for, named so a test can tell the two apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    /// The gap the engine's provider policy keeps between requests.
    Spacing(Duration),
    /// A cooldown before the same item is sent again.
    Cooldown(Duration),
}

impl Wait {
    /// How long to wait, whichever kind it is.
    pub(crate) const fn duration(self) -> Duration {
        match self {
            Self::Spacing(duration) | Self::Cooldown(duration) => duration,
        }
    }
}

/// What the loop settled for a batch of items.
#[derive(Debug)]
pub struct ItemsOutcome {
    /// One result per item, in order.
    pub(crate) results: Vec<Result<AdmittedTranslation, TranslateItemFailure>>,
    /// How many translations came back identical to their source text. An
    /// engine echoing its input is admitted (names, numbers and loanwords
    /// match legitimately), so this is counted and reported, never refused.
    pub(crate) echoed: usize,
}

/// The two effects the loop has: one worker request for one item, and a wait.
///
/// A trait rather than two closures because a boxed `Send` future through
/// the gateway's `async_trait` needs the futures named here to be `Send`,
/// which a closure over borrowed arguments cannot promise for every lifetime,
/// and because a test transport is a struct with a script and a log, not a
/// pair of `RefCell`s captured by closures.
pub trait ItemTransport {
    /// Send one item to the worker and return what it reported.
    fn send(
        &mut self,
        item: &TranslateBatchItem,
    ) -> impl Future<Output = Result<TranslationItemResultV2, ServerError>> + Send;

    /// Wait, or return the job's cancellation as `Err`.
    fn pause(&mut self, wait: Wait) -> impl Future<Output = Result<(), ServerError>> + Send;
}

/// Translate `items` one request at a time under `engine`'s provider policy.
pub async fn translate_items<Transport: ItemTransport>(
    items: &[TranslateBatchItem],
    engine: &TranslateEngineName,
    punct_refs: &[&str],
    transport: &mut Transport,
) -> Result<ItemsOutcome, ServerError> {
    let policy = engine.provider_policy();
    let mut throttle = policy.fresh_throttle();
    let mut results = Vec::with_capacity(items.len());
    let mut echoed = 0;
    // The wait owed before the next request, if any: the spacing after a
    // request that reached the provider, or the cooldown a verdict asked for.
    let mut pending: Option<Wait> = None;

    for (index, item) in items.iter().enumerate() {
        let settled = loop {
            if let Some(wait) = pending.take() {
                transport.pause(wait).await?;
            }
            let (provider_answer, error) = match transport.send(item).await? {
                TranslationItemResultV2::Translated {
                    raw_translation,
                    engine: reported,
                } => {
                    throttle = policy.fresh_throttle();
                    pending = policy.spacing().map(Wait::Spacing);
                    if raw_translation.trim() == item.text.trim() {
                        echoed += 1;
                    }
                    let postprocessed = postprocess_translation(&raw_translation, punct_refs);
                    break match TranslationText::admit(&postprocessed) {
                        Ok(text) => Ok(AdmittedTranslation::Translated {
                            text,
                            engine: reported,
                        }),
                        // The engine answered with nothing to write. This used
                        // to be skipped at injection, so the file was written
                        // with the `%xtra` tier missing and nothing said so.
                        Err(EmptyTranslation) => {
                            Err(ItemFailure::Command(TranslateFailure::EmptyTranslation {
                                engine: reported,
                            }))
                        }
                    };
                }
                // Blank text never reaches the provider, so it neither resets
                // the throttle nor owes the next request a gap.
                TranslationItemResultV2::BlankInput => break Ok(AdmittedTranslation::BlankInput),
                TranslationItemResultV2::ProviderStatus {
                    status,
                    retry_after_s,
                    error,
                } => (
                    ProviderAnswer::Status {
                        status,
                        // A Retry-After the type proves finite may still not
                        // fit a Duration; one that does not is longer than any
                        // ceiling, which is what `Duration::MAX` says.
                        retry_after: retry_after_s
                            .map(|s| Duration::try_from_secs_f64(s.get()).unwrap_or(Duration::MAX)),
                    },
                    error,
                ),
                TranslationItemResultV2::NoResponse { error } => {
                    (ProviderAnswer::NoResponse, error)
                }
                TranslationItemResultV2::Failed { error } => {
                    break Err(ItemFailure::EngineReported(error));
                }
            };
            match policy.verdict(provider_answer, throttle) {
                Verdict::WaitAndRetry {
                    cooldown,
                    throttle: next,
                } => {
                    warn!(
                        engine = %engine.as_wire_name(),
                        item = index,
                        answer = %provider_answer,
                        error = %error,
                        cooldown_s = cooldown.as_secs_f64(),
                        "translate provider answered something transient; waiting, then sending the item again"
                    );
                    pending = Some(Wait::Cooldown(cooldown));
                    throttle = next;
                }
                Verdict::GiveUp(give_up) => {
                    break Err(ItemFailure::Command(TranslateFailure::Provider {
                        engine: engine.clone(),
                        give_up,
                        error,
                    }));
                }
            }
        };

        match settled {
            Ok(admitted) => results.push(Ok(admitted)),
            Err(failure) => {
                // One bad utterance abandons the file, so the rest is not
                // sent: a persistent refusal is paid for once, not once per
                // remaining utterance.
                results.push(Err(failure));
                results.extend((index + 1..items.len()).map(|_| {
                    Err(ItemFailure::Command(TranslateFailure::NotAttempted {
                        after_item: index,
                    }))
                }));
                break;
            }
        }
    }

    Ok(ItemsOutcome { results, echoed })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::api::{NonNegativeSeconds, ReportedEngineName};
    use crate::types::worker_v2::HttpStatusCodeV2;

    fn item(text: &str) -> TranslateBatchItem {
        TranslateBatchItem {
            text: text.to_owned(),
        }
    }

    fn translated(text: &str) -> TranslationItemResultV2 {
        TranslationItemResultV2::Translated {
            raw_translation: text.to_owned(),
            engine: ReportedEngineName::try_from("googletrans-v1").expect("valid engine name"),
        }
    }

    fn status(code: u16, retry_after_s: Option<f64>) -> TranslationItemResultV2 {
        TranslationItemResultV2::ProviderStatus {
            status: HttpStatusCodeV2::known(code),
            retry_after_s: retry_after_s
                .map(|s| NonNegativeSeconds::try_from(s).expect("test cooldown")),
            error: format!("Unexpected status code \"{code}\""),
        }
    }

    /// A transport that answers from a script and records every item sent
    /// and every wait asked for, without a clock.
    struct Script {
        answers: VecDeque<TranslationItemResultV2>,
        sent: Vec<String>,
        waits: Vec<Wait>,
    }

    impl ItemTransport for Script {
        async fn send(
            &mut self,
            item: &TranslateBatchItem,
        ) -> Result<TranslationItemResultV2, ServerError> {
            self.sent.push(item.text.clone());
            Ok(self
                .answers
                .pop_front()
                .expect("the script has an answer for every request"))
        }

        async fn pause(&mut self, wait: Wait) -> Result<(), ServerError> {
            self.waits.push(wait);
            Ok(())
        }
    }

    async fn run(
        items: &[&str],
        engine: TranslateEngineName,
        answers: Vec<TranslationItemResultV2>,
    ) -> (ItemsOutcome, Script) {
        let items: Vec<TranslateBatchItem> = items.iter().map(|text| item(text)).collect();
        let punct = batchalign_transform::translate::chat_punct_chars();
        let punct_refs: Vec<&str> = punct.iter().map(String::as_str).collect();
        let mut script = Script {
            answers: VecDeque::from(answers),
            sent: Vec::new(),
            waits: Vec::new(),
        };
        let outcome = translate_items(&items, &engine, &punct_refs, &mut script)
            .await
            .expect("no cancellation in these scripts");
        (outcome, script)
    }

    fn text_of(result: &Result<AdmittedTranslation, TranslateItemFailure>) -> Option<&str> {
        match result {
            Ok(AdmittedTranslation::Translated { text, .. }) => Some(text.as_str()),
            Ok(AdmittedTranslation::BlankInput) | Err(_) => None,
        }
    }

    /// Guards: Google translation preserves ordered sends and places only one spacing wait between successful items.
    /// Kind: contract
    /// Not a type: A typed provider policy can still be applied one step late or after the last item.
    /// State: A scripted transport gives two item replies and records send and wait effects.
    /// Concurrency: One awaited deterministic script, with no wall-clock waits.
    #[tokio::test]
    async fn items_are_sent_one_at_a_time_with_the_engine_spacing_between_them() {
        let (outcome, script) = run(
            &["hola.", "adios."],
            TranslateEngineName::Google,
            vec![translated("Hello."), translated("Goodbye.")],
        )
        .await;
        assert_eq!(script.sent, vec!["hola.", "adios."]);
        assert_eq!(
            script.waits,
            vec![Wait::Spacing(Duration::from_millis(1500))]
        );
        assert_eq!(text_of(&outcome.results[0]), Some("Hello ."));
        assert_eq!(text_of(&outcome.results[1]), Some("Goodbye ."));
        assert_eq!(outcome.echoed, 0);
    }

    /// Guards: Local translations send all three items without requesting provider pacing.
    /// Kind: contract
    /// Not a type: An engine's local spacing policy must reach the actual translation scheduling seam.
    /// State: A local-engine transport double records three sends and all waits.
    /// Concurrency: One awaited deterministic script, with no worker or timers.
    #[tokio::test]
    async fn a_local_engine_keeps_no_gap() {
        let (_, script) = run(
            &["a", "b", "c"],
            TranslateEngineName::Nllb,
            vec![translated("x"), translated("y"), translated("z")],
        )
        .await;
        assert_eq!(script.sent.len(), 3);
        assert!(script.waits.is_empty(), "{:?}", script.waits);
    }

    /// Guards: Transient HTTP status and no-response receipts both wait once and re-send only the original item.
    /// Kind: contract
    /// Not a type: The two independently typed provider observations can still schedule a fresh item or an extra delay.
    /// State: A scripted transport captures the repeated input and admitted cooldown for each response variant.
    /// Concurrency: Every script owns its sequence and uses no clock.
    #[tokio::test]
    async fn a_transient_answer_waits_and_sends_the_same_item_again() {
        for transient in [
            status(429, Some(2.0)),
            TranslationItemResultV2::NoResponse {
                error: "ConnectionResetError".into(),
            },
        ] {
            let (outcome, script) = run(
                &["hola."],
                TranslateEngineName::Google,
                vec![transient, translated("Hello.")],
            )
            .await;
            assert_eq!(script.sent, vec!["hola.", "hola."]);
            assert_eq!(script.waits, vec![Wait::Cooldown(Duration::from_secs(5))]);
            assert_eq!(text_of(&outcome.results[0]), Some("Hello ."));
        }
    }

    /// Guards: Exhausted cooldowns stop dispatch at the first item and mark each later item by that original index.
    /// Kind: contract
    /// Not a type: Server replies can spend all allowed retries while additional items remain queued.
    /// State: A scripted sequence of three transient status responses records sends and waits.
    /// Concurrency: Awaited scripted transport and no real provider.
    #[tokio::test]
    async fn a_spent_schedule_fails_the_item_and_the_rest_are_not_sent() {
        let (outcome, script) = run(
            &["uno.", "dos.", "tres."],
            TranslateEngineName::Google,
            vec![status(429, None), status(429, None), status(429, None)],
        )
        .await;
        assert_eq!(
            script.sent,
            vec!["uno.", "uno.", "uno."],
            "only the first item is ever sent"
        );
        assert_eq!(
            script.waits,
            vec![
                Wait::Cooldown(Duration::from_secs(5)),
                Wait::Cooldown(Duration::from_secs(15))
            ]
        );
        let first = outcome.results[0]
            .as_ref()
            .expect_err("the throttled item fails");
        assert!(
            first.to_string().contains("again after 2 retries"),
            "the failure says the schedule was spent, got: {first}"
        );
        assert!(first.to_string().contains("--translate-engine"), "{first}");
        for (index, later) in outcome.results.iter().enumerate().skip(1) {
            assert_eq!(
                later.as_ref().expect_err("not attempted"),
                &ItemFailure::Command(TranslateFailure::NotAttempted { after_item: 0 }),
                "item {index}"
            );
        }
    }

    /// Guards: A final provider status aborts the file without waiting and preserves the original diagnostic.
    /// Kind: contract
    /// Not a type: A status return and diagnostic come from the provider independently of its scheduling policy.
    /// State: A scripted final HTTP response with exact text observes sent items and waits.
    /// Concurrency: Awaited scripted transport and no real provider.
    #[tokio::test]
    async fn a_final_status_stops_the_batch_without_a_wait() {
        let (outcome, script) = run(
            &["uno.", "dos."],
            TranslateEngineName::Google,
            vec![status(403, None)],
        )
        .await;
        assert_eq!(script.sent, vec!["uno."]);
        assert!(script.waits.is_empty(), "{:?}", script.waits);
        let first = outcome.results[0].as_ref().expect_err("refused");
        assert!(first.to_string().contains("HTTP 403"), "{first}");
        assert!(
            first.to_string().contains("Unexpected status code"),
            "the provider library's own message travels: {first}"
        );
        assert!(matches!(
            outcome.results[1],
            Err(ItemFailure::Command(TranslateFailure::NotAttempted {
                after_item: 0
            }))
        ));
    }

    /// Guards: A completed item hands the following item a fresh retry schedule.
    /// Kind: contract
    /// Not a type: Adjacent valid worker responses can exhaust one item's schedule without exhausting the next.
    /// State: A two-item scripted transport provides two long retry sequences and records cooldown and spacing.
    /// Concurrency: One awaited script and no real clock.
    #[tokio::test]
    async fn a_success_hands_the_next_item_a_fresh_schedule() {
        let (outcome, script) = run(
            &["uno.", "dos."],
            TranslateEngineName::Google,
            vec![
                status(429, None),
                status(429, None),
                translated("One."),
                status(429, None),
                translated("Two."),
            ],
        )
        .await;
        assert_eq!(script.sent, vec!["uno.", "uno.", "uno.", "dos.", "dos."]);
        assert_eq!(
            script.waits,
            vec![
                Wait::Cooldown(Duration::from_secs(5)),
                Wait::Cooldown(Duration::from_secs(15)),
                Wait::Spacing(Duration::from_millis(1500)),
                Wait::Cooldown(Duration::from_secs(5)),
            ]
        );
        assert_eq!(text_of(&outcome.results[1]), Some("Two ."));
    }

    /// Guards: A terminal worker-supplied item message survives verbatim and stops later sends.
    /// Kind: wire
    /// Not a type: The worker can return a whole message independently of the operation's typed disposition.
    /// State: A scripted local-engine failure and two text items witness the absence of another send.
    /// Concurrency: Private scripted transport, no live inference.
    #[tokio::test]
    async fn an_engine_reported_failure_is_final_and_verbatim() {
        let (outcome, script) = run(
            &["uno.", "dos."],
            TranslateEngineName::Nllb,
            vec![TranslationItemResultV2::Failed {
                error: "Translation failed: unmapped source language".into(),
            }],
        )
        .await;
        assert_eq!(script.sent, vec!["uno."]);
        assert!(
            matches!(
                &outcome.results[0],
                Err(ItemFailure::EngineReported(error))
                    if error == "Translation failed: unmapped source language"
            ),
            "{:?}",
            outcome.results[0]
        );
    }

    /// Guards: A provider reply that normalizes to no words stops the batch at that item.
    /// Kind: contract
    /// Not a type: A syntactically successful response may contain punctuation alone and fail the usable-text policy.
    /// State: A scripted punctuation-only reply and second queued item observe the stopped sequence.
    /// Concurrency: Pure scripted attempts and no real provider.
    #[tokio::test]
    async fn an_empty_translation_fails_the_item_and_stops_the_batch() {
        let (outcome, script) = run(
            &["uno.", "dos."],
            TranslateEngineName::Google,
            vec![translated(".")],
        )
        .await;
        assert_eq!(script.sent, vec!["uno."]);
        let first = outcome.results[0]
            .as_ref()
            .expect_err("an empty translation is refused");
        assert!(first.to_string().contains("empty translation"), "{first}");
    }

    /// Guards: Source-identical translations are counted separately from genuinely changed text.
    /// Kind: policy
    /// Not a type: The worker supplies actual strings that can equal the source text after the chosen normalization.
    /// State: Three scripted outputs exercise the result counter.
    /// Concurrency: No real clock or watcher.
    #[tokio::test]
    async fn translations_identical_to_their_source_are_counted() {
        let (outcome, _) = run(
            &["OK.", "mama.", "hola."],
            TranslateEngineName::Google,
            vec![translated("OK."), translated("mama."), translated("Hello.")],
        )
        .await;
        assert_eq!(outcome.echoed, 2);
    }

    /// A transport whose every wait is the job's cancellation.
    struct Cancelled;

    impl ItemTransport for Cancelled {
        async fn send(
            &mut self,
            _item: &TranslateBatchItem,
        ) -> Result<TranslationItemResultV2, ServerError> {
            Ok(status(429, None))
        }

        async fn pause(&mut self, _wait: Wait) -> Result<(), ServerError> {
            Err(ServerError::Cancelled)
        }
    }

    /// Guards: A cancelled transport wait ends translation with the original cancellation.
    /// Kind: contract
    /// Not a type: An awaited pacing or cooldown effect can independently report job cancellation before the next attempt.
    /// State: A cancellation-returning double implements the actual transport interface.
    /// Concurrency: One local async invocation with no wall-clock wait.
    #[tokio::test]
    async fn a_cancelled_pause_stops_the_loop_with_the_cancellation() {
        let items = [item("hola.")];
        let outcome =
            translate_items(&items, &TranslateEngineName::Google, &[], &mut Cancelled).await;
        assert!(
            matches!(outcome, Err(ServerError::Cancelled)),
            "{outcome:?}"
        );
    }
}
