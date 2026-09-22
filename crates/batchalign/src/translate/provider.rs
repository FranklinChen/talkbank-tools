//! How a translate engine's provider is approached: the gap kept between
//! consecutive requests, and which of its answers are waited out.
//!
//! The policy is data on [`TranslateEngineName`], read by the per-item loop
//! in [`super::items`]. It lives on the control plane rather than in the
//! worker because a wait spent inside a worker request is invisible to the
//! job: it counts against the request's transport deadline, it cannot be
//! cancelled, and the worker cannot know whether a later item is worth
//! sending. Here every wait is a `tokio` sleep raced against the job's
//! cancellation, and a refusal stops the file before the next utterance is
//! paid for.
//!
//! The throttle is one value per file, not per item: a provider throttles
//! the client, so once the schedule is spent the next item does not re-spend
//! it, and a success hands the next item a fresh schedule.

use std::time::Duration;

use crate::types::engines::TranslateEngineName;
use crate::types::worker_v2::HttpStatusCodeV2;

/// What a provider answered instead of a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderAnswer {
    /// An HTTP status, with the cooldown the provider asked for, if any.
    Status {
        /// The status the provider answered.
        status: HttpStatusCodeV2,
        /// The provider's own `Retry-After`, when it named one. A value too
        /// large for a `Duration` arrives as `Duration::MAX`, which every
        /// ceiling refuses.
        retry_after: Option<Duration>,
    },
    /// No reply at all: the request failed in transport before any response.
    NoResponse,
}

impl std::fmt::Display for ProviderAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Status {
                status,
                retry_after,
            } => {
                write!(f, "HTTP {status}")?;
                if let Some(cooldown) = retry_after {
                    write!(f, " (Retry-After {} s)", cooldown.as_secs_f64())?;
                }
                Ok(())
            }
            Self::NoResponse => {
                f.write_str("no HTTP response (a transport failure before any reply)")
            }
        }
    }
}

/// Which answers an HTTP provider is given time to recover from, and how much.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransientPolicy {
    /// Statuses worth waiting out. A transport failure with no response is
    /// always among the transient answers.
    statuses: &'static [HttpStatusCodeV2],
    /// The cooldowns spent in order before giving up; its length is the retry
    /// budget. A `Retry-After` longer than the scheduled cooldown replaces it.
    schedule: &'static [Duration],
    /// A `Retry-After` above this is refused rather than slept: the operator
    /// reads the cooldown instead of watching a job that appears hung.
    ceiling: Duration,
}

/// How to approach one engine's provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderPolicy {
    /// The gap kept between consecutive requests. `None` for a model in the
    /// worker process, which has no rate limit to respect.
    spacing: Option<Duration>,
    /// The answers waited out. `None` for an engine whose every failure is
    /// final: a local model, or a provider whose SDK reports no status.
    transient: Option<TransientPolicy>,
}

impl ProviderPolicy {
    /// The gap kept between consecutive requests, if any.
    pub(crate) fn spacing(&self) -> Option<Duration> {
        self.spacing
    }

    /// A throttle with the whole schedule still available.
    pub(crate) fn fresh_throttle(&self) -> Throttle {
        Throttle {
            remaining: self.transient.map_or(&[], |transient| transient.schedule),
        }
    }

    /// What to do about an answer that was not a translation.
    pub(crate) fn verdict(&self, answer: ProviderAnswer, throttle: Throttle) -> Verdict {
        let give_up = |why| Verdict::GiveUp(ProviderGiveUp { answer, why });
        let Some(transient) = self.transient else {
            return give_up(GiveUpReason::Final);
        };
        let (transient_answer, retry_after) = match answer {
            ProviderAnswer::Status {
                status,
                retry_after,
            } => (transient.statuses.contains(&status), retry_after),
            ProviderAnswer::NoResponse => (true, None),
        };
        if !transient_answer {
            return give_up(GiveUpReason::Final);
        }
        let [scheduled, rest @ ..] = throttle.remaining else {
            return give_up(GiveUpReason::Exhausted {
                retries: transient.schedule.len(),
            });
        };
        let cooldown = retry_after.map_or(*scheduled, |asked| asked.max(*scheduled));
        if cooldown > transient.ceiling {
            return give_up(GiveUpReason::CooldownTooLong {
                requested: cooldown,
                ceiling: transient.ceiling,
            });
        }
        Verdict::WaitAndRetry {
            cooldown,
            throttle: Throttle { remaining: rest },
        }
    }
}

/// The cooldowns still available to a file.
///
/// Minted by [`ProviderPolicy::fresh_throttle`], advanced only by a
/// [`Verdict::WaitAndRetry`], and replaced by a fresh one on a success, so the
/// number of retries a file has spent is never a counter kept beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Throttle {
    remaining: &'static [Duration],
}

/// The policy's answer to a provider answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Sleep `cooldown`, then send the same item again with `throttle`.
    WaitAndRetry {
        /// How long to wait before the next attempt.
        cooldown: Duration,
        /// The schedule left after this cooldown.
        throttle: Throttle,
    },
    /// Stop: this item is not going to be translated.
    GiveUp(ProviderGiveUp),
}

/// Why an item was given up on: the last answer, and what the policy made of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderGiveUp {
    /// The answer the policy stopped on.
    pub(crate) answer: ProviderAnswer,
    /// Why it stopped.
    pub(crate) why: GiveUpReason,
}

/// What the policy made of the last answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GiveUpReason {
    /// Not an answer this engine recovers from.
    Final,
    /// Transient, but every scheduled cooldown has been spent.
    Exhausted {
        /// How many cooldowns were spent.
        retries: usize,
    },
    /// The provider asked for a longer cooldown than the ceiling allows.
    CooldownTooLong {
        /// The cooldown it asked for.
        requested: Duration,
        /// The most this policy will sleep.
        ceiling: Duration,
    },
}

impl std::fmt::Display for ProviderGiveUp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "answered {}", self.answer)?;
        match self.why {
            GiveUpReason::Final => Ok(()),
            GiveUpReason::Exhausted { retries } => write!(f, " again after {retries} retries"),
            GiveUpReason::CooldownTooLong { requested, ceiling } => write!(
                f,
                " and asked for a {} s cooldown, above the {} s ceiling",
                requested.as_secs_f64(),
                ceiling.as_secs_f64()
            ),
        }
    }
}

/// Google's public endpoint throttles by client: 429 on a burst, 502 to 504
/// when the edge is overloaded. Two cooldowns, then the file is refused.
const GOOGLE_TRANSIENT: TransientPolicy = TransientPolicy {
    statuses: &[
        HttpStatusCodeV2::known(429),
        HttpStatusCodeV2::known(502),
        HttpStatusCodeV2::known(503),
        HttpStatusCodeV2::known(504),
    ],
    schedule: &[Duration::from_secs(5), Duration::from_secs(15)],
    ceiling: Duration::from_secs(60),
};

impl TranslateEngineName {
    /// How this engine's provider is approached.
    pub(crate) fn provider_policy(&self) -> ProviderPolicy {
        match self {
            // The 1.5 s gap is the rate the endpoint tolerates from one client
            // without answering 429; it used to be a sleep inside the worker.
            Self::Google => ProviderPolicy {
                spacing: Some(Duration::from_millis(1500)),
                transient: Some(GOOGLE_TRANSIENT),
            },
            // Tencent TMT's free tier allows 5 QPS on `TextTranslate`; a tight
            // loop answers `RequestLimitExceeded`. The SDK raises without an
            // HTTP status the worker can report, so nothing is waited out.
            Self::Tencent => ProviderPolicy {
                spacing: Some(Duration::from_millis(200)),
                transient: None,
            },
            // Aliyun MT has shown no rate limit at corpus volumes; its SDK
            // likewise raises without a reportable status.
            Self::Aliyun => ProviderPolicy {
                spacing: None,
                transient: None,
            },
            // In-process models: no provider, nothing to space or wait out.
            Self::Seamless | Self::Nllb => ProviderPolicy {
                spacing: None,
                transient: None,
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn status(code: u16, retry_after: Option<Duration>) -> ProviderAnswer {
        ProviderAnswer::Status {
            status: HttpStatusCodeV2::known(code),
            retry_after,
        }
    }

    fn google() -> ProviderPolicy {
        TranslateEngineName::Google.provider_policy()
    }

    fn cooldown_of(verdict: Verdict) -> Duration {
        match verdict {
            Verdict::WaitAndRetry { cooldown, .. } => cooldown,
            Verdict::GiveUp(give_up) => panic!("expected a wait, got: {give_up}"),
        }
    }

    #[test]
    fn a_rate_limit_waits_the_scheduled_cooldown_first() {
        let policy = google();
        assert_eq!(
            cooldown_of(policy.verdict(status(429, None), policy.fresh_throttle())),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn a_longer_retry_after_replaces_the_schedule_and_a_shorter_one_does_not() {
        let policy = google();
        assert_eq!(
            cooldown_of(policy.verdict(
                status(503, Some(Duration::from_secs(12))),
                policy.fresh_throttle()
            )),
            Duration::from_secs(12)
        );
        assert_eq!(
            cooldown_of(policy.verdict(
                status(503, Some(Duration::from_secs(1))),
                policy.fresh_throttle()
            )),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn the_second_cooldown_is_longer_and_the_third_answer_is_refused() {
        let policy = google();
        let Verdict::WaitAndRetry { throttle, .. } =
            policy.verdict(status(429, None), policy.fresh_throttle())
        else {
            panic!("first 429 is waited out");
        };
        let Verdict::WaitAndRetry { cooldown, throttle } =
            policy.verdict(status(429, None), throttle)
        else {
            panic!("second 429 is waited out");
        };
        assert_eq!(cooldown, Duration::from_secs(15));
        assert_eq!(
            policy.verdict(status(429, None), throttle),
            Verdict::GiveUp(ProviderGiveUp {
                answer: status(429, None),
                why: GiveUpReason::Exhausted { retries: 2 },
            })
        );
    }

    #[test]
    fn a_cooldown_above_the_ceiling_is_refused_rather_than_slept() {
        let policy = google();
        for asked in [Duration::from_secs(61), Duration::MAX] {
            assert_eq!(
                policy.verdict(status(429, Some(asked)), policy.fresh_throttle()),
                Verdict::GiveUp(ProviderGiveUp {
                    answer: status(429, Some(asked)),
                    why: GiveUpReason::CooldownTooLong {
                        requested: asked,
                        ceiling: Duration::from_secs(60),
                    },
                })
            );
        }
    }

    #[test]
    fn a_final_status_is_refused_at_once() {
        let policy = google();
        for code in [400, 403, 404, 500] {
            assert_eq!(
                policy.verdict(status(code, None), policy.fresh_throttle()),
                Verdict::GiveUp(ProviderGiveUp {
                    answer: status(code, None),
                    why: GiveUpReason::Final,
                }),
                "{code}"
            );
        }
    }

    #[test]
    fn no_response_is_waited_out_on_the_same_schedule() {
        let policy = google();
        assert_eq!(
            cooldown_of(policy.verdict(ProviderAnswer::NoResponse, policy.fresh_throttle())),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn an_engine_without_a_transient_policy_refuses_every_answer_and_keeps_its_spacing() {
        for engine in [
            TranslateEngineName::Tencent,
            TranslateEngineName::Aliyun,
            TranslateEngineName::Nllb,
            TranslateEngineName::Seamless,
        ] {
            let policy = engine.provider_policy();
            for answer in [status(429, None), ProviderAnswer::NoResponse] {
                assert_eq!(
                    policy.verdict(answer, policy.fresh_throttle()),
                    Verdict::GiveUp(ProviderGiveUp {
                        answer,
                        why: GiveUpReason::Final,
                    }),
                    "{engine:?} {answer}"
                );
            }
        }
        assert_eq!(
            TranslateEngineName::Tencent.provider_policy().spacing(),
            Some(Duration::from_millis(200))
        );
        assert_eq!(TranslateEngineName::Nllb.provider_policy().spacing(), None);
    }

    /// The operator reads these in the file's failure; the book quotes them.
    #[test]
    fn give_up_reasons_read_as_the_operator_will_see_them() {
        assert_eq!(
            ProviderGiveUp {
                answer: status(403, None),
                why: GiveUpReason::Final
            }
            .to_string(),
            "answered HTTP 403"
        );
        assert_eq!(
            ProviderGiveUp {
                answer: status(429, Some(Duration::from_secs(7))),
                why: GiveUpReason::Exhausted { retries: 2 }
            }
            .to_string(),
            "answered HTTP 429 (Retry-After 7 s) again after 2 retries"
        );
    }
}
