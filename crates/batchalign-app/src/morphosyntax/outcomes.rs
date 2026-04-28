//! Aggregation of per-language-group morphosyntax batch outcomes.
//!
//! Owns the pure logic that decides what happens when one (or several)
//! language groups fail mid-batch. A failed language group means some
//! utterances received no UD response — their %mor was cleared before
//! dispatch and will not be re-populated. Writing the file in that
//! state is silent corruption: `clear_morphosyntax` resets %mor to
//! empty, `inject_results` never fires for bailouts, and
//! `remove_empty_morphosyntax_placeholders` then strips the empty
//! placeholders on the way to disk. The file lands on disk with
//! authoritative-looking CHAT output that is missing %mor/%gra on
//! every utterance in the failed language.
//!
//! This module exists as a separate seam so the corruption boundary
//! is unit-testable without a live worker pool. The caller (`batch.rs`)
//! converts worker-pool outcomes into [`LanguageGroupOutcome`] values
//! and calls [`aggregate_language_group_outcomes`]. A failure returns
//! a typed [`LanguageGroupFailure`] that the orchestrator must surface
//! as per-file errors rather than silently substituting empty UD
//! responses.

use batchalign_chat_ops::nlp::UdResponse;

use crate::error::ServerError;

/// Per-language-group dispatch outcome.
///
/// The caller constructs one of these per language group after
/// `infer_batch` returns; [`aggregate_language_group_outcomes`]
/// decides whether the cross-group batch is recoverable.
#[derive(Debug)]
pub(crate) struct LanguageGroupOutcome {
    /// 3-letter ISO language code for this group (`"eng"`, `"deu"`, ...).
    pub lang3: String,
    /// Miss-table indices (into the flat `all_misses` vector) for the
    /// utterances this group was responsible for.
    pub global_indices: Vec<usize>,
    /// Success: one UD response per global index, in order. Failure:
    /// the worker-pool error that bailed the group out (deadlock
    /// prevention, timeout, crash, ...).
    pub result: Result<Vec<UdResponse>, ServerError>,
}

/// One failed language group surfaced as part of [`LanguageGroupFailure`].
#[derive(Debug, Clone)]
#[allow(dead_code)] // `error_message` and `lang3` inspected by tests + future telemetry.
pub(crate) struct FailedLanguageGroup {
    /// 3-letter ISO language code of the group that failed.
    pub lang3: String,
    /// Global miss indices that would have received empty %mor.
    pub global_indices: Vec<usize>,
    /// Human-readable description of the worker-pool failure.
    pub error_message: String,
}

/// Typed failure describing which language groups bailed out.
///
/// The orchestrator must propagate this as a typed error so every
/// file contributing to a failed group is marked failed. Silent
/// substitution of empty UD responses is banned — it produced
/// thousands of corrupted CHAT files during the 2026-04-14 resume
/// run (see `docs/postmortems/`).
#[derive(Debug, thiserror::Error)]
#[error(
    "{num_failed} language group(s) failed morphosyntax dispatch: {languages}; \
     files contributing utterances to these groups must be marked failed \
     because their %mor/%gra tiers would otherwise ship empty"
)]
pub(crate) struct LanguageGroupFailure {
    /// Number of failed groups. Redundant with `failed.len()` but
    /// kept on the struct so the `Display` impl can format it
    /// without allocating.
    pub num_failed: usize,
    /// Comma-separated language list, pre-rendered for the `Display`
    /// impl.
    pub languages: String,
    /// Detail per failed group.
    pub failed: Vec<FailedLanguageGroup>,
}

impl LanguageGroupFailure {
    /// 3-letter ISO codes of every failed language, in iteration order.
    #[allow(dead_code)] // inspected by tests + future telemetry hooks.
    pub fn failed_languages(&self) -> Vec<&str> {
        self.failed.iter().map(|f| f.lang3.as_str()).collect()
    }

    /// Global miss indices that would have been written with empty
    /// UD — these identify which files need per-file error marking.
    pub fn affected_global_indices(&self) -> std::collections::BTreeSet<usize> {
        self.failed
            .iter()
            .flat_map(|f| f.global_indices.iter().copied())
            .collect()
    }
}

/// Aggregated outcome of every language group in one batch.
///
/// Carries both the partial response vector (so files whose languages
/// all succeeded can still be injected) *and* the typed failure (so
/// the caller can mark files contributing to a failed group as
/// per-file errors). The caller must consult `failure` before
/// injecting: slots corresponding to `failure.affected_global_indices()`
/// will be empty `UdResponse` and injecting them directly would
/// silently strip %mor/%gra — the 2026-04-14 corruption regression.
#[derive(Debug)]
pub(crate) struct AggregatedOutcomes {
    /// Responses indexed by global miss index. Slots belonging to
    /// failed groups (or skipped unsupported languages) are left as
    /// an empty `UdResponse`.
    pub responses: Vec<UdResponse>,
    /// `None` when every language group succeeded. `Some` when any
    /// group returned `Err`; the caller must mark every file that
    /// contributed to the failed groups as a per-file error.
    pub failure: Option<LanguageGroupFailure>,
}

/// What the orchestrator should do with one file after language-group
/// outcomes have been aggregated.
///
/// The corruption regression this guards: when a language group fails
/// dispatch, the `clear_morphosyntax` step has already reset every
/// utterance's `%mor`/`%gra` in place; injecting against the failed
/// group's empty `UdResponse` would leave those tiers unpopulated, and
/// the final empty-placeholder sweep would strip them. Files with any
/// utterance in a failed group MUST be marked as per-file errors and
/// skipped by the injection loop, never serialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FileInjectionDecision {
    /// No utterance in this file belongs to a failed group. Inject
    /// normally; serialize; record success.
    Inject,
    /// One or more utterances in this file belong to a failed group.
    /// Record a per-file error naming the failed languages; do NOT
    /// call `inject_results`; do NOT serialize.
    SkipFailed {
        /// Formatted language list for the per-file error message.
        /// Matches `LanguageGroupFailure::languages`.
        failed_languages: String,
    },
}

/// Decide whether a file should be injected or marked failed given a
/// batch's aggregated outcomes.
///
/// `global_start..global_start+count` is the file's contiguous range in
/// the flat `all_misses` vector. A file contributes utterances to a
/// failed language group iff that range intersects
/// `failure.affected_global_indices()`.
///
/// When `failure` is `None` (every group succeeded) the decision is
/// always `Inject`. When `failure` is `Some` but the file's range is
/// disjoint from the affected indices, the decision is also `Inject` —
/// only files that actually contribute utterances to a failed group
/// get marked, so a batch of 100 English-only files next to one Dutch
/// file does not fail all 101.
pub(super) fn classify_file_for_injection(
    global_start: usize,
    item_count: usize,
    failure: Option<&LanguageGroupFailure>,
) -> FileInjectionDecision {
    let Some(failure) = failure else {
        return FileInjectionDecision::Inject;
    };
    let affected = failure.affected_global_indices();
    let file_intersects = (global_start..global_start + item_count).any(|i| affected.contains(&i));
    if file_intersects {
        FileInjectionDecision::SkipFailed {
            failed_languages: failure.languages.clone(),
        }
    } else {
        FileInjectionDecision::Inject
    }
}

/// Aggregate per-language-group outcomes into a flat response vector
/// plus an optional typed failure.
pub(crate) fn aggregate_language_group_outcomes(
    outcomes: Vec<LanguageGroupOutcome>,
    total_miss_count: usize,
) -> AggregatedOutcomes {
    let mut all_responses: Vec<Option<UdResponse>> = (0..total_miss_count).map(|_| None).collect();
    let mut failed: Vec<FailedLanguageGroup> = Vec::new();

    for outcome in outcomes {
        match outcome.result {
            Ok(responses) => {
                for (global_idx, ud) in outcome.global_indices.into_iter().zip(responses) {
                    if let Some(slot) = all_responses.get_mut(global_idx) {
                        *slot = Some(ud);
                    }
                }
            }
            Err(e) => {
                failed.push(FailedLanguageGroup {
                    lang3: outcome.lang3,
                    global_indices: outcome.global_indices,
                    error_message: e.to_string(),
                });
            }
        }
    }

    let failure = if failed.is_empty() {
        None
    } else {
        let languages = failed
            .iter()
            .map(|f| f.lang3.clone())
            .collect::<Vec<_>>()
            .join(", ");
        Some(LanguageGroupFailure {
            num_failed: failed.len(),
            languages,
            failed,
        })
    };

    AggregatedOutcomes {
        responses: all_responses
            .into_iter()
            .map(|slot| {
                slot.unwrap_or(UdResponse {
                    sentences: Vec::new(),
                })
            })
            .collect(),
        failure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_chat_ops::nlp::UdSentence;

    fn mk_response(n: usize) -> Vec<UdResponse> {
        (0..n)
            .map(|_| UdResponse {
                sentences: vec![UdSentence { words: Vec::new() }],
            })
            .collect()
    }

    /// Baseline: every group succeeds → full response vector, no failure.
    #[test]
    fn all_groups_succeed_returns_responses_in_global_order_with_no_failure() {
        let outcomes = vec![
            LanguageGroupOutcome {
                lang3: "eng".into(),
                global_indices: vec![0, 2],
                result: Ok(mk_response(2)),
            },
            LanguageGroupOutcome {
                lang3: "deu".into(),
                global_indices: vec![1, 3],
                result: Ok(mk_response(2)),
            },
        ];
        let agg = aggregate_language_group_outcomes(outcomes, 4);
        assert!(agg.failure.is_none(), "no group failed");
        assert_eq!(agg.responses.len(), 4);
        for r in &agg.responses {
            assert_eq!(r.sentences.len(), 1, "every slot should be filled");
        }
    }

    /// Corruption regression guard: a failed language group must surface
    /// as a typed `failure`, not silently substitute empty UdResponses
    /// that the injection path would then strip. This is the
    /// 2026-04-14 corruption pattern.
    #[test]
    fn failed_language_group_surfaces_typed_failure_with_affected_indices() {
        let dispatched_err =
            ServerError::Validation("worker cap reached and no workers exist for deu".into());
        let outcomes = vec![
            LanguageGroupOutcome {
                lang3: "eng".into(),
                global_indices: vec![0, 2],
                result: Ok(mk_response(2)),
            },
            LanguageGroupOutcome {
                lang3: "deu".into(),
                global_indices: vec![1, 3],
                result: Err(dispatched_err),
            },
        ];

        let agg = aggregate_language_group_outcomes(outcomes, 4);

        let failure = agg.failure.expect(
            "aggregation must surface failure when any language group failed — silent \
             empty fill caused the 2026-04-14 corruption regression",
        );
        assert_eq!(failure.num_failed, 1);
        assert_eq!(failure.failed_languages(), vec!["deu"]);
        let affected: Vec<usize> = failure.affected_global_indices().into_iter().collect();
        assert_eq!(affected, vec![1usize, 3]);
        assert!(
            failure.to_string().contains("deu"),
            "error message must name the failed language; got: {failure}"
        );

        // Partial responses still have the successful eng slots populated
        // so the caller can inject succeeding files.
        assert_eq!(
            agg.responses[0].sentences.len(),
            1,
            "successful eng slot 0 still carries its response"
        );
        assert_eq!(
            agg.responses[2].sentences.len(),
            1,
            "successful eng slot 2 still carries its response"
        );
        assert!(
            agg.responses[1].sentences.is_empty(),
            "failed deu slot 1 must be empty so a caller skipping the failure check \
             still does not inject garbage"
        );
    }

    /// Multiple failed groups: the failure must enumerate every one of them.
    #[test]
    fn multiple_failed_language_groups_all_appear_in_failure() {
        let outcomes = vec![
            LanguageGroupOutcome {
                lang3: "eng".into(),
                global_indices: vec![0],
                result: Ok(mk_response(1)),
            },
            LanguageGroupOutcome {
                lang3: "deu".into(),
                global_indices: vec![1],
                result: Err(ServerError::Validation("deadlock".into())),
            },
            LanguageGroupOutcome {
                lang3: "nld".into(),
                global_indices: vec![2],
                result: Err(ServerError::Validation("deadlock".into())),
            },
        ];
        let agg = aggregate_language_group_outcomes(outcomes, 3);
        let failure = agg.failure.expect("two groups failed");
        assert_eq!(failure.num_failed, 2);
        let langs = failure.failed_languages();
        assert!(langs.contains(&"deu"));
        assert!(langs.contains(&"nld"));
        assert!(!langs.contains(&"eng"));
    }

    /// Empty outcomes (no language groups dispatched at all) returns
    /// an empty response vector with no failure.
    #[test]
    fn empty_outcomes_returns_empty_vec_on_total_zero() {
        let agg = aggregate_language_group_outcomes(Vec::new(), 0);
        assert!(agg.failure.is_none());
        assert!(agg.responses.is_empty());
    }

    /// Gaps between dispatched indices (e.g., skipped unsupported
    /// languages) get filled with an empty UdResponse — that path is
    /// intentionally legal because the caller has already decided the
    /// utterance is unprocessable.
    #[test]
    fn gaps_left_by_skipped_languages_fill_with_empty_response() {
        let outcomes = vec![LanguageGroupOutcome {
            lang3: "eng".into(),
            global_indices: vec![0, 2],
            result: Ok(mk_response(2)),
        }];
        let agg = aggregate_language_group_outcomes(outcomes, 3);
        assert!(agg.failure.is_none());
        assert_eq!(agg.responses.len(), 3);
        assert_eq!(agg.responses[0].sentences.len(), 1, "eng slot 0 filled");
        assert!(
            agg.responses[1].sentences.is_empty(),
            "gap at 1 stays empty"
        );
        assert_eq!(agg.responses[2].sentences.len(), 1, "eng slot 2 filled");
    }

    // ────────────────────────────────────────────────────────────────
    // File-level injection-decision tests.
    //
    // These are the end-to-end corruption-regression guards for the
    // pre-fix silent-corruption pattern: a language group fails, the
    // orchestrator fills empty UdResponses, `inject_results` runs
    // against empty data, the empty-placeholder sweep strips the tier,
    // and a file ships with `rc=0` but missing `%mor`/`%gra`.
    //
    // The old orchestrator had no equivalent of these tests; adding
    // them is the "so it doesn't happen again" answer. Any future
    // change that regresses the per-file error-marking loop will
    // produce `Inject` where `SkipFailed` is required, and these
    // tests will turn RED.
    // ────────────────────────────────────────────────────────────────

    fn mk_failure(lang: &str, indices: Vec<usize>) -> LanguageGroupFailure {
        LanguageGroupFailure {
            num_failed: 1,
            languages: lang.to_string(),
            failed: vec![FailedLanguageGroup {
                lang3: lang.to_string(),
                global_indices: indices,
                error_message: format!("simulated failure for {lang}"),
            }],
        }
    }

    #[test]
    fn classify_with_no_failure_always_injects() {
        assert_eq!(
            classify_file_for_injection(0, 10, None),
            FileInjectionDecision::Inject
        );
        assert_eq!(
            classify_file_for_injection(100, 0, None),
            FileInjectionDecision::Inject
        );
    }

    /// Regression guard: a file whose utterance range intersects the
    /// failed group's indices MUST be marked `SkipFailed`. The old
    /// orchestrator silently injected empty UD here, which stripped
    /// `%mor` in place after `clear_morphosyntax`. Never again.
    #[test]
    fn classify_skips_file_when_its_range_intersects_failed_indices() {
        let failure = mk_failure("deu", vec![5, 6, 7]);
        let decision = classify_file_for_injection(3, 5, Some(&failure));
        match decision {
            FileInjectionDecision::SkipFailed { failed_languages } => {
                assert_eq!(failed_languages, "deu");
            }
            FileInjectionDecision::Inject => panic!(
                "file range 3..8 overlaps failed indices {{5,6,7}} — must be SkipFailed to \
                 prevent shipping a file with stripped %mor (2026-04-14 silent-corruption pattern)"
            ),
        }
    }

    /// Files with zero overlap into the failed group must still be
    /// injected. A batch of 100 English files plus one Dutch file
    /// should not fail the 100 English files when the Dutch group
    /// fails.
    #[test]
    fn classify_injects_files_whose_range_is_disjoint_from_failed_indices() {
        let failure = mk_failure("nld", vec![50, 51, 52]);
        assert_eq!(
            classify_file_for_injection(0, 10, Some(&failure)),
            FileInjectionDecision::Inject
        );
        assert_eq!(
            classify_file_for_injection(60, 40, Some(&failure)),
            FileInjectionDecision::Inject
        );
    }

    /// A single utterance at the exact boundary of the file's range
    /// still counts as intersection.
    #[test]
    fn classify_skips_file_at_range_start_boundary() {
        let failure = mk_failure("ita", vec![10]);
        let decision = classify_file_for_injection(10, 5, Some(&failure));
        assert!(
            matches!(decision, FileInjectionDecision::SkipFailed { .. }),
            "boundary intersection at global_start must count"
        );
    }

    #[test]
    fn classify_skips_file_at_range_end_boundary() {
        let failure = mk_failure("ita", vec![14]);
        let decision = classify_file_for_injection(10, 5, Some(&failure));
        assert!(
            matches!(decision, FileInjectionDecision::SkipFailed { .. }),
            "boundary intersection at global_start+count-1 must count"
        );
    }

    /// Just past the end of the range is outside.
    #[test]
    fn classify_injects_when_failure_is_exactly_one_past_range_end() {
        let failure = mk_failure("ita", vec![15]);
        let decision = classify_file_for_injection(10, 5, Some(&failure));
        assert_eq!(
            decision,
            FileInjectionDecision::Inject,
            "range 10..15 does not include index 15 — file must inject"
        );
    }

    /// Multi-language failure: the `failed_languages` string carries
    /// every failed language so the per-file error message accurately
    /// explains what went wrong.
    #[test]
    fn classify_skipped_file_names_all_failed_languages() {
        let failure = LanguageGroupFailure {
            num_failed: 2,
            languages: "deu, nld".into(),
            failed: vec![
                FailedLanguageGroup {
                    lang3: "deu".into(),
                    global_indices: vec![2],
                    error_message: "x".into(),
                },
                FailedLanguageGroup {
                    lang3: "nld".into(),
                    global_indices: vec![5],
                    error_message: "y".into(),
                },
            ],
        };
        match classify_file_for_injection(0, 10, Some(&failure)) {
            FileInjectionDecision::SkipFailed { failed_languages } => {
                assert!(failed_languages.contains("deu"));
                assert!(failed_languages.contains("nld"));
            }
            _ => panic!("must skip — file overlaps both failed groups"),
        }
    }

    /// End-to-end: round-trip an `AggregatedOutcomes` through the
    /// decision function. This is the property test that encodes the
    /// corruption contract: for every file, if any of its indices is
    /// in `failure.affected_global_indices()`, the file MUST skip
    /// injection. The pre-fix orchestrator silently violated this by
    /// injecting empty UD; this test would have failed RED before
    /// the fix and passes GREEN after.
    #[test]
    fn aggregator_plus_classifier_protects_every_file_whose_range_fails() {
        let failing_err = ServerError::Validation("deadlock".into());
        let outcomes = vec![
            LanguageGroupOutcome {
                lang3: "eng".into(),
                global_indices: vec![0, 1, 2, 3],
                result: Ok(mk_response(4)),
            },
            LanguageGroupOutcome {
                lang3: "deu".into(),
                global_indices: vec![4, 5],
                result: Err(failing_err),
            },
        ];
        let agg = aggregate_language_group_outcomes(outcomes, 6);
        let failure = agg.failure.as_ref().expect("deu group failed");

        // File A: utterances 0..4 — all eng, all succeeded.
        assert_eq!(
            classify_file_for_injection(0, 4, Some(failure)),
            FileInjectionDecision::Inject,
            "pure-eng file must inject when only deu failed"
        );

        // File B: utterances 4..6 — both deu, the failed group.
        let decision_b = classify_file_for_injection(4, 2, Some(failure));
        assert!(
            matches!(decision_b, FileInjectionDecision::SkipFailed { .. }),
            "pure-deu file must skip — injecting empty UD would strip its %mor"
        );

        // File C: utterances 2..6 — mixed eng+deu. Must still skip
        // because even one affected utterance would leave the file
        // with partial empty tiers.
        let decision_c = classify_file_for_injection(2, 4, Some(failure));
        assert!(
            matches!(decision_c, FileInjectionDecision::SkipFailed { .. }),
            "mixed-language file containing any failed-group utterance must skip"
        );
    }
}
