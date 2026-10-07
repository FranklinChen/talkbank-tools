//! Synthetic timing assessment: no-drift is not complete recovery.

use crate::chat_ops::fa::utr::UtrResult;
use talkbank_model::model::{ChatFile, Line};

use super::fixture::ExpectedWindow;

#[derive(Debug)]
enum WindowObservation {
    WithinWindow,
    OutsideWindow,
    Unrecovered { surviving_asr_words: usize },
}

/// Produced only by observing the whole admitted scenario after injection.
/// A caller cannot obtain a completion proof from an empty violation list.
#[derive(Debug)]
pub(super) struct DriftObservation {
    injection_summary: String,
    windows: Vec<WindowObservation>,
    violations: Vec<String>,
}

/// Successful admission requires both sound timing and complete recovery.
#[derive(Debug)]
pub(super) struct CompleteDriftObservation(DriftObservation);

impl CompleteDriftObservation {
    pub(super) fn summary(&self) -> String {
        self.0.summary()
    }
}

impl DriftObservation {
    pub(super) fn observe(
        chat: &ChatFile,
        expected: &[ExpectedWindow],
        result: UtrResult,
        mut violations: Vec<String>,
    ) -> Self {
        const SLACK_MS: u64 = 500;
        let utterances: Vec<_> = chat
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance),
                _ => None,
            })
            .collect();
        // A fixture/scorer population mismatch is a producer fault, not a
        // missing observation to silently discard (including release builds).
        assert_eq!(
            utterances.len(),
            expected.len(),
            "fixture/scorer population mismatch"
        );
        let windows = utterances
            .iter()
            .zip(expected)
            .enumerate()
            .map(|(ordinal, (utterance, exp))| {
                assert_eq!(exp.utt_index, ordinal, "fixture/scorer order mismatch");
                let Some(bullet) = utterance.main.content.bullet.as_ref() else {
                    return WindowObservation::Unrecovered {
                        surviving_asr_words: exp.surviving_asr_words,
                    };
                };
                let lo = exp.expected_start_ms.saturating_sub(SLACK_MS);
                let hi = exp.expected_end_ms.saturating_add(SLACK_MS);
                let s = bullet.timing.start_ms;
                let e = bullet.timing.end_ms;
                if s < lo || e > hi {
                    violations.push(format!(
                        "utt #{ordinal}: bullet [{s},{e}]ms outside ground-truth window \
                     [{lo},{hi}]ms (expected [{exp_s},{exp_e}]ms ± {SLACK_MS}ms slack)",
                        exp_s = exp.expected_start_ms,
                        exp_e = exp.expected_end_ms,
                    ));
                    WindowObservation::OutsideWindow
                } else {
                    WindowObservation::WithinWindow
                }
            })
            .collect();
        Self {
            injection_summary: format!(
                "injected={} skipped={} unmatched={}",
                result.injected, result.skipped, result.unmatched
            ),
            windows,
            violations,
        }
    }

    pub(super) fn violations(&self) -> &[String] {
        &self.violations
    }

    pub(super) fn try_complete(self) -> Result<CompleteDriftObservation, Self> {
        if self.violations.is_empty()
            && self
                .windows
                .iter()
                .all(|window| matches!(window, WindowObservation::WithinWindow))
        {
            Ok(CompleteDriftObservation(self))
        } else {
            Err(self)
        }
    }

    pub(super) fn summary(&self) -> String {
        let mut within = 0;
        let mut outside = 0;
        let mut unrecovered_with_words = 0;
        let mut unrecovered_without_words = 0;
        for window in &self.windows {
            match window {
                WindowObservation::WithinWindow => within += 1,
                WindowObservation::OutsideWindow => outside += 1,
                WindowObservation::Unrecovered {
                    surviving_asr_words: 0,
                } => {
                    unrecovered_without_words += 1;
                }
                WindowObservation::Unrecovered { .. } => unrecovered_with_words += 1,
            }
        }
        format!(
            "{} windows={} within={} outside={} unrecovered_with_words={} unrecovered_without_words={}",
            self.injection_summary,
            self.windows.len(),
            within,
            outside,
            unrecovered_with_words,
            unrecovered_without_words
        )
    }
}
