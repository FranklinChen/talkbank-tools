use std::collections::HashMap;
use std::time::Duration;

use crate::text_batch::TextBatchFileInput;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WindowRange {
    pub(super) start: usize,
    pub(super) end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WindowPlan {
    pub(super) windows: Vec<WindowRange>,
    pub(super) group_timeout: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MorphotagExecutionMode {
    Incremental,
    Windowed(WindowPlan),
}

pub(super) fn select_execution_mode(
    file_texts: &[TextBatchFileInput],
    before_texts: &HashMap<String, String>,
    batch_window: usize,
    timeout_secs: u64,
) -> MorphotagExecutionMode {
    if !before_texts.is_empty() {
        return MorphotagExecutionMode::Incremental;
    }

    MorphotagExecutionMode::Windowed(WindowPlan {
        windows: build_windows(file_texts, batch_window),
        group_timeout: Duration::from_secs(timeout_secs),
    })
}

fn build_windows(file_texts: &[TextBatchFileInput], batch_window: usize) -> Vec<WindowRange> {
    let utterance_counts: Vec<usize> = file_texts
        .iter()
        .map(|file| utterance_count(file.chat_text.as_ref()))
        .collect();
    let batch_window_size = match batch_window {
        0 => file_texts.len(),
        1..=1000 => batch_window,
        _ => 1000,
    };
    let utterance_budget = if batch_window == 0 {
        usize::MAX
    } else {
        batch_window_size * 80
    };
    chunk_by_utterance_budget(&utterance_counts, utterance_budget)
}

fn utterance_count(chat_text: &str) -> usize {
    chat_text
        .lines()
        .filter(|line| line.starts_with('*'))
        .count()
}

pub(super) fn chunk_by_utterance_budget(
    utterance_counts: &[usize],
    budget: usize,
) -> Vec<WindowRange> {
    let mut windows = Vec::new();
    let mut start = 0;
    let mut window_utts = 0;
    for (i, &count) in utterance_counts.iter().enumerate() {
        if window_utts + count > budget && i > start {
            windows.push(WindowRange { start, end: i });
            start = i;
            window_utts = 0;
        }
        window_utts += count;
    }
    if start < utterance_counts.len() {
        windows.push(WindowRange {
            start,
            end: utterance_counts.len(),
        });
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_with_utterances(count: usize) -> String {
        let mut lines = vec!["@UTF8".to_string(), "@Begin".to_string()];
        for idx in 0..count {
            lines.push(format!("*PAR:\tutt{idx} ."));
        }
        lines.push("@End".to_string());
        lines.join("\n")
    }

    #[test]
    fn chunking_respects_utterance_budget_boundaries() {
        let windows = chunk_by_utterance_budget(&[100, 30, 90, 10], 120);

        assert_eq!(
            windows,
            vec![
                WindowRange { start: 0, end: 1 },
                WindowRange { start: 1, end: 3 },
                WindowRange { start: 3, end: 4 },
            ]
        );
    }

    #[test]
    fn window_selection_caps_large_batch_window_at_1000() {
        let files = vec![
            TextBatchFileInput::new("a.cha", chat_with_utterances(90_000)),
            TextBatchFileInput::new("b.cha", chat_with_utterances(10)),
        ];

        let mode = select_execution_mode(&files, &HashMap::new(), 5_000, 1800);

        assert_eq!(
            mode,
            MorphotagExecutionMode::Windowed(WindowPlan {
                windows: vec![
                    WindowRange { start: 0, end: 1 },
                    WindowRange { start: 1, end: 2 },
                ],
                group_timeout: Duration::from_secs(1800),
            })
        );
    }
}
