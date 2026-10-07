//! Local recovery shares the global proposal's checked timing producer.

use super::*;

#[test]
fn local_overlap_hull_uses_the_same_matched_interval_admission() {
    let words = vec!["hello".to_owned(), "world".to_owned(), "again".to_owned()];
    let recover = |tokens: &[(&str, u64, u64)]| {
        let tokens = tokens
            .iter()
            .map(|&(text, start_ms, end_ms)| AsrTimingToken {
                text: text.to_owned(),
                start_ms,
                end_ms,
            })
            .collect::<Vec<_>>();
        recover_overlap_timing(&words, &tokens, 0, 2_000, MatchMode::CaseInsensitive)
            .map(|interval| (interval.start_ms(), interval.end_ms()))
    };
    assert_eq!(
        recover(&[
            ("hello", 100, 200),
            ("world", 150, 1_200),
            ("again", 300, 400)
        ]),
        Some((100, 1_200))
    );
    assert_eq!(
        recover(&[
            ("hello", 100, 200),
            ("world", 300, 300),
            ("again", 400, 500)
        ]),
        None
    );
    assert_eq!(
        recover(&[
            ("hello", 100, 200),
            ("unrelated", 0, 2_000),
            ("world", 200, 300),
            ("again", 400, 500)
        ]),
        Some((100, 500))
    );
}
