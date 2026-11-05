//! Abbreviation merging over speaker-attributed ASR words, before any CHAT
//! exists.
//!
//! Recognizers often emit an abbreviation as separate letters (`F B I`). This
//! stage collapses such a run into one word (`FBI`) when the letters spell an
//! entry of the abbreviation list (`data/abbrev.json`), and it does so on the
//! recognizer's own words, while each still carries the timing it was
//! recognized with. That is the only point where the merged word's timing can
//! be the truth: the union of its letters' intervals, from the first letter's
//! start to the last letter's end.
//!
//! # The transition
//!
//! ```text
//! Vec<PreparedMonologueChunk> --merge_abbreviations--> (Vec<PreparedMonologueChunk>,
//!                                                       AbbreviationMergeReport)
//! ```
//!
//! The [`AbbreviationMergeReport`] counts what the merge did and what it
//! declined to do.
//!
//! # Rules
//!
//! - **A run** is a maximal sequence of consecutive single-letter words in one
//!   chunk. A chunk has one speaker, so a speaker change ends a run; so does
//!   any other chunk boundary (a long-turn or long-pause split), and so does
//!   any word that is not a single letter, punctuation included.
//! - **A letter** is a regular word whose text is exactly one alphabetic
//!   character. A word already marked as part of a retrace is not a letter.
//! - **Matching** is greedy from the left: at each position the longest span
//!   of two or more letters whose concatenation, uppercased, is an entry of
//!   the list (itself uppercased) is merged; otherwise the letter stays and
//!   matching resumes at the next one. The merged word keeps each letter's
//!   case as recognized (`f b i` becomes `fbi`).
//! - **Timing** is the union of the letters' intervals. A letter is timed
//!   when it has an admitted, non-empty span ([`AsrWord::timed_interval`]):
//!   the same rule every ASR stage uses, since a span that covers no time
//!   locates nothing.
//! - **A matched span with an untimed letter is not merged.** Its letters are
//!   kept exactly as recognized, with whatever timing each had, and the span
//!   is counted in [`AbbreviationMergeReport::left_untimed`]. The two other
//!   choices both state something nobody observed: a union over only the
//!   timed letters claims the word ends (or starts) where a letter that was
//!   heard at an unknown time did not, and an untimed merged word throws away
//!   the timing its other letters did have. Keeping the letters loses nothing
//!   and invents nothing; the count says where a merge was declined.

use std::collections::{HashSet, VecDeque};
use std::sync::LazyLock;

use super::{AsrWord, PreparedMonologueChunk, covering_all_timed};

/// Known abbreviations, uppercased, from the crate's one parse of the list.
// The list is compile-time-constant and the crate's own tests load it, so an
// empty or malformed list fails there.
#[allow(clippy::expect_used)]
static ABBREVIATIONS: LazyLock<AbbreviationList> = LazyLock::new(|| {
    let entries: HashSet<String> = crate::abbreviation_data::ENTRIES
        .iter()
        .map(|text| text.to_uppercase())
        .collect();
    let longest = entries
        .iter()
        .map(|entry| entry.chars().count())
        .max()
        .expect("embedded abbrev.json is not empty");
    AbbreviationList { entries, longest }
});

/// The uppercased abbreviation list, with the length of its longest entry so
/// a run is never searched for a span no entry can match.
struct AbbreviationList {
    entries: HashSet<String>,
    /// Characters in the longest entry.
    longest: usize,
}

/// What one abbreviation merge did, for the log and the caller's report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbbreviationMergeReport {
    merged: usize,
    left_untimed: usize,
}

impl AbbreviationMergeReport {
    /// How many letter spans were merged into one word.
    pub const fn merged(self) -> usize {
        self.merged
    }

    /// How many letter spans spelled a known abbreviation but were kept as
    /// letters because at least one letter had no timing.
    pub const fn left_untimed(self) -> usize {
        self.left_untimed
    }
}

/// Merge single-letter runs that spell a known abbreviation, chunk by chunk,
/// and report what the merge did.
///
/// Total: a chunk with nothing to merge comes back unchanged, and no chunk is
/// added, removed or reordered. See the module documentation for the rules.
pub fn merge_abbreviations(
    chunks: Vec<PreparedMonologueChunk>,
) -> (Vec<PreparedMonologueChunk>, AbbreviationMergeReport) {
    let mut report = AbbreviationMergeReport {
        merged: 0,
        left_untimed: 0,
    };
    let chunks = chunks
        .into_iter()
        .map(
            |PreparedMonologueChunk { speaker, words }| PreparedMonologueChunk {
                speaker,
                words: merge_in_words(words, &mut report),
            },
        )
        .collect();
    (chunks, report)
}

/// One word that can join an abbreviation: the word and its single letter.
struct Letter {
    letter: char,
    word: AsrWord,
}

/// Merge every letter run in one chunk's words.
fn merge_in_words(words: Vec<AsrWord>, report: &mut AbbreviationMergeReport) -> Vec<AsrWord> {
    let mut merged = Vec::with_capacity(words.len());
    // One run buffer for the chunk: `resolve_run` always leaves it empty.
    let mut run: VecDeque<Letter> = VecDeque::new();
    for word in words {
        match letter_of(&word) {
            Some(letter) => run.push_back(Letter { letter, word }),
            None => {
                resolve_run(&mut run, &mut merged, report);
                merged.push(word);
            }
        }
    }
    resolve_run(&mut run, &mut merged, report);
    merged
}

/// The word's letter, when the word is a regular single alphabetic character.
fn letter_of(word: &AsrWord) -> Option<char> {
    word.sole_char().filter(|only| only.is_alphabetic())
}

/// Write one run of letters to `out`, greedily from the left: the longest
/// known abbreviation at the front of the run is merged (or, with an untimed
/// letter, kept as letters), otherwise the front letter is kept. Empties `run`.
fn resolve_run(
    run: &mut VecDeque<Letter>,
    out: &mut Vec<AsrWord>,
    report: &mut AbbreviationMergeReport,
) {
    let mut candidate = String::new();
    loop {
        match longest_known_prefix(run.make_contiguous(), &mut candidate) {
            Some(len) => {
                let span: Vec<Letter> = run.drain(..len).collect();
                match merged_word(&span) {
                    Some(word) => {
                        report.merged += 1;
                        out.push(word);
                    }
                    None => {
                        report.left_untimed += 1;
                        out.extend(span.into_iter().map(|letter| letter.word));
                    }
                }
            }
            None => match run.pop_front() {
                Some(letter) => out.push(letter.word),
                None => break,
            },
        }
    }
}

/// The length (at least two) of the longest known abbreviation the letters
/// start with. Never tries a span longer than the list's longest entry, so a
/// run is resolved in time linear in its length; `candidate` is the reused
/// buffer each span is uppercased into.
fn longest_known_prefix(letters: &[Letter], candidate: &mut String) -> Option<usize> {
    (2..=letters.len().min(ABBREVIATIONS.longest))
        .rev()
        .find(|&len| {
            candidate.clear();
            candidate.extend(
                letters[..len]
                    .iter()
                    .flat_map(|letter| letter.letter.to_uppercase()),
            );
            ABBREVIATIONS.entries.contains(candidate.as_str())
        })
}

/// The one word a fully timed span becomes, or `None` when a letter in it has
/// no admitted interval (see the module documentation for why that span is
/// not merged).
fn merged_word(span: &[Letter]) -> Option<AsrWord> {
    let union = covering_all_timed(span.iter().map(|letter| letter.word.timed_interval()))?;
    let text: String = span.iter().map(|letter| letter.letter).collect();
    Some(AsrWord::new(
        text,
        Some(union.start_ms()),
        Some(union.end_ms()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr_postprocess::{SpeakerIndex, WordKind};

    fn word(text: &str, start_ms: i64, end_ms: i64) -> AsrWord {
        AsrWord::new(text, Some(start_ms), Some(end_ms))
    }

    fn chunk(speaker: usize, words: Vec<AsrWord>) -> PreparedMonologueChunk {
        PreparedMonologueChunk {
            speaker: SpeakerIndex(speaker),
            words,
        }
    }

    fn texts(chunk: &PreparedMonologueChunk) -> Vec<&str> {
        chunk.words.iter().map(|word| word.text.as_str()).collect()
    }

    #[test]
    fn letters_spelling_an_abbreviation_become_one_word_timed_by_their_union() {
        let (chunks, report) = merge_abbreviations(vec![chunk(
            0,
            vec![
                word("the", 0, 200),
                word("F", 300, 400),
                word("B", 450, 520),
                word("I", 600, 700),
                word("is", 800, 900),
            ],
        )]);
        assert_eq!(texts(&chunks[0]), ["the", "FBI", "is"]);
        let fbi = &chunks[0].words[1];
        assert_eq!((fbi.start_ms, fbi.end_ms), (Some(300), Some(700)));
        assert_eq!(fbi.kind, WordKind::Regular);
        assert_eq!((report.merged(), report.left_untimed()), (1, 0));
    }

    /// The union is the covering interval, not the first letter's start and
    /// the last letter's end taken blindly: a recognizer that reports
    /// overlapping letter spans still yields the true extent.
    #[test]
    fn the_union_covers_every_letter_even_when_intervals_overlap() {
        let (chunks, _) = merge_abbreviations(vec![chunk(
            0,
            vec![
                word("C", 100, 900),
                word("I", 200, 300),
                word("A", 400, 500),
            ],
        )]);
        assert_eq!(texts(&chunks[0]), ["CIA"]);
        assert_eq!(
            (chunks[0].words[0].start_ms, chunks[0].words[0].end_ms),
            (Some(100), Some(900))
        );
    }

    #[test]
    fn a_letter_run_that_spells_no_abbreviation_is_left_alone() {
        let words = vec![word("Q", 0, 100), word("W", 100, 200), word("Z", 200, 300)];
        let (chunks, report) = merge_abbreviations(vec![chunk(0, words.clone())]);
        assert_eq!(report.merged(), 0);
        assert_eq!(chunks[0].words, words);
    }

    /// Greedy longest match from the left, then the rest of the run is
    /// matched afresh: the list holds `XYZ`, and `Q` and `W` start nothing.
    #[test]
    fn a_known_prefix_merges_and_the_unknown_tail_stays_as_letters() {
        let (chunks, report) = merge_abbreviations(vec![chunk(
            0,
            vec![
                word("X", 0, 100),
                word("Y", 100, 200),
                word("Z", 200, 300),
                word("Q", 300, 400),
                word("W", 400, 500),
            ],
        )]);
        assert_eq!(texts(&chunks[0]), ["XYZ", "Q", "W"]);
        assert_eq!(report.merged(), 1);
    }

    /// Chunks are per speaker, so letters on either side of a speaker change
    /// are in different chunks and never merge across it.
    #[test]
    fn a_speaker_change_ends_the_run() {
        let (chunks, report) = merge_abbreviations(vec![
            chunk(0, vec![word("F", 0, 100), word("B", 100, 200)]),
            chunk(1, vec![word("I", 200, 300), word("agreed", 300, 600)]),
        ]);
        assert_eq!(texts(&chunks[0]), ["F", "B"]);
        assert_eq!(texts(&chunks[1]), ["I", "agreed"]);
        assert_eq!(chunks[1].speaker, SpeakerIndex(1));
        assert_eq!(report.merged(), 0);
    }

    #[test]
    fn a_span_with_an_untimed_letter_keeps_its_letters_and_is_counted() {
        let words = vec![
            word("F", 0, 100),
            AsrWord::new("B", None, None),
            word("I", 200, 300),
        ];
        let (chunks, report) = merge_abbreviations(vec![chunk(0, words.clone())]);
        assert_eq!(chunks[0].words, words, "letters kept exactly as recognized");
        assert_eq!((report.merged(), report.left_untimed()), (0, 1));
    }

    #[test]
    fn a_letter_missing_only_one_bound_is_untimed_too() {
        let words = vec![
            word("F", 0, 100),
            word("B", 100, 200),
            AsrWord::new("I", Some(200), None),
        ];
        let (chunks, report) = merge_abbreviations(vec![chunk(0, words.clone())]);
        assert_eq!(chunks[0].words, words);
        assert_eq!(report.left_untimed(), 1);
    }

    /// A zero-width span locates nothing, here as in every ASR stage, so a
    /// letter with one is untimed and its span is kept as letters.
    #[test]
    fn a_letter_with_a_zero_width_span_is_untimed() {
        let words = vec![word("F", 0, 100), word("B", 150, 150), word("I", 200, 300)];
        let (chunks, report) = merge_abbreviations(vec![chunk(0, words.clone())]);
        assert_eq!(chunks[0].words, words);
        assert_eq!(report.left_untimed(), 1);
    }

    /// Matching is case-insensitive against the list; the merged word keeps
    /// the letters' case as recognized, as the list's mixed-case entries
    /// (`PhD`) are matched by their uppercase form.
    #[test]
    fn matching_ignores_case_and_the_merged_word_keeps_the_recognized_case() {
        let (chunks, _) = merge_abbreviations(vec![chunk(
            0,
            vec![
                word("f", 0, 100),
                word("b", 100, 200),
                word("i", 200, 300),
                word("P", 400, 500),
                word("h", 500, 600),
                word("D", 600, 700),
            ],
        )]);
        assert_eq!(texts(&chunks[0]), ["fbi", "PhD"]);
    }

    /// A word that is not a single letter ends the run, punctuation included,
    /// and a retrace-marked letter is not a letter.
    #[test]
    fn a_non_letter_word_or_a_retrace_ends_the_run() {
        let mut retraced = word("B", 100, 200);
        retraced.kind = WordKind::Retrace;
        let (chunks, report) = merge_abbreviations(vec![chunk(
            0,
            vec![
                word("F", 0, 100),
                word(".", 100, 100),
                word("B", 100, 200),
                word("I", 200, 300),
                word("F", 300, 400),
                retraced,
                word("I", 500, 600),
            ],
        )]);
        assert_eq!(texts(&chunks[0]), ["F", ".", "B", "I", "F", "B", "I"]);
        assert_eq!(report.merged(), 0);
    }

    /// The search never tries a span longer than the longest entry, and a
    /// run longer than that still has each of its abbreviations found.
    #[test]
    fn a_run_longer_than_any_entry_still_merges_each_abbreviation() {
        let letters = ["F", "B", "I", "X", "Y", "Z", "C", "I", "A", "Q"];
        let words = letters
            .iter()
            .enumerate()
            .map(|(at, letter)| {
                let start = i64::try_from(at).expect("small") * 100;
                word(letter, start, start + 80)
            })
            .collect();
        let (chunks, report) = merge_abbreviations(vec![chunk(0, words)]);
        assert!(letters.len() > ABBREVIATIONS.longest);
        assert_eq!(texts(&chunks[0]), ["FBI", "XYZ", "CIA", "Q"]);
        assert_eq!(report.merged(), 3);
    }

    #[test]
    fn the_embedded_list_loads_and_is_uppercased() {
        assert!(ABBREVIATIONS.entries.contains("FBI"));
        assert!(ABBREVIATIONS.entries.contains("PHD"));
        assert!(!ABBREVIATIONS.entries.contains("PhD"));
        assert!(ABBREVIATIONS.longest >= "FBI".len());
    }
}
