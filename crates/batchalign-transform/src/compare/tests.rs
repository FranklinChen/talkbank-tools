use super::engine::{conform_file_words, is_punct_or_filler};
use super::*;
use talkbank_model::{ChatFile, ErrorCollector, WriteChat};
use talkbank_parser::TreeSitterParser;

pub(super) fn parse_lenient(
    parser: &TreeSitterParser,
    chat_text: &str,
) -> (ChatFile, Vec<talkbank_model::ParseError>) {
    let errors = ErrorCollector::new();
    let chat_file = parser.parse_chat_file_streaming(chat_text, &errors);
    let error_vec = errors.into_vec();
    (chat_file, error_vec)
}

/// Build a minimal English CHAT file with given utterance lines.
fn make_chat(utterances: &[(&str, &str)]) -> String {
    chat_text("eng", utterances)
}

/// Build a minimal CHAT file declaring `languages` (the `@Languages` value,
/// first entry primary) with given utterance lines.
pub(super) fn chat_text(languages: &str, utterances: &[(&str, &str)]) -> String {
    let mut lines = vec![
        "@UTF8".to_string(),
        "@Begin".to_string(),
        format!("@Languages:\t{languages}"),
        "@Participants:\tCHI Target_Child, MOT Mother".to_string(),
        "@ID:\teng|test|CHI|3;|female|||Target_Child|||".to_string(),
        "@ID:\teng|test|MOT||female|||Mother|||".to_string(),
    ];
    for (speaker, text) in utterances {
        lines.push(format!("*{speaker}:\t{text}"));
    }
    lines.push("@End".to_string());
    lines.join("\n")
}

#[test]
fn identical_transcripts() {
    let parser = TreeSitterParser::new().unwrap();
    let chat = make_chat(&[("CHI", "hello world ."), ("MOT", "good morning .")]);
    let (main_file, _) = parse_lenient(&parser, &chat);
    let (gold_file, _) = parse_lenient(&parser, &chat);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.wer(), 0.0);
    assert_eq!(result.metrics.accuracy(), 1.0);
    assert_eq!(result.metrics.matches(), 4);
    assert_eq!(result.metrics.insertions(), 0);
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.total_gold_words(), 4);
    assert_eq!(result.metrics.total_main_words(), 4);
}

/// Main tokens outside the matched stretch must be reported as insertions, not
/// dropped.
///
/// The defect fixed on 2026-07-30. Placement decided
/// WHICH main material corresponded to a gold utterance; it was not a licence
/// to ignore the main tokens it did not select. While it was (placement was then a
/// bag-of-words window), every hypothesis word the window missed went uncounted, so the
/// reported WER was systematically friendlier than the truth, in an amount that
/// varied with how ragged the transcript was. That makes the metric unusable as
/// a headline accuracy number, which is how it was rediscovered: someone tried
/// to use it as one.
#[test]
fn main_tokens_outside_the_matched_stretch_are_insertions_not_silently_dropped() {
    let parser = TreeSitterParser::new().unwrap();
    // Two leading and two trailing main tokens have no gold counterpart.
    let main = make_chat(&[("CHI", "alpha beta the cat sat gamma delta .")]);
    let gold = make_chat(&[("CHI", "the cat sat .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 3, "the/cat/sat still match");
    assert_eq!(
        result.metrics.insertions(),
        4,
        "alpha, beta, gamma and delta are hypothesis words the reference does \
                  not contain; each is an insertion however placement was decided"
    );
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.total_main_words(), 7);
}

#[test]
fn single_substitution() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello earth .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    // A substitution is one insertion plus one deletion. Before 2026-07-30,
    // "earth" fell outside the chosen window and was not counted at
    // all, so a substitution registered as a bare deletion.
    assert!(result.metrics.wer() > 0.0);
    assert_eq!(result.metrics.matches(), 1); // "hello" matches
    assert_eq!(result.metrics.insertions(), 1); // "earth"
    assert_eq!(result.metrics.deletions(), 1); // "world"
    // "earth" and "world" are different words, so nothing cancels: a genuine
    // substitution costs the same under `cwer` as under `wer`.
    assert_eq!(result.metrics.cwer(), result.metrics.wer());
}

#[test]
fn extra_word_in_main() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello big world .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 2); // "hello", "world"
    assert_eq!(result.metrics.insertions(), 1); // "big"
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.total_gold_words(), 2);
    assert_eq!(result.metrics.total_main_words(), 3);
}

#[test]
fn missing_word_in_main() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 1); // "hello"
    assert_eq!(result.metrics.insertions(), 0);
    assert_eq!(result.metrics.deletions(), 1); // "world"
    assert_eq!(result.metrics.total_gold_words(), 2);
    assert_eq!(result.metrics.total_main_words(), 1);
}

#[test]
fn empty_main() {
    let parser = TreeSitterParser::new().unwrap();
    // Main has an utterance but no content words (just terminator)
    let main = make_chat(&[("CHI", ".")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 0);
    assert_eq!(result.metrics.deletions(), 2);
    assert_eq!(result.metrics.wer(), 1.0);
}

#[test]
fn empty_gold() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello world .")]);
    let gold = make_chat(&[("CHI", ".")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 0);
    // A COMPLETE gold containing nothing claims the recording contains nothing,
    // so every word the system produced is an insertion. Before the gold
    // coverage was stated, these two were counted nowhere.
    assert_eq!(result.metrics.insertions(), 2);
    assert_eq!(result.metrics.total_gold_words(), 0);
    // WER's denominator is zero here, so the rate itself is not meaningful and
    // is reported as 0.0; read `insertions` instead in this degenerate case.
    assert_eq!(result.metrics.wer(), 0.0);

    // The same files under a PARTIAL gold: the reference claims nothing about
    // this material, so nothing is charged.
    let partial = compare(&main_file, &gold_file, GoldCoverage::Partial);
    assert_eq!(partial.metrics.insertions(), 0);
}

#[test]
fn case_insensitive_matching() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "Hello World .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.wer(), 0.0);
    assert_eq!(result.metrics.matches(), 2);
}

#[test]
fn conform_normalizes_contractions() {
    let parser = TreeSitterParser::new().unwrap();
    // "he's" should be expanded to "he is" by conform_words
    let main = make_chat(&[("CHI", "he's going .")]);
    let gold = make_chat(&[("CHI", "he is going .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    // After conform: main = ["he", "is", "going"], gold = ["he", "is", "going"]
    assert_eq!(result.metrics.wer(), 0.0);
}

#[test]
fn multiple_utterances() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
    let gold = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.wer(), 0.0);
    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(result.main_utterances.len(), 2);
    assert_eq!(result.gold_utterances.len(), 2);
}

#[test]
fn xsrep_tier_content_serializes_through_write_chat() {
    let utt = UtteranceComparison {
        utterance_index: 0,
        speaker: "CHI".to_string(),
        tokens: vec![
            CompareToken {
                text: "hello".to_string(),
                pos: Some("INTJ".to_string()),
                status: CompareStatus::Match,
            },
            CompareToken {
                text: "big".to_string(),
                pos: Some("ADJ".to_string()),
                status: CompareStatus::ExtraMain,
            },
            CompareToken {
                text: "world".to_string(),
                pos: Some("NOUN".to_string()),
                status: CompareStatus::Match,
            },
            CompareToken {
                text: "today".to_string(),
                pos: Some("NOUN".to_string()),
                status: CompareStatus::ExtraGold,
            },
        ],
    };
    let xsrep = XsrepTierContent::try_from(&utt).expect("xsrep tier");
    assert_eq!(xsrep.to_chat_string(), "hello +big world -today");
}

#[test]
fn xsmor_tier_content_serializes_through_write_chat() {
    let utt = UtteranceComparison {
        utterance_index: 0,
        speaker: "CHI".to_string(),
        tokens: vec![
            CompareToken {
                text: "hello".to_string(),
                pos: Some("INTJ".to_string()),
                status: CompareStatus::Match,
            },
            CompareToken {
                text: "big".to_string(),
                pos: Some("ADJ".to_string()),
                status: CompareStatus::ExtraMain,
            },
            CompareToken {
                text: "today".to_string(),
                pos: None,
                status: CompareStatus::ExtraGold,
            },
        ],
    };
    let xsmor = XsmorTierContent::try_from(&utt).expect("xsmor tier");
    assert_eq!(xsmor.to_chat_string(), "INTJ +ADJ -?");
}

#[test]
fn xsmor_serializes_final_punctuation_as_surface_delimiter() {
    let utt = UtteranceComparison {
        utterance_index: 0,
        speaker: "CHI".to_string(),
        tokens: vec![
            CompareToken {
                text: "hello".to_string(),
                pos: Some("INTJ".to_string()),
                status: CompareStatus::Match,
            },
            CompareToken {
                text: ".".to_string(),
                pos: Some("PUNCT".to_string()),
                status: CompareStatus::Match,
            },
        ],
    };
    let xsmor = XsmorTierContent::try_from(&utt).expect("xsmor tier");
    assert_eq!(xsmor.to_chat_string(), "INTJ .");
}

#[test]
fn compare_metrics_csv_table_serializes_with_csv_writer() {
    let parser = TreeSitterParser::new().unwrap();
    let gold = make_chat(&[("CHI", "the dog cat .")]);
    let main = make_chat(&[("CHI", "the dog cat mouse .")]);
    let main = main.replace(
        "@End",
        "%mor:\tdet|the noun|dog noun|cat noun|mouse .\n@End",
    );
    let (gold, _) = parse_lenient(&parser, &gold);
    let (main, _) = parse_lenient(&parser, &main);
    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;
    let csv = CompareMetricsCsvTable::from_metrics(&metrics)
        .expect("table")
        .to_csv_string()
        .expect("csv");
    assert!(csv.contains("wer,0.3333"));
    assert!(csv.contains("accuracy,0.6667"));
    assert!(csv.contains("matches,3"));
    assert!(csv.contains("insertions,1"));
    assert!(csv.contains("deletions,0"));
    assert!(csv.contains("NOUN:matches,2"));
    assert!(csv.contains("NOUN:insertions,1"));
}

#[test]
fn wer_computation_is_correct() {
    let expected_wer = 2.0 / 3.0;

    // Test via actual compare
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello big world .")]);
    let gold = make_chat(&[("CHI", "hello world today .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    // main: hello, big, world
    // gold: hello, world, today
    // align: hello=match, big=extra_main, world=match, today=extra_gold
    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(result.metrics.insertions(), 1);
    assert_eq!(result.metrics.deletions(), 1);
    assert!((result.metrics.wer() - expected_wer).abs() < 0.001);
}

#[test]
fn is_punct_or_filler_works() {
    assert!(is_punct_or_filler("."));
    assert!(is_punct_or_filler("?"));
    assert!(is_punct_or_filler("!"));
    assert!(is_punct_or_filler("+/."));
    assert!(is_punct_or_filler("um"));
    assert!(is_punct_or_filler("uh"));
    assert!(!is_punct_or_filler("hello"));
    assert!(!is_punct_or_filler("world"));
}

/// A short utterance that recurs later in the main transcript does not strand
/// the rest of the file.
///
/// The regression (2026-09-16, found scoring real Bangor/Miami files): gold
/// utterances were placed by a window search over all of the remaining main
/// transcript that broke ties toward the LATEST window, followed by a
/// forward-only cursor. The gold's opening `yeah .` was placed on the main's
/// closing `yeah .`, the cursor jumped to the end of the file, and every later
/// gold utterance was left unplaced. On this input the comparison reported 1
/// match instead of 10; on 30-minute recordings it reported about 100 percent
/// error whatever the recognizer did.
#[test]
fn an_early_repeated_utterance_does_not_strand_the_rest_of_the_file() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[
        ("MOT", "yeah ."),
        ("MOT", "the cat sat on the mat ."),
        ("MOT", "we went home ."),
        ("MOT", "yeah ."),
    ]);
    let gold = make_chat(&[
        ("MOT", "yeah ."),
        ("MOT", "the cat sat on the mat ."),
        ("MOT", "we went home ."),
    ]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let metrics = compare(&main_file, &gold_file, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.matches(), 10, "every gold word is in the main");
    assert_eq!(metrics.insertions(), 1, "only the closing yeah is extra");
    assert_eq!(metrics.deletions(), 0);
    assert_eq!(metrics.languages().utterances().placed().scored(), 3);
    assert!(metrics.languages().utterances().unplaced().is_empty());
}

/// The same property when the files start and end differently, so the
/// aligner's own recursion places the utterances rather than its common
/// prefix and suffix stripping: the main opens with `okay .`, and the gold
/// closes with `bye .`, which the main never says.
#[test]
fn placement_holds_when_the_files_start_and_end_differently() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[
        ("MOT", "okay ."),
        ("MOT", "yeah ."),
        ("MOT", "the cat sat on the mat ."),
        ("MOT", "we went home ."),
        ("MOT", "yeah ."),
    ]);
    let gold = make_chat(&[
        ("MOT", "yeah ."),
        ("MOT", "the cat sat on the mat ."),
        ("MOT", "we went home ."),
        ("MOT", "bye ."),
    ]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let metrics = compare(&main_file, &gold_file, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.matches(), 10);
    assert_eq!(metrics.insertions(), 2, "okay and the closing yeah");
    assert_eq!(metrics.deletions(), 1, "bye");
    assert_eq!(metrics.languages().utterances().placed().scored(), 3);
    let unplaced: usize = metrics.languages().utterances().unplaced().values().sum();
    assert_eq!(unplaced, 1);
}

/// A gold utterance with no counterpart is unplaced on its own, and the gold
/// utterances after it still find theirs.
#[test]
fn an_unmatched_gold_utterance_does_not_displace_later_ones() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("MOT", "hello there ."), ("MOT", "good night .")]);
    let gold = make_chat(&[
        ("MOT", "hello there ."),
        ("MOT", "completely different words ."),
        ("MOT", "good night ."),
    ]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let metrics = compare(&main_file, &gold_file, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.matches(), 4);
    assert_eq!(metrics.deletions(), 3);
    assert_eq!(metrics.insertions(), 0);
    assert_eq!(metrics.languages().utterances().placed().scored(), 2);
    let unplaced: usize = metrics.languages().utterances().unplaced().values().sum();
    assert_eq!(unplaced, 1);
}

/// Scores do not depend on where either transcript put utterance boundaries.
///
/// A recognizer that splits one reference utterance in two has transcribed the
/// same words, and a benchmark of transcription must score it that way: every
/// metric comes from one alignment of the whole file, so a matched word counts
/// wherever the boundary fell. Aligning each main utterance only against the
/// gold placed on it charged the split as an error.
///
/// Scenario:
///   main: "the sky ." (utt 0) + "this dog ran fast ." (utt 1)
///   gold: "the dog ran ."
///
/// All three gold words match; `sky`, `this` and `fast` are insertions. The
/// gold utterance is still placed, for utterance language agreement, on utt 1,
/// which holds two of its three matched words.
#[test]
fn scores_do_not_depend_on_utterance_boundaries() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "the sky ."), ("CHI", "this dog ran fast .")]);
    let gold = make_chat(&[("CHI", "the dog ran .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 3, "the, dog and ran all match");
    assert_eq!(result.metrics.insertions(), 3, "sky, this and fast");
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.total_gold_words(), 3);
    assert_eq!(result.metrics.languages().utterances().placed().scored(), 1);
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "the +sky +this dog ran +fast .",
        "the gold view shows every word matched, with the insertions in order"
    );
}

/// Under a complete gold, splitting the main transcript differently changes
/// nothing: not the counts, not `cwer`, not language attribution, not the gold
/// view.
///
/// `ran the dog .` and `ran .` + `the dog .` are the same words in the same
/// order. Deciding anything from which main utterance a word is in (whether it
/// "matched nothing", say) would score them differently, and a benchmark would
/// then be measuring segmentation while claiming to measure recognition.
#[test]
fn a_main_transcript_split_differently_scores_the_same() {
    let parser = TreeSitterParser::new().unwrap();
    let (gold_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", "the dog ran .")]));
    let (joined, _) = parse_lenient(&parser, &make_chat(&[("CHI", "ran the dog .")]));
    let (split, _) = parse_lenient(
        &parser,
        &make_chat(&[("CHI", "ran ."), ("CHI", "the dog .")]),
    );

    let joined = compare(&joined, &gold_file, GoldCoverage::Complete);
    let split = compare(&split, &gold_file, GoldCoverage::Complete);

    assert_eq!(joined.metrics, split.metrics);
    assert_eq!(
        joined.metrics.cwer(),
        0.0,
        "`ran` only moved within its utterance"
    );
    let gold_view = |bundle: &ComparisonBundle| {
        XsrepTierContent::try_from(&bundle.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string()
    };
    assert_eq!(gold_view(&joined), gold_view(&split));
}

/// A word moved from inside its gold utterance to just before the next one
/// still cancels in `cwer`.
///
/// The alignment reads `home` as deleted before `i` and inserted before `you`,
/// the first word of the next gold utterance. An insertion at a boundary may
/// cancel on either side of it, so the displaced word is not charged twice.
#[test]
fn a_word_moved_to_the_end_of_its_utterance_cancels() {
    let parser = TreeSitterParser::new().unwrap();
    let (gold_file, _) = parse_lenient(
        &parser,
        &make_chat(&[("CHI", "home i went ."), ("CHI", "you .")]),
    );
    let (main_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", "i went home you .")]));

    let metrics = compare(&main_file, &gold_file, GoldCoverage::Complete).metrics;
    assert_eq!(metrics.insertions(), 1);
    assert_eq!(metrics.deletions(), 1);
    assert_eq!(metrics.cwer(), 0.0);
}

/// The mirror case: a word moved from the next gold utterance back to just
/// after the previous one cancels too.
#[test]
fn a_word_moved_to_the_start_of_the_next_utterance_cancels() {
    let parser = TreeSitterParser::new().unwrap();
    let (gold_file, _) = parse_lenient(
        &parser,
        &make_chat(&[("CHI", "i went ."), ("CHI", "you home .")]),
    );
    let (main_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", "i went home you .")]));

    let metrics = compare(&main_file, &gold_file, GoldCoverage::Complete).metrics;
    assert_eq!(metrics.insertions(), 1);
    assert_eq!(metrics.deletions(), 1);
    assert_eq!(metrics.cwer(), 0.0);
}

/// Under a partial gold, a deletion is never shown in a main utterance that is
/// outside what the gold covers.
///
/// The gold covers only the child. `c` is missing before any covered main word
/// has been shown, so it waits and is shown before the child's first word,
/// not in the mother's unscored utterance, which gets no annotation at all.
#[test]
fn a_partial_gold_shows_no_deletion_in_an_uncovered_main_utterance() {
    let parser = TreeSitterParser::new().unwrap();
    let (main_file, _) = parse_lenient(&parser, &make_chat(&[("MOT", "x ."), ("CHI", "d .")]));
    let (gold_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", "c d .")]));

    let result = compare(&main_file, &gold_file, GoldCoverage::Partial);
    assert!(result.main_utterances[0].tokens.is_empty());
    assert_eq!(
        XsrepTierContent::try_from(&result.main_utterances[1])
            .expect("xsrep tier")
            .to_chat_string(),
        "-c d"
    );
    assert_eq!(result.metrics.insertions(), 0);
    assert_eq!(result.metrics.deletions(), 1);
}

/// The gold terminator is shown after every word of its utterance.
///
/// A word's slot is its extracted-word position, which counts the separators
/// compare leaves out, so `yes , no , maybe` puts `maybe` at position 4 while
/// the utterance holds three compared words. The float sort key this replaced
/// put the terminator at that word COUNT, ahead of `maybe`.
#[test]
fn the_gold_terminator_follows_every_word_after_separators() {
    let parser = TreeSitterParser::new().unwrap();
    let text = make_chat(&[("CHI", "yes , no , maybe .")]);
    let (main_file, _) = parse_lenient(&parser, &text);
    let (gold_file, _) = parse_lenient(&parser, &text);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 3);
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "yes no maybe ."
    );
}

/// A gold word the main transcript lacks is shown in the main view where it
/// was missed: after the main word before it, or ahead of the first main word
/// when nothing precedes it.
#[test]
fn the_main_view_shows_a_deletion_where_the_word_was_missed() {
    let parser = TreeSitterParser::new().unwrap();
    let main_view = |main: &str, gold: &str| {
        let (main_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", main)]));
        let (gold_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", gold)]));
        let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
        XsrepTierContent::try_from(&result.main_utterances[0])
            .expect("xsrep tier")
            .to_chat_string()
    };

    assert_eq!(
        main_view("hello world .", "hello big world ."),
        "hello -big world"
    );
    assert_eq!(main_view("world .", "hello world ."), "-hello world");
}

/// One word can conform to several tokens; the mapping must point each token
/// back at the word it came from, or insertions and matches land on the wrong
/// word. Built through a parsed English file, because conforming is only
/// reachable from a word whose language has been resolved.
#[test]
fn conform_with_mapping_tracks_indices() {
    let parser = TreeSitterParser::new().unwrap();
    let (chat_file, _) = parse_lenient(&parser, &make_chat(&[("CHI", "he's going .")]));
    let (conformed, mapping) = conform_file_words(&chat_file);
    // "he's" -> ["he", "is"], "going" -> ["going"]
    assert_eq!(conformed, vec!["he", "is", "going"]);
    assert_eq!(mapping, vec![0, 0, 1]);
}

#[test]
fn inject_comparison_adds_xsrep_tiers() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello big world .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (mut main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");

    // Find the utterance and check it has an %xsrep tier
    let serialized = main_file.to_chat_string();
    assert!(
        serialized.contains("%xsrep:"),
        "Output should contain %xsrep tier"
    );
    assert!(
        serialized.contains("+big"),
        "Should mark 'big' as extra_main"
    );
    assert!(
        serialized.contains("%xsmor:"),
        "Output should contain %xsmor tier"
    );
}

#[test]
fn clear_comparison_removes_compare_tiers() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello world .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (mut main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");

    // Verify xsrep was added
    let serialized = main_file.to_chat_string();
    assert!(serialized.contains("%xsrep:"));
    assert!(serialized.contains("%xsmor:"));

    // Clear and verify removal
    clear_comparison(&mut main_file);
    let serialized = main_file.to_chat_string();
    assert!(!serialized.contains("%xsrep:"));
    assert!(!serialized.contains("%xsmor:"));
}

#[test]
fn inject_comparison_idempotent() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello big world .")]);
    let gold = make_chat(&[("CHI", "hello world .")]);
    let (mut main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");
    let first = main_file.to_chat_string();

    // Inject again: should produce the same output (replace, not duplicate)
    inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");
    let second = main_file.to_chat_string();
    assert_eq!(first, second);
}

#[test]
fn format_metrics_csv_has_header() {
    let parser = TreeSitterParser::new().unwrap();
    let (main, _) = parse_lenient(&parser, &make_chat(&[("CHI", "a big dog .")]));
    let (gold, _) = parse_lenient(&parser, &make_chat(&[("CHI", "a dog .")]));
    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;
    let csv = CompareMetricsCsvTable::from_metrics(&metrics)
        .expect("table")
        .to_csv_string()
        .expect("csv");
    assert!(csv.starts_with("metric,value\n"));
    assert!(csv.contains("wer,0.5000"));
}

#[test]
fn inject_comparison_rejects_empty_compare_tokens() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello .")]);
    let (mut main_file, _) = parse_lenient(&parser, &main);

    let utterances = vec![UtteranceComparison {
        utterance_index: 0,
        speaker: "CHI".to_string(),
        tokens: vec![CompareToken {
            text: String::new(),
            pos: Some("INTJ".to_string()),
            status: CompareStatus::Match,
        }],
    }];

    let err =
        inject_comparison(&mut main_file, &utterances).expect_err("should reject empty token");
    assert!(err.to_string().contains("empty content"));
}

#[test]
fn compare_uses_mor_pos_for_xsmor_output() {
    // Both files carry %mor with the same POS tags, so attribution direction
    // (main vs gold) is invisible. This test verifies POS extraction from
    // %mor in general; see compare_attributes_gold_pos_to_matches for the
    // BA2-parity test that pins gold-side attribution explicitly.
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(
        XsmorTierContent::try_from(&result.main_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "INTJ NOUN"
    );
    assert_eq!(result.metrics.pos_counts()["INTJ"].matches, 1);
    assert_eq!(result.metrics.pos_counts()["NOUN"].matches, 1);
}

/// BA2 attributes the gold-side form's POS to every Match and
/// ExtraReference, not the main-side form's (compare.py:540-550, via
/// `_get_pos(gold_form)`). When the two transcripts have a %mor disagreement
/// on the same matched word, which is the entire point of running
/// `compare`: the xsmor tier and pos_counts must reflect the gold-standard
/// POS so the reviewer can see the gold tag the transcriber missed.
#[test]
fn compare_attributes_gold_pos_to_matches() {
    let parser = TreeSitterParser::new().unwrap();
    // Main %mor: noun|hello adj|world, the disagreed POS tags.
    // Gold %mor: intj|hello noun|world, the gold-standard reference.
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tnoun|hello adj|world .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(
        XsmorTierContent::try_from(&result.main_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "INTJ NOUN",
        "matches should carry gold-side POS (INTJ, NOUN) per BA2 \
         compare.py:540-550, not main-side POS (NOUN, ADJ)",
    );
    assert_eq!(result.metrics.pos_counts()["INTJ"].matches, 1);
    assert_eq!(result.metrics.pos_counts()["NOUN"].matches, 1);
    assert!(
        !result.metrics.pos_counts().contains_key("ADJ"),
        "main-side ADJ should not appear in match counts",
    );
}

/// The ORDINARY compare run: a morphotagged main transcript against a gold
/// companion read off disk, which carries no `%mor` at all.
///
/// RED FIRST (2026-09-16): every matched word used to report the literal `?`,
/// because the rule read the gold form's tag and an untagged gold side has
/// none. The `%xsmor` tier came out as a row of question marks and the entire
/// per-POS breakdown was one `?` bucket, so every metric derived from it said
/// nothing at all. This is compare's normal case, not a corner of it: nothing
/// on the released path morphotags the gold companion.
///
/// batchalign2 does the same thing and is deliberately not matched here: its
/// `_get_pos` returns `"?"` for a form carrying no morphology and it has no
/// file-level notion of a tagged side (`d8bb0cd0`, compare.py:544-588).
#[test]
fn compare_reports_the_main_tag_for_matches_when_the_gold_side_is_untagged() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
    // No `%mor`: exactly what a gold companion looks like.
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);

    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(
        XsmorTierContent::try_from(&result.main_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "INTJ NOUN",
        "an untagged gold side must not turn every matched word into `?`",
    );
    assert_eq!(result.metrics.pos_counts()["INTJ"].matches, 1);
    assert_eq!(result.metrics.pos_counts()["NOUN"].matches, 1);
    assert!(
        !result.metrics.pos_counts().contains_key("?"),
        "no matched word may land in the unknown bucket when a tag exists for it",
    );
}

/// The other half of the same rule, and the reason it is one rule rather than
/// "use the main tag whenever the gold has none".
///
/// A deletion is a gold word the main transcript does not contain, so there is
/// no main word to have been tagged and no tag anywhere in either document. It
/// still reports `?`, which is the honest answer. Asserted beside the test
/// above because an implementation that borrowed the neighbouring main tag for
/// deletions too would pass that one and fabricate here.
#[test]
fn compare_still_reports_an_unknown_tag_for_a_deletion_on_an_untagged_gold() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello .\n%mor:\tintj|hello .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);

    assert_eq!(result.metrics.matches(), 1);
    assert_eq!(result.metrics.deletions(), 1);
    assert_eq!(result.metrics.pos_counts()["INTJ"].matches, 1);
    assert_eq!(
        result.metrics.pos_counts()["?"].deletions,
        1,
        "a word neither document tagged reports no tag",
    );
}

#[test]
fn compare_ignores_pos_punct_even_when_surface_is_not_punctuation() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello comma .\n%mor:\tintj|hello PUNCT|comma .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 1);
    assert_eq!(result.metrics.insertions(), 0);
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.wer(), 0.0);
}

/// A reordered utterance costs WER but not `cwer`.
///
/// This pinned the old rotation step, which cyclically re-phased the chosen
/// window and so scored "world hello" against "hello world" as two matches.
/// That was right for a WINDOW, whose start offset was an artifact of the
/// search, and wrong for a whole utterance, where the order is the data: it let
/// a genuinely scrambled utterance score perfectly. Compare now aligns without
/// rotating, so the displacement is visible.
///
/// It is exactly what `cwer` exists to separate. Both words were recognised
/// correctly and merely placed wrongly, so `cwer` is 0 while `wer` is 1.0. The
/// pair says "the ASR heard this fine, the ordering is what went wrong".
#[test]
fn reordered_utterance_costs_wer_but_not_cwer() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tworld hello .\n%mor:\tnoun|world intj|hello .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 1, "one word survives in place");
    assert_eq!(result.metrics.insertions(), 1);
    assert_eq!(result.metrics.deletions(), 1);
    assert_eq!(result.metrics.wer(), 1.0);
    assert_eq!(
        result.metrics.cwer(),
        0.0,
        "the displaced word cancels: right word, wrong position"
    );
    // The replay reads as what actually happened: an inserted "world" ahead of
    // the matched "hello", and the reference "world" missing after it.
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "+world hello -world ."
    );
}

#[test]
fn batchalign2_master_simple_gold_projection_shape() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello big world .\n%mor:\tintj|hello adj|big noun|world .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world today .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "hello +big world -today ."
    );
    // The gold companion carries no %mor, so a matched word reports the tag of
    // the main word it matched, and only the deletion, which no document
    // tagged, reports `?`.
    //
    // This read `"? +ADJ ? -? ."` until 2026-09-16, as strict parity with BA2's
    // `_get_pos(gold_form)`. The parity expectation was the thing that was
    // wrong: measured that day at `TalkBank/batchalign2` `master` `d8bb0cd0`,
    // blob `37270401`, compare.py lines 355-359 and 544-588, `_get_pos` returns
    // the literal `"?"` for a form carrying no morphology and the file has no
    // notion of a tagged gold side anywhere. Reproducing it turned every
    // matched word into `?` on compare's ordinary input, which is a tagged main
    // transcript against an untagged gold companion. See `super::pos`.
    assert_eq!(
        XsmorTierContent::try_from(&result.gold_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "INTJ +ADJ NOUN -? ."
    );
    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(result.metrics.insertions(), 1);
    assert_eq!(result.metrics.deletions(), 1);
    assert!((result.metrics.wer() - (2.0 / 3.0)).abs() < 0.001);
    assert_eq!(result.metrics.pos_counts()["ADJ"].insertions, 1);
    assert_eq!(result.metrics.pos_counts()["?"].deletions, 1);
    assert_eq!(result.metrics.pos_counts()["INTJ"].matches, 1);
    assert_eq!(result.metrics.pos_counts()["NOUN"].matches, 1);
}

/// Repeated main tokens before the matched stretch are insertions.
///
/// Named `..._ignores_skipped_prefix_tokens` until 2026-07-30, when it pinned
/// the opposite: the two leading "dog"s scored as nothing at all. They are
/// hypothesis words with no reference counterpart, so they are insertions.
#[test]
fn repeated_leading_main_tokens_are_insertions() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tdog dog the dog .\n%mor:\tnoun|dog noun|dog det|the noun|dog .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tthe dog .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 2);
    assert_eq!(
        result.metrics.insertions(),
        2,
        "the two leading \"dog\" tokens"
    );
    assert_eq!(result.metrics.deletions(), 0);
    assert_eq!(result.metrics.wer(), 1.0, "2 insertions over 2 gold words");
    // `%xsrep` is the COMPARISON replay, not a copy of the gold line, so the
    // recovered insertions appear in it with the `+` marker. They were absent
    // before only because they were never emitted at all.
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "+dog +dog the dog ."
    );
    // The gold companion carries no %mor, so each matched word reports the tag
    // of the main word it matched: `the` is DET and `dog` is NOUN.
    //
    // This read `"+NOUN +NOUN ? ? ."` until 2026-09-16, and its comment
    // recorded that BA3 had once lifted DET/NOUN from main and was changed away
    // from that to match BA2. The change went the wrong way: BA2 prints `?`
    // because `_get_pos` has no file-level notion of a tagged gold side
    // (`d8bb0cd0`, compare.py:544-588), not because `?` is a better answer for
    // a word whose tag BA3 is holding.
    assert_eq!(
        XsmorTierContent::try_from(&result.gold_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "+NOUN +NOUN DET NOUN ."
    );
}

#[test]
fn batchalign2_master_multi_utterance_compare_metrics() {
    let parser = TreeSitterParser::new().unwrap();
    let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tone fish two fish .\n%mor:\tnum|one noun|fish num|two noun|fish .\n*CHI:\tred fish blue fish .\n%mor:\tadj|red noun|fish adj|blue noun|fish .\n@End\n";
    let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tone fish fish .\n*CHI:\tred fish green fish .\n@End\n";
    let (main_file, _) = parse_lenient(&parser, main);
    let (gold_file, _) = parse_lenient(&parser, gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[0])
            .expect("xsrep tier")
            .to_chat_string(),
        "one fish +two fish ."
    );
    // Neither gold utterance carries %mor, so matches report the tag of the
    // main word they matched and only the deletion, which no document tagged,
    // reports `?`. These two assertions read `"? ? +NUM ? ."` and
    // `"? ? -? +ADJ ? ."` until 2026-09-16, as BA2 parity; see
    // `batchalign2_master_simple_gold_projection_shape` for the measurement
    // that retired that expectation.
    assert_eq!(
        XsmorTierContent::try_from(&result.gold_utterances[0])
            .expect("xsmor tier")
            .to_chat_string(),
        "NUM NOUN +NUM NOUN ."
    );
    assert_eq!(
        XsrepTierContent::try_from(&result.gold_utterances[1])
            .expect("xsrep tier")
            .to_chat_string(),
        "red fish -green +blue fish ."
    );
    assert_eq!(
        XsmorTierContent::try_from(&result.gold_utterances[1])
            .expect("xsmor tier")
            .to_chat_string(),
        "ADJ NOUN -? +ADJ NOUN ."
    );
    assert_eq!(result.metrics.matches(), 6);
    assert_eq!(result.metrics.insertions(), 2);
    assert_eq!(result.metrics.deletions(), 1);
    assert!((result.metrics.wer() - (3.0 / 7.0)).abs() < 0.001);
}

#[test]
fn gold_anchored_projection_attaches_diff_to_gold_transcript() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[("CHI", "hello big world .")]);
    let gold = make_chat(&[("CHI", "hello world today .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (mut gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    inject_comparison(&mut gold_file, &result.gold_utterances).expect("inject comparison");

    let serialized = gold_file.to_chat_string();
    assert!(serialized.contains("*CHI:\thello world today ."));
    assert!(serialized.contains("%xsrep:\thello +big world -today"));
    assert!(serialized.contains("%xsmor:"));
}

/// With a COMPLETE gold, a main utterance no gold maps to is all insertions.
///
/// The caller states what the gold claims to cover, because only the caller
/// knows. A complete gold is a full reference for this transcript, so main
/// material it does not account for is material the system produced and the
/// reference does not contain: insertions, by the definition of WER.
#[test]
fn unmapped_main_utterance_is_insertions_under_complete_gold() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[
        ("CHI", "the cat sat ."),
        ("MOT", "entirely unrelated invented material ."),
    ]);
    let gold = make_chat(&[("CHI", "the cat sat .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Complete);
    assert_eq!(result.metrics.matches(), 3);
    assert_eq!(
        result.metrics.insertions(),
        4,
        "the four words of the unmapped MOT utterance"
    );
    assert_eq!(
        result.metrics.total_main_words(),
        7,
        "the main-word total is the whole transcript"
    );
}

/// With a PARTIAL gold, the same utterance is correctly ignored.
///
/// A gold that deliberately covers one slice of a recording says nothing about
/// the rest, so charging the rest as insertions would penalise the system for
/// transcribing material the reference never claimed. The same input as the
/// test above, and the opposite right answer: which is exactly why this is a
/// caller's declaration and not a default.
#[test]
fn unmapped_main_utterance_is_ignored_under_partial_gold() {
    let parser = TreeSitterParser::new().unwrap();
    let main = make_chat(&[
        ("CHI", "the cat sat ."),
        ("MOT", "material outside the sampled slice ."),
    ]);
    let gold = make_chat(&[("CHI", "the cat sat .")]);
    let (main_file, _) = parse_lenient(&parser, &main);
    let (gold_file, _) = parse_lenient(&parser, &gold);

    let result = compare(&main_file, &gold_file, GoldCoverage::Partial);
    assert_eq!(result.metrics.matches(), 3);
    assert_eq!(result.metrics.insertions(), 0);
    assert_eq!(result.metrics.wer(), 0.0, "the covered slice is perfect");
}
