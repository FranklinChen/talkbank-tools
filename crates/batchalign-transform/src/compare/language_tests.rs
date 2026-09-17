//! Language scoring through the real comparison: substitutions counted once,
//! errors attributed to the language of the reference, utterance and word
//! language agreement, and code-switch detection.
//!
//! Every test parses CHAT with the real parser and calls [`compare`], because
//! the language a word is scored in comes from the parsed tree (a precode, a
//! word's own `@s`, an enclosing `<...> [@s]` span, the `@Languages` header),
//! and a test that built the tallies by hand would skip exactly the resolution
//! these metrics exist to report.

use super::tests::{chat_text, parse_lenient};
use super::*;
use talkbank_model::{ChatFile, WriteChat};
use talkbank_parser::TreeSitterParser;

/// A parsed CHAT file declaring `languages` with the given utterance lines.
fn chat(languages: &str, utterances: &[(&str, &str)]) -> ChatFile {
    let parser = TreeSitterParser::new().expect("parser");
    parse_lenient(&parser, &chat_text(languages, utterances)).0
}

/// The error counts for one gold language.
fn counts<'s>(scores: &'s LanguageScores, code: &str) -> Option<&'s LanguageErrorCounts> {
    let bucket = language(code);
    scores
        .by_language()
        .find(|(language, _)| **language == bucket)
        .map(|(_, counts)| counts)
}

fn language(code: &str) -> LanguageBucket {
    LanguageBucket::Language(
        talkbank_model::model::LanguageCode::new(code).expect("a valid ISO 639-3 code"),
    )
}

/// A substituted word is one error, not two.
///
/// The legacy `wer` is `(insertions + deletions) / gold`, so the hypothesis
/// "the dog sat" against "the cat sat" scores 2/3: `dog` is charged as an
/// insertion and `cat` as a deletion. It is kept unchanged for comparability
/// with every number already published from it, and the substitution-paired
/// rate sits beside it.
#[test]
fn a_substituted_word_is_one_error_in_the_substitution_paired_rate() {
    let main = chat("eng", &[("CHI", "the dog sat .")]);
    let gold = chat("eng", &[("CHI", "the cat sat .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.insertions(), 1);
    assert_eq!(metrics.deletions(), 1);
    assert!(
        (metrics.wer() - 2.0 / 3.0).abs() < 1e-9,
        "legacy wer unchanged"
    );
    assert_eq!(metrics.languages().substitutions(), 1);
    assert_eq!(
        metrics.wer_with_substitutions(),
        Rate::Defined(1.0 / 3.0),
        "one substituted word out of three reference words"
    );
}

/// Pairing happens only between the same two matches.
///
/// An insertion before `the` and a deletion after it are not one wrong word:
/// a match separates them, so they are an insertion and a deletion.
#[test]
fn an_insertion_and_a_deletion_either_side_of_a_match_are_not_paired() {
    let main = chat("eng", &[("CHI", "dog the sat .")]);
    let gold = chat("eng", &[("CHI", "the cat sat .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.languages().substitutions(), 0);
    assert_eq!(metrics.wer_with_substitutions(), Rate::Defined(2.0 / 3.0));
}

/// Errors are charged to the language of the reference words they occur in.
///
/// An English-only recognizer hears a Spanish utterance, keeps `yo` and turns
/// `quiero agua` into three English words. Both substitutions and the leftover
/// insertion belong to Spanish: the reference says that stretch was Spanish,
/// whatever language the hypothesis wrote it in. Charging the insertion to the
/// hypothesis word's language would put a Spanish recognition failure into the
/// English error rate.
#[test]
fn errors_are_charged_to_the_language_of_the_reference() {
    let main = chat("eng", &[("MOT", "yo can i go .")]);
    let gold = chat("eng, spa", &[("MOT", "[- spa] yo quiero agua .")]);

    let scores = compare(&main, &gold, GoldCoverage::Complete)
        .metrics
        .languages()
        .clone();

    let spanish = counts(&scores, "spa").expect("Spanish reference words were scored");
    assert_eq!(spanish.matches(), 1);
    assert_eq!(spanish.substitutions(), 2);
    assert_eq!(spanish.deletions(), 0);
    assert_eq!(spanish.insertions(), 1);
    assert_eq!(spanish.gold_words(), 3);
    assert_eq!(spanish.wer_with_substitutions(), Rate::Defined(1.0));
    assert!(
        counts(&scores, "eng").is_none(),
        "nothing in the reference is English, so nothing is charged to English"
    );
}

/// An insertion is charged to a gold language through its place in the
/// alignment, never through the main transcript's utterances or language.
///
/// `can i go` is a main utterance matching no gold word, after the last gold
/// word, so its words belong to that word, `agua`, and are charged to Spanish.
/// Splitting the main transcript differently must not change that, and the
/// main transcript's own language is exactly the evidence under test in
/// code-switched speech, so it is never consulted.
#[test]
fn trailing_insertions_belong_to_the_last_gold_word_whatever_utterance_holds_them() {
    let main = chat(
        "eng, spa",
        &[("MOT", "[- spa] quiero agua ."), ("MOT", "can i go .")],
    );
    let gold = chat("eng, spa", &[("MOT", "[- spa] quiero agua .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    let spanish = counts(metrics.languages(), "spa").expect("Spanish reference words were scored");
    assert_eq!(spanish.matches(), 2);
    assert_eq!(spanish.insertions(), 3);
    assert_eq!(metrics.languages().unattributed_insertions(), 0);
    assert_eq!(metrics.wer_with_substitutions(), Rate::Defined(3.0 / 2.0));
}

/// Only a gold with no compared words leaves insertions unattributed: there
/// is no gold language anywhere to charge them to.
#[test]
fn insertions_are_unattributed_only_against_a_gold_with_no_words() {
    let main = chat("eng", &[("MOT", "can i go .")]);
    let gold = chat("eng, spa", &[("MOT", ".")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.languages().unattributed_insertions(), 3);
    assert!(metrics.languages().by_language().next().is_none());
    assert_eq!(metrics.wer_with_substitutions(), Rate::NoDenominator);
}

/// Unmatched hypothesis words opposite unmatched reference words are
/// substitutions, charged to the reference word's language.
///
/// Nothing in `can i go` matches `quiero agua`, and standard WER scores that
/// stretch as two substitutions and one insertion, not as two deletions and
/// three insertions. All three errors belong to Spanish, the language of the
/// reference words around them.
#[test]
fn an_unmatched_stretch_opposite_reference_words_scores_as_substitutions() {
    let main = chat("eng", &[("MOT", "can i go .")]);
    let gold = chat("eng, spa", &[("MOT", "[- spa] quiero agua .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    let spanish = counts(metrics.languages(), "spa").expect("Spanish reference words were scored");
    assert_eq!(spanish.substitutions(), 2);
    assert_eq!(spanish.deletions(), 0);
    assert_eq!(spanish.insertions(), 1);
    assert_eq!(metrics.languages().unattributed_insertions(), 0);
    assert_eq!(metrics.wer_with_substitutions(), Rate::Defined(3.0 / 2.0));
}

/// An insertion belongs to the gold word the alignment reaches next.
///
/// `there` falls between the Spanish gold utterance and the English one. One
/// owner decides both where the insertion is shown and which language its
/// error is charged to, so the two can never disagree: here it is the English
/// `you`, the next gold word, and the insertion appears at the start of that
/// utterance. Only an insertion after the last gold word falls back to the
/// word before it.
#[test]
fn an_insertion_belongs_to_the_gold_word_reached_next() {
    let main = chat("eng, spa", &[("MOT", "hola there you .")]);
    let gold = chat("eng, spa", &[("MOT", "[- spa] hola ."), ("MOT", "you .")]);

    let result = compare(&main, &gold, GoldCoverage::Complete);

    let english = counts(result.metrics.languages(), "eng").expect("English reference words");
    assert_eq!(english.insertions(), 1);
    let spanish = counts(result.metrics.languages(), "spa").expect("Spanish reference words");
    assert_eq!(spanish.insertions(), 0);
    let gold_view = |index: usize| {
        XsrepTierContent::try_from(&result.gold_utterances[index])
            .expect("xsrep tier")
            .to_chat_string()
    };
    assert_eq!(gold_view(0), "hola .");
    assert_eq!(gold_view(1), "+there you .");
}

/// A gold utterance whose matched words fall evenly in two main utterances is
/// placed on the earlier one.
///
/// Placement only decides utterance language agreement, which is why the two
/// main utterances here are in different languages: the Spanish gold utterance
/// agrees with the Spanish main utterance it is placed on, and would disagree
/// had the tie reached forward to the English one.
#[test]
fn a_gold_utterance_matched_evenly_by_two_is_placed_on_the_earlier() {
    let main = chat(
        "eng, spa",
        &[("MOT", "[- spa] uno dos ."), ("MOT", "three four .")],
    );
    let gold = chat("eng, spa", &[("MOT", "[- spa] uno dos three four .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    let placed = metrics.languages().utterances().placed();
    assert_eq!(placed.scored(), 1);
    assert_eq!(placed.agreeing(), 1);
}

/// Name collapsing is for English-only comparisons.
///
/// The WER normalizer replaces about 6,700 names with `name`, and that list
/// holds everyday Spanish words: `linda` and `clara` are both on it. Applied
/// to Spanish, "qué linda" and "qué clara" became "qué name" twice and scored
/// as a perfect match. A gold transcript declaring any language other than
/// English turns the rule off; the English control shows it still applies
/// where it was meant to.
#[test]
fn name_collapsing_applies_only_when_the_gold_declares_english_alone() {
    let spanish_main = chat("spa", &[("MOT", "qué clara .")]);
    let spanish_gold = chat("spa, eng", &[("MOT", "qué linda .")]);
    let spanish = compare(&spanish_main, &spanish_gold, GoldCoverage::Complete).metrics;
    assert_eq!(spanish.matches(), 1, "only qué matches");
    assert_eq!(spanish.languages().substitutions(), 1);

    let english_main = chat("eng", &[("MOT", "she is clara .")]);
    let english_gold = chat("eng", &[("MOT", "she is linda .")]);
    let english = compare(&english_main, &english_gold, GoldCoverage::Complete).metrics;
    assert_eq!(
        english.matches(),
        3,
        "in English both names still normalize to the same token"
    );
}

/// A word's language comes from what governs it, including a span.
///
/// `<más agua> [@s]` marks both words as the other language; neither word
/// carries its own `@s`. Reading only a word's own marker scores `agua` as
/// English, which is the trap chatter's resolver exists to close. The
/// hypothesis marks `más` itself and leaves `agua` unmarked, so it switched one
/// of the two gold switches.
#[test]
fn word_language_and_switches_follow_spans_and_own_markers() {
    let main = chat("eng, spa", &[("CHI", "we want más@s agua please .")]);
    let gold = chat("eng, spa", &[("CHI", "we want <más agua> [@s] please .")]);

    let scores = compare(&main, &gold, GoldCoverage::Complete)
        .metrics
        .languages()
        .clone();

    assert_eq!(scores.words().scored(), 5);
    assert_eq!(scores.words().agreeing(), 4);
    assert!(
        scores
            .words()
            .pairs()
            .any(|(pair, count)| pair.gold == language("spa")
                && pair.main == language("eng")
                && count == 1),
        "agua is Spanish in gold and English in main"
    );

    let switches = scores.switches();
    assert_eq!(switches.both_switched(), 1);
    assert_eq!(switches.gold_only(), 1);
    assert_eq!(switches.main_only(), 0);
    assert_eq!(switches.neither(), 3);
    assert_eq!(switches.precision(), Rate::Defined(1.0));
    assert_eq!(switches.recall(), Rate::Defined(0.5));
}

/// Utterance language is scored on placed utterances, and an utterance that
/// could not be placed is counted apart rather than as a disagreement.
#[test]
fn utterance_language_agreement_and_unplaced_utterances() {
    let main = chat(
        "eng, spa",
        &[("MOT", "we want water ."), ("MOT", "[- spa] quiero agua .")],
    );
    let gold = chat(
        "eng, spa",
        &[
            ("MOT", "we want water ."),
            ("MOT", "[- spa] quiero agua ."),
            ("MOT", "[- spa] muchas gracias ."),
        ],
    );

    let languages = compare(&main, &gold, GoldCoverage::Complete)
        .metrics
        .languages()
        .clone();
    let utterances = languages.utterances();

    assert_eq!(utterances.placed().scored(), 2);
    assert_eq!(utterances.placed().agreeing(), 2);
    assert_eq!(utterances.placed().accuracy(), Rate::Defined(1.0));
    assert_eq!(utterances.unplaced().get(&language("spa")), Some(&1));
}

/// The CSV carries every new count, keyed so a corpus roll-up can sum rows,
/// and says `NA` for a rate with nothing to divide by rather than printing 0.
#[test]
fn language_scores_reach_the_metrics_csv() {
    let main = chat("eng", &[("MOT", "yo can i go .")]);
    let gold = chat("eng, spa", &[("MOT", "[- spa] yo quiero agua .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;
    let csv = format_metrics_csv(&metrics).expect("csv");

    for expected in [
        "substitutions,2",
        "wer_with_substitutions,1.0000",
        "unattributed_insertions,0",
        "language:spa:gold_words,3",
        "language:spa:matches,1",
        "language:spa:substitutions,2",
        "language:spa:deletions,0",
        "language:spa:insertions,1",
        "language:spa:wer_with_substitutions,1.0000",
        "utterance_language:scored,1",
        "utterance_language:agreeing,0",
        "utterance_language:indeterminate,0",
        "utterance_language:accuracy,0.0000",
        "utterance_language:gold=spa:main=eng,1",
        "word_language:scored,1",
        "word_language:gold=spa:main=eng,1",
        "switch:both,0",
        "switch:precision,NA",
        "switch:recall,NA",
    ] {
        assert!(
            csv.lines().any(|line| line == expected),
            "missing CSV row {expected:?} in:\n{csv}"
        );
    }
}

/// A word the recognizer got right is a match whatever language it labeled it.
///
/// An English-only recognizer writes Spanish `qué linda` into an English
/// transcript. Normalizing each side by its own labels would collapse the
/// hypothesis `linda` to `name` and keep the gold `linda`, charging a correctly
/// recognized word as a substitution. Normalization is chosen once, from the
/// gold, and applied to both sides; the wrong label is still reported, as
/// utterance language disagreement.
#[test]
fn a_correctly_recognized_word_with_the_wrong_language_label_still_matches() {
    let main = chat("eng", &[("MOT", "qué linda .")]);
    let gold = chat("eng, spa", &[("MOT", "[- spa] qué linda .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;

    assert_eq!(metrics.matches(), 2, "both words were recognized");
    assert_eq!(metrics.wer_with_substitutions(), Rate::Defined(0.0));
    let utterances = metrics.languages().utterances().placed();
    assert_eq!(utterances.agreeing(), 0);
    assert_eq!(utterances.accuracy(), Rate::Defined(0.0));
}

/// Two transcripts that both fail to say what language an utterance is in
/// have not agreed.
///
/// With no `@Languages` header and no precode, both utterances resolve to no
/// language. That pair is indeterminate: counted, but neither agreeing nor
/// differing, so accuracy has nothing to divide by rather than reading 1.0.
#[test]
fn unresolved_languages_are_indeterminate_not_agreement() {
    let without_languages = |text: &str| {
        let parser = TreeSitterParser::new().expect("parser");
        let with_header = chat_text("eng", &[("MOT", text)]);
        let without_header: String = with_header
            .lines()
            .filter(|line| !line.starts_with("@Languages:"))
            .collect::<Vec<_>>()
            .join("\n");
        parse_lenient(&parser, &without_header).0
    };
    let main = without_languages("we want water .");
    let gold = without_languages("we want water .");

    let languages = compare(&main, &gold, GoldCoverage::Complete)
        .metrics
        .languages()
        .clone();
    let placed = languages.utterances().placed();

    assert_eq!(placed.scored(), 1);
    assert_eq!(placed.indeterminate(), 1);
    assert_eq!(placed.agreeing(), 0);
    assert_eq!(placed.accuracy(), Rate::NoDenominator);
}

/// A word marked as a mix keeps its languages, in CHAT's own spelling.
#[test]
fn a_mixed_word_keeps_its_languages() {
    let main = chat("eng, spa", &[("CHI", "we want tortilla .")]);
    let gold = chat("eng, spa", &[("CHI", "we want tortilla@s:eng+spa .")]);

    let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;
    let csv = format_metrics_csv(&metrics).expect("csv");

    assert!(
        csv.lines()
            .any(|line| line == "language:eng+spa:gold_words,1"),
        "missing the mixed-language row in:\n{csv}"
    );
}
