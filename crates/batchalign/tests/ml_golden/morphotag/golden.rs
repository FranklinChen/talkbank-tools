use crate::common::assert_completed_without_errors;
use batchalign::api::ReleasedCommand;
use batchalign::options::{CommandOptions, CommonOptions, MorphotagOptions};
use batchalign::worker::InferTask;

use crate::ml_golden::golden::fixtures::{
    ENG_DISFLUENCY, ENG_MULTI_UTT, ENG_RETOKENIZE, ENG_SIMPLE, ITA_MULTI_WORD_UTTERANCES,
    ITA_SINGLE_WORD_UTTERANCES, SPA_SIMPLE,
};
use crate::ml_golden::golden::helpers::{
    assert_golden_snapshot, find_mor_line_for, has_gra_tier, has_mor_tier, parse_output,
    pin_provenance_timestamps, require_managed_session_warmed,
};

fn morphotag_options(override_media_cache: bool, retokenize: bool) -> CommandOptions {
    CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache,
            ..CommonOptions::default()
        },
        retokenize,

        ..Default::default()
    })
}

/// Guards: English subject, verb and object MOR/GRA bytes for ENG_SIMPLE.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove English subject, verb and object MOR/GRA bytes for ENG_SIMPLE from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_eng_simple() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "eng",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_simple.cha",
            ENG_SIMPLE,
            morphotag_options(true, false),
        )
        .await;

    assert_completed_without_errors("morphotag_eng_simple", &info, &results);
    let output = results[0].content.as_text().expect("text command result");
    let file = parse_output(output, "morphotag_eng_simple");
    assert!(has_mor_tier(&file), "no %mor tier in:\n{output}");
    assert!(has_gra_tier(&file), "no %gra tier in:\n{output}");
    assert_golden_snapshot!("morphotag_eng_simple", output);
}

/// Guards: utterance ordering and English MOR/GRA bytes for ENG_MULTI_UTT.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove utterance ordering and English MOR/GRA bytes for ENG_MULTI_UTT from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_eng_multi_utt() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "eng",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_multi_utt.cha",
            ENG_MULTI_UTT,
            morphotag_options(true, false),
        )
        .await;

    assert_completed_without_errors("morphotag_eng_multi_utt", &info, &results);
    assert_golden_snapshot!(
        "morphotag_eng_multi_utt",
        results[0].content.as_text().expect("text command result")
    );
}

/// Guards: dictionary lemmas and normalized CHAT equality across repeated mixed-length contextual analyses.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove dictionary lemmas and normalized CHAT equality across repeated mixed-length contextual analyses from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_repeated_analysis() {
    const TARGET: &str = "she's gone home .";
    const PREFIX: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n";
    let mixed = format!(
        "{PREFIX}*PAR:\t{TARGET}\n*PAR:\tthat's her book .\n*PAR:\tthe housewives made deep dives .\n@End\n"
    );
    let alone = format!("{PREFIX}*PAR:\t{TARGET}\n@End\n");
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "eng",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info1, results1) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "lemma_repeated.cha",
            &mixed,
            morphotag_options(false, false),
        )
        .await;
    assert_completed_without_errors("morphotag_first_analysis", &info1, &results1);

    let (info2, results2) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "lemma_repeated.cha",
            &mixed,
            morphotag_options(false, false),
        )
        .await;
    assert_completed_without_errors("morphotag_repeated_analysis", &info2, &results2);

    // The two runs may stamp different seconds; everything else, the stamp's
    // fields included, must be identical.
    assert_eq!(
        pin_provenance_timestamps(results1[0].content.as_text().expect("text command result")),
        pin_provenance_timestamps(results2[0].content.as_text().expect("text command result"))
    );
    let first = results1[0].content.as_text().expect("text command result");
    let lexicon_line = find_mor_line_for(first, "the housewives made deep dives")
        .expect("dictionary probe MOR tier");
    assert!(
        lexicon_line.contains("noun|housewife-Plur"),
        "{lexicon_line}"
    );
    assert!(lexicon_line.contains("noun|dive-Plur"), "{lexicon_line}");
    let (alone_info, alone_results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "lemma_alone.cha",
            &alone,
            morphotag_options(false, false),
        )
        .await;
    assert_completed_without_errors("morphotag_context_alone", &alone_info, &alone_results);
    let alone_output = alone_results[0]
        .content
        .as_text()
        .expect("text command result");
    assert_eq!(
        find_mor_line_for(first, TARGET).expect("mixed target MOR tier"),
        find_mor_line_for(alone_output, TARGET).expect("single target MOR tier"),
        "contextual lemma output must be independent of other utterance lengths"
    );
}

/// Guards: Spanish inflection and dependency bytes for SPA_SIMPLE.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove Spanish inflection and dependency bytes for SPA_SIMPLE from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_spa_simple() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "spa",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "spa",
            "spa_simple.cha",
            SPA_SIMPLE,
            morphotag_options(true, false),
        )
        .await;

    if info.status == batchalign::api::JobStatus::Failed {
        panic!("PREREQUISITE MISSING: Spanish morphotag failed (model likely not downloaded)")
    }

    assert_completed_without_errors("morphotag_spa_simple", &info, &results);
    let output = results[0].content.as_text().expect("text command result");
    let file = parse_output(output, "morphotag_spa_simple");
    assert!(has_mor_tier(&file), "no %mor tier in:\n{output}");
    assert!(has_gra_tier(&file), "no %gra tier in:\n{output}");
    assert_golden_snapshot!(
        "morphotag_spa_simple",
        results[0].content.as_text().expect("text command result")
    );
}

/// Guards: English contraction splitting and dependent-tier bytes in retokenize mode.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove English contraction splitting and dependent-tier bytes in retokenize mode from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_retokenize_eng() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "eng",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_retokenize.cha",
            ENG_RETOKENIZE,
            morphotag_options(true, true),
        )
        .await;

    assert_completed_without_errors("morphotag_retokenize_eng", &info, &results);
    let output = results[0].content.as_text().expect("text command result");
    let file = parse_output(output, "morphotag_retokenize_eng");
    assert!(has_mor_tier(&file), "no %mor tier in:\n{output}");
    assert!(has_gra_tier(&file), "no %gra tier in:\n{output}");
    assert_golden_snapshot!(
        "morphotag_retokenize_eng",
        results[0].content.as_text().expect("text command result")
    );
}

/// Guards: the explicit that subject plus be copula after mm-hmm disfluency.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove the explicit that subject plus be copula after mm-hmm disfluency from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn morphotag_disfluency_preserves_thats_subject_and_copula() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "eng",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_disfluency.cha",
            ENG_DISFLUENCY,
            morphotag_options(true, false),
        )
        .await;

    assert_completed_without_errors(
        "morphotag_disfluency_preserves_thats_subject_and_copula",
        &info,
        &results,
    );
    assert_eq!(results.len(), 1);

    let mmhmm_mor = find_mor_line_for(
        results[0].content.as_text().expect("text command result"),
        "mm-hmm that's right",
    )
    .expect("expected %mor line for mm-hmm that's right");
    assert!(
        mmhmm_mor.contains("pron|that-Dem~aux|be-Fin-Ind-Pres-S3"),
        "expected explicit subject+copula analysis for \"that's right\", got: {mmhmm_mor}"
    );
    assert!(
        !mmhmm_mor.contains("aux|that-Fin-Ind-Pres-S3"),
        "unexpected collapsed auxiliary-only analysis for \"that's right\": {mmhmm_mor}"
    );
}

/// Italian single-word utterances must not be shredded into invented verbs.
///
/// Stanza's Italian MWT treats multi-word tokens as an open class and, on
/// utterances with no syntactic context, split roughly a third of real corpus
/// words into a nonexistent verb plus a clitic: `attenzione` -> *attenzi* +
/// *ne*, `cavallo` -> *cava* + *lo*, `gallina` -> *galli* + *na* (where "na"
/// is not even a clitic), `mucche` -> *mu* + *cce* + *he*. That corrupted
/// `%mor` with verbs that do not exist in Italian, not merely the `%gra`
/// relation, and it destroyed plural lemmas along the way.
///
/// Measured 2026-07-28 on stanza 1.13.0 against real single-word utterances
/// pulled from the CHILDES Italian corpora. Fixed by suppressing MWT expansion
/// on single-word utterances outside Italian's closed MWT classes; see
/// `batchalign/inference/_italian_mwt.py`.
///
/// The controls are the point of this test, not an afterthought. Suppressing
/// every single-word expansion trivially satisfies "no invented verbs" while
/// destroying real analysis, so the fixture also carries genuine multi-word
/// tokens that MUST still split:
///
/// * `dammelo`, `diglielo`, `giralo`, `prendilo`, `guardalo` are verb+enclitic
///   imperatives, an OPEN class. No surface pattern separates `giralo` (turn it)
///   from `cavallo` (horse): both are a verb-shaped base plus a real clitic. A
///   closed-class allowlist therefore cannot admit them, and one shipped briefly
///   that destroyed roughly 55 such utterances across the corpora.
/// * `eccolo` is `ecco`+clitic, a genuinely CLOSED class. It is also the form on
///   which a naive part-of-speech probe fails: stanza tags the unsplit `eccolo`
///   ADJ rather than VERB.
///
/// Between them these controls fail BOTH tempting cheap fixes, which is why they
/// are here.
/// Guards: Italian isolated lexical words and legitimate enclitic controls through the model.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove Italian isolated lexical words and legitimate enclitic controls through the model from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_ita_single_word_utterances_are_not_split() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "ita",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "ita",
            "ita_single_word.cha",
            ITA_SINGLE_WORD_UTTERANCES,
            morphotag_options(true, false),
        )
        .await;

    assert_completed_without_errors("morphotag_ita_single_word", &info, &results);
    let output = results[0].content.as_text().expect("text command result");
    let file = parse_output(output, "morphotag_ita_single_word");
    assert!(has_mor_tier(&file), "no %mor tier in:\n{output}");
    assert!(has_gra_tier(&file), "no %gra tier in:\n{output}");

    // Each lexical word must keep its own part of speech, with no clitic split.
    for (utterance, expected_pos, expected_lemma) in [
        ("attenzione", "noun|", "attenzione"),
        ("macchine", "noun|", "macchina"),
        ("gallina", "noun|", "gallina"),
        ("cavallo", "noun|", "cavallo"),
        ("mucche", "noun|", "mucca"),
        ("persone", "noun|", "persona"),
        ("bello", "adj|", "bello"),
    ] {
        let mor = find_mor_line_for(output, utterance)
            .unwrap_or_else(|| panic!("no %mor tier for {utterance:?}"));
        assert!(
            mor.contains(expected_pos),
            "{utterance:?} must be tagged {expected_pos:?}, got {mor:?}"
        );
        assert!(
            !mor.contains('~'),
            "{utterance:?} must not be split into a multi-word token, got {mor:?}"
        );
        assert!(
            mor.contains(expected_lemma),
            "{utterance:?} should carry lemma {expected_lemma:?} (the split \
             destroyed plural lemmas), got {mor:?}"
        );
    }

    // Controls: genuine multi-word tokens must STILL expand. These are the
    // assertions a closed-class allowlist cannot satisfy, and `eccolo` is the
    // one a bare part-of-speech probe cannot satisfy either.
    for utterance in [
        "dammelo", "diglielo", "giralo", "prendilo", "guardalo", "eccolo",
    ] {
        let mor = find_mor_line_for(output, utterance)
            .unwrap_or_else(|| panic!("no %mor tier for {utterance:?}"));
        assert!(
            mor.contains('~'),
            "{utterance:?} is a genuine multi-word token and must still split, got {mor:?}"
        );
    }
}

/// Italian multi-word utterances: genuine MWTs expand, verb forms do not.
///
/// The single-word suppression must not leak into multi-word context, where
/// stanza analyzes these same nouns correctly and where preposition+article
/// contractions carry real linguistic information.
///
/// `dai` appears twice on purpose, in both of its readings: as the 2sg verb
/// *dare* ("dai il libro a me") and as the contraction *da* + *il* ("vieni dai
/// bambini"). `hai` is subjectless because Italian is pro-drop, which is the
/// normal spoken form and the context where a mis-analysis is most likely.
/// Guards: Italian multiword utterances retaining legitimate MWT components.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove Italian multiword utterances retaining legitimate MWT components from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_morphotag_ita_multi_word_keeps_genuine_mwts() {
    let jobs = require_managed_session_warmed(
        InferTask::Morphosyntax,
        ReleasedCommand::Morphotag,
        "ita",
        "Managed session does not support morphosyntax infer",
    )
    .await;

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "ita",
            "ita_multi_word.cha",
            ITA_MULTI_WORD_UTTERANCES,
            morphotag_options(true, false),
        )
        .await;

    assert_completed_without_errors("morphotag_ita_multi_word", &info, &results);
    let output = results[0].content.as_text().expect("text command result");
    // Parsed for its side effect: parse_output asserts the output is valid CHAT.
    let _file = parse_output(output, "morphotag_ita_multi_word");

    // Preposition + article contractions must still expand.
    let contractions =
        find_mor_line_for(output, "vado").expect("no %mor tier for the contraction utterance");
    assert!(
        contractions.matches('~').count() >= 2,
        "alla and della are genuine contractions and must both expand, got {contractions:?}"
    );

    // Pro-drop 2sg `hai` is never a contraction in Italian: no `ha` + `i`.
    let hai = find_mor_line_for(output, "opinione").expect("no %mor tier for hai");
    assert!(
        hai.contains("avere"),
        "hai is 2sg of avere and must not be split into ha + i, got {hai:?}"
    );

    // Nouns in context keep their correct lemmas.
    let persone = find_mor_line_for(output, "sono").expect("no %mor tier for persone");
    assert!(
        persone.contains("persona"),
        "persone should lemmatize to persona in context, got {persone:?}"
    );

    // Limitation 3: over-splitting that survives IN context. Italian has exactly
    // three legitimate multi-word patterns (preposition+article, `ecco`+enclitic,
    // verb+enclitic); none of these lines contains one, so any `~` on them is a
    // multi-word token stanza invented. Each context was verified to trigger the
    // defect on stanza 1.13.0, so these assertions are real gates and not
    // decoration.
    for (anchor, what) in [
        ("grande", "la must stay the article `la`, never il + i"),
        ("opinione", "hai must stay 2sg of avere, never ha + i"),
        ("mozzarella", "mozzarella is a noun, never mozzar + la"),
        ("tagliatelle", "tagliatelle is a noun, never tagliate + le"),
        ("pennarello", "pennarello is a noun, never pennar + lo"),
    ] {
        let mor = find_mor_line_for(output, anchor)
            .unwrap_or_else(|| panic!("no %mor tier for the {anchor:?} utterance"));
        assert!(
            !mor.contains('~'),
            "{what}; this utterance carries no genuine multi-word token, got {mor:?}"
        );
    }

    // Defects 12 and 13: the OPPOSITE failure. Stanza declines to split a
    // genuine imperative+enclitic and invents a verb for the whole surface;
    // raw stanza 1.13.0 gives `aprilo` -> `verb|aprilare` and `leggila` ->
    // `verb|leggilare`, neither of which is an Italian verb.
    //
    // These two are repaired downstream by `IT_COMPOUND_IMPERATIVES` in
    // `lang_it.rs`, an 11-entry hand-curated allowlist, so what they pin is
    // that the existing repair keeps working. The uncovered forms, which is
    // where the defect still bites, are asserted separately below.
    for (anchor, expected_lemma, fabricated) in [
        ("aprilo", "aprire", "aprilare"),
        ("leggila", "leggere", "leggilare"),
    ] {
        let mor = find_mor_line_for(output, anchor)
            .unwrap_or_else(|| panic!("no %mor tier for {anchor:?}"));
        assert!(
            !mor.contains(fabricated),
            "{anchor:?} must not be labelled with the invented verb \
             {fabricated:?}, got {mor:?}"
        );
        assert!(
            mor.contains('~'),
            "{anchor:?} is a real imperative+enclitic and must split, got {mor:?}"
        );
        assert!(
            mor.contains(expected_lemma),
            "{anchor:?} should carry lemma {expected_lemma:?}, got {mor:?}"
        );
    }

    // The uncovered half of the same defect: genuine imperative+enclitic forms
    // that are NOT in the 11-entry allowlist, so nothing repaired them. Raw
    // stanza 1.13.0 leaves `dimmi` whole (losing the clitic entirely) and gives
    // `buttalo` the unanalyzed lemma `buttalo`. Verb+enclitic is an open class,
    // so no finite allowlist can cover it, which is why the fix has to be
    // general rather than another row in a table.
    for (anchor, expected_lemma) in [("verita", "dire"), ("buttalo", "buttare")] {
        let mor = find_mor_line_for(output, anchor)
            .unwrap_or_else(|| panic!("no %mor tier for the {anchor:?} utterance"));
        assert!(
            mor.contains('~'),
            "the {anchor:?} utterance carries an imperative+enclitic that must \
             split, got {mor:?}"
        );
        assert!(
            mor.contains(expected_lemma),
            "expected lemma {expected_lemma:?} in the {anchor:?} utterance, got {mor:?}"
        );
    }
}
