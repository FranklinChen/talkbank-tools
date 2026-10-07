use crate::common::assert_completed_without_errors;
use batchalign::api::ReleasedCommand;
use batchalign::options::{CommandOptions, CommonOptions, MorphotagOptions};
use batchalign::worker::InferTask;

use crate::ml_golden::golden::fixtures::{
    CAT_SPA_L2, DAN_ENG_L2, DEU_ENG_CONTRACTIONS, DEU_ENG_L2, DEU_ENG_PHRASAL, ENG_SPA_L2,
    FRA_NLD_L2,
};
use crate::ml_golden::golden::helpers::{
    assert_golden_snapshot, find_mor_line_for, require_direct_session_warmed_many,
};

fn l2_enabled_options() -> CommandOptions {
    CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },

        ..Default::default()
    })
}

fn l2_disabled_options() -> CommandOptions {
    CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        no_l2_morphotag: true,

        ..Default::default()
    })
}

/// The `%mor` line of the utterance containing `surface`; a missing line
/// fails with the whole output.
fn mor_line(output: &str, surface: &str) -> String {
    find_mor_line_for(output, surface).unwrap_or_else(|| {
        panic!("no %mor line for the utterance containing {surface:?} in:\n{output}")
    })
}

/// Assert the `%mor` line of `surface`'s utterance contains `needle`,
/// printing the line on failure.
fn assert_mor_has(output: &str, surface: &str, needle: &str) {
    let line = mor_line(output, surface);
    assert!(
        line.contains(needle),
        "%mor for {surface:?} must contain {needle:?}; the line is: {line}"
    );
}

/// Assert the `%mor` line of `surface`'s utterance lacks `needle`,
/// printing the line on failure.
fn assert_mor_lacks(output: &str, surface: &str, needle: &str) {
    let line = mor_line(output, surface);
    assert!(
        !line.contains(needle),
        "%mor for {surface:?} must not contain {needle:?}; the line is: {line}"
    );
}

/// Assert no `L2|xxx` placeholder survives, printing every line holding one.
fn assert_no_placeholder(output: &str) {
    let left: Vec<&str> = output
        .lines()
        .filter(|line| line.contains("L2|xxx"))
        .collect();
    assert!(
        left.is_empty(),
        "every @s word must be analysed; L2|xxx is left on:\n{}",
        left.join("\n")
    );
}

#[tokio::test]
async fn golden_l2_morphotag_eng_spa() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "eng"),
            (ReleasedCommand::Morphotag, "spa"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_spa_l2.cha",
            ENG_SPA_L2,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_eng_spa", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    assert_mor_has(output, "tienda@s:spa", "noun|tienda");
    assert_mor_has(output, "muy@s:spa", "adv|");
    assert_mor_has(output, "niños@s:spa", "niño");
    assert_golden_snapshot!("l2_morphotag_eng_spa", output);
}

#[tokio::test]
async fn golden_l2_morphotag_deu_eng() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "deu"),
            (ReleasedCommand::Morphotag, "eng"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "deu",
            "deu_eng_l2.cha",
            DEU_ENG_L2,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_deu_eng", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    assert_mor_has(output, "film@s", "noun|film");
    assert_mor_has(output, "drug@s", "noun|drug");
    assert_golden_snapshot!("l2_morphotag_deu_eng", output);
}

#[tokio::test]
async fn golden_l2_morphotag_eng_contractions() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "deu"),
            (ReleasedCommand::Morphotag, "eng"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "deu",
            "deu_eng_contractions.cha",
            DEU_ENG_CONTRACTIONS,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_eng_contractions", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_mor_has(output, "it's@s:eng", "~");
    assert_mor_lacks(output, "it's@s:eng", "L2|xxx");
    assert_mor_has(output, "don't@s:eng", "~");
    assert_mor_lacks(output, "working@s:eng", "L2|xxx");
    // The secondary's verb, not a category read off the primary's guess.
    assert_mor_has(output, "working@s:eng", "verb|work");
    assert_golden_snapshot!("l2_morphotag_eng_contractions", output);
}

#[tokio::test]
async fn golden_l2_morphotag_phrasal_verbs() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "deu"),
            (ReleasedCommand::Morphotag, "eng"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "deu",
            "deu_eng_phrasal.cha",
            DEU_ENG_PHRASAL,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_phrasal_verbs", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    for (surface, verb) in [
        ("wake@s up@s", "verb|wake"),
        ("give@s up@s", "verb|give"),
        ("pick@s up@s", "verb|pick"),
    ] {
        assert_mor_has(output, surface, verb);
        assert_mor_has(output, surface, "part|up");
    }
    // A compound noun, not a phrasal verb: `out` keeps its ADP.
    assert_mor_has(output, "time@s out@s", "noun|time");
    assert_mor_has(output, "time@s out@s", "adp|out");
    assert_golden_snapshot!("l2_morphotag_phrasal_verbs", output);
}

#[tokio::test]
async fn golden_l2_morphotag_off_produces_l2_xxx() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "eng"),
            (ReleasedCommand::Morphotag, "spa"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "eng",
            "eng_spa_l2_off.cha",
            ENG_SPA_L2,
            l2_disabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_off", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert!(
        output.contains("L2|xxx"),
        "with L2 morphotag off every @s word keeps L2|xxx; the output is:\n{output}"
    );
    assert_mor_has(output, "tienda@s:spa", "L2|xxx");
}

#[tokio::test]
async fn golden_l2_morphotag_cat_spa() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "cat"),
            (ReleasedCommand::Morphotag, "spa"),
        ],
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "cat",
            "cat_spa_l2.cha",
            CAT_SPA_L2,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_cat_spa", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    assert_mor_lacks(output, "cole@s", "L2|xxx");
    assert_mor_lacks(output, "bonita@s", "L2|xxx");
    assert_golden_snapshot!("l2_morphotag_cat_spa", output);
}

#[tokio::test]
async fn golden_l2_morphotag_dan_eng() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "dan"),
            (ReleasedCommand::Morphotag, "eng"),
        ],
        "Direct session does not support morphotag infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "dan",
            "dan_eng_l2.cha",
            DAN_ENG_L2,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_dan_eng", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    assert_mor_lacks(output, "computer@s", "L2|xxx");
    assert_mor_lacks(output, "happy@s", "L2|xxx");
    assert_golden_snapshot!("l2_morphotag_dan_eng", output);
}

#[tokio::test]
async fn golden_l2_morphotag_fra_nld() {
    let Some(jobs) = require_direct_session_warmed_many(
        InferTask::Morphosyntax,
        vec![
            (ReleasedCommand::Morphotag, "fra"),
            (ReleasedCommand::Morphotag, "nld"),
        ],
        "Direct session does not support morphotag infer",
    )
    .await
    else {
        return;
    };
    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Morphotag,
            "fra",
            "fra_nld_l2.cha",
            FRA_NLD_L2,
            l2_enabled_options(),
        )
        .await;
    assert_completed_without_errors("l2_morphotag_fra_nld", &info, &results);
    let output = &results[0].content.as_text().expect("text command result");
    assert_no_placeholder(output);
    assert_mor_lacks(output, "opa@s", "L2|xxx");
    assert_mor_lacks(output, "ja@s:nld", "L2|xxx");
    // Both models tag `ja` INTJ.
    assert_mor_has(output, "ja@s:nld", "intj|ja");
    assert_golden_snapshot!("l2_morphotag_fra_nld", output);
}
