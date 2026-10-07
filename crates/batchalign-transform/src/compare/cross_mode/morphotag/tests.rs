use super::*;
use talkbank_model::model::TranscriptName;
use talkbank_model::validation::{AlignmentValidation, ValidationPolicy};
use talkbank_model::{ErrorCollector, RuleSelection};
use talkbank_parser::TreeSitterParser;

fn source(speaker: &str, tiers: &str) -> String {
    format!(
        "@UTF8\n@Begin\n@Languages:\tfra\n@Participants:\t{speaker} Participant\n@ID:\tfra|test|{speaker}|||||Participant|||\n*{speaker}:\tl'escargot dort .\n{tiers}@End\n"
    )
}

fn admitted(text: &str) -> ValidChatFile {
    let parser = TreeSitterParser::new().unwrap();
    let errors = ErrorCollector::new();
    crate::parse_validated_with_parser(
        &parser,
        text,
        ValidationPolicy::new(
            RuleSelection::new(),
            AlignmentValidation::IncludeTierAlignment,
        ),
        TranscriptName::Anonymous,
        &errors,
    )
    .unwrap_or_else(|error| panic!("{error}: {:?}", errors.into_vec()))
}

const COMPLETE: &str = "%mor:\tnoun|escargot-Masc~det|le-Masc-Def-Art-Sing verb|dormir-Fin-Ind-Pres-S3 .\n%gra:\t1|3|NSUBJ 2|1|DET 3|0|ROOT 4|3|PUNCT\n";

fn rows(left: &str, right: &str) -> Vec<MorphotagTokenDifference> {
    let left = morph_tokens(&admitted(&source("PAR", left)), "left").unwrap();
    let right = morph_tokens(&admitted(&source("OTH", right)), "right").unwrap();
    left["PAR"][0]
        .iter()
        .zip(&right["OTH"][0])
        .enumerate()
        .map(|(i, (l, r))| compare_token(Some(l), Some(r), "PAR", "OTH", 0, i))
        .collect()
}

#[test]
fn jointly_absent_annotations_do_not_claim_agreement() {
    let result = MorphotagPairResult {
        tokens: rows("", ""),
    };
    assert_eq!(result.compared_tokens(), 2);
    assert_eq!(result.fully_annotated_tokens(), 0);
    assert_eq!(result.differences().count(), 0);
    assert!(
        result
            .tokens()
            .iter()
            .all(|row| row.analysis_agreement().is_none())
    );
    let wire = serde_json::to_value(&result).unwrap();
    assert_eq!(wire["fully_annotated_tokens"], 0);
    assert!(wire["tokens"][0]["analysis_agreement"].is_null());
    assert_eq!(wire["tokens"][0]["left_annotation"], "absent");
}

#[test]
fn missing_one_analysis_is_presence_not_complete_agreement() {
    let result = MorphotagPairResult {
        tokens: rows(COMPLETE, ""),
    };
    assert_eq!(result.fully_annotated_tokens(), 0);
    assert_eq!(
        result.tokens()[0].left_annotation,
        AnnotationPresence::Complete
    );
    assert_eq!(
        result.tokens()[0].right_annotation,
        AnnotationPresence::Absent
    );
    assert!(result.tokens()[0].analysis_agreement().is_none());
}

#[test]
fn mor_without_gra_remains_explicitly_incomplete() {
    let mor_only = COMPLETE.split("%gra:").next().unwrap();
    let result = MorphotagPairResult {
        tokens: rows(mor_only, mor_only),
    };
    assert_eq!(result.fully_annotated_tokens(), 0);
    assert_eq!(
        result.tokens()[0].left_annotation,
        AnnotationPresence::MorphologyOnly
    );
}

#[test]
fn mapped_speaker_codes_are_not_dependency_identity() {
    let result = MorphotagPairResult {
        tokens: rows(COMPLETE, COMPLETE),
    };
    assert_eq!(result.fully_annotated_tokens(), 2);
    assert_eq!(result.differences().count(), 0);
    assert!(
        result
            .tokens()
            .iter()
            .all(|row| row.analysis_agreement() == Some(true))
    );
}

#[test]
fn every_post_clitic_analysis_is_compared() {
    for (replacement, axis) in [
        ("det|un-Masc-Def-Art-Sing", MorphotagDifference::Lemma),
        ("pron|le-Masc-Def-Art-Sing", MorphotagDifference::Pos),
        ("det|le-Fem-Def-Art-Sing", MorphotagDifference::FeatureSet),
    ] {
        let altered = COMPLETE.replace("det|le-Masc-Def-Art-Sing", replacement);
        let compared = rows(COMPLETE, &altered);
        assert!(compared[0].differences.contains(&axis));
        assert_eq!(compared[0].analysis_agreement(), Some(false));
        assert!(compared[1].differences.is_empty());
    }
}

#[test]
fn post_clitic_edges_have_local_chunk_addresses() {
    for (replacement, axis) in [
        ("2|3|DET", MorphotagDifference::DependencyHead),
        ("2|1|AMOD", MorphotagDifference::Relation),
    ] {
        let altered = COMPLETE.replace("2|1|DET", replacement);
        let compared = rows(COMPLETE, &altered);
        assert!(compared[0].differences.contains(&axis));
        assert_eq!(compared[0].analysis_agreement(), Some(false));
    }
    // An edge into a post-clitic, not just an item's first chunk, is retained.
    let altered = COMPLETE
        .replace("1|3|NSUBJ", "1|2|NSUBJ")
        .replace("2|1|DET", "2|3|DET");
    assert!(
        rows(COMPLETE, &altered)[0]
            .differences
            .contains(&MorphotagDifference::DependencyHead)
    );
}

#[test]
fn absent_main_position_cannot_claim_annotation_agreement() {
    let complete = morph_tokens(&admitted(&source("PAR", COMPLETE)), "left").unwrap();
    let row = compare_token(Some(&complete["PAR"][0][0]), None, "PAR", "OTH", 0, 0);
    assert_eq!(row.right_annotation, AnnotationPresence::NoMainToken);
    assert!(row.analysis_agreement().is_none());
}
