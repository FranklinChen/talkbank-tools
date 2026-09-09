//! POS-specific feature handlers for UD-to-CHAT morphosyntax mapping.

use super::mor_word::{push_feat, push_feature};
use super::{MappingContext, UdPunctable, UdWord, UniversalPos, lang_en, lang_fr, lang2};
use smallvec::SmallVec;
use std::collections::HashMap;
use talkbank_model::model::dependent_tier::mor::MorFeature;

/// Whether the person/number agreement in a UD feature bundle is spelled out on
/// the word itself, or was inferred by the tagger from somewhere else.
///
/// Why this exists: Stanza propagates `Person=3|Number=Sing` onto an English
/// present-tense verb from its SUBJECT, so it labels the child utterance
/// "it bang ." exactly as it labels "it turns .". CHAT `%mor` describes the
/// REALIZED word, so the third-person suffix belongs on "turns" and must not
/// appear on "bang".
///
/// This is an enum rather than a bool so the feature builder must MATCH on the
/// distinction, and so the conditions live in one classifier instead of being
/// re-derived (and re-mis-derived) at each place that reads `Person`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RealizedAgreement {
    /// The word carries its agreement, or the bundle is not the one narrow case
    /// below. This is the ordinary path: render `Person` as given.
    Overt,
    /// English, a lexical `VERB` (never an `AUX`), finite, present, third-person
    /// singular, AND a surface form equal to the lemma ignoring case. The person
    /// came from the subject; the word does not spell it.
    Inferred,
}

impl RealizedAgreement {
    /// Classify one token. Every condition is required; any miss is `Overt`,
    /// which is the pre-existing behaviour, so this can only ever narrow.
    fn classify(feats: &HashMap<String, String>, ud: &UdWord, ctx: &MappingContext) -> Self {
        // English only: the rule is about English present-tense agreement being
        // marked on exactly one cell of the paradigm.
        if lang2(&ctx.lang) != "en" {
            return Self::Overt;
        }
        // Lexical verbs only. `verb_features` also serves `AUX`, whose forms are
        // a closed class that spells agreement suppletively ("is", "has"); we do
        // not second-guess those. Matching on the ORIGINAL UPOS is what keeps
        // the two apart, since `effective_pos` is a string that merges them.
        if !matches!(ud.upos, UdPunctable::Value(UniversalPos::Verb)) {
            return Self::Overt;
        }
        let feat_is = |key: &str, value: &str| feats.get(key).map(String::as_str) == Some(value);
        if !feat_is("VerbForm", "Fin")
            || !feat_is("Tense", "Pres")
            || !feat_is("Number", "Sing")
            || !feat_is("Person", "3")
        {
            return Self::Overt;
        }
        // The evidence itself: a third-singular present form that is spelled
        // like its own lemma has no agreement suffix on it. Case-insensitive so
        // an utterance-initial "Bang" is treated like "bang".
        if ud.text.to_lowercase() == ud.lemma.to_lowercase() {
            Self::Inferred
        } else {
            Self::Overt
        }
    }
}

pub(super) fn verb_features(
    feats: &HashMap<String, String>,
    effective_pos: &str,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    if effective_pos.contains("sconj") {
        return SmallVec::new();
    }
    if ud.text == "\u{308D}" {
        return SmallVec::new();
    }
    if !effective_pos.contains("verb") && !effective_pos.contains("aux") {
        if ud.text == "\u{305F}\u{308A}" {
            let mut s = SmallVec::new();
            push_feature(&mut s, "Inf");
            push_feature(&mut s, "S");
            return s;
        }
        return SmallVec::new();
    }

    let mut suffixes = SmallVec::new();
    let verb_form = feats
        .get("VerbForm")
        .cloned()
        .unwrap_or_else(|| "Inf".to_string());
    push_feature(&mut suffixes, &verb_form);
    push_feat(&mut suffixes, feats, "Aspect");
    push_feat(&mut suffixes, feats, "Mood");
    push_feat(&mut suffixes, feats, "Tense");
    push_feat(&mut suffixes, feats, "Polarity");
    push_feat(&mut suffixes, feats, "Polite");

    if let Some(v) = feats.get("HebBinyan") {
        push_feature(&mut suffixes, &v.to_lowercase());
    }
    if let Some(v) = feats.get("HebExistential") {
        push_feature(&mut suffixes, &v.to_lowercase());
    }

    let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("");
    let number_raw = feats.get("Number").map(|s| s.as_str()).unwrap_or("Sing");
    let number_char = number_raw.chars().next().unwrap_or('S');
    // Decide ONCE whether the agreement Stanza reported is actually spelled out
    // on this word, then match on the answer. The classification is the only
    // route to the decision, so no later code can re-derive it differently.
    let person_str = match RealizedAgreement::classify(feats, ud, ctx) {
        RealizedAgreement::Overt => {
            // `Person=0` is UD's impersonal person; CHAT spells it "4".
            if person_raw == "0" { "4" } else { person_raw }
        }
        // The person was inferred from the subject, not realized on the word,
        // so render the number alone. That is byte-identical to what an absent
        // `Person` feature already produces here, which keeps the uninflected
        // form inside the existing scheme rather than inventing a new marking.
        RealizedAgreement::Inferred => "",
    };
    let num_person = format!("{}{}", number_char, person_str);
    push_feature(&mut suffixes, &num_person);

    if lang2(&ctx.lang) == "en"
        && let Some(tense) = feats.get("Tense")
        && tense == "Past"
        && lang_en::is_irregular(&ud.lemma, &ud.text)
    {
        push_feature(&mut suffixes, "irr");
    }

    suffixes
}

pub(super) fn pron_features(
    feats: &HashMap<String, String>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut parts = Vec::new();
    let pron_type = feats.get("PronType").map(|s| s.as_str()).unwrap_or("Int");
    parts.push(pron_type.to_string());

    let case = if lang2(&ctx.lang) == "fr" {
        lang_fr::french_pronoun_case(&ud.text).to_string()
    } else {
        feats.get("Case").cloned().unwrap_or_default()
    };
    if !case.is_empty() {
        parts.push(case);
    }

    if let Some(reflex) = feats.get("Reflex")
        && reflex == "Yes"
    {
        parts.push("reflx".to_string());
    }

    if ud.text != "that" && ud.text != "who" {
        let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("1");
        let person_str = if person_raw == "0" { "4" } else { person_raw };
        let number = feats
            .get("Number")
            .map(|n| if n.starts_with('P') { "P" } else { "S" })
            .unwrap_or("S");
        parts.push(format!("{}{}", number, person_str));
    }

    let non_empty: Vec<&str> = parts
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    if non_empty.is_empty() {
        SmallVec::new()
    } else {
        let mut suffixes = SmallVec::new();
        for part in non_empty {
            push_feature(&mut suffixes, part);
        }
        suffixes
    }
}

pub(super) fn det_features(
    feats: &HashMap<String, String>,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();
    let number = feats.get("Number").map(|s| s.as_str()).unwrap_or("");
    let gender_default = if lang2(&ctx.lang) == "fr" {
        if number == "Plur" { "" } else { "Masc" }
    } else {
        ""
    };
    let gender = feats
        .get("Gender")
        .cloned()
        .unwrap_or_else(|| gender_default.to_string());
    if !gender.is_empty() && gender != "Com,Neut" && gender != "Com" {
        push_feature(&mut suffixes, &gender);
    }

    let definite = feats.get("Definite").map(|s| s.as_str()).unwrap_or("Def");
    push_feature(&mut suffixes, definite);
    push_feat(&mut suffixes, feats, "PronType");
    push_feature(&mut suffixes, number);

    let np = feats
        .get("Number[psor]")
        .and_then(|s| s.chars().next())
        .map(|c| c.to_string())
        .unwrap_or_default();
    let pp = feats.get("Person[psor]").map(|s| s.as_str()).unwrap_or("");
    let psor = format!("{}{}", np, pp);
    push_feature(&mut suffixes, &psor);
    suffixes
}

pub(super) fn adj_features(feats: &HashMap<String, String>) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();
    let degree = feats.get("Degree").map(|s| s.as_str()).unwrap_or("Pos");
    if degree != "Pos" {
        push_feature(&mut suffixes, degree);
    }
    if let Some(case) = feats.get("Case") {
        push_feature(&mut suffixes, case);
    }
    let number = feats
        .get("Number")
        .and_then(|s| s.chars().next())
        .unwrap_or('S');
    let person_raw = feats.get("Person").map(|s| s.as_str()).unwrap_or("1");
    let person_str = if person_raw == "0" { "4" } else { person_raw };
    push_feature(&mut suffixes, &format!("{}{}", number, person_str));
    suffixes
}

pub(super) fn noun_features(
    feats: &HashMap<String, String>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    let mut suffixes = SmallVec::new();
    let gender = feats
        .get("Gender")
        .cloned()
        .unwrap_or_else(|| "Com,Neut".to_string());
    if gender != "Com,Neut" && gender != "Com" {
        push_feature(&mut suffixes, &gender);
    }

    let number = feats.get("Number").map(|s| s.as_str()).unwrap_or("Sing");
    if number != "Sing" {
        push_feature(&mut suffixes, number);
    }

    let case = feats.get("Case").cloned().unwrap_or_else(|| {
        if ud.deprel == "obj" {
            "Acc".to_string()
        } else {
            String::new()
        }
    });
    push_feature(&mut suffixes, &case);
    push_feat(&mut suffixes, feats, "PronType");

    if lang2(&ctx.lang) == "en" && ud.text.ends_with("ing") {
        push_feature(&mut suffixes, "Ger");
    }
    if lang2(&ctx.lang) == "fr" && number == "Plur" && lang_fr::is_apm_noun(&ud.text) {
        push_feature(&mut suffixes, "Apm");
    }
    suffixes
}

#[cfg(test)]
mod tests {
    use super::super::{UdId, UdPunctable, UniversalPos, map_ud_word_to_mor};
    use super::*;
    use talkbank_model::model::LanguageCode;

    /// The feature bundle Stanza emits for an English present-tense verb whose
    /// subject is third-person singular. Stanza assigns it from the SUBJECT, so
    /// it appears identically on "it turns" and on "it bang".
    const EN_PRES_3SG: &str = "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin";

    /// Render one Stanza-shaped token through the real mapping seam and return
    /// the `%mor` item text, which is exactly what a CHAT file would carry.
    fn render(lang: &str, text: &str, lemma: &str, upos: UniversalPos, feats: &str) -> String {
        let ctx = MappingContext {
            lang: LanguageCode::new(lang).expect("valid test language code"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: text.to_string(),
            lemma: lemma.to_string(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: Some(feats.to_string()),
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).expect("mapping must succeed");
        let mut out = String::new();
        mor.write_chat(&mut out)
            .expect("serialization must succeed");
        out
    }

    #[test]
    fn an_uninflected_english_present_verb_does_not_carry_third_person() {
        // A child says "it bang ." Stanza infers Person=3 from "it", but the
        // word itself is bare, so `%mor` (which describes the REALIZED word)
        // must render the number alone, exactly as an absent Person does.
        assert_eq!(
            render("eng", "bang", "bang", UniversalPos::Verb, EN_PRES_3SG),
            "verb|bang-Fin-Ind-Pres-S"
        );
    }

    #[test]
    fn an_overtly_inflected_english_present_verb_keeps_third_person() {
        // "it turns ." carries the agreement on the surface form, so the S3
        // suffix is real morphology and must survive.
        assert_eq!(
            render("eng", "turns", "turn", UniversalPos::Verb, EN_PRES_3SG),
            "verb|turn-Fin-Ind-Pres-S3"
        );
    }

    #[test]
    fn a_non_english_present_verb_with_a_lemma_shaped_surface_keeps_third_person() {
        // Same token shape as the English case, only the language differs, so
        // this pins the language gate rather than the surface comparison.
        assert_eq!(
            render("deu", "bang", "bang", UniversalPos::Verb, EN_PRES_3SG),
            "verb|bang-Fin-Ind-Pres-S3"
        );
    }

    #[test]
    fn an_english_auxiliary_with_a_lemma_shaped_surface_keeps_third_person() {
        // The rule is about lexical verbs Stanza failed to see as uninflected.
        // An AUX is a closed-class form; leave its agreement alone.
        assert_eq!(
            render("eng", "be", "be", UniversalPos::Aux, EN_PRES_3SG),
            "aux|be-Fin-Ind-Pres-S3"
        );
    }
}
