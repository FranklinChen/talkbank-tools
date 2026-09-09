//! Single-word UD-to-CHAT MOR mapping.

use super::features::{adj_features, det_features, noun_features, pron_features, verb_features};
use super::{
    MappingContext, MappingError, UdPunctable, UdWord, UniversalPos, japanese_verbform, lang2,
    sanitize_mor_text,
};
use smallvec::SmallVec;
use std::collections::HashMap;
use talkbank_model::model::dependent_tier::mor::{Mor, MorFeature, MorStem, MorWord, PosCategory};

/// Map a single UD word into a CHAT `%mor` item.
pub fn map_ud_word_to_mor(ud: &UdWord, ctx: &MappingContext) -> Result<Mor, MappingError> {
    if matches!(ud.lemma.as_str(), "." | "!" | "?" | "," | "$,") {
        return Ok(map_actual_punct(ud));
    }

    let feats = parse_feats(ud.feats.as_deref());
    let mut cleaned_lemma = clean_lemma(&ud.lemma, &ud.text);
    let mut effective_pos = upos_to_name(&ud.upos).to_string();
    if lang2(&ctx.lang) == "ja"
        && let Some(ovr) = japanese_verbform(&effective_pos, &cleaned_lemma, &ud.text)
    {
        effective_pos = ovr.pos.to_string();
        cleaned_lemma = ovr.lemma.to_string();
        cleaned_lemma = cleaned_lemma.replace(',', "cm");
    }

    if lang2(&ctx.lang) == "ja" {
        if matches!(ud.upos, UdPunctable::Value(UniversalPos::Punct)) {
            effective_pos = "cm".to_string();
        }
        if ud.lemma == "、" || ud.lemma == "," {
            effective_pos = "cm".to_string();
        }
    }

    let features = compute_features(&ud.upos, &feats, &effective_pos, ud, ctx);
    let sanitized_lemma = sanitize_mor_text(&cleaned_lemma);
    if sanitized_lemma.is_empty() {
        return Err(MappingError::EmptyStem {
            word: ud.text.clone(),
            lemma: ud.lemma.clone(),
            upos: format!("{:?}", ud.upos),
        });
    }

    let mor_word = MorWord::new(
        PosCategory::new(&effective_pos),
        MorStem::new(sanitized_lemma),
    )
    .with_features(features);
    Ok(Mor::new(mor_word))
}

fn map_actual_punct(ud: &UdWord) -> Mor {
    let (pos_name, stem) = if ud.lemma == "," || ud.lemma == "$," {
        ("cm", "cm")
    } else {
        ("punct", ud.lemma.as_str())
    };

    let mor_word = MorWord::new(PosCategory::new(pos_name), MorStem::new(stem));
    Mor::new(mor_word)
}

pub(super) fn parse_feats(feats: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(f) = feats {
        for pair in f.split('|') {
            let mut parts = pair.split('=');
            if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                map.insert(key.to_string(), value.to_string());
            }
        }
    }
    map
}

/// Clean a UD lemma for use as a CHAT `%mor` stem.
///
/// # The zero-prefix branch that used to live here, and why it is gone
///
/// This function used to special-case a lemma starting with `0`: it replaced
/// the lemma with `text[1..]`, dropping the first character of the WORD, and
/// set an `is_unknown` flag no caller read. The deletion is right; the reason
/// first given for it ("a CHAT word cannot begin with a digit") is not, and it
/// is corrected here so the next reader does not inherit it.
///
/// The real reasons, in order:
///
/// - A leading `0` in CHAT is the OMISSION marker, not a digit in a word:
///   `0the` is "the word *the*, omitted by the speaker", modelled as
///   `WordCategory::Omission` (`talkbank-model`,
///   `model/content/action.rs`, which distinguishes it from the bare `0`
///   non-verbal action token).
/// - An omission never reaches a `%mor` payload. It carries no surface form to
///   tag, and the alignment rules say so directly: "Omissions never align"
///   (`talkbank-model`, `alignment/helpers/rules.rs`). So a `%mor` item whose
///   lemma begins with `0` is not a thing this code can be handed.
/// - Measured rather than assumed: no `%mor` item in the whole data tree
///   carries a zero-prefixed POS. The branch was dead in production and
///   destructive if it ever fired.
///
/// A lemma that does begin with `0` is therefore cleaned like any other, with
/// no character dropped. [`tests::clean_lemma_keeps_a_leading_zero_intact`]
/// pins that, so the deletion is a stated behaviour rather than a silent one.
pub fn clean_lemma(lemma: &str, text: &str) -> String {
    let mut target = lemma.to_string();

    if target.trim() == "\u{300D}" || target.trim() == "\u{300C}" {
        target = text.to_string();
    }
    if target == "\"" {
        target = text.to_string();
    }
    if target.is_empty() {
        target = text.to_string();
    }
    target = target.replace(['\u{300D}', '\u{300C}'], "");

    if target.contains("<SOS>") {
        target = text.to_string();
    }
    target = target.replace(['$', '.'], "");
    if target.starts_with('-') && target.len() > 1 {
        target = target[1..].to_string();
    }
    if target.ends_with('-') && target.len() > 1 {
        target = target[..target.len() - 1].to_string();
    }
    target = target.replace("--", "-");
    target = target.replace("--", "-");
    target = target.replace("<unk>", "");
    target = target.replace("<SOS>", "");
    target = target.replace("/100", "");
    target = target.replace("/r", "");
    target = target.replace([',', '\'', '~', '(', ')'], "");

    if target.contains('|') {
        target = target.split('|').next().unwrap_or("").trim().to_string();
    }

    target = target.replace(['_', '+'], "");
    if target == "door zogen" {
        target = text.to_string();
    }
    target = target.replace('-', "\u{2013}");
    if target.contains('\u{201C}') {
        target = text.to_string();
    }

    let chars: Vec<char> = target.chars().collect();
    if chars.len() >= 2
        && chars[chars.len() - 2] == '@'
        && (chars[chars.len() - 1].is_alphanumeric() || chars[chars.len() - 1] == '_')
    {
        target = chars[..chars.len() - 2].iter().collect::<String>();
    }

    target = target.trim().to_string();
    if target.is_empty() && !text.is_empty() {
        target = text.to_string();
    }
    if target.is_empty() {
        target = "x".to_string();
    }

    target
}

fn upos_to_name(upos: &UdPunctable<UniversalPos>) -> &'static str {
    match upos {
        UdPunctable::Value(v) => v.to_chat_pos_name(),
        UdPunctable::Punct(_) => "punct",
    }
}

pub(super) fn push_feature(features: &mut SmallVec<[MorFeature; 4]>, value: &str) {
    if !value.is_empty() {
        features.push(MorFeature::flat(value));
    }
}

pub(super) fn push_feat(
    features: &mut SmallVec<[MorFeature; 4]>,
    feats: &HashMap<String, String>,
    key: &str,
) {
    if let Some(val) = feats.get(key) {
        push_feature(features, val);
    }
}

fn compute_features(
    original_upos: &UdPunctable<UniversalPos>,
    feats: &HashMap<String, String>,
    effective_pos: &str,
    ud: &UdWord,
    ctx: &MappingContext,
) -> SmallVec<[MorFeature; 4]> {
    match original_upos {
        UdPunctable::Value(UniversalPos::Verb | UniversalPos::Aux) => {
            verb_features(feats, effective_pos, ud, ctx)
        }
        UdPunctable::Value(UniversalPos::Pron) => pron_features(feats, ud, ctx),
        UdPunctable::Value(UniversalPos::Det) => det_features(feats, ctx),
        UdPunctable::Value(UniversalPos::Adj) => adj_features(feats),
        UdPunctable::Value(UniversalPos::Noun | UniversalPos::Propn) => {
            noun_features(feats, ud, ctx)
        }
        UdPunctable::Value(UniversalPos::Sym | UniversalPos::Punct) => SmallVec::new(),
        _ => SmallVec::new(),
    }
}

/// Return `true` if the token text represents a known clitic for the given
/// language.
pub fn is_clitic(text: &str, ctx: &MappingContext) -> bool {
    match lang2(&ctx.lang) {
        "en" => text == "n't" || text == "'s" || text == "'ve" || text == "'ll",
        "fr" => text.ends_with('\'') || text == "-ce" || text == "-être" || text == "-là",
        "it" => text.ends_with('\''),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_feats_preserves_multi_value_commas() {
        let feats = parse_feats(Some("PronType=Int,Rel|Person=3"));
        assert_eq!(feats.get("PronType"), Some(&"Int,Rel".to_string()));
        assert_eq!(feats.get("Person"), Some(&"3".to_string()));
    }

    /// The zero-prefix branch was DELETED (see `clean_lemma`'s own docs): a
    /// leading `0` is CHAT's omission marker, omissions never reach a `%mor`
    /// payload, and no `%mor` item in the data tree carries a zero-prefixed
    /// POS. This pins what the deletion means for a lemma that does arrive
    /// with one: it is cleaned like any other lemma, and no character of
    /// either the lemma or the text is dropped. The old branch returned
    /// `text[1..]`, so it would answer "ero" here.
    #[test]
    fn clean_lemma_keeps_a_leading_zero_intact() {
        assert_eq!(clean_lemma("0zero", "zero"), "0zero");
    }

    #[test]
    fn clean_lemma_falls_back_from_empty_to_text() {
        let lemma = clean_lemma("'", "Claus'");
        assert_eq!(lemma, "Claus'");
    }

    #[test]
    fn is_clitic_dispatches_by_language() {
        let en = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng")
                .expect("valid test language code"),
        };
        let fr = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fra")
                .expect("valid test language code"),
        };
        assert!(is_clitic("n't", &en));
        assert!(is_clitic("l'", &fr));
        assert!(!is_clitic("hello", &en));
    }
}
