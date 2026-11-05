//! Single-word UD-to-CHAT MOR mapping.

use super::features::{
    FeatureRendering, RenderedFeatures, UdFeats, adj_features, det_features, noun_features,
    pron_features, verb_features,
};
use super::{
    MappingContext, MappingError, UdPunctable, UdWord, UniversalPos, japanese_verbform, lang2,
    sanitize_mor_text,
};
use talkbank_model::model::dependent_tier::mor::{Mor, MorStem, MorWord, PosCategory};

/// The category a `%mor` item is written with: a UD category, or Japanese
/// `cm` for a comma. Typed, so the feature handlers match on it instead of
/// searching a category name for `verb` or `sconj`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MorPos {
    Upos(UniversalPos),
    JapaneseComma,
}

impl MorPos {
    /// The name written before `|`.
    fn name(self) -> &'static str {
        match self {
            Self::Upos(upos) => upos.to_chat_pos_name(),
            Self::JapaneseComma => "cm",
        }
    }
}

/// A CHAT tag-marker separator that the morphosyntax payload keeps as a
/// `%mor` item: the vocative marker `‡` (U+2021) and the tag marker `„`
/// (U+201E). The payload sends them to the model like words (the parse keeps
/// its punctuation token), and the model echoes them back as tokens; this type
/// is what such a token is recognized AS, once, at [`Self::classify`].
///
/// The comma `,` is also a tag-marker separator in CHAT (Chatter's
/// `is_tag_marker_separator`, `talkbank_model::alignment::helpers`, keeps all
/// three as `%mor` items), but it is written through its own lemma branch
/// (`cm|cm`, [`map_actual_punct`]) and is not one of these. This is the
/// model-token side, keyed by the surface text the payload sent; the resting
/// shape, which needs Chatter's extractor to carry the separator kind on the
/// extracted word, is a `WordRole` decided where the word is born, as special
/// forms are. Until then the token is classified here, once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagMarker {
    /// `‡` (U+2021), the vocative or summons marker.
    Vocative,
    /// `„` (U+201E), the tag marker.
    Tag,
}

impl TagMarker {
    /// The vocative marker's one spelling in CHAT.
    const VOCATIVE: &'static str = "\u{2021}";
    /// The tag marker's one spelling in CHAT.
    const TAG: &'static str = "\u{201E}";

    /// The one classifier: the marker `text` is, or `None` for every other
    /// text. The crate spells the same two marks by hand elsewhere
    /// (`build_chat::tag_marker_separator`, `retokenize::parse_helpers`,
    /// `asr_postprocess::MOR_PUNCT`, `compare::engine`); this is the primitive
    /// those sites can be brought onto.
    pub fn of_text(text: &str) -> Option<Self> {
        match text {
            Self::VOCATIVE => Some(Self::Vocative),
            Self::TAG => Some(Self::Tag),
            _ => None,
        }
    }

    /// What a model token is to the `%mor` writer, decided once: a marker, or
    /// a lexical word the generic path may write. Classified on the text rather
    /// than the model's lemma, because the text is ours and the lemma is the
    /// model's. The only way to mint a [`Lexical`] from a model token, so the
    /// generic path cannot be handed a marker.
    pub fn classify(ud: &UdWord) -> Classified<'_> {
        match Self::of_text(&ud.text) {
            Some(marker) => Classified::Marker(marker),
            None => Classified::Lexical(Lexical(ud)),
        }
    }

    /// The marker as written in CHAT.
    pub const fn chat_text(self) -> &'static str {
        match self {
            Self::Vocative => Self::VOCATIVE,
            Self::Tag => Self::TAG,
        }
    }

    /// The `%mor` item written for this marker: the ONE place its spelling
    /// lives.
    ///
    /// PENDING O5a (the maintainer's open decision): today the marker is written as
    /// `punct|‡` and `punct|„`, and in Japanese as `cm|‡` and `cm|„`, which is
    /// exactly what the generic word path produced for a `PUNCT` token before
    /// this type existed (the Japanese branch writes every `PUNCT` token's
    /// category as `cm`). CLAN's convention is `beg|beg` for `‡` and `end|end`
    /// for `„` (Chatter's `is_tag_marker_separator` names it). This function
    /// keeps the current spelling so that O5a changes exactly one function,
    /// and nothing else, when it is decided.
    pub fn mor(self, ctx: &MappingContext) -> MorWord {
        let category = match lang2(&ctx.lang) {
            "ja" => MorPos::JapaneseComma,
            _ => MorPos::Upos(UniversalPos::Punct),
        };
        MorWord::new(
            PosCategory::new(category.name()),
            MorStem::new(self.chat_text()),
        )
    }
}

/// A model token that is not a tag marker: the only input the generic word
/// path ([`map_ud_mor_word`]) accepts, so a marker cannot reach it and its
/// spelling cannot arise there. Minted by [`TagMarker::classify`] for a model
/// token, and by [`Self::curated`] for a reading a curated table synthesized.
#[derive(Debug, Clone, Copy)]
pub struct Lexical<'a>(&'a UdWord);

impl<'a> Lexical<'a> {
    /// A curated table's reading (`UdWord::curated_reading`): its surface
    /// comes from the table's own allowlist (Italian compound imperatives and
    /// mis-splits), never from the model, so it is lexical by provenance.
    /// Scoped to this module tree: a model token takes the classifier.
    pub(super) fn curated(word: &'a UdWord) -> Self {
        Self(word)
    }

    /// The word.
    pub fn word(self) -> &'a UdWord {
        self.0
    }
}

/// What one model token is to the `%mor` writer.
#[derive(Debug, Clone, Copy)]
pub enum Classified<'a> {
    /// A tag marker, written by [`TagMarker::mor`].
    Marker(TagMarker),
    /// A word, written by the generic word path.
    Lexical(Lexical<'a>),
}

/// Map a single UD word into a CHAT `%mor` item: a marker through
/// [`TagMarker::mor`], every other token through the generic word path.
pub fn map_ud_word(ud: &UdWord, ctx: &MappingContext) -> Result<Mor, MappingError> {
    match TagMarker::classify(ud) {
        Classified::Marker(marker) => Ok(Mor::new(marker.mor(ctx))),
        Classified::Lexical(word) => map_ud_mor_word(word, ctx).map(Mor::new),
    }
}

/// Map a single lexical UD word into one `%mor` word: one chunk, by type.
///
/// Takes a [`Lexical`], never a bare `UdWord`: a tag marker cannot be given
/// to this path, so its spelling belongs to [`TagMarker::mor`] alone.
pub(super) fn map_ud_mor_word(
    word: Lexical<'_>,
    ctx: &MappingContext,
) -> Result<MorWord, MappingError> {
    let ud = word.word();
    if matches!(ud.lemma.as_str(), "." | "!" | "?" | "," | "$,") {
        return Ok(map_actual_punct(ud));
    }

    let feats = UdFeats::of_word(ud);
    let mut cleaned_lemma = clean_lemma(&ud.lemma, &ud.text);
    // Stanza's own punctuation marker is written as the `punct` category.
    let tagged = match ud.upos {
        UdPunctable::Value(upos) => upos,
        UdPunctable::Punct(_) => UniversalPos::Punct,
    };
    let mut written = MorPos::Upos(tagged);
    if lang2(&ctx.lang) == "ja" {
        if let Some(ovr) = japanese_verbform(tagged, &cleaned_lemma, &ud.text) {
            written = MorPos::Upos(ovr.pos);
            cleaned_lemma = ovr.lemma.replace(',', "cm");
        }
        if matches!(ud.upos, UdPunctable::Value(UniversalPos::Punct))
            || ud.lemma == "\u{3001}"
            || ud.lemma == ","
        {
            written = MorPos::JapaneseComma;
        }
    }

    let suffixes = compute_features(&ud.upos, &feats, written, ud, ctx)?.into_suffixes();
    let sanitized_lemma = sanitize_mor_text(&cleaned_lemma);
    if sanitized_lemma.is_empty() {
        return Err(MappingError::EmptyStem {
            word: ud.text.clone(),
            lemma: ud.lemma.clone(),
            upos: format!("{:?}", ud.upos),
        });
    }

    let mor_word = MorWord::new(
        PosCategory::new(written.name()),
        MorStem::new(sanitized_lemma),
    )
    .with_features(suffixes);
    Ok(mor_word)
}

fn map_actual_punct(ud: &UdWord) -> MorWord {
    let (pos_name, stem) = if ud.lemma == "," || ud.lemma == "$," {
        (MorPos::JapaneseComma.name(), "cm")
    } else {
        (UniversalPos::Punct.to_chat_pos_name(), ud.lemma.as_str())
    };

    MorWord::new(PosCategory::new(pos_name), MorStem::new(stem))
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

fn compute_features(
    original_upos: &UdPunctable<UniversalPos>,
    feats: &UdFeats<'_>,
    written: MorPos,
    ud: &UdWord,
    ctx: &MappingContext,
) -> FeatureRendering {
    // Every UPOS named, so a new one fails to compile here instead of falling
    // through to "no features".
    match original_upos {
        UdPunctable::Value(UniversalPos::Verb | UniversalPos::Aux) => {
            verb_features(feats, written, ud, ctx)
        }
        UdPunctable::Value(UniversalPos::Pron) => pron_features(feats, ud, ctx),
        UdPunctable::Value(UniversalPos::Det) => det_features(feats),
        UdPunctable::Value(UniversalPos::Adj) => adj_features(feats),
        UdPunctable::Value(UniversalPos::Noun | UniversalPos::Propn) => {
            noun_features(feats, ud, ctx)
        }
        UdPunctable::Value(
            UniversalPos::Adp
            | UniversalPos::Adv
            | UniversalPos::Cconj
            | UniversalPos::Intj
            | UniversalPos::Num
            | UniversalPos::Part
            | UniversalPos::Punct
            | UniversalPos::Sconj
            | UniversalPos::Sym
            | UniversalPos::X,
        )
        | UdPunctable::Punct(_) => Ok(RenderedFeatures::default()),
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

    /// A model token for one of the two markers, as Stanza echoes it: text
    /// and lemma the marker, category `PUNCT`, attached to word `head`.
    fn marker_token(id: usize, text: &str, head: usize) -> UdWord {
        UdWord::try_from(crate::morphosyntax::UdWordAnalysis {
            id: crate::morphosyntax::UdId::Single(id),
            text: text.to_string(),
            lemma: text.to_string(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head,
            deprel: "punct".to_string(),
            deps: None,
            misc: None,
        })
        .expect("a marker token carries nothing the admission refuses")
    }

    fn rendered(mor: &Mor) -> String {
        let mut out = String::new();
        mor.write_chat(&mut out).expect("a %mor item writes");
        out
    }

    /// PENDING O5a: the spelling written for each marker today, pinned so
    /// the decision (`beg|beg` / `end|end` versus `punct|`) changes this
    /// test and `TagMarker::mor` together, and nothing else. Japanese writes
    /// `cm|`, as the generic path did for every `PUNCT` token there.
    #[test]
    fn tag_marker_spelling_is_pending_o5a() {
        let lang = |code: &str| MappingContext {
            lang: talkbank_model::model::LanguageCode::new(code).expect("valid test language code"),
        };
        let (eng, jpn) = (lang("eng"), lang("jpn"));
        assert_eq!(
            rendered(&Mor::new(TagMarker::Vocative.mor(&eng))),
            "punct|\u{2021}"
        );
        assert_eq!(
            rendered(&Mor::new(TagMarker::Tag.mor(&eng))),
            "punct|\u{201E}"
        );
        assert_eq!(
            rendered(&Mor::new(TagMarker::Vocative.mor(&jpn))),
            "cm|\u{2021}"
        );
        assert_eq!(rendered(&Mor::new(TagMarker::Tag.mor(&jpn))), "cm|\u{201E}");
        assert_eq!(TagMarker::of_text("\u{2021}"), Some(TagMarker::Vocative));
        assert!(matches!(
            TagMarker::classify(&marker_token(1, "\u{2021}", 0)),
            Classified::Marker(TagMarker::Vocative)
        ));
        assert!(matches!(
            TagMarker::classify(&marker_token(1, "\u{201E}", 0)),
            Classified::Marker(TagMarker::Tag)
        ));
        assert!(matches!(
            TagMarker::classify(&marker_token(1, ",", 0)),
            Classified::Lexical(_)
        ));
    }

    /// RED FIRST: the generic word path wrote `punct|„` itself from the
    /// model's lemma, so the marker's spelling lived in the same code as
    /// every word's. The generic path now takes a `Lexical`, which only the
    /// classifier mints for a model token, so it cannot be given a marker:
    /// `map_ud_mor_word(&marker_token(..), &ctx)` no longer compiles. The
    /// one place a marker can still arrive is as a component of a multi-word
    /// token, and that is refused there, by name.
    #[test]
    fn a_tag_marker_cannot_be_a_clitic() {
        use crate::morphosyntax::assemble_mors;
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng")
                .expect("valid test language code"),
        };
        let main = UdWord::try_from(crate::morphosyntax::UdWordAnalysis {
            id: crate::morphosyntax::UdId::Single(1),
            text: "hello".to_string(),
            lemma: "hello".to_string(),
            upos: UdPunctable::Value(UniversalPos::Intj),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        })
        .expect("test analysis is admitted");
        let refused = assemble_mors(&[main, marker_token(2, "\u{201E}", 1)], &ctx);
        assert!(
            matches!(
                refused,
                Err(MappingError::TagMarkerAsClitic {
                    marker: TagMarker::Tag
                })
            ),
            "{refused:?}"
        );
    }

    /// A sentence with a marker token writes the marker through its own
    /// renderer, with the relation the model gave it, and every other token
    /// through the generic path: the output is what it was, by a route that
    /// classifies the token once.
    #[test]
    fn a_sentence_writes_a_tag_marker_through_its_own_renderer() {
        use crate::morphosyntax::{UdId, UdSentence, UdWordAnalysis, map_ud_sentence};
        let word = |id: usize, text: &str, upos: UniversalPos, head: usize, deprel: &str| {
            UdWord::try_from(UdWordAnalysis {
                id: UdId::Single(id),
                text: text.to_string(),
                lemma: text.to_string(),
                upos: UdPunctable::Value(upos),
                xpos: None,
                feats: None,
                head,
                deprel: deprel.to_string(),
                deps: None,
                misc: None,
            })
            .expect("test analysis is admitted")
        };
        let sentence = UdSentence {
            words: vec![
                word(1, "hello", UniversalPos::Intj, 0, "root"),
                marker_token(2, "\u{201E}", 1),
                word(3, "there", UniversalPos::Adv, 1, "advmod"),
                word(4, ".", UniversalPos::Punct, 1, "punct"),
            ],
        };
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng")
                .expect("valid test language code"),
        };
        let (mors, gras) = map_ud_sentence(&sentence, &ctx).expect("the sentence maps");
        // The terminator is no `%mor` item; its relation closes the `%gra`.
        let written: Vec<String> = mors.iter().map(rendered).collect();
        assert_eq!(written, ["intj|hello", "punct|\u{201E}", "adv|there"]);
        assert_eq!(gras.len(), 4, "three chunks and the mapped terminator");
        assert_eq!((gras[1].index, gras[1].head), (2, 1));
        assert_eq!(gras[1].relation.as_str(), "PUNCT");
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
