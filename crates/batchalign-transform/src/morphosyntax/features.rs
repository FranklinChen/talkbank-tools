//! POS-specific feature handlers for UD-to-CHAT morphosyntax mapping.
//!
//! # Where a suffix's value comes from
//!
//! Every `%mor` suffix a handler emits is a [`Suffix`], and a `Suffix` says
//! where its value came from:
//!
//! - [`Suffix::Feature`]: a feature value exactly as the word's features hold
//!   it. It holds a [`FeatValue`], which only the word's typed features
//!   ([`super::WordFeatures::get`]) can make, so no literal can pass for one,
//!   and the value carries its own source ([`super::FeatSource`]): the
//!   analysis that returned the word, or one of our curated tables (the
//!   English contraction expansions' `me` is
//!   `Case=Acc|Number=Sing|Person=1|PronType=Prs` because the table says so,
//!   not because Stanza did). Both render the same way; the source is what
//!   tells them apart.
//! - [`Suffix::Derived`]: a feature value rendered by a fixed rule
//!   (`Reflex=Yes` as `reflx`, a Hebrew binyan lowercased).
//! - [`Suffix::Lexical`]: a mark from one of our curated word tables.
//! - [`Suffix::Agreement`]: number and person joined (`S3`), each a [`Slot`]
//!   holding a feature value or empty.
//!
//! There is no other kind. A feature neither the analysis nor a curated table
//! contains is not written: there is no variant that could carry an invented
//! value. The book's morphosyntax reference ("POS Mapping") records what is
//! written.

use super::lang_fr::FrenchPronounCase;
use super::mor_word::MorPos;
use super::{
    FeatName, FeatValue, MappingContext, UdPunctable, UdWord, UniversalPos, WordFeatures, lang_en,
    lang_fr, lang2,
};
use smallvec::SmallVec;
use talkbank_model::model::dependent_tier::mor::MorFeature;

/// How the renderer classifies feature values: each distinction named once,
/// so no call site compares a value with a string literal.
impl<'a> FeatValue<'a> {
    /// The first character as a sub-slice (`Plur` gives `P`); `None` if empty.
    fn initial(self) -> Option<&'a str> {
        let text = self.text();
        text.chars().next().map(|c| &text[..c.len_utf8()])
    }

    fn person(self) -> PersonClass {
        match self.text() {
            "3" => PersonClass::Third,
            "0" => PersonClass::Impersonal,
            _ => PersonClass::Other,
        }
    }

    fn gender(self) -> GenderClass {
        match self.text() {
            "Com" | "Com,Neut" => GenderClass::Common,
            _ => GenderClass::Other,
        }
    }

    fn degree(self) -> DegreeClass {
        match self.text() {
            "Pos" => DegreeClass::Positive,
            _ => DegreeClass::Other,
        }
    }

    fn reflex(self) -> ReflexClass {
        match self.text() {
            "Yes" => ReflexClass::Reflexive,
            _ => ReflexClass::Other,
        }
    }
}

/// One word's features as the renderer reads them: a borrow of the word's
/// typed features with the classifications the handlers share.
pub(super) struct UdFeats<'a> {
    features: &'a WordFeatures,
}

impl<'a> UdFeats<'a> {
    /// One word's features.
    pub(super) fn of_word(ud: &'a UdWord) -> Self {
        Self {
            features: ud.features(),
        }
    }

    /// The value of `name`, if the word has one, with where it came from.
    pub(super) fn get(&self, name: FeatName) -> Option<FeatValue<'a>> {
        self.features.get(name)
    }

    fn verb_form(&self) -> Option<VerbFormClass> {
        self.get(FeatName::VerbForm).map(|v| match v.text() {
            "Fin" => VerbFormClass::Finite,
            _ => VerbFormClass::Other,
        })
    }

    fn tense(&self) -> Option<TenseClass> {
        self.get(FeatName::Tense).map(|v| match v.text() {
            "Pres" => TenseClass::Present,
            "Past" => TenseClass::Past,
            _ => TenseClass::Other,
        })
    }

    fn number(&self) -> Option<NumberClass> {
        self.get(FeatName::Number).map(|v| match v.text() {
            "Sing" => NumberClass::Singular,
            "Plur" => NumberClass::Plural,
            _ => NumberClass::Other,
        })
    }

    fn person(&self) -> Option<PersonClass> {
        self.get(FeatName::Person).map(FeatValue::person)
    }
}

/// The feature-value distinctions the renderer makes, named once each, so no
/// call site compares a feature value with a string literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerbFormClass {
    Finite,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TenseClass {
    Present,
    Past,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberClass {
    Singular,
    Plural,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersonClass {
    Third,
    /// UD's impersonal person `0`, which CHAT writes `4`.
    Impersonal,
    Other,
}

/// Common gender (`Com`, `Com,Neut`) is not written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenderClass {
    Common,
    Other,
}

/// The positive degree is the unmarked one and is not written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DegreeClass {
    Positive,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReflexClass {
    Reflexive,
    Other,
}

/// A feature value rendered by a fixed rule.
enum Derived<'a> {
    /// `Reflex=Yes` is written `reflx`.
    Reflexive,
    /// Hebrew `HebBinyan` and `HebExistential` are written in lower case.
    Lowercase(FeatValue<'a>),
}

/// A mark from one of our curated word tables, not from the analysis.
enum LexicalMark {
    /// English irregular past (`lang_en::is_irregular`): `irr`.
    IrregularPast,
    /// French noun with auditory plural marking (`lang_fr::is_apm_noun`): `Apm`.
    FrenchApm,
    /// French pronoun case from the surface form (`lang_fr::french_pronoun_case`).
    FrenchPronounCase(FrenchPronounCase),
}

impl LexicalMark {
    const fn text(self) -> &'static str {
        match self {
            Self::IrregularPast => "irr",
            Self::FrenchApm => "Apm",
            Self::FrenchPronounCase(case) => case.text(),
        }
    }
}

/// One slot of a joined suffix (`S3`; a possessor's `S1`).
enum Slot<'a> {
    /// A value's first character (`Number=Plur` gives `P`).
    Initial(FeatValue<'a>),
    /// A value as written.
    Value(FeatValue<'a>),
    /// `Person` as written, except UD's impersonal `0`, which CHAT writes `4`.
    Person(FeatValue<'a>),
    /// Nothing written: the analysis has no value for the slot.
    Empty,
}

impl Slot<'_> {
    fn write(&self, out: &mut String) {
        match self {
            Self::Initial(value) => out.push_str(value.initial().unwrap_or("")),
            Self::Value(value) => out.push_str(value.text()),
            Self::Person(value) => out.push_str(match value.person() {
                PersonClass::Impersonal => "4",
                PersonClass::Third | PersonClass::Other => value.text(),
            }),
            Self::Empty => {}
        }
    }
}

/// One `%mor` suffix before rendering, with where its value came from.
enum Suffix<'a> {
    /// A feature value as the word holds it; the value carries its source.
    Feature(FeatValue<'a>),
    Derived(Derived<'a>),
    Lexical(LexicalMark),
    Agreement {
        number: Slot<'a>,
        person: Slot<'a>,
    },
}

type Suffixes<'a> = SmallVec<[Suffix<'a>; 6]>;

/// A number slot from the value's first character; empty when the analysis
/// has no number or an empty one.
fn number_initial(value: Option<FeatValue<'_>>) -> Slot<'_> {
    match value {
        Some(value) if value.initial().is_some() => Slot::Initial(value),
        Some(_) | None => Slot::Empty,
    }
}

/// A word's suffixes as written. Only [`render`] makes one with content.
#[derive(Default)]
pub(super) struct RenderedFeatures {
    suffixes: SmallVec<[MorFeature; 4]>,
}

impl RenderedFeatures {
    pub(super) fn into_suffixes(self) -> SmallVec<[MorFeature; 4]> {
        self.suffixes
    }
}

/// Write suffixes in order. A suffix whose text is empty is skipped, as it
/// always was (an empty feature value, an `Agreement` of two empty slots).
fn render(suffixes: Suffixes<'_>) -> RenderedFeatures {
    let mut rendered = RenderedFeatures::default();
    let mut text = String::new();
    for suffix in suffixes {
        text.clear();
        match suffix {
            Suffix::Feature(value) => text.push_str(value.text()),
            Suffix::Derived(Derived::Reflexive) => text.push_str("reflx"),
            Suffix::Derived(Derived::Lowercase(value)) => {
                text.push_str(&value.text().to_lowercase());
            }
            Suffix::Lexical(mark) => text.push_str(mark.text()),
            Suffix::Agreement { number, person } => {
                number.write(&mut text);
                person.write(&mut text);
            }
        }
        if !text.is_empty() {
            rendered.suffixes.push(MorFeature::flat(&text));
        }
    }
    rendered
}

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
    fn classify(feats: &UdFeats<'_>, ud: &UdWord, ctx: &MappingContext) -> Self {
        // English only: the rule is about English present-tense agreement being
        // marked on exactly one cell of the paradigm.
        if lang2(&ctx.lang) != "en" {
            return Self::Overt;
        }
        // Lexical verbs only. `verb_features` also serves `AUX`, whose forms are
        // a closed class that spells agreement suppletively ("is", "has"); we do
        // not second-guess those. Matching on the ORIGINAL UPOS keeps the two
        // apart even where an override changed the written category.
        if !matches!(ud.upos, UdPunctable::Value(UniversalPos::Verb)) {
            return Self::Overt;
        }
        let present_third_singular = feats.verb_form() == Some(VerbFormClass::Finite)
            && feats.tense() == Some(TenseClass::Present)
            && feats.number() == Some(NumberClass::Singular)
            && feats.person() == Some(PersonClass::Third);
        if !present_third_singular {
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

/// Features of a word Stanza tagged VERB or AUX. `written` is the category
/// the word is written with, which a Japanese override can change: one written
/// as anything but a verb or auxiliary (a subordinator, `cm`) carries none.
pub(super) fn verb_features(
    feats: &UdFeats<'_>,
    written: MorPos,
    ud: &UdWord,
    ctx: &MappingContext,
) -> RenderedFeatures {
    // `ろ` is written bare.
    if ud.text == "\u{308D}"
        || !matches!(
            written,
            MorPos::Upos(UniversalPos::Verb | UniversalPos::Aux)
        )
    {
        return RenderedFeatures::default();
    }
    let mut suffixes = Suffixes::new();
    for name in [
        FeatName::VerbForm,
        FeatName::Aspect,
        FeatName::Mood,
        FeatName::Tense,
        FeatName::Polarity,
        FeatName::Polite,
    ] {
        if let Some(value) = feats.get(name) {
            suffixes.push(Suffix::Feature(value));
        }
    }
    for name in [FeatName::HebBinyan, FeatName::HebExistential] {
        if let Some(value) = feats.get(name) {
            suffixes.push(Suffix::Derived(Derived::Lowercase(value)));
        }
    }

    let number = number_initial(feats.get(FeatName::Number));
    // Decide ONCE whether the agreement Stanza reported is actually spelled out
    // on this word, then match on the answer. The classification is the only
    // route to the decision, so no later code can re-derive it differently.
    let person = match RealizedAgreement::classify(feats, ud, ctx) {
        RealizedAgreement::Overt => feats
            .get(FeatName::Person)
            .map_or(Slot::Empty, Slot::Person),
        // The person was inferred from the subject, not realized on the word,
        // so render the number alone. That is byte-identical to what an absent
        // `Person` feature already produces here, which keeps the uninflected
        // form inside the existing scheme rather than inventing a new marking.
        RealizedAgreement::Inferred => Slot::Empty,
    };
    suffixes.push(Suffix::Agreement { number, person });

    if lang2(&ctx.lang) == "en"
        && feats.tense() == Some(TenseClass::Past)
        && lang_en::is_irregular(&ud.lemma, &ud.text)
    {
        suffixes.push(Suffix::Lexical(LexicalMark::IrregularPast));
    }
    render(suffixes)
}

pub(super) fn pron_features(
    feats: &UdFeats<'_>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> RenderedFeatures {
    let mut suffixes = Suffixes::new();
    if let Some(kind) = feats.get(FeatName::PronType) {
        suffixes.push(Suffix::Feature(kind));
    }

    if lang2(&ctx.lang) == "fr" {
        if let Some(case) = lang_fr::french_pronoun_case(&ud.text) {
            suffixes.push(Suffix::Lexical(LexicalMark::FrenchPronounCase(case)));
        }
    } else if let Some(case) = feats.get(FeatName::Case) {
        suffixes.push(Suffix::Feature(case));
    }

    if feats
        .get(FeatName::Reflex)
        .is_some_and(|v| v.reflex() == ReflexClass::Reflexive)
    {
        suffixes.push(Suffix::Derived(Derived::Reflexive));
    }

    if ud.text != "that" && ud.text != "who" {
        // The number's initial, as on every other category: `Sing` is `S`,
        // `Plur` is `P`, and `Dual` is `D` (it used to be written `S`).
        let number = number_initial(feats.get(FeatName::Number));
        let person = feats
            .get(FeatName::Person)
            .map_or(Slot::Empty, Slot::Person);
        suffixes.push(Suffix::Agreement { number, person });
    }
    render(suffixes)
}

pub(super) fn det_features(feats: &UdFeats<'_>) -> RenderedFeatures {
    let mut suffixes = Suffixes::new();
    match feats.get(FeatName::Gender) {
        // Common gender (`Com`, `Com,Neut`) is not written.
        Some(gender) if gender.gender() == GenderClass::Common => {}
        Some(gender) => suffixes.push(Suffix::Feature(gender)),
        None => {}
    }
    if let Some(definite) = feats.get(FeatName::Definite) {
        suffixes.push(Suffix::Feature(definite));
    }
    if let Some(kind) = feats.get(FeatName::PronType) {
        suffixes.push(Suffix::Feature(kind));
    }
    if let Some(number) = feats.get(FeatName::Number) {
        suffixes.push(Suffix::Feature(number));
    }
    suffixes.push(Suffix::Agreement {
        number: feats
            .get(FeatName::NumberPsor)
            .map_or(Slot::Empty, Slot::Initial),
        person: feats
            .get(FeatName::PersonPsor)
            .map_or(Slot::Empty, Slot::Value),
    });
    render(suffixes)
}

pub(super) fn adj_features(feats: &UdFeats<'_>) -> RenderedFeatures {
    let mut suffixes = Suffixes::new();
    // The positive degree is not written; nor is an absent degree.
    match feats.get(FeatName::Degree) {
        Some(degree) if degree.degree() == DegreeClass::Positive => {}
        Some(degree) => suffixes.push(Suffix::Feature(degree)),
        None => {}
    }
    if let Some(case) = feats.get(FeatName::Case) {
        suffixes.push(Suffix::Feature(case));
    }
    suffixes.push(Suffix::Agreement {
        number: number_initial(feats.get(FeatName::Number)),
        person: feats
            .get(FeatName::Person)
            .map_or(Slot::Empty, Slot::Person),
    });
    render(suffixes)
}

pub(super) fn noun_features(
    feats: &UdFeats<'_>,
    ud: &UdWord,
    ctx: &MappingContext,
) -> RenderedFeatures {
    let mut suffixes = Suffixes::new();
    match feats.get(FeatName::Gender) {
        Some(gender) if gender.gender() == GenderClass::Common => {}
        Some(gender) => suffixes.push(Suffix::Feature(gender)),
        None => {}
    }
    // Singular is the unmarked number and is not written.
    if let (Some(number), Some(NumberClass::Plural | NumberClass::Other)) =
        (feats.get(FeatName::Number), feats.number())
    {
        suffixes.push(Suffix::Feature(number));
    }
    if let Some(case) = feats.get(FeatName::Case) {
        suffixes.push(Suffix::Feature(case));
    }
    if let Some(kind) = feats.get(FeatName::PronType) {
        suffixes.push(Suffix::Feature(kind));
    }
    if lang2(&ctx.lang) == "fr"
        && feats.number() == Some(NumberClass::Plural)
        && lang_fr::is_apm_noun(&ud.text)
    {
        suffixes.push(Suffix::Lexical(LexicalMark::FrenchApm));
    }
    render(suffixes)
}

#[cfg(test)]
mod tests {
    use super::super::{
        CuratedFeats, FeatSource, UdHead, UdId, UdPunctable, UdWordAnalysis, UniversalPos,
        map_ud_word,
    };
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
        let ud = UdWord::from(UdWordAnalysis {
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
        });
        let mor = map_ud_word(&ud, &ctx).expect("mapping must succeed");
        let mut out = String::new();
        mor.write_chat(&mut out)
            .expect("serialization must succeed");
        out
    }

    /// The `%mor` item for one token with a chosen relation and no lemma
    /// difference.
    fn render_bare(
        lang: &str,
        text: &str,
        upos: UniversalPos,
        feats: Option<&str>,
        deprel: &str,
    ) -> String {
        let ctx = MappingContext {
            lang: LanguageCode::new(lang).expect("valid test language code"),
        };
        let ud = UdWord::from(UdWordAnalysis {
            id: UdId::Single(1),
            text: text.to_string(),
            lemma: text.to_string(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: feats.map(str::to_string),
            head: 0,
            deprel: deprel.to_string(),
            deps: None,
            misc: None,
        });
        let mor = map_ud_word(&ud, &ctx).expect("mapping must succeed");
        let mut out = String::new();
        mor.write_chat(&mut out)
            .expect("serialization must succeed");
        out
    }

    /// A pronoun's number is written as the analysis assigned it: `Dual` is
    /// `D`. It used to be written `S`, a singular nobody assigned.
    #[test]
    fn a_dual_pronoun_is_written_dual() {
        assert_eq!(
            render(
                "slv",
                "midva",
                "jaz",
                UniversalPos::Pron,
                "Case=Nom|Number=Dual|Person=1|PronType=Prs"
            ),
            "pron|jaz-Prs-Nom-D1"
        );
        assert_eq!(
            render(
                "eng",
                "we",
                "we",
                UniversalPos::Pron,
                "Case=Nom|Number=Plur|Person=1|PronType=Prs"
            ),
            "pron|we-Prs-Nom-P1"
        );
    }

    /// `me` as the contraction table synthesizes it and `me` as the analysis
    /// returns it render identically; each value says which made it.
    #[test]
    fn curated_and_analysed_values_render_alike_and_keep_their_source() {
        const ME: &str = "Case=Acc|Number=Sing|Person=1|PronType=Prs";
        let curated = UdWord::synthetic(
            UdId::Single(1),
            "me",
            "I",
            UniversalPos::Pron,
            CuratedFeats::new(ME),
            UdHead::Root,
            "obj",
        );
        let analysed = UdWord::from(UdWordAnalysis {
            id: UdId::Single(1),
            text: "me".to_string(),
            lemma: "I".to_string(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some(ME.to_string()),
            head: 0,
            deprel: "obj".to_string(),
            deps: None,
            misc: None,
        });
        let case_source = |word: &UdWord| {
            UdFeats::of_word(word)
                .get(FeatName::Case)
                .map(FeatValue::source)
        };
        assert_eq!(case_source(&curated), Some(FeatSource::Curated));
        assert_eq!(case_source(&analysed), Some(FeatSource::Analysis));

        let ctx = MappingContext {
            lang: LanguageCode::new("eng").expect("valid test language code"),
        };
        let render = |word: &UdWord| {
            let mut out = String::new();
            map_ud_word(word, &ctx)
                .expect("mapping must succeed")
                .write_chat(&mut out)
                .expect("serialization must succeed");
            out
        };
        assert_eq!(render(&curated), "pron|I-Prs-Acc-S1");
        assert_eq!(render(&analysed), render(&curated));
    }

    /// The analysis cannot claim a value is curated: provenance is not read
    /// from MISC, where an engine's own annotation (or a record kept from an
    /// earlier rewrite) could say anything.
    #[test]
    fn a_misc_entry_does_not_make_a_value_curated() {
        let word = UdWord::from(UdWordAnalysis {
            id: UdId::Single(1),
            text: "me".to_string(),
            lemma: "I".to_string(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Case=Acc".to_string()),
            head: 0,
            deprel: "obj".to_string(),
            deps: None,
            misc: Some("CuratedFeats=Case".to_string()),
        });
        assert_eq!(
            word.features().get(FeatName::Case).map(FeatValue::source),
            Some(FeatSource::Analysis)
        );
    }

    #[test]
    fn a_fully_analyzed_word_is_written_as_analyzed() {
        assert_eq!(
            render_bare(
                "deu",
                "dreht",
                UniversalPos::Verb,
                Some(EN_PRES_3SG),
                "root"
            ),
            "verb|dreht-Fin-Ind-Pres-S3"
        );
    }

    /// The values the renderer used to invent are not written: a feature the
    /// analysis does not contain leaves no suffix. Each assertion names one
    /// of the eleven retired fills.
    #[test]
    fn a_feature_the_analysis_lacks_is_not_written() {
        // Verb: no `Inf`, no `S`.
        assert_eq!(
            render_bare("zho", "说", UniversalPos::Verb, None, "root"),
            "verb|说"
        );
        // Pronoun: no `Int`, no `S1`.
        assert_eq!(
            render_bare("eng", "there", UniversalPos::Pron, None, "expl"),
            "pron|there"
        );
        // Determiner: no `Def`; French: no `Masc` either.
        assert_eq!(
            render_bare("eng", "this", UniversalPos::Det, None, "det"),
            "det|this"
        );
        assert_eq!(
            render_bare("fra", "ce", UniversalPos::Det, None, "det"),
            "det|ce"
        );
        assert_eq!(
            render_bare("fra", "ces", UniversalPos::Det, Some("Number=Plur"), "det"),
            "det|ces-Plur"
        );
        // Adjective: no `S1`; an observed number is written alone.
        assert_eq!(
            render_bare("ita", "forte", UniversalPos::Adj, None, "amod"),
            "adj|forte"
        );
        assert_eq!(
            render_bare(
                "ita",
                "forte",
                UniversalPos::Adj,
                Some("Number=Sing"),
                "amod"
            ),
            "adj|forte-S"
        );
        // Object noun: no `Acc`.
        assert_eq!(
            render_bare("eng", "dish", UniversalPos::Noun, None, "obj"),
            "noun|dish"
        );
        // English noun spelled `-ing`: no `Ger`.
        assert_eq!(
            render_bare("eng", "king", UniversalPos::Noun, None, "nsubj"),
            "noun|king"
        );
    }

    /// An observed value is still written exactly where a fill used to stand.
    #[test]
    fn observed_values_are_still_written() {
        assert_eq!(
            render_bare(
                "eng",
                "this",
                UniversalPos::Det,
                Some("Definite=Def|PronType=Dem"),
                "det"
            ),
            "det|this-Def-Dem"
        );
        assert_eq!(
            render_bare("eng", "dish", UniversalPos::Noun, Some("Case=Acc"), "obj"),
            "noun|dish-Acc"
        );
        assert_eq!(
            render_bare(
                "eng",
                "go",
                UniversalPos::Verb,
                Some("VerbForm=Inf"),
                "root"
            ),
            "verb|go-Inf"
        );
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
