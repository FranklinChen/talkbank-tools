//! Typed views over Universal Dependencies values: the 17 UPOS tags,
//! the dependency-relation labels, the `VerbForm` feature, and the
//! per-token record (`UdWord`/`UdSentence`/`UdResponse`) that the
//! pipeline marshals back from the Stanza worker, with its typed features
//! ([`WordFeatures`], in `word_features`). Plus the small cleanup pass
//! (`validate_and_clean`, `is_bogus_lemma`, `sanitize_mor_text`).

mod word_features;

use std::num::NonZeroUsize;

pub(crate) use word_features::FeatName;
pub use word_features::{CuratedFeats, FeatSource, FeatValue, WordFeatures};

/// The 17 Universal POS tags as defined by Universal Dependencies v2.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "UPPERCASE")]
pub enum UniversalPos {
    /// Adjective.
    Adj,
    /// Adposition.
    Adp,
    /// Adverb.
    Adv,
    /// Auxiliary verb.
    Aux,
    /// Coordinating conjunction.
    Cconj,
    /// Determiner.
    Det,
    /// Pronoun.
    Pron,
    /// Common noun.
    Noun,
    /// Proper noun.
    Propn,
    /// Numeral.
    Num,
    /// Particle.
    Part,
    /// Main verb.
    Verb,
    /// Subordinating conjunction.
    Sconj,
    /// Punctuation.
    Punct,
    /// Symbol.
    Sym,
    /// Interjection.
    Intj,
    /// Other / unknown.
    X,
}

impl UniversalPos {
    /// The lowercase CHAT POS category name for this UPOS.
    pub fn to_chat_pos_name(self) -> &'static str {
        match self {
            Self::Adj => "adj",
            Self::Adp => "adp",
            Self::Adv => "adv",
            Self::Aux => "aux",
            Self::Cconj => "cconj",
            Self::Det => "det",
            Self::Intj => "intj",
            Self::Noun => "noun",
            Self::Num => "num",
            Self::Part => "part",
            Self::Pron => "pron",
            Self::Propn => "propn",
            Self::Punct => "punct",
            Self::Sconj => "sconj",
            Self::Sym | Self::X => "x",
            Self::Verb => "verb",
        }
    }

    /// Parse a POS category name into a `UniversalPos`.
    pub fn from_pos_name(name: &str) -> Option<Self> {
        let eq = |s: &str| name.eq_ignore_ascii_case(s);
        if eq("adj") {
            Some(Self::Adj)
        } else if eq("adp") {
            Some(Self::Adp)
        } else if eq("adv") {
            Some(Self::Adv)
        } else if eq("aux") {
            Some(Self::Aux)
        } else if eq("cconj") {
            Some(Self::Cconj)
        } else if eq("det") {
            Some(Self::Det)
        } else if eq("intj") {
            Some(Self::Intj)
        } else if eq("noun") {
            Some(Self::Noun)
        } else if eq("num") {
            Some(Self::Num)
        } else if eq("part") {
            Some(Self::Part)
        } else if eq("pron") {
            Some(Self::Pron)
        } else if eq("propn") {
            Some(Self::Propn)
        } else if eq("punct") {
            Some(Self::Punct)
        } else if eq("sconj") {
            Some(Self::Sconj)
        } else if eq("verb") {
            Some(Self::Verb)
        } else if eq("sym") || eq("x") {
            Some(Self::X)
        } else {
            None
        }
    }
}

/// Universal Dependencies relation label.
///
/// Known values get dedicated variants so call sites compile-check against
/// typos. Unknown relations land in `Other(String)` so round-tripping stays
/// lossless without allocating on the known-value hot path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DepRel {
    /// `root`: the sentence-level root.
    Root,
    /// `nsubj`: nominal subject.
    NSubj,
    /// `nsubj:pass`: nominal passive subject.
    NSubjPass,
    /// `obj`: direct object.
    Obj,
    /// `aux`: auxiliary.
    Aux,
    /// `aux:pass`: passive auxiliary.
    AuxPass,
    /// `cop`: copula.
    Cop,
    /// `case`: case-marking word, including possessive `'s`.
    Case,
    /// `nmod:poss`: possessive nominal modifier.
    NmodPoss,
    /// `det`: determiner.
    Det,
    /// `cc`: coordinating conjunction.
    Cc,
    /// `conj`: conjoined element.
    Conj,
    /// `compound`: compound modifier.
    Compound,
    /// `compound:prt`: phrasal verb particle.
    CompoundPrt,
    /// `amod`: adjectival modifier.
    Amod,
    /// `advmod`: adverbial modifier.
    AdvMod,
    /// `punct`: punctuation.
    Punct,
    /// `discourse`: discourse element.
    Discourse,
    /// `mark`: subordinating marker.
    Mark,
    /// `expl`: expletive, such as existential `there`.
    Expl,
    /// Any other UD relation, preserved as its original string.
    Other(String),
}

impl DepRel {
    /// Parse a UD relation string into a typed variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "root" => Self::Root,
            "nsubj" => Self::NSubj,
            "nsubj:pass" => Self::NSubjPass,
            "obj" => Self::Obj,
            "aux" => Self::Aux,
            "aux:pass" => Self::AuxPass,
            "cop" => Self::Cop,
            "case" => Self::Case,
            "nmod:poss" => Self::NmodPoss,
            "det" => Self::Det,
            "cc" => Self::Cc,
            "conj" => Self::Conj,
            "compound" => Self::Compound,
            "compound:prt" => Self::CompoundPrt,
            "amod" => Self::Amod,
            "advmod" => Self::AdvMod,
            "punct" => Self::Punct,
            "discourse" => Self::Discourse,
            "mark" => Self::Mark,
            "expl" => Self::Expl,
            other => Self::Other(other.to_string()),
        }
    }

    /// Serialize back to the UD relation string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Root => "root",
            Self::NSubj => "nsubj",
            Self::NSubjPass => "nsubj:pass",
            Self::Obj => "obj",
            Self::Aux => "aux",
            Self::AuxPass => "aux:pass",
            Self::Cop => "cop",
            Self::Case => "case",
            Self::NmodPoss => "nmod:poss",
            Self::Det => "det",
            Self::Cc => "cc",
            Self::Conj => "conj",
            Self::Compound => "compound",
            Self::CompoundPrt => "compound:prt",
            Self::Amod => "amod",
            Self::AdvMod => "advmod",
            Self::Punct => "punct",
            Self::Discourse => "discourse",
            Self::Mark => "mark",
            Self::Expl => "expl",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Typed view over UD's `VerbForm` feature values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VerbForm {
    /// Finite form.
    Fin,
    /// Participle form.
    Part,
    /// Gerund.
    Ger,
    /// Infinitive form.
    Inf,
    /// Supine form.
    Sup,
    /// Converb.
    Conv,
    /// Verbal noun.
    Vnoun,
    /// Any other UD `VerbForm` value, preserved as written.
    Other(String),
}

impl VerbForm {
    /// Parse a UD `VerbForm` value into a typed variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "Fin" => Self::Fin,
            "Part" => Self::Part,
            "Ger" => Self::Ger,
            "Inf" => Self::Inf,
            "Sup" => Self::Sup,
            "Conv" => Self::Conv,
            "Vnoun" => Self::Vnoun,
            other => Self::Other(other.to_string()),
        }
    }

    /// Serialize this typed value back to its UD string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Fin => "Fin",
            Self::Part => "Part",
            Self::Ger => "Ger",
            Self::Inf => "Inf",
            Self::Sup => "Sup",
            Self::Conv => "Conv",
            Self::Vnoun => "Vnoun",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// The value of `key` in a UD MISC field (`Key=Value|Key=Value`). Splits on
/// the first `=` only, so a value may itself contain `=`. FEATS is not read
/// this way: it is a typed [`WordFeatures`] on the word.
pub fn ud_pair_value<'a>(field: Option<&'a str>, key: &str) -> Option<&'a str> {
    field?.split('|').find_map(|pair| {
        pair.split_once('=')
            .filter(|(k, _)| *k == key)
            .map(|(_, v)| v)
    })
}

/// UD IDs can be single integers (`1`), ranges (`1-2`) for MWTs, or decimals
/// (`1.1`) for empty nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdId {
    /// Regular word index.
    Single(usize),
    /// Multi-word token range.
    Range(usize, usize),
    /// Empty-node index.
    Decimal(f64),
}

/// The 1-based `ID` of a syntactic UD word (a `UdId::Single`, or the first
/// component a multi-word token's range starts at).
///
/// Never a position in `UdSentence::words`: range rows of multi-word tokens
/// occupy rows of their own, so the two diverge after the first contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UdWordId(NonZeroUsize);

impl UdWordId {
    /// The raw id, for diagnostics and for building a [`UdId`].
    pub fn get(self) -> usize {
        self.0.get()
    }

    /// The id a UD row stands for in the `%gra` builder: a syntactic word's
    /// own id, a multi-word token's first component; none for an empty node
    /// or the reserved id 0.
    pub fn of_row(word: &UdWord) -> Option<Self> {
        match word.id {
            UdId::Single(id) | UdId::Range(id, _) => NonZeroUsize::new(id).map(Self),
            UdId::Decimal(_) => None,
        }
    }

    /// Every id of a multi-word token's range `start..=end` (0 excluded).
    pub fn range(start: usize, end: usize) -> impl Iterator<Item = Self> {
        (start..=end).filter_map(NonZeroUsize::new).map(Self)
    }

    /// The id `by` places further on: where this word lands when `by` words
    /// are inserted before it.
    pub(crate) fn after(self, by: usize) -> Self {
        Self(self.0.saturating_add(by))
    }
}

impl std::fmt::Display for UdWordId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Where a UD word's `HEAD` points: the sentence root, or another word.
///
/// The analysis's `HEAD = 0` becomes [`UdHead::Root`] where the analysis is
/// admitted (`From<UdWordAnalysis> for UdWord`), once; nothing downstream
/// compares a head with 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdHead {
    /// The word is the sentence root.
    Root,
    /// The word depends on this word.
    Word(UdWordId),
}

impl UdHead {
    /// The CoNLL-U `HEAD` column: `0` is the root.
    fn from_conllu(head: usize) -> Self {
        match NonZeroUsize::new(head) {
            None => Self::Root,
            Some(id) => Self::Word(UdWordId(id)),
        }
    }

    /// The word this head names; `None` for the root.
    pub fn word(self) -> Option<UdWordId> {
        match self {
            Self::Root => None,
            Self::Word(id) => Some(id),
        }
    }

    /// Whether this head names `id`.
    pub fn is(self, id: UdWordId) -> bool {
        self == Self::Word(id)
    }

    /// This head with the word it names moved by `f`; the root stays the
    /// root.
    pub(crate) fn map_word(self, f: impl FnOnce(UdWordId) -> UdWordId) -> Self {
        match self {
            Self::Root => Self::Root,
            Self::Word(id) => Self::Word(f(id)),
        }
    }
}

/// Test fixtures state heads as the CoNLL-U column does.
#[cfg(test)]
impl UdHead {
    /// The CoNLL-U `HEAD` value: `0` for the root.
    pub(crate) fn conllu(self) -> usize {
        match self {
            Self::Root => 0,
            Self::Word(id) => id.get(),
        }
    }

    /// The head a CoNLL-U `HEAD` value names.
    pub(crate) fn of_conllu(head: usize) -> Self {
        Self::from_conllu(head)
    }
}

/// Wrapper for UD fields that may contain either a semantic value or raw
/// punctuation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdPunctable<T> {
    /// A semantic value.
    Value(T),
    /// A punctuation token with no semantic category.
    Punct(String),
}

/// One word exactly as the analysis engine produced it: the record the worker
/// sends, and the only route by which a word's features are admitted as the
/// analysis's ([`FeatSource::Analysis`]). Deserializing a [`UdWord`] goes
/// through it.
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
pub struct UdWordAnalysis {
    /// Word index within the sentence.
    pub id: UdId,
    /// Surface form.
    pub text: String,
    /// Lemma or stem.
    pub lemma: String,
    /// Universal part-of-speech tag.
    pub upos: UdPunctable<UniversalPos>,
    /// Language-specific part-of-speech tag, if present.
    pub xpos: Option<String>,
    /// The FEATS column as the engine wrote it, if present.
    pub feats: Option<String>,
    /// Head token index, with `0` meaning root.
    pub head: usize,
    /// Universal dependency relation to the head.
    pub deprel: String,
    /// Enhanced dependency information, if present.
    pub deps: Option<String>,
    /// Miscellaneous annotation, if present.
    pub misc: Option<String>,
}

/// Every feature of an analysis record is the analysis's: this conversion
/// is where that is stated, once.
impl From<UdWordAnalysis> for UdWord {
    fn from(word: UdWordAnalysis) -> Self {
        Self {
            id: word.id,
            text: word.text,
            lemma: word.lemma,
            upos: word.upos,
            xpos: word.xpos,
            feats: word.feats.as_deref().map_or(WordFeatures::NONE, |feats| {
                WordFeatures::parse(feats, FeatSource::Analysis)
            }),
            head: UdHead::from_conllu(word.head),
            deprel: word.deprel,
            deps: word.deps,
            misc: word.misc,
        }
    }
}

/// Typed UD token record used by the morphosyntax mapping layer.
///
/// Its features are private: they enter from the analysis
/// ([`UdWordAnalysis`]) or from a curated table ([`Self::apply_curated`]),
/// and every value carries which ([`FeatSource`]).
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
#[serde(from = "UdWordAnalysis")]
pub struct UdWord {
    /// Word index within the sentence.
    pub id: UdId,
    /// Surface form.
    pub text: String,
    /// Lemma or stem.
    pub lemma: String,
    /// Universal part-of-speech tag.
    pub upos: UdPunctable<UniversalPos>,
    /// Language-specific part-of-speech tag, if present.
    pub xpos: Option<String>,
    /// UD features, each with its source.
    feats: WordFeatures,
    /// Where the word attaches: the root or another word.
    pub head: UdHead,
    /// Universal dependency relation to the head.
    pub deprel: String,
    /// Enhanced dependency information, if present.
    pub deps: Option<String>,
    /// Miscellaneous annotation, if present. Carries what the worker sends
    /// beside the analysis (`VerbReadingLemma`); never feature provenance.
    pub misc: Option<String>,
}

impl UdWord {
    /// Typed view over this word's dependency relation.
    pub fn dep_rel(&self) -> DepRel {
        DepRel::parse(&self.deprel)
    }

    /// This word's features.
    pub fn features(&self) -> &WordFeatures {
        &self.feats
    }

    /// Whether this word carries a finite-verb marker (`VerbForm=Fin`).
    pub fn has_finite_verb_form(&self) -> bool {
        self.feats.value(FeatName::VerbForm) == Some("Fin")
    }

    /// Replace this word's features with a curated table's entry, every value
    /// [`FeatSource::Curated`]. The only writer of curated values.
    pub(crate) fn apply_curated(&mut self, table: CuratedFeats) {
        self.feats = table.features();
    }

    /// Drop every feature: a rewrite gave the word a category the analysis's
    /// features do not describe (an interjection).
    pub(crate) fn clear_features(&mut self) {
        self.feats = WordFeatures::NONE;
    }

    /// Take the analysis's values of `names` from `source` (the word this one
    /// was split from) where this word's table does not fix them.
    pub(crate) fn adopt_observed(&mut self, source: &WordFeatures, names: &[FeatName]) {
        self.feats.adopt(source, names);
    }

    /// A word one of our tables synthesizes at `id`, attached at `head`: its
    /// analysis, features included, is the table's.
    pub(crate) fn synthetic(
        id: UdId,
        text: impl Into<String>,
        lemma: impl Into<String>,
        upos: UniversalPos,
        feats: CuratedFeats,
        head: UdHead,
        deprel: impl Into<String>,
    ) -> Self {
        let mut word = Self {
            id,
            text: text.into(),
            lemma: lemma.into(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: WordFeatures::NONE,
            head,
            deprel: deprel.into(),
            deps: None,
            misc: None,
        };
        word.apply_curated(feats);
        word
    }

    /// A table's reading of the word `row` holds: the table's text, lemma,
    /// category and features, at the row's id, head and relation. For the
    /// `%mor` mapper, which reads only the reading; where the word sits in
    /// the tree stays the row's.
    pub(crate) fn curated_reading(
        row: &UdWord,
        text: impl Into<String>,
        lemma: impl Into<String>,
        upos: UniversalPos,
        feats: CuratedFeats,
    ) -> Self {
        Self::synthetic(
            row.id.clone(),
            text,
            lemma,
            upos,
            feats,
            row.head,
            row.deprel.clone(),
        )
    }
}

/// A single UD sentence: an ordered sequence of token records.
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
pub struct UdSentence {
    /// Ordered token records for this sentence.
    pub words: Vec<UdWord>,
}

impl UdSentence {
    /// The id spans of every multi-word token in this sentence (`gonna`
    /// arrives as a `Range(1, 2)` parent followed by `gon` = 1 and `na` = 2).
    /// Components carry Stanza's sub-token analysis, not a CHAT word's, which
    /// is why word-level rewrites must not touch them.
    pub fn mwt_component_ranges(
        &self,
    ) -> impl Iterator<Item = std::ops::RangeInclusive<usize>> + '_ {
        self.words.iter().filter_map(|w| match w.id {
            UdId::Range(start, end) => Some(start..=end),
            UdId::Single(_) | UdId::Decimal(_) => None,
        })
    }

    /// Whether the single-token id `id` is a component of a multi-word token.
    pub fn is_mwt_component(&self, id: usize) -> bool {
        self.mwt_component_ranges().any(|r| r.contains(&id))
    }
}

/// Top-level UD response for one utterance.
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
pub struct UdResponse {
    /// One or more UD sentences produced by the NLP engine.
    pub sentences: Vec<UdSentence>,
}

/// Apply post-parse validation and cleanup to one Stanza-produced UD word.
pub fn validate_and_clean(word: &mut UdWord) {
    if word.deprel.starts_with('<') && word.deprel.ends_with('>') {
        tracing::warn!(
            deprel = %word.deprel,
            text = %word.text,
            "Stanza emitted pad deprel: replacing with 'dep'"
        );
        word.deprel = "dep".to_string();
    }

    if !matches!(word.id, UdId::Range(_, _)) && is_bogus_lemma(&word.text, &word.lemma) {
        tracing::warn!(
            lemma = %word.lemma,
            text = %word.text,
            "Stanza returned bogus lemma: falling back to surface form"
        );
        word.lemma = word.text.clone();
    }
}

/// Detect when Stanza returns a pure-punctuation lemma for a word with letters.
pub fn is_bogus_lemma(text: &str, lemma: &str) -> bool {
    if text == lemma || lemma.is_empty() {
        return false;
    }

    let text_has_letters = text.chars().any(|c| c.is_alphabetic());
    let lemma_all_punct = lemma
        .chars()
        .all(|c| !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control());

    text_has_letters && lemma_all_punct
}

/// Sanitize a string for use in a `%mor` field by replacing structural
/// separators with underscores and stripping whitespace.
pub fn sanitize_mor_text(s: &str) -> String {
    let mut result = s.replace(['|', '#', '-', '&', '$', '~'], "_");
    result.retain(|c| !c.is_whitespace());
    result
}
