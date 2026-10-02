//! A UD word's FEATS column as a typed value, each feature carrying where its
//! value came from.
//!
//! A feature value has exactly two possible sources, and nothing else can make
//! one:
//!
//! - **the analysis**: the FEATS column the engine (Stanza) returned, admitted
//!   once where a word enters the pipeline ([`super::UdWordAnalysis`], which is
//!   also what deserialization goes through);
//! - **a curated table of ours** ([`CuratedFeats`]): a `&'static str` compiled
//!   into the binary and checked at compile time, written onto a word only by
//!   [`super::UdWord::apply_curated`].
//!
//! The source is a field of the value, not an annotation beside it (UD's MISC
//! column used to carry it), so no value the engine sends can claim to be
//! curated, and a later rewrite that replaces a word's features cannot leave a
//! record of the old ones behind.
//!
//! There is one FEATS splitter, [`WordFeatures::parse`]; every reader goes
//! through the typed lookups below by [`FeatName`].

use std::fmt;

/// The UD feature names the pipeline reads: a closed set, so a misspelled name
/// is a compile error rather than a silently absent feature. Features with
/// other names are kept (and written back by [`WordFeatures`]'s `Display`) but
/// nothing looks them up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatName {
    VerbForm,
    Aspect,
    Mood,
    Tense,
    Polarity,
    Polite,
    HebBinyan,
    HebExistential,
    Number,
    Person,
    PronType,
    Case,
    Reflex,
    Gender,
    Definite,
    NumberPsor,
    PersonPsor,
    Degree,
}

impl FeatName {
    /// The name as UD writes it in a FEATS string.
    const fn ud_name(self) -> &'static str {
        match self {
            Self::VerbForm => "VerbForm",
            Self::Aspect => "Aspect",
            Self::Mood => "Mood",
            Self::Tense => "Tense",
            Self::Polarity => "Polarity",
            Self::Polite => "Polite",
            Self::HebBinyan => "HebBinyan",
            Self::HebExistential => "HebExistential",
            Self::Number => "Number",
            Self::Person => "Person",
            Self::PronType => "PronType",
            Self::Case => "Case",
            Self::Reflex => "Reflex",
            Self::Gender => "Gender",
            Self::Definite => "Definite",
            Self::NumberPsor => "Number[psor]",
            Self::PersonPsor => "Person[psor]",
            Self::Degree => "Degree",
        }
    }
}

/// Where one feature value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatSource {
    /// The analysis engine assigned it (the word's FEATS as received).
    Analysis,
    /// One of our curated tables supplied it ([`CuratedFeats`]).
    Curated,
}

/// One `Name=Value` pair with its source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Feature {
    name: Box<str>,
    value: Box<str>,
    source: FeatSource,
}

/// A feature value borrowed from a word, with where it came from. Only
/// [`WordFeatures::get`] makes one, so no literal can pass for a value either
/// source assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatValue<'a> {
    text: &'a str,
    source: FeatSource,
}

impl<'a> FeatValue<'a> {
    /// The value as written (`Plur`).
    pub fn text(self) -> &'a str {
        self.text
    }

    /// Where the value came from.
    pub fn source(self) -> FeatSource {
        self.source
    }
}

/// A word's features in FEATS order, each with its source.
///
/// Constructed only by [`Self::parse`] (from the analysis, or from a curated
/// table), so every value has exactly one source and a word's features are
/// replaced whole by each writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordFeatures(Vec<Feature>);

impl WordFeatures {
    /// No features.
    pub(super) const NONE: Self = Self(Vec::new());

    /// The one FEATS splitter: `Name=Value|Name=Value`, every value from
    /// `source`. An empty string or UD's `_` is no features; a later pair with
    /// a repeated name overrides the earlier one; a pair without `=` is not a
    /// feature and is skipped (the analysis boundary logs it, and a curated
    /// table cannot contain one: [`CuratedFeats::new`] refuses it at compile
    /// time). The value is everything after the first `=`.
    pub(super) fn parse(text: &str, source: FeatSource) -> Self {
        let mut features: Vec<Feature> = Vec::new();
        for pair in text.split('|') {
            if pair.is_empty() || pair == "_" {
                continue;
            }
            let Some((name, value)) = pair.split_once('=') else {
                tracing::warn!(
                    pair,
                    feats = text,
                    "FEATS pair without `=` is not a feature"
                );
                continue;
            };
            let feature = Feature {
                name: name.into(),
                value: value.into(),
                source,
            };
            match features.iter_mut().find(|f| *f.name == *name) {
                Some(existing) => *existing = feature,
                None => features.push(feature),
            }
        }
        Self(features)
    }

    /// The value of `name`, with where it came from.
    pub(crate) fn get(&self, name: FeatName) -> Option<FeatValue<'_>> {
        let ud_name = name.ud_name();
        self.0
            .iter()
            .find(|f| *f.name == *ud_name)
            .map(|f| FeatValue {
                text: &f.value,
                source: f.source,
            })
    }

    /// The text of `name`'s value, when only the text matters.
    pub(crate) fn value(&self, name: FeatName) -> Option<&str> {
        self.get(name).map(FeatValue::text)
    }

    /// Every feature in FEATS order: its UD name, and its value with where
    /// the value came from.
    pub fn iter(&self) -> impl Iterator<Item = (&str, FeatValue<'_>)> {
        self.0.iter().map(|f| {
            (
                &*f.name,
                FeatValue {
                    text: &f.value,
                    source: f.source,
                },
            )
        })
    }

    /// Whether the word has no features.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Take from `source` the values of `names` this word does not have,
    /// keeping their source; then order the features by name, as UD writes
    /// them.
    pub(super) fn adopt(&mut self, source: &Self, names: &[FeatName]) {
        for name in names {
            let ud_name = name.ud_name();
            if self.0.iter().any(|f| *f.name == *ud_name) {
                continue;
            }
            if let Some(feature) = source.0.iter().find(|f| *f.name == *ud_name) {
                self.0.push(feature.clone());
            }
        }
        self.0.sort_by(|a, b| {
            a.name
                .to_ascii_lowercase()
                .cmp(&b.name.to_ascii_lowercase())
        });
    }
}

/// The FEATS string: `Name=Value|Name=Value`, empty for no features.
impl fmt::Display for WordFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, value)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str("|")?;
            }
            write!(f, "{name}={}", value.text())?;
        }
        Ok(())
    }
}

/// A curated table's feature bundle: a FEATS string compiled into the binary.
///
/// [`Self::new`] is a `const fn` that refuses a malformed string, so a table
/// entry with a pair lacking `=`, an empty name or an empty value fails to
/// compile. Being `&'static str`, it cannot be built from data received at run
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuratedFeats(&'static str);

impl CuratedFeats {
    /// A table entry that supplies no features.
    pub const NONE: Self = Self("");

    /// A table's FEATS string (`Mood=Imp|Number=Sing`), checked at compile
    /// time when used in a constant (every table is one).
    pub const fn new(feats: &'static str) -> Self {
        let bytes = feats.as_bytes();
        let mut i = 0;
        // Per pair: where it starts and where its `=` is.
        let mut start = 0;
        let mut eq: Option<usize> = None;
        while i <= bytes.len() {
            let at_end = i == bytes.len();
            if at_end || bytes[i] == b'|' {
                if !(at_end && i == 0) {
                    assert!(eq.is_some(), "curated FEATS pair without `=`");
                    if let Some(at) = eq {
                        assert!(at > start, "curated FEATS pair with an empty name");
                        assert!(at + 1 < i, "curated FEATS pair with an empty value");
                    }
                }
                start = i + 1;
                eq = None;
            } else if bytes[i] == b'=' && eq.is_none() {
                eq = Some(i);
            }
            i += 1;
        }
        Self(feats)
    }

    /// The table's features, every one curated.
    pub(super) fn features(self) -> WordFeatures {
        WordFeatures::parse(self.0, FeatSource::Curated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_keeps_commas_inside_a_value() {
        let feats = WordFeatures::parse("PronType=Int,Rel|Person=3", FeatSource::Analysis);
        assert_eq!(feats.value(FeatName::PronType), Some("Int,Rel"));
        assert_eq!(feats.value(FeatName::Person), Some("3"));
    }

    #[test]
    fn parse_reads_no_features_from_empty_or_underscore() {
        assert!(WordFeatures::parse("", FeatSource::Analysis).is_empty());
        assert!(WordFeatures::parse("_", FeatSource::Analysis).is_empty());
    }

    #[test]
    fn a_later_duplicate_overrides_and_a_pair_without_equals_is_skipped() {
        let feats = WordFeatures::parse("Number=Sing|Bogus|Number=Plur", FeatSource::Analysis);
        assert_eq!(feats.value(FeatName::Number), Some("Plur"));
        assert_eq!(feats.to_string(), "Number=Plur");
    }

    #[test]
    fn every_value_carries_its_source() {
        let curated = CuratedFeats::new("Case=Acc|Number=Sing").features();
        assert_eq!(
            curated.get(FeatName::Case).map(FeatValue::source),
            Some(FeatSource::Curated)
        );
        let mut adopted = CuratedFeats::new("VerbForm=Fin").features();
        adopted.adopt(
            &WordFeatures::parse("Person=3|Number=Sing|Case=Nom", FeatSource::Analysis),
            &[FeatName::Person, FeatName::Number],
        );
        assert_eq!(adopted.to_string(), "Number=Sing|Person=3|VerbForm=Fin");
        assert_eq!(
            adopted.get(FeatName::Person).map(FeatValue::source),
            Some(FeatSource::Analysis)
        );
        assert_eq!(
            adopted.get(FeatName::VerbForm).map(FeatValue::source),
            Some(FeatSource::Curated)
        );
        assert_eq!(adopted.get(FeatName::Case), None);
    }

    #[test]
    fn adopt_never_overrides_what_the_table_fixed() {
        let mut feats = CuratedFeats::new("Number=Sing|Person=3|VerbForm=Fin").features();
        feats.adopt(
            &WordFeatures::parse("Number=Plur|Person=1", FeatSource::Analysis),
            &[FeatName::Person, FeatName::Number],
        );
        assert_eq!(feats.to_string(), "Number=Sing|Person=3|VerbForm=Fin");
        assert_eq!(
            feats.get(FeatName::Number).map(FeatValue::source),
            Some(FeatSource::Curated)
        );
    }
}
