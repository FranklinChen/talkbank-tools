//! French-specific morphosyntax rules.

/// The case our French pronoun table assigns a surface form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrenchPronounCase {
    /// Nominative (`je`, `il`, ...).
    Nom,
    /// Accusative (`me`, `le`, ...).
    Acc,
}

impl FrenchPronounCase {
    /// The `%mor` suffix text.
    pub const fn text(self) -> &'static str {
        match self {
            Self::Nom => "Nom",
            Self::Acc => "Acc",
        }
    }
}

/// French pronoun case from the surface word form; `None` for a form the
/// table does not list. (`lui` is listed as both and reads as nominative.)
pub fn french_pronoun_case(text: &str) -> Option<FrenchPronounCase> {
    let clean = text.to_lowercase();
    let clean = clean.trim_matches('\'').trim();

    if PRON_NOM.contains(&clean) {
        Some(FrenchPronounCase::Nom)
    } else if PRON_ACC.contains(&clean) {
        Some(FrenchPronounCase::Acc)
    } else {
        None
    }
}

const PRON_NOM: &[&str] = &[
    "qui", "je", "tu", "il", "elle", "ils", "elles", "moi", "toi", "lui", "eux",
];

const PRON_ACC: &[&str] = &["que", "quoi", "me", "te", "le", "la", "les", "lui", "leur"];

/// Return `true` if the word is a French noun with auditory plural marking.
pub fn is_apm_noun(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    APM_NOUNS.contains(&lower.as_str())
}

const APM_NOUNS: &[&str] = &[
    "amiral",
    "amiraux",
    "animal",
    "animaux",
    "annal",
    "annaux",
    "anormal",
    "anormaux",
    "anticlérical",
    "anticléricaux",
    "arsenal",
    "arsenaux",
    "bocal",
    "bocaux",
    "canal",
    "canaux",
    "cantal",
    "cantaux",
    "capital",
    "capitaux",
    "caporal",
    "caporaux",
    "cardinal",
    "cardinaux",
    "central",
    "centraux",
    "chenal",
    "chenaux",
    "cheval",
    "chevaux",
    "clérical",
    "cléricaux",
    "collatéral",
    "collatéraux",
    "colonial",
    "coloniaux",
    "commensal",
    "commensaux",
    "communal",
    "communaux",
    "confessionnal",
    "confessionnaux",
    "cordial",
    "cordiaux",
    "corporal",
    "corporaux",
    "cristal",
    "cristaux",
    "cérébral",
    "cérébraux",
    "fanal",
    "fanaux",
    "frontal",
    "frontaux",
    "fédéral",
    "fédéraux",
    "féodal",
    "féodaux",
    "gardénal",
    "gardénaux",
    "général",
    "généraux",
    "hôpital",
    "hôpitaux",
    "idéal",
    "idéaux",
    "international",
    "internationaux",
    "journal",
    "journaux",
    "libéral",
    "libéraux",
    "local",
    "locaux",
    "madrigal",
    "madrigaux",
    "marsupial",
    "marsupiaux",
    "maréchal",
    "maréchaux",
    "mal",
    "maux",
    "minéral",
    "minéraux",
    "moral",
    "moraux",
    "méridional",
    "méridionaux",
    "métal",
    "métaux",
    "nasal",
    "nasaux",
    "national",
    "nationaux",
    "normal",
    "normaux",
    "numéral",
    "numéraux",
    "occidental",
    "occidentaux",
    "occipital",
    "occipitaux",
    "oral",
    "oraux",
    "ordinal",
    "ordinaux",
    "oriental",
    "orientaux",
    "original",
    "originaux",
    "piédestal",
    "piédestaux",
    "principal",
    "principaux",
    "provincial",
    "provinciaux",
    "quintal",
    "quintaux",
    "radical",
    "radicaux",
    "rival",
    "rivaux",
    "rural",
    "ruraux",
    "régional",
    "régionaux",
    "sentimental",
    "sentimentaux",
    "signal",
    "signaux",
    "social",
    "sociaux",
    "sénéchal",
    "sénéchaux",
    "temporal",
    "temporaux",
    "tergal",
    "tergaux",
    "total",
    "totaux",
    "tribunal",
    "tribunaux",
    "urinal",
    "urinaux",
    "vassal",
    "vassaux",
    "val",
    "vaux",
    "végétal",
    "végétaux",
    "véronal",
    "véronals",
    "éditorial",
    "éditoriaux",
    "égal",
    "égaux",
    "étal",
    "étau",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_french_pronoun_case_nom() {
        assert_eq!(french_pronoun_case("je"), Some(FrenchPronounCase::Nom));
        assert_eq!(french_pronoun_case("il"), Some(FrenchPronounCase::Nom));
        assert_eq!(french_pronoun_case("moi"), Some(FrenchPronounCase::Nom));
    }

    #[test]
    fn test_french_pronoun_case_acc() {
        assert_eq!(french_pronoun_case("me"), Some(FrenchPronounCase::Acc));
        assert_eq!(french_pronoun_case("le"), Some(FrenchPronounCase::Acc));
        assert_eq!(french_pronoun_case("leur"), Some(FrenchPronounCase::Acc));
    }

    #[test]
    fn test_french_pronoun_case_unknown() {
        assert_eq!(french_pronoun_case("nous"), None);
        assert_eq!(french_pronoun_case("vous"), None);
    }

    #[test]
    fn test_french_pronoun_case_with_apostrophe() {
        assert_eq!(french_pronoun_case("qu'"), None);
        assert_eq!(french_pronoun_case("je'"), Some(FrenchPronounCase::Nom));
    }

    #[test]
    fn test_apm_noun_singular() {
        assert!(is_apm_noun("cheval"));
        assert!(is_apm_noun("animal"));
    }

    #[test]
    fn test_apm_noun_plural() {
        assert!(is_apm_noun("chevaux"));
        assert!(is_apm_noun("animaux"));
    }

    #[test]
    fn test_apm_noun_not_apm() {
        assert!(!is_apm_noun("maison"));
        assert!(!is_apm_noun("chat"));
    }

    #[test]
    fn test_apm_noun_case_insensitive() {
        assert!(is_apm_noun("Cheval"));
        assert!(is_apm_noun("ANIMAL"));
    }
}
