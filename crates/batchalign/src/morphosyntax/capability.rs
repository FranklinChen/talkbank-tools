//! Model capability refusal is not a verdict about CHAT validity.

use crate::api::LanguageCode3;
#[cfg(test)]
use crate::stanza_registry::StanzaRegistry;
use crate::stanza_registry::{
    Admitted, Morphosyntax, StanzaRefusal, StanzaRegistryLookup, admit_from,
};

/// The source language whose required analysis cannot be supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisLanguageRequirement {
    /// The file's primary declaration, outside CA pass-through.
    PrimaryHeader,
    /// An outstanding utterance's effective language, including precodes.
    EffectiveUtterance,
}

impl std::fmt::Display for AnalysisLanguageRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrimaryHeader => f.write_str("primary @Languages"),
            Self::EffectiveUtterance => f.write_str("effective utterance language"),
        }
    }
}

/// Why capability admission declined, not a guessed interpretation of prose:
/// no admitted registry (at the file's first admission only), or the
/// processors morphotag's requirement lacks for the language.
pub type AnalysisUnavailableReason = StanzaRefusal<Morphosyntax>;

/// Issued only by capability admission. Truthful CHAT must not be relabeled
/// to obtain analysis from a model for a different language.
#[derive(Debug, thiserror::Error)]
#[error(
    "morphotag analysis unavailable for {requirement} '{language}': {reason}. Keep truthful language declarations; use a backend with the required language models."
)]
pub struct AnalysisUnavailable {
    language: LanguageCode3,
    requirement: AnalysisLanguageRequirement,
    reason: AnalysisUnavailableReason,
}

impl AnalysisUnavailable {
    /// Truthful source language whose required analysis was declined.
    pub fn language(&self) -> &LanguageCode3 {
        &self.language
    }

    /// Distinguishes the file-level policy from outstanding utterance work.
    pub fn requirement(&self) -> AnalysisLanguageRequirement {
        self.requirement
    }

    /// Capability admission's typed reason, without parsing diagnostic prose.
    pub fn reason(&self) -> &AnalysisUnavailableReason {
        &self.reason
    }

    /// The file's morphosyntax language admission: the admitted runtime
    /// registry decides, and its absence refuses. There is no static list to
    /// fall back on. Admission does not certify CHAT validity, and CA
    /// pass-through declines before it is reached. Every later language of
    /// the file is admitted from the proof this returns
    /// ([`Admitted::sibling`], through [`Self::of_sibling`]).
    pub(crate) fn admit(
        requirement: AnalysisLanguageRequirement,
        language: &LanguageCode3,
        registry: &StanzaRegistryLookup,
    ) -> Result<Admitted<Morphosyntax>, Self> {
        admit_from(registry, language).map_err(|reason| Self {
            language: language.clone(),
            requirement,
            reason,
        })
    }

    /// Admit another language of an admitted file, by the registry that
    /// admitted the file: a refusal is a gap, never "no registry".
    pub(crate) fn of_sibling(
        admitted: &Admitted<Morphosyntax>,
        requirement: AnalysisLanguageRequirement,
        language: &LanguageCode3,
    ) -> Result<Admitted<Morphosyntax>, Self> {
        admitted.sibling(language).map_err(|gap| Self {
            language: language.clone(),
            requirement,
            reason: StanzaRefusal::Gap(gap),
        })
    }
}

/// A refusal as admission issues it, for tests of how one travels.
#[cfg(test)]
impl AnalysisUnavailable {
    /// `language` refused as a primary declaration by a registry that serves
    /// no language.
    pub(crate) fn primary_refusal_for_test(language: &LanguageCode3) -> Self {
        Self::admit(
            AnalysisLanguageRequirement::PrimaryHeader,
            language,
            &Ok(StanzaRegistry::admitting(&[]).shared()),
        )
        .expect_err("an empty registry admits nothing")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stanza_registry::StanzaProcessor::{Depparse, Lemma, Mwt, Pos, Tokenize};
    use crate::stanza_registry::{GapKind, StanzaRegistryUnavailable, reason};

    fn language(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("test language is ISO 639-3")
    }

    /// Kazakh with the processors Stanza 1.15's catalog ships for it, and
    /// Telugu without a lemmatizer, as that catalog ships it.
    fn registry() -> StanzaRegistryLookup {
        Ok(StanzaRegistry::serving(&[
            ("kaz", &[Tokenize, Mwt, Pos, Lemma, Depparse]),
            ("tel", &[Tokenize, Pos, Depparse]),
        ])
        .shared())
    }

    fn refusal(
        requirement: AnalysisLanguageRequirement,
        code: &str,
        registry: &StanzaRegistryLookup,
    ) -> Option<AnalysisUnavailableReason> {
        AnalysisUnavailable::admit(requirement, &language(code), registry)
            .err()
            .map(|refusal| refusal.reason().clone())
    }

    const BOTH: [AnalysisLanguageRequirement; 2] = [
        AnalysisLanguageRequirement::PrimaryHeader,
        AnalysisLanguageRequirement::EffectiveUtterance,
    ];

    /// RED FIRST: `kaz` was refused as a primary language by a static list
    /// that never named it, although the catalog ships every processor
    /// morphotag loads.
    #[test]
    fn a_registry_with_kazakh_processors_admits_kazakh() {
        let registry = registry();
        for requirement in BOTH {
            assert_eq!(refusal(requirement, "kaz", &registry), None);
        }
    }

    #[test]
    fn a_language_with_no_catalog_entry_is_refused_as_not_in_the_catalog() {
        let registry = registry();
        for requirement in BOTH {
            let Some(StanzaRefusal::Gap(gap)) = refusal(requirement, "que", &registry) else {
                panic!("Quechua has no catalog entry");
            };
            assert_eq!(gap.kind(), &GapKind::NotInCatalog);
        }
    }

    #[test]
    fn an_entry_without_a_morphotag_processor_is_refused_for_its_processors() {
        let registry = registry();
        for requirement in BOTH {
            let Some(StanzaRefusal::Gap(gap)) = refusal(requirement, "tel", &registry) else {
                panic!("Telugu ships no lemmatizer");
            };
            assert_eq!(
                gap.kind(),
                &GapKind::ProcessorsUnavailable {
                    missing: std::collections::BTreeSet::from([Lemma])
                }
            );
        }
    }

    /// No admitted registry refuses everything, English included: there is
    /// no static list to fall back on. The refusal carries the reason the
    /// registry is missing, so a failed worker report is not mistaken for a
    /// catalog without the language.
    #[test]
    fn without_an_admitted_registry_nothing_is_admitted_and_the_reason_is_kept() {
        let unavailable = StanzaRegistryUnavailable::ReportFailed {
            worker: "infer:morphosyntax:eng".to_owned(),
            reason: reason("ImportError: ISO_639_1_PAIRS"),
        };
        let lookup = Err(unavailable.clone());
        for requirement in BOTH {
            assert_eq!(
                refusal(requirement, "eng", &lookup),
                Some(StanzaRefusal::RegistryUnavailable(unavailable.clone()))
            );
        }
        let message = AnalysisUnavailable::admit(
            AnalysisLanguageRequirement::PrimaryHeader,
            &language("eng"),
            &lookup,
        )
        .expect_err("no registry admits nothing")
        .to_string();
        assert!(message.contains("ISO_639_1_PAIRS"), "{message}");
    }

    /// An utterance language is admitted from the file's proof: a refusal is
    /// a gap by construction.
    #[test]
    fn a_sibling_refusal_is_a_gap() {
        let file = AnalysisUnavailable::admit(
            AnalysisLanguageRequirement::PrimaryHeader,
            &language("kaz"),
            &registry(),
        )
        .expect("Kazakh");
        let refusal = AnalysisUnavailable::of_sibling(
            &file,
            AnalysisLanguageRequirement::EffectiveUtterance,
            &language("que"),
        )
        .expect_err("Quechua");
        assert!(matches!(refusal.reason(), StanzaRefusal::Gap(_)));
        assert_eq!(refusal.language(), &language("que"));
    }
}
