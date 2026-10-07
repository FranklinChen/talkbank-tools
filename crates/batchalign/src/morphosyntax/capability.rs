//! Model capability refusal is not a verdict about CHAT validity.

use crate::api::LanguageCode3;
use crate::stanza_registry::StanzaRegistry;

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

/// Why capability admission declined, not a guessed interpretation of prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisUnavailableReason {
    /// The retained file-level supported-language policy excludes this code.
    #[error("not supported by Stanza")]
    UnsupportedPrimary,
    /// No worker has supplied an admitted language capability inventory.
    #[error("the runtime has no admitted Stanza language registry")]
    RegistryUnavailable,
    /// The admitted runtime inventory lacks required morphology processors.
    #[error("the runtime lacks the required Stanza morphology processors")]
    ProcessorsUnavailable,
}

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
    pub fn reason(&self) -> AnalysisUnavailableReason {
        self.reason
    }

    /// Retain the file-level support policy even for nonlexical transcripts.
    /// CA declines before this boundary; this does not certify CHAT validity.
    pub(crate) fn admit_primary(language: &LanguageCode3) -> Result<(), Self> {
        if batchalign_transform::morphosyntax::supported_iso3_codes()
            .binary_search(&language.as_ref())
            .is_ok()
        {
            Ok(())
        } else {
            Err(Self {
                language: language.clone(),
                requirement: AnalysisLanguageRequirement::PrimaryHeader,
                reason: AnalysisUnavailableReason::UnsupportedPrimary,
            })
        }
    }

    /// Required primary work must never acquire a secondary-placeholder result.
    pub(super) fn admit_effective(
        language: &LanguageCode3,
        registry: Option<&StanzaRegistry>,
    ) -> Result<(), Self> {
        let reason = match registry {
            Some(registry) if registry.supports_morphosyntax(language.as_ref()) => return Ok(()),
            Some(_) => AnalysisUnavailableReason::ProcessorsUnavailable,
            None => AnalysisUnavailableReason::RegistryUnavailable,
        };
        Err(Self {
            language: language.clone(),
            requirement: AnalysisLanguageRequirement::EffectiveUtterance,
            reason,
        })
    }
}
