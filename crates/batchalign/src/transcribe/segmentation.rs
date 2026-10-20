//! Transcription can retain acoustic/ASR boundaries without a text NLP model.
//!
//! This route is deliberately separate from standalone `utseg`: preserving
//! source ASR punctuation and diarization is not a claim to segment arbitrary
//! existing CHAT with a Taiwanese constituency or boundary model.
use crate::api::LanguageCode3;
use crate::params::UtsegFallbackPolicy;
use crate::utseg_route::{UtsegRoute, UtsegUnavailable};

#[derive(Clone, Debug)]
pub(crate) struct TranscribeSegmentation(SegmentationRoute);

#[derive(Clone, Debug)]
enum SegmentationRoute {
    Model(UtsegRoute),
    AsrBoundaries { language: LanguageCode3 },
}

impl TranscribeSegmentation {
    pub(crate) fn resolve(
        language: &LanguageCode3,
        fallback: UtsegFallbackPolicy,
    ) -> Result<Self, UtsegUnavailable> {
        // Taiwanese has no boundary model; its boundaries are the recognizer's
        // own punctuation, which the Qwen profile restores for `nan`
        // (`RecognitionPunctuation.RESTORE` in `qwen_asr_profile.py`). The two
        // sides must agree: a profile that stopped restoring would leave this
        // route with speaker changes as its only boundaries.
        if language.as_ref() == "nan" {
            Ok(Self(SegmentationRoute::AsrBoundaries {
                language: language.clone(),
            }))
        } else {
            UtsegRoute::resolve(language, fallback)
                .map(|route| Self(SegmentationRoute::Model(route)))
        }
    }

    pub(crate) fn language(&self) -> &LanguageCode3 {
        match &self.0 {
            SegmentationRoute::Model(route) => route.language(),
            SegmentationRoute::AsrBoundaries { language } => language,
        }
    }

    pub(crate) fn model(&self) -> Option<&UtsegRoute> {
        match &self.0 {
            SegmentationRoute::Model(route) => Some(route),
            SegmentationRoute::AsrBoundaries { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn taiwanese_preserves_asr_boundaries_without_claiming_a_text_model() {
        let nan = LanguageCode3::try_new("nan").unwrap();
        let route = TranscribeSegmentation::resolve(&nan, UtsegFallbackPolicy::Refuse).unwrap();
        assert_eq!(route.language(), &nan);
        assert!(route.model().is_none());
        assert!(UtsegRoute::resolve(&nan, UtsegFallbackPolicy::Refuse).is_err());
    }
}
