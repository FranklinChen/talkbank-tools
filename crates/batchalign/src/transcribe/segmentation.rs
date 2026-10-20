//! Transcription can retain acoustic/ASR boundaries without a text NLP model.
//!
//! This route is deliberately separate from standalone `utseg`: preserving
//! source ASR punctuation and diarization is not a claim to segment arbitrary
//! existing CHAT with a Taiwanese constituency or boundary model.
use crate::api::LanguageCode3;
use crate::params::UtsegFallbackPolicy;
use crate::types::worker_v2::{DecodeWindowV2, QwenRecognitionV2, RecognitionPunctuationV2};
use crate::utseg_route::{UtsegRoute, UtsegUnavailable};

/// A language whose transcription keeps the recognizer's own utterance
/// boundaries, because no boundary model exists for it.
///
/// The ONE table of such languages. The segmentation route
/// ([`TranscribeSegmentation::resolve`]) and the Qwen recognition policy sent
/// to the worker ([`qwen_recognition`]) both read it, so the route that relies
/// on recognizer punctuation and the recognizer that restores it cannot
/// disagree.
struct RecognizerBoundaryLanguage {
    code: &'static str,
    /// Qwen decoding window: long windows let spontaneous speech in this
    /// language drift into repetition.
    qwen_window_seconds: std::num::NonZeroU32,
}

const RECOGNIZER_BOUNDARY_LANGUAGES: [RecognizerBoundaryLanguage; 1] =
    [RecognizerBoundaryLanguage {
        code: "nan",
        qwen_window_seconds: std::num::NonZeroU32::new(30).expect("nonzero"),
    }];

fn recognizer_boundary_language(
    language: &LanguageCode3,
) -> Option<&'static RecognizerBoundaryLanguage> {
    RECOGNIZER_BOUNDARY_LANGUAGES
        .iter()
        .find(|entry| entry.code == language.as_ref())
}

/// How a Qwen recognizer runs for `language`: restored punctuation and a
/// bounded window where the language keeps recognizer boundaries, otherwise
/// the chunker default with punctuation left to segmentation. An auto-detect
/// request (`None`) has no recognizer-boundary language.
pub(crate) fn qwen_recognition(language: Option<&LanguageCode3>) -> QwenRecognitionV2 {
    match language.and_then(recognizer_boundary_language) {
        Some(entry) => QwenRecognitionV2 {
            window: DecodeWindowV2::BoundedSeconds {
                seconds: entry.qwen_window_seconds,
            },
            punctuation: RecognitionPunctuationV2::Restore,
        },
        None => QwenRecognitionV2 {
            window: DecodeWindowV2::ChunkerDefault,
            punctuation: RecognitionPunctuationV2::Discard,
        },
    }
}

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
        // The same table decides the Qwen policy (`qwen_recognition`), so a
        // route that relies on recognizer punctuation always gets a recognizer
        // told to restore it.
        if recognizer_boundary_language(language).is_some() {
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

    /// The route and the recognizer policy come from one table: a language
    /// that keeps recognizer boundaries is told to restore punctuation, and a
    /// language with a boundary model is told to discard it.
    #[test]
    fn the_route_and_the_qwen_policy_agree_for_every_language() {
        // Every table language plus languages with and without a boundary
        // model of their own (`spa` falls back to Stanza, not to recognizer
        // boundaries).
        let codes = RECOGNIZER_BOUNDARY_LANGUAGES
            .iter()
            .map(|entry| entry.code)
            .chain(["yue", "zho", "eng", "spa"]);
        for code in codes {
            let language = LanguageCode3::try_new(code).unwrap();
            let keeps_boundaries =
                TranscribeSegmentation::resolve(&language, UtsegFallbackPolicy::AllowStanza)
                    .map(|route| route.model().is_none())
                    .unwrap_or(false);
            let restores =
                qwen_recognition(Some(&language)).punctuation == RecognitionPunctuationV2::Restore;
            assert_eq!(keeps_boundaries, restores, "{code}");
        }
        assert_eq!(
            qwen_recognition(Some(&LanguageCode3::try_new("nan").unwrap())).window,
            DecodeWindowV2::BoundedSeconds {
                seconds: std::num::NonZeroU32::new(30).unwrap()
            }
        );
        assert_eq!(
            qwen_recognition(None).window,
            DecodeWindowV2::ChunkerDefault
        );
    }
}
