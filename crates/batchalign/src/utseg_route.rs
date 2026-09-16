//! Which utterance-segmentation route a language takes, decided once.
//!
//! Utterance segmentation has two segmenters and they are not interchangeable:
//!
//! - A **TalkBank boundary model**, a per-word BERT classifier, which exists
//!   for a few languages only.
//! - **Stanza constituency parsing**, the legacy segmenter, which covers more
//!   languages at varying quality and is therefore an explicit operator opt-in
//!   ([`UtsegFallbackPolicy::AllowStanza`]) rather than a silent substitution.
//!
//! A language with neither has no route, and that is a property of the request
//! that is knowable the moment the language is known. It used to be discovered
//! at the far end of the pipeline instead: the Python worker re-derived model
//! availability from its own resolver table and raised only once a batch of
//! already-transcribed words reached it, so a `transcribe` job in an
//! unsupported language ran an entire ASR pass, paid for it, and then failed
//! with `invalid utseg V2 result`, which names the wire format rather than the
//! missing model.
//!
//! That refusal also had a hole. The worker refused only if some item still
//! needed a segmenter, and it excluded items of one word (those get a trivial
//! single-group assignment first), so a batch whose chunks were all single
//! words skipped the refusal entirely and reported success for a language with
//! no segmenter at all.
//!
//! [`UtsegRoute::resolve`] answers the question once, in Rust, from the one
//! table below, before any work is dispatched. It answers from the language
//! alone, so no content can route around it.

use crate::api::LanguageCode3;
use crate::params::UtsegFallbackPolicy;

/// Whether one language has a TalkBank utterance-boundary model.
///
/// **There is one table, and it is the manifest's.** This fact used to live in
/// three places: a `matches!` in `pipeline/transcribe.rs`, a `&[&str]` list
/// here, and the key set of `_RESOLVER["utterance"]` in
/// `batchalign/models/resolve.py`, which the worker consulted to decide whether
/// to refuse. Three tables of one fact is how they come to disagree, and the
/// Python copy was the one that decided a job's fate after the expensive stage
/// had already run.
///
/// Availability is now a CONSEQUENCE of the pin rather than a parallel list:
/// a language has a boundary model exactly when
/// [`crate::model_manifest::utseg_boundary_model`] pins one for it. Adding a
/// language is therefore one edit, to the manifest, and it cannot produce a
/// language this build claims to segment but cannot name a model for.
fn has_boundary_model(lang: &LanguageCode3) -> bool {
    crate::model_manifest::utseg_boundary_model(lang).is_some()
}

/// How utterance segmentation will be performed for one language.
///
/// Both variants are RUNNABLE. "No route" is not a third variant here: it is
/// [`UtsegUnavailable`], the error arm of [`UtsegRoute::resolve`], so a route
/// that cannot run cannot be constructed, cannot be stored in a plan, and
/// cannot be handed to a worker. A `Refused` variant would have been a value
/// every consumer had to remember to reject, which is the shape that let the
/// old refusal be skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UtsegRoute {
    /// The TalkBank per-word boundary model for this language.
    BoundaryModel,
    /// Stanza constituency parsing, explicitly authorized by the operator.
    StanzaFallback,
}

/// No segmenter exists for this language under the job's fallback policy.
///
/// Carries the language, because naming what was actually asked for is the
/// whole point: the failure this replaces named `invalid utseg V2 result`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "utterance segmentation has no TalkBank boundary model for language '{lang}', and the \
     Stanza constituency fallback was not authorized. Re-run with `--utseg-fallback-stanza` to \
     use it (quality varies by language), or run without utterance segmentation."
)]
pub(crate) struct UtsegUnavailable {
    /// The language that has no segmenter.
    lang: LanguageCode3,
}

impl UtsegRoute {
    /// Decide how this language will be segmented, or refuse.
    ///
    /// The ONE constructor, and it consults only the language and the operator's
    /// policy. Deliberately independent of the payload: the refusal it replaces
    /// was conditional on a batch still having multi-word items to segment, so a
    /// batch of single words bypassed it.
    pub(crate) fn resolve(
        lang: &LanguageCode3,
        fallback: UtsegFallbackPolicy,
    ) -> Result<Self, UtsegUnavailable> {
        if has_boundary_model(lang) {
            return Ok(Self::BoundaryModel);
        }
        match fallback {
            UtsegFallbackPolicy::AllowStanza => Ok(Self::StanzaFallback),
            UtsegFallbackPolicy::Refuse => Err(UtsegUnavailable { lang: lang.clone() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lang(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("test language code must be valid ISO 639-3")
    }

    /// Every language the boundary-model table names takes the model route,
    /// under either policy: an available model is never substituted.
    #[test]
    fn boundary_model_languages_take_the_model_route() {
        for code in ["eng", "cmn", "zho", "yue"] {
            for policy in [
                UtsegFallbackPolicy::Refuse,
                UtsegFallbackPolicy::AllowStanza,
            ] {
                assert_eq!(
                    UtsegRoute::resolve(&lang(code), policy),
                    Ok(UtsegRoute::BoundaryModel),
                    "{code} has a boundary model and must use it"
                );
            }
        }
    }

    /// RED FIRST: the defect. A language with no boundary model and no opt-in
    /// is refused HERE, by the language alone, rather than after a full ASR
    /// pass, and the message names the language instead of the wire format.
    #[test]
    fn an_unsupported_language_is_refused_by_the_language_alone() {
        let refusal = UtsegRoute::resolve(&lang("spa"), UtsegFallbackPolicy::Refuse)
            .expect_err("Spanish has no TalkBank boundary model");
        let message = refusal.to_string();
        assert!(
            message.contains("spa"),
            "the refusal must name the language it refused: {message}"
        );
        assert!(
            message.contains("--utseg-fallback-stanza"),
            "the refusal must name the remedy: {message}"
        );
        assert!(
            !message.contains("utseg V2"),
            "the refusal must not name the wire format, which is what the late \
             failure did: {message}"
        );
    }

    /// The opt-in is what makes an unsupported language runnable, and it
    /// selects Stanza rather than silently borrowing another language's model.
    #[test]
    fn the_operator_opt_in_selects_the_stanza_fallback() {
        for code in ["spa", "fra", "jpn", "cat", "nld"] {
            assert_eq!(
                UtsegRoute::resolve(&lang(code), UtsegFallbackPolicy::AllowStanza),
                Ok(UtsegRoute::StanzaFallback),
                "{code} has no boundary model, so the opt-in selects Stanza"
            );
        }
    }

    /// The hole in the failure this replaces: the worker refused only when a
    /// batch still had multi-word items, so single-word content skipped it.
    /// The route cannot be reached through content at all, which is what this
    /// pins: the answer is a function of the language and policy only.
    #[test]
    fn the_route_does_not_depend_on_payload_content() {
        let first = UtsegRoute::resolve(&lang("spa"), UtsegFallbackPolicy::Refuse);
        let second = UtsegRoute::resolve(&lang("spa"), UtsegFallbackPolicy::Refuse);
        assert_eq!(
            first, second,
            "the route is decided by language and policy; there is no payload input"
        );
        assert!(first.is_err());
    }
}
