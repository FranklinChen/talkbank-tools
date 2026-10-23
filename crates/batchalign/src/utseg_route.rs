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
use crate::stanza_registry::{
    Admitted, StanzaGap, StanzaRefusal, StanzaRegistry, StanzaRegistryLookup,
    UtteranceSegmentation, admit_from,
};
use crate::types::worker_v2::UtsegSegmenterV2;

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
/// Both private route kinds are runnable. "No route" is not a third variant:
/// it is
/// [`UtsegUnavailable`], the error arm of [`UtsegRoute::resolve`], so a route
/// that cannot run cannot be constructed, cannot be stored in a plan, and
/// cannot be handed to a worker. A `Refused` variant would have been a value
/// every consumer had to remember to reject, which is the shape that let the
/// old refusal be skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtsegRoute {
    language: LanguageCode3,
    kind: RouteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteKind {
    BoundaryModel,
    StanzaFallback,
}

/// No segmenter exists for this language under the job's fallback policy.
///
/// Carries the language, because naming what was actually asked for is the
/// whole point: the failure this replaces named `invalid utseg V2 result`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "utterance segmentation has no TalkBank boundary model for language '{lang}', and the \
     Stanza constituency fallback was not authorized. Authorize it to proceed (its quality \
     varies by language): `--utseg-fallback-stanza` on the command line, or \
     `\"utseg_fallback\": true` in a job request's options."
)]
pub(crate) struct UtsegUnavailable {
    /// The language that has no segmenter.
    lang: LanguageCode3,
}

/// A route this runtime can serve: proven by [`UtsegRoute::admit`] against
/// the admitted Stanza registry when the route runs Stanza, and needing no
/// proof when it runs a boundary model. The only route a job is dispatched
/// with, so a language whose catalog entry cannot segment is refused before
/// any work rather than failing item by item at the worker.
///
/// It is what the worker request carries: [`Self::segmenter`] names the one
/// segmenter the worker runs, so the worker never re-derives a route from
/// what it happens to have loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServableUtsegRoute(ServableSegmenter);

/// The servable segmenter, with what proves it: a boundary-model language
/// needs no Stanza proof; the Stanza route carries the registry's admission.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ServableSegmenter {
    BoundaryModel(LanguageCode3),
    StanzaConstituency(Admitted<UtteranceSegmentation>),
}

impl ServableUtsegRoute {
    /// The language this route segments.
    pub(crate) fn language(&self) -> &LanguageCode3 {
        match &self.0 {
            ServableSegmenter::BoundaryModel(language) => language,
            ServableSegmenter::StanzaConstituency(admitted) => admitted.language(),
        }
    }

    /// The segmenter the worker request names.
    pub(crate) fn segmenter(&self) -> UtsegSegmenterV2 {
        match &self.0 {
            ServableSegmenter::BoundaryModel(_) => UtsegSegmenterV2::BoundaryModel,
            ServableSegmenter::StanzaConstituency(_) => UtsegSegmenterV2::StanzaConstituency,
        }
    }
}

/// The Stanza constituency route chosen for a language this runtime's
/// Stanza catalog cannot serve.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "utterance segmentation for language '{lang}' takes the Stanza constituency route, which \
     this runtime cannot serve: {reason}"
)]
pub(crate) struct UtsegStanzaUnservable {
    lang: LanguageCode3,
    reason: StanzaRefusal<UtteranceSegmentation>,
}

impl UtsegRoute {
    /// Prove this route servable, or refuse it before any work.
    ///
    /// # Errors
    ///
    /// The route runs Stanza and the admitted registry does not serve the
    /// language's constituency requirement, or there is no admitted registry.
    pub(crate) fn admit(
        self,
        registry: &StanzaRegistryLookup,
    ) -> Result<ServableUtsegRoute, UtsegStanzaUnservable> {
        match self.kind {
            RouteKind::BoundaryModel => Ok(ServableUtsegRoute(ServableSegmenter::BoundaryModel(
                self.language,
            ))),
            RouteKind::StanzaFallback => admit_from(registry, &self.language)
                .map(|admitted| ServableUtsegRoute(ServableSegmenter::StanzaConstituency(admitted)))
                .map_err(|reason| UtsegStanzaUnservable {
                    lang: self.language,
                    reason,
                }),
        }
    }

    /// Prove this route servable by an admitted registry in hand (the
    /// submission check): the registry's own admission, so the only refusal
    /// is the registry's gap. There is no "no registry" case to answer.
    ///
    /// # Errors
    ///
    /// The route runs Stanza and the registry does not serve the language.
    pub(crate) fn admit_by(
        self,
        registry: &std::sync::Arc<StanzaRegistry>,
    ) -> Result<ServableUtsegRoute, StanzaGap<UtteranceSegmentation>> {
        match self.kind {
            RouteKind::BoundaryModel => Ok(ServableUtsegRoute(ServableSegmenter::BoundaryModel(
                self.language,
            ))),
            RouteKind::StanzaFallback => registry.admit(&self.language).map(|admitted| {
                ServableUtsegRoute(ServableSegmenter::StanzaConstituency(admitted))
            }),
        }
    }

    /// The servable route of a boundary-model language, which needs no
    /// Stanza proof; `None` for the Stanza route, which must be admitted.
    ///
    /// Transcribe's pre-CHAT pass runs the boundary model or nothing, so this
    /// is how it reaches a route to dispatch without consulting a registry.
    pub(crate) fn boundary_model(&self) -> Option<ServableUtsegRoute> {
        match self.kind {
            RouteKind::BoundaryModel => Some(ServableUtsegRoute(ServableSegmenter::BoundaryModel(
                self.language.clone(),
            ))),
            RouteKind::StanzaFallback => None,
        }
    }

    /// The language whose segmentation capability this value proves.
    pub(crate) fn language(&self) -> &LanguageCode3 {
        &self.language
    }

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
            return Ok(Self {
                language: lang.clone(),
                kind: RouteKind::BoundaryModel,
            });
        }
        match fallback {
            UtsegFallbackPolicy::AllowStanza => Ok(Self {
                language: lang.clone(),
                kind: RouteKind::StanzaFallback,
            }),
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
                let route = UtsegRoute::resolve(&lang(code), policy).expect("boundary model");
                assert_eq!(
                    route.boundary_model().map(|servable| servable.segmenter()),
                    Some(UtsegSegmenterV2::BoundaryModel)
                );
                assert_eq!(route.language(), &lang(code));
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
            "the refusal must name the command-line remedy: {message}"
        );
        assert!(
            message.contains("\"utseg_fallback\": true"),
            "the refusal must name the job request's option by its wire name, which is not \
             the flag's name; a caller who sent the flag's name was refused again with the \
             same message: {message}"
        );
        assert!(
            !message.contains("without utterance segmentation"),
            "the refusal must not offer a remedy transcribe does not have: {message}"
        );
        assert!(
            !message.contains("utseg V2"),
            "the refusal must not name the wire format, which is what the late \
             failure did: {message}"
        );
    }

    /// RED FIRST: the Stanza route was admitted by tokenize and pos alone, so
    /// a language whose catalog entry has no constituency parser ran, and
    /// every multi-word item failed at the worker with "produced no tree". It
    /// is now refused before any work, naming the processors it lacks; a
    /// boundary-model route needs no Stanza, registry or not.
    #[test]
    fn the_stanza_route_is_admitted_only_with_a_constituency_parser() {
        use crate::stanza_registry::{GapKind, StanzaProcessor::*};
        let admitted = StanzaRegistry::serving(&[
            ("nld", &[Tokenize, Pos, Lemma, Depparse, Mwt]),
            ("spa", &[Tokenize, Pos, Constituency, Mwt]),
        ])
        .shared();
        let gap = UtsegRoute::resolve(&lang("nld"), UtsegFallbackPolicy::AllowStanza)
            .expect("authorized fallback")
            .admit_by(&admitted)
            .expect_err("no constituency parser, no Stanza segmentation");
        assert_eq!(
            gap.kind(),
            &GapKind::ProcessorsUnavailable {
                missing: std::collections::BTreeSet::from([Constituency])
            }
        );
        let registry: StanzaRegistryLookup = Ok(admitted);
        let refusal = UtsegRoute::resolve(&lang("nld"), UtsegFallbackPolicy::AllowStanza)
            .expect("authorized fallback")
            .admit(&registry)
            .expect_err("the lookup refuses the same gap");
        assert!(refusal.to_string().contains("constituency"), "{refusal}");

        let spanish = UtsegRoute::resolve(&lang("spa"), UtsegFallbackPolicy::AllowStanza)
            .expect("authorized fallback")
            .admit(&registry)
            .expect("Spanish ships a constituency parser");
        assert_eq!(spanish.language(), &lang("spa"));
        assert_eq!(spanish.segmenter(), UtsegSegmenterV2::StanzaConstituency);

        let english =
            UtsegRoute::resolve(&lang("eng"), UtsegFallbackPolicy::Refuse).expect("boundary model");
        assert!(english.boundary_model().is_some());
        english
            .admit(&Err(
                crate::stanza_registry::StanzaRegistryUnavailable::NoReport,
            ))
            .expect("a boundary-model route reads no Stanza output");
    }

    /// The opt-in is what makes an unsupported language runnable, and it
    /// selects Stanza rather than silently borrowing another language's model.
    #[test]
    fn the_operator_opt_in_selects_the_stanza_fallback() {
        for code in ["spa", "fra", "jpn", "cat", "nld"] {
            let route = UtsegRoute::resolve(&lang(code), UtsegFallbackPolicy::AllowStanza)
                .expect("authorized fallback");
            assert_eq!(route.boundary_model(), None);
            assert_eq!(route.language(), &lang(code));
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
