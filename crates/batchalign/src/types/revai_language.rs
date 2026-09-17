//! Rev.AI language capability data: which ISO 639-3 codes the engine supports,
//! which options each takes, and which it is known to handle badly.
//!
//! Lives under `types/` rather than in the `revai` client module, because it is
//! static data that submission VALIDATION consults before any API call is made,
//! and `types/request.rs` is where that validation lives. While it sat in
//! `revai/` the typed model depended on the HTTP client module for a lookup
//! table, which is one of the edges that kept everything downstream of `types`
//! out of the core crate.
//!
//! The client still uses it; the dependency simply runs the other way now.
//!
//! # One table, read both ways
//!
//! [`REV_LANGUAGES`] is the only list of Rev.AI languages. Submission reads it
//! from ISO 639-3 to Rev.AI's code; Rev.AI's language identification reads it
//! back. These used to be two hand-kept tables (78 and 74 entries) that had to
//! agree, and a language admitted in one direction but missing in the other
//! was a state the code had to handle; it now has nowhere to exist.
//!
//! # History
//!
//! Earlier Python implementations used
//! `pycountry.languages.get(alpha_3=lang).alpha_2` for this conversion. The
//! Rust rewrite initially replaced it with a 13-entry hardcoded match and an
//! `&other[..2]` truncation fallback. That fallback was a regression bug: ISO
//! 639-3 first-two-characters do NOT reliably match ISO 639-1 codes (e.g.,
//! `pol` → `po` instead of `pl`, `hak` → `ha` which doesn't exist). Fixed
//! 2026-03-19 with a comprehensive mapping table covering all
//! Rev.AI-supported languages.

use crate::api::{AsrLanguageRequest, LanguageCode3, LanguagePair, TranscriptLanguage};

/// Which of Rev.AI's language-conditional submission options a request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RevOptionSupport {
    /// `speakers_count`, and `skip_postprocessing` so the transcript keeps
    /// spoken forms (`eighty percent`, not `80%`), which CHAT records.
    SpeakerCountAndSpokenForm,
    /// Neither option is sent.
    Neither,
}

/// One language Rev.AI recognizes.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RevLanguageRow {
    /// The code Rev.AI names the language by.
    rev_code: &'static str,
    /// The ISO 639-3 codes that map to it. The first is the one a code Rev.AI
    /// reports maps back to.
    iso3: &'static [&'static str],
    /// The options BA3 sends with it.
    options: RevOptionSupport,
}

const fn row(
    rev_code: &'static str,
    iso3: &'static [&'static str],
    options: RevOptionSupport,
) -> RevLanguageRow {
    RevLanguageRow {
        rev_code,
        iso3,
        options,
    }
}

/// Every language Rev.AI's asynchronous API recognizes, one row each.
///
/// Options: English and Spanish take both. The live API's refusal message
/// (2026-09-16) also lists French and Portuguese as taking them; BA3 has never
/// sent them there, and changing that changes existing output, so it is not
/// decided here.
static REV_LANGUAGES: &[RevLanguageRow] = &[
    // Major languages (explicit Rev.AI codes)
    row("en", &["eng"], RevOptionSupport::SpeakerCountAndSpokenForm),
    row("es", &["spa"], RevOptionSupport::SpeakerCountAndSpokenForm),
    row("fr", &["fra"], RevOptionSupport::Neither),
    row("de", &["deu"], RevOptionSupport::Neither),
    row("it", &["ita"], RevOptionSupport::Neither),
    row("pt", &["por"], RevOptionSupport::Neither),
    row("nl", &["nld"], RevOptionSupport::Neither),
    row("ja", &["jpn"], RevOptionSupport::Neither),
    row("ko", &["kor"], RevOptionSupport::Neither),
    row("ru", &["rus"], RevOptionSupport::Neither),
    row("ar", &["ara"], RevOptionSupport::Neither),
    row("tr", &["tur"], RevOptionSupport::Neither),
    row("cmn", &["zho", "cmn"], RevOptionSupport::Neither),
    // European languages
    row("pl", &["pol"], RevOptionSupport::Neither),
    row("cs", &["ces"], RevOptionSupport::Neither),
    row("ro", &["ron"], RevOptionSupport::Neither),
    row("hu", &["hun"], RevOptionSupport::Neither),
    row("bg", &["bul"], RevOptionSupport::Neither),
    row("hr", &["hrv"], RevOptionSupport::Neither),
    row("sr", &["srp"], RevOptionSupport::Neither),
    row("sk", &["slk"], RevOptionSupport::Neither),
    row("sl", &["slv"], RevOptionSupport::Neither),
    row("uk", &["ukr"], RevOptionSupport::Neither),
    row("lt", &["lit"], RevOptionSupport::Neither),
    row("lv", &["lav"], RevOptionSupport::Neither),
    row("et", &["est"], RevOptionSupport::Neither),
    row("fi", &["fin"], RevOptionSupport::Neither),
    row("da", &["dan"], RevOptionSupport::Neither),
    row("no", &["nor", "nob", "nno"], RevOptionSupport::Neither),
    row("sv", &["swe"], RevOptionSupport::Neither),
    row("is", &["isl"], RevOptionSupport::Neither),
    row("el", &["ell"], RevOptionSupport::Neither),
    row("ca", &["cat"], RevOptionSupport::Neither),
    row("gl", &["glg"], RevOptionSupport::Neither),
    row("eu", &["eus"], RevOptionSupport::Neither),
    row("cy", &["cym"], RevOptionSupport::Neither),
    row("sq", &["sqi"], RevOptionSupport::Neither),
    row("be", &["bel"], RevOptionSupport::Neither),
    row("bs", &["bos"], RevOptionSupport::Neither),
    row("mk", &["mkd"], RevOptionSupport::Neither),
    row("mt", &["mlt"], RevOptionSupport::Neither),
    // South/Southeast Asian languages
    row("hi", &["hin"], RevOptionSupport::Neither),
    row("ur", &["urd"], RevOptionSupport::Neither),
    row("bn", &["ben"], RevOptionSupport::Neither),
    row("ta", &["tam"], RevOptionSupport::Neither),
    row("te", &["tel"], RevOptionSupport::Neither),
    row("kn", &["kan"], RevOptionSupport::Neither),
    row("ml", &["mal"], RevOptionSupport::Neither),
    row("mr", &["mar"], RevOptionSupport::Neither),
    row("pa", &["pan"], RevOptionSupport::Neither),
    row("ne", &["nep"], RevOptionSupport::Neither),
    row("si", &["sin"], RevOptionSupport::Neither),
    row("th", &["tha"], RevOptionSupport::Neither),
    row("vi", &["vie"], RevOptionSupport::Neither),
    // `msa` (Malay) has always been sent as Indonesian `id`, although Rev.AI
    // lists Malay separately as `ms`; kept as it was, not re-decided here.
    row("id", &["ind", "msa"], RevOptionSupport::Neither),
    row("tl", &["tgl"], RevOptionSupport::Neither),
    row("my", &["mya"], RevOptionSupport::Neither),
    row("km", &["khm"], RevOptionSupport::Neither),
    row("lo", &["lao"], RevOptionSupport::Neither),
    row("su", &["sun"], RevOptionSupport::Neither),
    // Caucasian / Central Asian
    row("ka", &["kat"], RevOptionSupport::Neither),
    row("hy", &["hye"], RevOptionSupport::Neither),
    row("az", &["aze"], RevOptionSupport::Neither),
    row("kk", &["kaz"], RevOptionSupport::Neither),
    row("uz", &["uzb"], RevOptionSupport::Neither),
    row("tg", &["tgk"], RevOptionSupport::Neither),
    // Other
    row("fa", &["fas"], RevOptionSupport::Neither),
    row("he", &["heb"], RevOptionSupport::Neither),
    row("yi", &["yid"], RevOptionSupport::Neither),
    row("af", &["afr"], RevOptionSupport::Neither),
    row("sw", &["swa"], RevOptionSupport::Neither),
    row("ht", &["hat"], RevOptionSupport::Neither),
    row("gu", &["guj"], RevOptionSupport::Neither),
    row("mg", &["mlg"], RevOptionSupport::Neither),
];

/// A language Rev.AI recognizes: a row of [`REV_LANGUAGES`], reached only by
/// looking it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RevSupported(&'static RevLanguageRow);

impl RevSupported {
    /// The row a requested ISO 639-3 code maps to.
    pub(crate) fn for_iso3(code: &LanguageCode3) -> Option<Self> {
        REV_LANGUAGES
            .iter()
            .find(|row| row.iso3.contains(&code.as_ref()))
            .map(Self)
    }

    /// The row a code Rev.AI reported names.
    fn for_rev_code(rev_code: &str) -> Option<Self> {
        REV_LANGUAGES
            .iter()
            .find(|row| row.rev_code == rev_code)
            .map(Self)
    }

    /// The ISO 639-3 code a reported language maps back to.
    fn canonical_iso3(self) -> Option<LanguageCode3> {
        self.0
            .iso3
            .first()
            .and_then(|code| LanguageCode3::try_new(code).ok())
    }
}

/// A language request Rev.AI can take, admitted once.
///
/// The representation is private to this module: [`RevLanguage::admit`],
/// [`RevLanguage::detect`] and [`RevLanguage::identified`] are the only ways
/// to build one, so no value pairs a code with another language's Rev.AI
/// code, or holds a pair Rev.AI has no model for. This replaced a conversion
/// that sent any unmapped language to Rev.AI as `auto` with a logged warning:
/// a job asking for one language and silently transcribed as another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RevLanguage(RevLanguageKind);

#[derive(Debug, Clone, PartialEq, Eq)]
enum RevLanguageKind {
    /// One language, as requested, with the row Rev.AI knows it by.
    One {
        code: LanguageCode3,
        supported: RevSupported,
    },
    /// Rev.AI detects the language.
    Detect,
    /// Rev.AI's multilingual English/Spanish model (`en/es`). The pair keeps
    /// the requested order, which is the transcript's `@Languages` order;
    /// Rev.AI's model has none.
    EnglishSpanish(LanguagePair),
}

/// Why Rev.AI cannot take a language request.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RevLanguageRefusal {
    /// Rev.AI has no model for this language.
    #[error("language '{0}' is not supported by Rev.AI ASR")]
    UnsupportedLanguage(LanguageCode3),
    /// Rev.AI has no multilingual model for this pair.
    #[error(
        "Rev.AI has a multilingual model only for English and Spanish; the pair '{0}' is not supported"
    )]
    UnsupportedPair(LanguagePair),
}

/// A stored transcript language that the request it is keyed under could not
/// have produced.
#[derive(Debug, Clone, thiserror::Error)]
#[error("stored transcript language '{stored}' cannot come from a request for '{requested}'")]
pub(crate) struct InconsistentResolution {
    requested: AsrLanguageRequest,
    stored: TranscriptLanguage,
}

impl RevLanguage {
    /// The code Rev.AI names its English/Spanish multilingual model by.
    const ENGLISH_SPANISH: &'static str = "en/es";

    /// Admit a request Rev.AI can take, or say why not.
    pub(crate) fn admit(request: &AsrLanguageRequest) -> Result<Self, RevLanguageRefusal> {
        match request {
            AsrLanguageRequest::One(code) => match RevSupported::for_iso3(code) {
                Some(supported) => Ok(Self(RevLanguageKind::One {
                    code: code.clone(),
                    supported,
                })),
                None => Err(RevLanguageRefusal::UnsupportedLanguage(code.clone())),
            },
            AsrLanguageRequest::Detect => Ok(Self::detect()),
            AsrLanguageRequest::Pair(pair) => {
                match (pair.primary().as_ref(), pair.secondary().as_ref()) {
                    ("eng", "spa") | ("spa", "eng") => {
                        Ok(Self(RevLanguageKind::EnglishSpanish(pair.clone())))
                    }
                    _ => Err(RevLanguageRefusal::UnsupportedPair(pair.clone())),
                }
            }
        }
    }

    /// Detection.
    pub(crate) fn detect() -> Self {
        Self(RevLanguageKind::Detect)
    }

    /// The language Rev.AI's language identification named, when BA3 knows it.
    pub(crate) fn identified(rev_code: &str) -> Option<Self> {
        let supported = RevSupported::for_rev_code(rev_code)?;
        let code = supported.canonical_iso3()?;
        Some(Self(RevLanguageKind::One { code, supported }))
    }

    /// The `language` value submitted to Rev.AI.
    pub(crate) fn provider_code(&self) -> &'static str {
        match &self.0 {
            RevLanguageKind::One { supported, .. } => supported.0.rev_code,
            RevLanguageKind::Detect => "auto",
            RevLanguageKind::EnglishSpanish(_) => Self::ENGLISH_SPANISH,
        }
    }

    /// Which language-conditional options the submission carries.
    ///
    /// The multilingual model takes neither: the live API refuses both
    /// `speakers_count` and `skip_postprocessing` for `en/es` with HTTP 400,
    /// measured on 2026-09-16.
    pub(crate) fn options(&self) -> RevOptionSupport {
        match &self.0 {
            RevLanguageKind::One { supported, .. } => supported.0.options,
            RevLanguageKind::Detect | RevLanguageKind::EnglishSpanish(_) => {
                RevOptionSupport::Neither
            }
        }
    }

    /// The request this was admitted from, which is what evidence is keyed by.
    pub(crate) fn request(&self) -> AsrLanguageRequest {
        match &self.0 {
            RevLanguageKind::One { code, .. } => AsrLanguageRequest::One(code.clone()),
            RevLanguageKind::Detect => AsrLanguageRequest::Detect,
            RevLanguageKind::EnglishSpanish(pair) => AsrLanguageRequest::Pair(pair.clone()),
        }
    }

    /// The transcript's language, the one route from a request to it.
    ///
    /// One language or the pair is what was requested. Detection resolves
    /// only to a language Rev.AI reported that BA3 maps back; `None` means the
    /// response names no usable language, never that English is assumed.
    pub(crate) fn resolve(&self, reported: Option<&str>) -> Option<TranscriptLanguage> {
        match &self.0 {
            RevLanguageKind::One { code, .. } => Some(TranscriptLanguage::One(code.clone())),
            RevLanguageKind::EnglishSpanish(pair) => Some(TranscriptLanguage::Pair(pair.clone())),
            RevLanguageKind::Detect => reported
                .and_then(RevSupported::for_rev_code)
                .and_then(RevSupported::canonical_iso3)
                .map(TranscriptLanguage::One),
        }
    }

    /// Whether stored evidence's transcript language is one this request
    /// could have resolved to: exactly the requested language or pair, or,
    /// under detection, one mapped provider language.
    pub(crate) fn check_stored_resolution(
        &self,
        stored: &TranscriptLanguage,
    ) -> Result<(), InconsistentResolution> {
        let consistent = match (&self.0, stored) {
            (RevLanguageKind::One { code, .. }, TranscriptLanguage::One(stored_code)) => {
                code == stored_code
            }
            (RevLanguageKind::EnglishSpanish(pair), TranscriptLanguage::Pair(stored_pair)) => {
                pair == stored_pair
            }
            (RevLanguageKind::Detect, TranscriptLanguage::One(code)) => {
                RevSupported::for_iso3(code).is_some()
            }
            (RevLanguageKind::One { .. }, TranscriptLanguage::Pair(_))
            | (RevLanguageKind::EnglishSpanish(_), TranscriptLanguage::One(_))
            | (RevLanguageKind::Detect, TranscriptLanguage::Pair(_)) => false,
        };
        match consistent {
            true => Ok(()),
            false => Err(InconsistentResolution {
                requested: self.request(),
                stored: stored.clone(),
            }),
        }
    }
}

impl std::fmt::Display for RevLanguage {
    /// The admitted request's text (`eng`, `auto`, `eng,spa`), NOT the Rev.AI
    /// code: evidence cache keys are built from this.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.request().fmt(f)
    }
}

impl serde::Serialize for RevLanguage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

/// Entry in the Rev.AI known-broken `(engine, language)` deny-list.
///
/// Each entry records a language whose Rev.AI model we have observed to
/// produce output unusable for CHAT construction, cross-script tokens,
/// embedded replacement characters, or other CHAT-illegal content that the
/// downstream validator (`ChatWordText::try_from_lang`) refuses. Listing
/// the pair here causes `validate_language_support()` to reject the job
/// at preflight with an error that names a working alternative, instead of
/// letting the failure surface as confusing per-token validation errors.
///
/// The deny-list is Option A from
/// [`book/src/batchalign/reference/revai-language-quality-strategy.md`]. Each entry
/// carries a dated provenance comment so a successor reading this table
/// can see *why* it exists and *when* it should be re-evaluated against
/// Rev.AI's current model.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RevAiKnownBroken {
    /// ISO 639-3 language code for which Rev.AI output is unusable.
    pub(crate) iso3: &'static str,
    /// One-line reason (appears in the operator-facing error message).
    pub(crate) reason: &'static str,
    /// Engine name string (matches `AsrEngineName::wire_name()`) that the
    /// error message should recommend as an alternative. Must be a name
    /// the user can pass to `--asr-engine`.
    pub(crate) recommended_engine: &'static str,
}

/// Rev.AI `(engine, language)` deny-list.
///
/// Keep entries alphabetized by `iso3` for readability. When adding an
/// entry, include a provenance comment with the incident date and a short
/// description of the observed failure. When Rev.AI's model for a listed
/// language is known to have improved (e.g. a changelog entry from Rev.AI
/// or a successful re-test), remove the entry in the same patch that
/// records the verification.
///
/// Escalation criteria (when the deny-list stops being enough and we
/// should build the runtime script-coherence gate or the empirical
/// capability probe) live in the strategy document.
pub(crate) const REVAI_KNOWN_BROKEN: &[RevAiKnownBroken] = &[
    // 2026-04-22: Rev.AI's Malayalam model (language=ml) returns tokens
    // in unrelated scripts (Hangul, Gurmukhi, Latin, Cyrillic) mixed
    // with U+FFFD replacement characters and bare punctuation. A
    // 1-minute test sample produced 55 tokens, effectively zero of
    // which were in Malayalam script. Submission-side mapping is
    // correct (mal → ml); this is a model-quality issue on Rev.AI's
    // side.
    //
    // The recommendation is ``whisper_hub`` rather than ``whisper``: a
    // follow-up empirical evaluation on the same sample showed stock
    // OpenAI Whisper (both medium and large-v3) also fails on Malayalam
    //: medium collapsed into Khmer/Gurmukhi character loops, large-v3
    // hallucinated "Thank you for watching." Only the community
    // fine-tune ``thennal/whisper-medium-ml`` (routed through the
    // ``whisper_hub`` engine) produced coherent Malayalam output. See
    // ``book/src/batchalign/reference/whisper-hub-asr.md`` for the comparison.
    RevAiKnownBroken {
        iso3: "mal",
        reason: "Malayalam ASR returns cross-script tokens (Hangul, Gurmukhi, Latin) and \
                 replacement characters; output cannot be represented as CHAT",
        recommended_engine: "whisper_hub",
    },
];

/// Look up a language in the Rev.AI deny-list.
///
/// Returns the deny-list entry when the given code matches a known-broken
/// language, or `None` when Rev.AI is not known to be broken for this
/// language. Callers in request validation use this to reject submissions
/// before making any Rev.AI API call.
pub(crate) fn revai_known_broken(lang: &LanguageCode3) -> Option<&'static RevAiKnownBroken> {
    REVAI_KNOWN_BROKEN
        .iter()
        .find(|entry| entry.iso3 == lang.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one table is well-formed: every code is a valid ISO 639-3 code, no
    /// ISO code or Rev.AI code appears in two rows, and every row's canonical
    /// code finds its own row again, so both directions agree by construction.
    #[test]
    fn the_rev_language_table_reads_the_same_both_ways() {
        let mut iso_seen = std::collections::BTreeSet::new();
        let mut rev_seen = std::collections::BTreeSet::new();
        for row in REV_LANGUAGES {
            assert!(rev_seen.insert(row.rev_code), "{} twice", row.rev_code);
            assert!(
                !row.iso3.is_empty(),
                "{} maps from no ISO code",
                row.rev_code
            );
            for code in row.iso3 {
                assert!(LanguageCode3::try_new(code).is_ok(), "{code}");
                assert!(iso_seen.insert(*code), "{code} in two rows");
            }
            let supported = RevSupported::for_rev_code(row.rev_code).expect("own row");
            let canonical = supported.canonical_iso3().expect("valid canonical code");
            assert_eq!(RevSupported::for_iso3(&canonical), Some(supported));
        }
    }

    /// What Rev.AI's language identification names maps back to the canonical
    /// ISO code; a sentinel or an unknown code maps to nothing.
    #[test]
    fn an_identified_rev_code_maps_back_to_its_canonical_code() {
        let request = |code: &str| RevLanguage::identified(code).map(|lang| lang.request());
        assert_eq!(
            request("es"),
            Some(AsrLanguageRequest::One(LanguageCode3::spa()))
        );
        assert_eq!(
            request("cmn"),
            Some(AsrLanguageRequest::One(LanguageCode3::zho()))
        );
        for unknown in ["auto", "xx", "", "en/es"] {
            assert_eq!(request(unknown), None, "{unknown}");
        }
    }

    /// Stored evidence must be something its request could have resolved to.
    #[test]
    fn a_stored_resolution_must_match_its_request() {
        let english =
            RevLanguage::admit(&AsrLanguageRequest::One(LanguageCode3::eng())).expect("English");
        let pair = LanguagePair::new(LanguageCode3::eng(), LanguageCode3::spa()).expect("pair");
        let english_spanish =
            RevLanguage::admit(&AsrLanguageRequest::Pair(pair.clone())).expect("en/es");
        let one = |code: LanguageCode3| TranscriptLanguage::One(code);
        assert!(
            RevLanguage::detect()
                .check_stored_resolution(&one(LanguageCode3::try_new("zzz").unwrap()))
                .is_err()
        );

        assert!(
            english
                .check_stored_resolution(&one(LanguageCode3::eng()))
                .is_ok()
        );
        assert!(
            english
                .check_stored_resolution(&one(LanguageCode3::spa()))
                .is_err()
        );
        assert!(
            english
                .check_stored_resolution(&TranscriptLanguage::Pair(pair.clone()))
                .is_err()
        );
        assert!(
            english_spanish
                .check_stored_resolution(&TranscriptLanguage::Pair(pair))
                .is_ok()
        );
        assert!(
            english_spanish
                .check_stored_resolution(&one(LanguageCode3::eng()))
                .is_err()
        );
        assert!(
            RevLanguage::detect()
                .check_stored_resolution(&one(LanguageCode3::spa()))
                .is_ok()
        );
    }
}
