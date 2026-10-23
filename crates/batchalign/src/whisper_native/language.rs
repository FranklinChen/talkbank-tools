//! whisper.cpp's language table: which languages `whisper_rs` decodes, and
//! the token it names each one by.
//!
//! Ungated, unlike the backend that runs the model, so job admission can ask
//! the same table the runtime enforces whether or not this build links
//! whisper.cpp. Its tokens are a subset of stock Whisper's
//! ([`batchalign_types::whisper_language`]): the same model vocabulary,
//! deliberately narrowed to the languages listed below.

use super::error::WhisperNativeError;
use crate::api::LanguageCode3;

/// One table, both directions: canonical ISO 639-3 (TalkBank internal)
/// paired with whisper.cpp's language token, which is stock Whisper's token
/// (a test below proves each one is). Restricted to the languages
/// whisper.cpp's large models genuinely support well; a silent
/// default-to-English would mask real misconfiguration. Deriving this table
/// from stock Whisper's would widen `whisper_rs` to every Whisper language,
/// which is a support decision, not a refactor. A round-trip test below keeps
/// the two directions from drifting.
const LANG_TABLE: &[(&str, &str)] = &[
    ("eng", "en"),
    ("yue", "yue"),
    ("cmn", "zh"),
    ("fra", "fr"),
    ("deu", "de"),
    ("spa", "es"),
    ("ita", "it"),
    ("jpn", "ja"),
    ("kor", "ko"),
    ("nld", "nl"),
    ("por", "pt"),
    ("rus", "ru"),
    ("tur", "tr"),
    ("swe", "sv"),
    ("nor", "no"),
    ("dan", "da"),
    ("fin", "fi"),
    ("pol", "pl"),
    ("ell", "el"),
    ("hun", "hu"),
    ("heb", "he"),
    ("ara", "ar"),
];

/// Legacy ISO 639-2/B aliases accepted on INPUT (normalized to the
/// canonical 639-3 code before the table lookup). Note the asymmetry
/// this creates for auto-detect: an explicit `zho` job stays `zho`, but
/// a detected `zh` comes back as the canonical `cmn`.
const LANG_INPUT_ALIASES: &[(&str, &str)] = &[
    ("zho", "cmn"),
    ("fre", "fra"),
    ("ger", "deu"),
    ("dut", "nld"),
    ("gre", "ell"),
];

pub(super) fn lang_to_iso2(lang: &LanguageCode3) -> Result<&'static str, WhisperNativeError> {
    let canonical = LANG_INPUT_ALIASES
        .iter()
        .find(|(alias, _)| *alias == &**lang)
        .map_or(&**lang, |(_, canon)| *canon);
    LANG_TABLE
        .iter()
        .find(|(iso3, _)| *iso3 == canonical)
        .map(|(_, iso2)| *iso2)
        .ok_or_else(|| WhisperNativeError::UnsupportedLanguage((**lang).to_owned()))
}

/// Map whisper.cpp's detected ISO 639-1 token back to TalkBank's ISO
/// 639-3 canonical code: the reverse direction of `LANG_TABLE`,
/// restricted to the same supported set so detection cannot smuggle in
/// a language the rest of the pipeline has no support for. Only the
/// whisper-rs backend reads a detected language.
#[cfg(any(feature = "whisper-rs-backend", test))]
pub(super) fn iso2_to_lang(iso2: &str) -> Result<LanguageCode3, WhisperNativeError> {
    let (iso3, _) = LANG_TABLE
        .iter()
        .find(|(_, i2)| *i2 == iso2)
        .ok_or_else(|| WhisperNativeError::UnsupportedDetectedLanguage(iso2.to_owned()))?;
    // The table holds only valid three-letter codes; a TryFrom failure
    // here would mean the table itself is malformed.
    LanguageCode3::try_from(*iso3)
        .map_err(|_| WhisperNativeError::UnsupportedDetectedLanguage(iso2.to_owned()))
}

/// Whether `whisper_rs` can be asked for `lang`: the check its runtime makes,
/// asked at admission.
pub(crate) fn supports_language(lang: &LanguageCode3) -> bool {
    lang_to_iso2(lang).is_ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn lang_map_known_codes() {
        assert_eq!(lang_to_iso2(&LanguageCode3::eng()).unwrap(), "en");
    }

    #[test]
    fn lang_table_round_trips() {
        // Both directions come from one table; every entry must survive
        // iso3 -> iso2 -> iso3 unchanged.
        for (iso3, iso2) in LANG_TABLE {
            let fwd = lang_to_iso2(&LanguageCode3::try_from(*iso3).unwrap()).unwrap();
            assert_eq!(fwd, *iso2);
            let back = iso2_to_lang(iso2).unwrap();
            assert_eq!(&*back, *iso3);
        }
    }

    #[test]
    fn lang_aliases_normalize_on_input() {
        // Legacy 639-2/B aliases are accepted forward but never produced
        // in reverse (the documented auto-detect asymmetry).
        for (alias, canonical) in LANG_INPUT_ALIASES {
            let via_alias = lang_to_iso2(&LanguageCode3::try_from(*alias).unwrap()).unwrap();
            let via_canonical =
                lang_to_iso2(&LanguageCode3::try_from(*canonical).unwrap()).unwrap();
            assert_eq!(via_alias, via_canonical);
        }
    }

    #[test]
    fn lang_map_unknown_returns_error() {
        let bogus = LanguageCode3::try_new("xxx").expect("xxx is three ASCII letters");
        let err = lang_to_iso2(&bogus).unwrap_err();
        assert!(matches!(err, WhisperNativeError::UnsupportedLanguage(_)));
    }

    /// whisper.cpp runs the same model vocabulary as stock Whisper, so every
    /// token it is given must be one of stock Whisper's.
    #[test]
    fn every_whisper_cpp_token_is_a_stock_whisper_token() {
        for (iso3, token) in LANG_TABLE {
            assert!(
                batchalign_types::whisper_language::WhisperLanguageToken::from_token(token)
                    .is_some(),
                "{iso3} -> {token}"
            );
        }
    }
}
