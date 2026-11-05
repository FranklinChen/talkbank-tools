//! Tencent's recognition engines, as one closed set, and the account's choice
//! among them.
//!
//! # Why a catalog rather than a name built from the language
//!
//! Tencent's `EngineModelType` is not a function of the language. It names a
//! product the account pays for, in one of three billing tiers, and the same
//! language is served by several products: Cantonese by the general `16k_yue`,
//! by the Large Model 1.0 `16k_zh_en`, and by the Large Model 2.0
//! `16k_zh_en_2.0`. Until 2026-11-05 BA3 built the name from the language
//! (`16k_yue`, then `16k_zh_large` for other Chinese, then `16k_<iso 639-1>`),
//! so every change in what an account had bought needed a code change and a
//! release, and the third rule produced names, such as `16k_ja`, that Tencent
//! serves only without word timings, which BA3 cannot use.
//!
//! The facts are separated here by who owns them:
//!
//! - **What Tencent offers** ([`TencentEngine`]): which engines exist, which
//!   tier bills each, which languages each recognizes and whether it returns
//!   word timings. Tencent owns these and they change, so each is written once,
//!   below, with the page it came from.
//! - **What the account bought** ([`TencentEngineSetting`]): the account
//!   holder's choice, read from `engine.tencent.model` in the `[asr]` section
//!   of `~/.batchalign.ini`, next to the account's keys. A purchase changes the
//!   setting, never the code.
//!
//! # Sources
//!
//! Both pages were read on 2026-11-05:
//!
//! - The `CreateRecTask` API reference,
//!   <https://cloud.tencent.com/document/product/1093/37823>: the engine list
//!   and its tiers, and, under `ResTextFormat`, the engines that accept only
//!   format 0 (no word-level results).
//! - The product feature table, <https://cloud.tencent.com/document/product/1093/35682>,
//!   row "录音文件识别" (recording-file recognition, the service BA3 calls):
//!   the languages and dialects each engine recognizes.
//!
//! `16k_zh_large`, which BA3 asked for until 2026-11-05, is on neither page.

use crate::api::LanguageCode3;
use crate::types::worker_v2::ProviderParameterV2;

/// The billing tier Tencent sells an engine under.
///
/// Tencent bills the three tiers from separate resource packs, so an account
/// that bought one tier is refused by an engine of another
/// (`FailedOperation.UserHasNoAmount`). That is why a billing refusal names
/// the tier ([`TencentEngine::tier`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TencentTier {
    /// 通用引擎, the general engines.
    General,
    /// 大模型1.0版引擎, Large Model 1.0 (formerly "大模型版", Large Model).
    LargeModel1,
    /// 大模型2.0版引擎, Large Model 2.0.
    LargeModel2,
}

impl std::fmt::Display for TencentTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::General => "General engine",
            Self::LargeModel1 => "Large Model 1.0",
            Self::LargeModel2 => "Large Model 2.0",
        })
    }
}

/// A Tencent recording-recognition engine BA3 can use.
///
/// Only engines that return word timings are variants: BA3 asks for
/// `ResTextFormat = 1` because transcription and utterance timing recovery
/// both need a time for every word. The engines Tencent serves only at format
/// 0 are named in [`NO_WORD_TIMINGS`], so choosing one is refused with that
/// reason rather than as an unknown name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TencentEngine {
    /// `16k_zh_en_2.0`: Mandarin, English and 31 dialects, Large Model 2.0.
    /// Tencent describes it as much more accurate on noisy, echoing, quiet or
    /// distant speech than the general Mandarin engine.
    ZhEn2_0,
    /// `16k_zh_en_meeting`: the same languages, Large Model 2.0, for
    /// multi-party conversation; Tencent says to use it with speaker
    /// separation on.
    ZhEnMeeting,
    /// `16k_zh_en`: Mandarin, English and 31 dialects, Large Model 1.0.
    ZhEn,
    /// `8k_zh_large`: Chinese and 27 dialects, telephone audio, Large Model 1.0.
    ZhLarge8k,
    /// `16k_en_large`: English, Large Model 1.0.
    EnLarge,
    /// `16k_zh`: general Mandarin (with a little English).
    Zh,
    /// `8k_zh`: Mandarin telephone audio.
    Zh8k,
    /// `16k_en`: general English.
    En,
    /// `8k_en`: English telephone audio.
    En8k,
    /// `16k_yue`: general Cantonese.
    Yue,
    /// `16k_zh-PY`: Mandarin, English and Cantonese in one engine.
    ZhPy,
    /// `16k_zh-TW`: Chinese written in traditional characters.
    ZhTw,
}

/// Pairs each engine with its `EngineModelType` and generates both
/// [`TencentEngine::ALL`] and the one exhaustive match over the enum from the
/// same list, so a variant missing here does not compile, and a variant here
/// is necessarily in `ALL`. The same device as `fa_engine_table!` in
/// `types/engines.rs`, for the same reason: a hand-written `ALL` beside a
/// hand-written match lets a new variant compile and then fail to parse.
macro_rules! tencent_engine_table {
    ($($variant:ident => $wire:literal),+ $(,)?) => {
        impl TencentEngine {
            /// Every engine, in the order Tencent's API reference lists them.
            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The `EngineModelType` Tencent is sent.
            pub(crate) const fn wire(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire),+
                }
            }

            /// The `EngineModelType` as the provider parameter a pinned plan
            /// carries, each one admitted at compile time.
            pub(crate) const fn parameter(self) -> ProviderParameterV2 {
                match self {
                    $(Self::$variant => const { ProviderParameterV2::from_static($wire) }),+
                }
            }
        }
    };
}

tencent_engine_table! {
    ZhEn2_0 => "16k_zh_en_2.0",
    ZhEnMeeting => "16k_zh_en_meeting",
    ZhLarge8k => "8k_zh_large",
    ZhEn => "16k_zh_en",
    EnLarge => "16k_en_large",
    Zh8k => "8k_zh",
    En8k => "8k_en",
    Zh => "16k_zh",
    En => "16k_en",
    Yue => "16k_yue",
    ZhPy => "16k_zh-PY",
    ZhTw => "16k_zh-TW",
}

/// The engines Tencent's `ResTextFormat` reference says accept only format 0,
/// which returns no word-level results.
const NO_WORD_TIMINGS: &[&str] = &[
    "16k_multi_lang",
    "16k_ja",
    "16k_ko",
    "16k_vi",
    "16k_ms",
    "16k_id",
    "16k_fil",
    "16k_th",
    "16k_pt",
    "16k_tr",
    "16k_ar",
    "16k_es",
    "16k_hi",
    "16k_fr",
    "16k_zh_medical",
    "16k_de",
];

/// A language some Tencent engine recognizes, as BA3 names languages.
///
/// A closed set rather than lists of ISO 639-3 strings: [`Self::of`] is the
/// one place a code is read, so a misspelt code cannot silently drop a
/// language from an engine, and the two Mandarin codes are one language here.
/// Among the 31 dialects Tencent lists for its Mandarin-English engines, the
/// ones BA3 has codes for are Wu (吴语, 上海话), Min Nan (闽南语) and Hakka
/// (客家话); the rest are varieties of Mandarin, Wu or Min.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TencentLanguage {
    /// `zho` and `cmn`.
    Mandarin,
    /// `eng`.
    English,
    /// `yue`.
    Cantonese,
    /// `wuu`.
    Wu,
    /// `nan`.
    MinNan,
    /// `hak`.
    Hakka,
}

impl TencentLanguage {
    /// The Tencent language `lang` is, if any engine recognizes it.
    fn of(lang: &LanguageCode3) -> Option<Self> {
        match lang.as_ref() {
            "zho" | "cmn" => Some(Self::Mandarin),
            "eng" => Some(Self::English),
            "yue" => Some(Self::Cantonese),
            "wuu" => Some(Self::Wu),
            "nan" => Some(Self::MinNan),
            "hak" => Some(Self::Hakka),
            _ => None,
        }
    }
}

use TencentLanguage::{Cantonese, English, Hakka, Mandarin, MinNan, Wu};

/// The Mandarin-English-dialect engines' languages.
const ZH_EN_DIALECTS: &[TencentLanguage] = &[Mandarin, English, Cantonese, Wu, MinNan, Hakka];
/// `8k_zh_large`'s: the same less English.
const ZH_DIALECTS: &[TencentLanguage] = &[Mandarin, Cantonese, Wu, MinNan, Hakka];

impl TencentEngine {
    /// The tier Tencent bills this engine under.
    pub(crate) const fn tier(self) -> TencentTier {
        match self {
            Self::ZhEn2_0 | Self::ZhEnMeeting => TencentTier::LargeModel2,
            Self::ZhLarge8k | Self::ZhEn | Self::EnLarge => TencentTier::LargeModel1,
            Self::Zh8k | Self::En8k | Self::Zh | Self::En | Self::Yue | Self::ZhPy | Self::ZhTw => {
                TencentTier::General
            }
        }
    }

    /// The languages this engine recognizes.
    const fn languages(self) -> &'static [TencentLanguage] {
        match self {
            Self::ZhEn2_0 | Self::ZhEnMeeting | Self::ZhEn => ZH_EN_DIALECTS,
            Self::ZhLarge8k => ZH_DIALECTS,
            Self::EnLarge | Self::En | Self::En8k => &[English],
            Self::Zh | Self::Zh8k | Self::ZhTw => &[Mandarin],
            Self::Yue => &[Cantonese],
            Self::ZhPy => &[Mandarin, English, Cantonese],
        }
    }

    /// Whether this engine recognizes `lang`.
    pub(crate) fn recognizes(self, lang: &LanguageCode3) -> bool {
        TencentLanguage::of(lang).is_some_and(|language| self.languages().contains(&language))
    }

    /// Whether this engine can separate speakers (`SpeakerDiarization = 1`).
    ///
    /// From the `SpeakerDiarization` entry of the `CreateRecTask` reference,
    /// which lists the engines that accept it; `16k_yue`, `16k_zh-PY`,
    /// `16k_zh-TW`, `16k_en_large` and `8k_en` are not among them.
    pub(crate) const fn separates_speakers(self) -> bool {
        match self {
            Self::ZhEn2_0
            | Self::ZhEnMeeting
            | Self::ZhEn
            | Self::ZhLarge8k
            | Self::Zh
            | Self::Zh8k
            | Self::En => true,
            Self::EnLarge | Self::En8k | Self::Yue | Self::ZhPy | Self::ZhTw => false,
        }
    }

    /// Whether ANY Tencent engine recognizes `lang`: the question job
    /// submission can answer before it knows which engine the account chose.
    /// The account's own engine is checked when the job is planned.
    pub(crate) fn some_engine_recognizes(lang: &LanguageCode3) -> bool {
        Self::ALL.iter().any(|engine| engine.recognizes(lang))
    }

    /// What an account holder must do when Tencent refuses to bill this
    /// engine. Here, beside the tiers and the setting it names, rather than in
    /// the generic error that renders it.
    pub(crate) fn billing_remedy(self) -> String {
        format!(
            "Buy or restore a {} resource pack, or set {TENCENT_ENGINE_INI_KEY} in \
             ~/.batchalign.ini to an engine the account has bought and restart the server.",
            self.tier()
        )
    }

    /// Parse an `EngineModelType`, refusing any engine BA3 cannot use and
    /// saying why. The name is taken as given; the setting trims it.
    pub(crate) fn from_wire(name: &str) -> Result<Self, TencentEngineRefusal> {
        if let Some(engine) = Self::ALL.iter().find(|engine| engine.wire() == name) {
            return Ok(*engine);
        }
        if NO_WORD_TIMINGS.contains(&name) {
            return Err(TencentEngineRefusal::NoWordTimings(name.to_owned()));
        }
        Err(TencentEngineRefusal::Unknown(name.to_owned()))
    }
}

impl std::fmt::Display for TencentEngine {
    /// `16k_zh_en_2.0 (Large Model 2.0)`: the wire name an account holder
    /// sees in Tencent's console, with the tier they would have to buy.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.wire(), self.tier())
    }
}

/// Why a named engine cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TencentEngineRefusal {
    /// Tencent serves this engine only without word timings.
    #[error(
        "Tencent engine `{0}` returns no word timings (Tencent accepts only ResTextFormat 0 \
         for it), and BA3 needs a time for every word"
    )]
    NoWordTimings(String),
    /// Not an engine on Tencent's current list.
    #[error(
        "`{0}` is not a Tencent recording-recognition engine BA3 knows; the engines it can use \
         are {known}",
        known = known_engines()
    )]
    Unknown(String),
}

/// The usable engine names, for a refusal that lists them.
fn known_engines() -> String {
    TencentEngine::ALL
        .iter()
        .map(|engine| engine.wire())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The ini key the account's engine is read from, in the `[asr]` section.
pub(crate) const TENCENT_ENGINE_INI_KEY: &str = "engine.tencent.model";

/// Which Tencent engine this server's account has chosen.
///
/// Read once, when the server builds its worker runtime, from the same file
/// as the account's keys, and fixed for the server's lifetime. That is what
/// keeps a worker's spawn-time pin a pure function of its key: two workers
/// spawned by one server can never be pinned to different engines.
///
/// No variant means "use a default". Tencent bills each engine separately, so
/// an engine BA3 picked would be a purchase decision nobody made, and that is
/// how the account came to spend weeks failing: BA3 kept asking for
/// `16k_zh_large`, then `16k_yue`, while the account had bought Large Model
/// 2.0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TencentEngineSetting {
    /// The account names no engine. A Tencent job is refused, with the line to
    /// add.
    Unset,
    /// The account names an engine BA3 can use.
    Chosen(TencentEngine),
    /// The account names an engine BA3 cannot use. Kept rather than failing
    /// the server: an account that never runs Tencent must not lose every
    /// other engine to a typo in an unused line. A Tencent job is refused
    /// with this reason.
    Unusable(TencentEngineRefusal),
}

impl TencentEngineSetting {
    /// The setting the server's account holds, read from `~/.batchalign.ini`
    /// the way the server reads it, for a caller outside the server (the
    /// evidence-cache inspection) that must key what it reads the same way.
    pub(crate) fn from_account() -> Self {
        crate::worker::provider_credentials::HkAsrCredentialSources::from_env()
            .read_tencent_engine()
    }

    /// The account PolyU's tests run as: Large Model 2.0.
    #[cfg(test)]
    pub(crate) const LARGE_MODEL_2: Self = Self::Chosen(TencentEngine::ZhEn2_0);

    /// Admit the raw setting: absent or blank is [`Self::Unset`].
    pub(crate) fn admit(raw: Option<&str>) -> Self {
        match raw.map(str::trim).filter(|value| !value.is_empty()) {
            None => Self::Unset,
            Some(name) => match TencentEngine::from_wire(name) {
                Ok(engine) => Self::Chosen(engine),
                Err(refusal) => Self::Unusable(refusal),
            },
        }
    }

    /// The engine a job in `lang` is sent to, or why there is none.
    pub(crate) fn engine_for(
        &self,
        lang: &LanguageCode3,
    ) -> Result<TencentEngine, TencentEngineUnavailable> {
        match self {
            Self::Unset => Err(TencentEngineUnavailable::Unset),
            Self::Unusable(refusal) => Err(TencentEngineUnavailable::Unusable(refusal.clone())),
            Self::Chosen(engine) if engine.recognizes(lang) => Ok(*engine),
            Self::Chosen(engine) => Err(TencentEngineUnavailable::LanguageNotRecognized {
                engine: *engine,
                lang: lang.clone(),
            }),
        }
    }
}

/// Why a Tencent job has no engine to run on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum TencentEngineUnavailable {
    /// The account names no engine.
    #[error(
        "this account has not chosen a Tencent engine. Tencent bills each engine separately, so \
         BA3 does not pick one: add `{TENCENT_ENGINE_INI_KEY} = <engine>` to the [asr] section \
         of ~/.batchalign.ini, naming the engine the account has bought (for example \
         16k_zh_en_2.0 for Large Model 2.0), and restart the BA3 server"
    )]
    Unset,
    /// The account names an engine BA3 cannot use.
    #[error(
        "the Tencent engine in ~/.batchalign.ini ({TENCENT_ENGINE_INI_KEY}) cannot be used: \
         {0}"
    )]
    Unusable(TencentEngineRefusal),
    /// The account's engine does not recognize the job's language.
    #[error(
        "the account's Tencent engine, {engine}, does not recognize language {lang}; use another \
         ASR engine for it, or choose a Tencent engine that recognizes it"
    )]
    LanguageNotRecognized {
        /// The engine the account chose.
        engine: TencentEngine,
        /// The job's language.
        lang: LanguageCode3,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lang(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("a valid ISO 639-3 code")
    }

    /// Every engine's wire name parses back to it: the parse reads the same
    /// generated list the match is built from.
    #[test]
    fn every_engine_round_trips_through_its_wire_name() {
        for engine in TencentEngine::ALL {
            assert_eq!(TencentEngine::from_wire(engine.wire()), Ok(*engine));
        }
    }

    /// Submission admits a language when any engine recognizes it; English
    /// is one, Japanese is not.
    #[test]
    fn submission_asks_whether_any_engine_recognizes_the_language() {
        assert!(TencentEngine::some_engine_recognizes(&lang("eng")));
        assert!(TencentEngine::some_engine_recognizes(&lang("hak")));
        assert!(!TencentEngine::some_engine_recognizes(&lang("jpn")));
    }

    /// The engine PolyU bought covers Cantonese, Mandarin and English, and is
    /// billed as Large Model 2.0.
    #[test]
    fn large_model_2_covers_cantonese_mandarin_and_english() {
        let engine = TencentEngine::from_wire("16k_zh_en_2.0").expect("a usable engine");
        assert_eq!(engine.tier(), TencentTier::LargeModel2);
        for code in ["yue", "cmn", "zho", "eng", "nan", "hak", "wuu"] {
            assert!(engine.recognizes(&lang(code)), "{code}");
        }
        assert!(!engine.recognizes(&lang("jpn")));
    }

    /// An engine Tencent serves only without word timings is refused with
    /// that reason, not as an unknown name. BA3 used to build these names
    /// (`16k_ja`) for any language with a two-letter code.
    #[test]
    fn an_engine_without_word_timings_is_refused_with_that_reason() {
        assert_eq!(
            TencentEngine::from_wire("16k_ja"),
            Err(TencentEngineRefusal::NoWordTimings("16k_ja".to_owned()))
        );
    }

    /// `16k_zh_large` is not on Tencent's current list.
    #[test]
    fn an_engine_not_on_the_list_is_refused_naming_the_usable_ones() {
        let refusal = TencentEngine::from_wire("16k_zh_large").expect_err("not on the list");
        assert!(matches!(refusal, TencentEngineRefusal::Unknown(_)));
        assert!(refusal.to_string().contains("16k_zh_en_2.0"), "{refusal}");
    }

    /// The engine PolyU bought separates speakers; general Cantonese does not.
    #[test]
    fn speaker_separation_follows_tencents_engine_list() {
        assert!(TencentEngine::ZhEn2_0.separates_speakers());
        assert!(!TencentEngine::Yue.separates_speakers());
    }

    #[test]
    fn a_blank_setting_is_unset_and_refuses_naming_the_ini_key() {
        for raw in [None, Some(""), Some("   ")] {
            let setting = TencentEngineSetting::admit(raw);
            assert_eq!(setting, TencentEngineSetting::Unset);
            let refusal = setting
                .engine_for(&lang("yue"))
                .expect_err("no engine chosen");
            assert!(
                refusal.to_string().contains(TENCENT_ENGINE_INI_KEY),
                "{refusal}"
            );
        }
    }

    #[test]
    fn a_chosen_engine_serves_its_languages_and_refuses_others() {
        let setting = TencentEngineSetting::admit(Some(" 16k_yue "));
        assert_eq!(setting.engine_for(&lang("yue")), Ok(TencentEngine::Yue));
        assert_eq!(
            setting.engine_for(&lang("cmn")),
            Err(TencentEngineUnavailable::LanguageNotRecognized {
                engine: TencentEngine::Yue,
                lang: lang("cmn"),
            })
        );
    }

    #[test]
    fn an_unusable_setting_refuses_every_job_with_its_reason() {
        let setting = TencentEngineSetting::admit(Some("16k_fr"));
        assert!(matches!(setting, TencentEngineSetting::Unusable(_)));
        let refusal = setting
            .engine_for(&lang("fra"))
            .expect_err("no word timings");
        assert!(refusal.to_string().contains("no word timings"), "{refusal}");
    }
}
