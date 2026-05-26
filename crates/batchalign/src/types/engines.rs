//! Engine backend types and traits.
//!
//! Closed enum sets for ASR, FA, and UTR engine selection.
//! No external plugin system — all engines are built-in.
//! The [`EngineBackend`] trait provides a common interface.

use serde::{Deserialize, Serialize};

/// Shared behavior for all engine backend selectors.
///
/// Implement this on each engine enum so generic code can work across
/// engine categories without knowing which specific enum it holds.
pub trait EngineBackend: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// Stable wire-format name used in JSON, CLI args, and SQLite.
    fn wire_name(&self) -> &str;

    /// Whether this engine's inference is fully Rust-owned (no Python worker).
    fn is_rust_owned(&self) -> bool;

    /// Parse a wire-format name. Returns `None` for unrecognized names.
    fn try_from_wire_name(name: &str) -> Option<Self>
    where
        Self: Sized;
}

/// Error returned when a wire-format engine name is not recognized.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown engine name \"{name}\" for {category}")]
pub struct UnknownEngineName {
    /// The unrecognized wire name.
    pub name: String,
    /// Which engine category was being parsed (e.g. "ASR", "FA", "UTR").
    pub category: &'static str,
}

/// Typed UTR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev_utr"`,
/// `"whisper_utr"`, or a plugin-provided name), but the server runtime works
/// with this enum so the control plane stops branching on anonymous strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UtrEngine {
    /// Rust-owned Rev.AI timed-word path.
    RevAi,
    /// Python-worker ASR path with the built-in Whisper profile.
    Whisper,
    /// Tencent UTR (HK/Cantonese).
    HkTencent,
}

impl EngineBackend for UtrEngine {
    fn wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev_utr",
            Self::Whisper => "whisper_utr",
            Self::HkTencent => "tencent_utr",
        }
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "rev_utr" => Some(Self::RevAi),
            "whisper_utr" => Some(Self::Whisper),
            "tencent_utr" => Some(Self::HkTencent),
            _ => None,
        }
    }
}

impl UtrEngine {
    /// Parse one persisted wire-format token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "UTR",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
    }

    /// Whether the current engine can reuse the worker-side segment strategy
    /// for partial-window UTR.
    pub fn supports_partial_windows(&self) -> bool {
        !self.is_rust_owned()
    }
}

impl Serialize for UtrEngine {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for UtrEngine {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed forced-alignment engine selector.
///
/// The wire format still uses the legacy string tokens (`"wav2vec_fa"`,
/// `"whisper_fa"`, or a plugin-provided name), but the control plane works
/// with this enum so dispatch does not branch on anonymous strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaEngineName {
    /// MMS Wave2Vec forced alignment.
    Wave2Vec,
    /// Whisper token-timestamp forced alignment.
    Whisper,
    /// Wav2Vec Cantonese forced alignment (HK).
    Wav2vecCanto,
}

impl EngineBackend for FaEngineName {
    fn wire_name(&self) -> &str {
        match self {
            Self::Wave2Vec => "wav2vec_fa",
            Self::Whisper => "whisper_fa",
            Self::Wav2vecCanto => "cantonese_fa",
        }
    }

    fn is_rust_owned(&self) -> bool {
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "wav2vec_fa" | "wave2vec" => Some(Self::Wave2Vec),
            "whisper_fa" | "whisper" => Some(Self::Whisper),
            "cantonese_fa" | "wav2vec_canto" | "wav2vec_fa_canto" => Some(Self::Wav2vecCanto),
            _ => None,
        }
    }
}

impl FaEngineName {
    /// The override name used in worker pool keys for dispatch.
    ///
    /// Must match `fa_backend_override_name()` in `worker/pool/execute_v2.rs`.
    /// These are the names the Python worker sees in its `--engine-overrides`
    /// JSON and uses to select which FA model to load.
    pub fn dispatch_override_name(&self) -> &'static str {
        match self {
            Self::Wave2Vec => "wave2vec",
            Self::Whisper => "whisper",
            Self::Wav2vecCanto => "wav2vec_canto",
        }
    }

    /// Parse one persisted wire-format token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "FA",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process running
    /// this FA engine, in MB. Used by the admission gate to reserve enough
    /// headroom for engines whose actual RSS exceeds the default GPU-profile
    /// reservation (``tier.gpu_startup_mb``: 6 GB Small / 3 GB Medium /
    /// 16 GB Large+Fleet). See
    /// [`super::super::worker::pool::memory_gate::engine_aware_startup_reservation_mb`].
    pub fn resident_memory_mb(&self) -> u64 {
        match self {
            // Whisper-large-v2 FA: ~3 GB weights + tokenizer + Python
            // runtime. Same shape as Whisper-large-v3 ASR, hence the
            // shared constant.
            Self::Whisper => WHISPER_LARGE_V3_RSS_MB,
            // MMS / torchaudio Wave2Vec FA models: ~1.2 GB + runtime
            // margin. Cantonese FA is the same shape.
            Self::Wave2Vec | Self::Wav2vecCanto => WAVE2VEC_FA_RSS_MB,
        }
    }
}

impl Serialize for FaEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for FaEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed ASR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev"`,
/// `"whisper"`, `"whisperx"`, `"whisper_oai"`, or a plugin-provided name), but
/// the control plane works with this enum so backend selection is explicit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsrEngineName {
    /// Rust-owned Rev.AI backend.
    RevAi,
    /// Local Whisper worker backend.
    Whisper,
    /// HuggingFace Whisper fine-tune backend. Loads community fine-tunes
    /// by model_id (resolved per-language, with an explicit override in
    /// ``engine_overrides.model_id``). See
    /// ``book/src/reference/whisper-hub-asr.md``.
    WhisperHub,
    /// WhisperX worker backend.
    WhisperX,
    /// OpenAI Whisper API backend.
    WhisperOai,
    /// Tencent Cloud ASR (HK/Cantonese).
    HkTencent,
    /// Aliyun ASR (HK/Cantonese).
    HkAliyun,
    /// FunAudio ASR (HK/Cantonese).
    HkFunaudio,
}

impl EngineBackend for AsrEngineName {
    fn wire_name(&self) -> &str {
        match self {
            Self::RevAi => "rev",
            Self::Whisper => "whisper",
            Self::WhisperHub => "whisper_hub",
            Self::WhisperX => "whisperx",
            Self::WhisperOai => "whisper_oai",
            Self::HkTencent => "tencent",
            Self::HkAliyun => "aliyun",
            Self::HkFunaudio => "funaudio",
        }
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "rev" => Some(Self::RevAi),
            "whisper" => Some(Self::Whisper),
            "whisper_hub" => Some(Self::WhisperHub),
            "whisperx" => Some(Self::WhisperX),
            "whisper_oai" => Some(Self::WhisperOai),
            "tencent" => Some(Self::HkTencent),
            "aliyun" => Some(Self::HkAliyun),
            "funaudio" => Some(Self::HkFunaudio),
            _ => None,
        }
    }
}

impl AsrEngineName {
    /// The override name used in worker pool keys for dispatch, or `None` for
    /// cloud-only engines (Rev.AI) that don't need a local worker.
    ///
    /// Must match `asr_backend_override_name()` in `worker/pool/execute_v2.rs`.
    pub fn dispatch_override_name(&self) -> Option<&'static str> {
        match self {
            Self::Whisper => Some("whisper"),
            Self::WhisperHub => Some("whisper_hub"),
            Self::HkTencent => Some("tencent"),
            Self::HkAliyun => Some("aliyun"),
            Self::HkFunaudio => Some("funaudio"),
            Self::RevAi | Self::WhisperX | Self::WhisperOai => None,
        }
    }

    /// Parse one persisted wire-format token. Falls back to `try_from_wire_name`.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "ASR",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process running
    /// this ASR engine, in MB. Used by the admission gate to reserve
    /// enough headroom for engines whose actual RSS exceeds the default
    /// GPU-profile reservation (``tier.gpu_startup_mb``: 6 GB Small /
    /// 3 GB Medium / 16 GB Large+Fleet). See
    /// [`super::super::worker::pool::memory_gate::engine_aware_startup_reservation_mb`].
    pub fn resident_memory_mb(&self) -> u64 {
        match self {
            // Whisper-large-v3 (and its WhisperHub fine-tunes): ~3 GB
            // model + tokenizer + Python runtime. WhisperX is included
            // here for symmetry/future-proofing even though
            // ``dispatch_override_name`` returns ``None`` for it today
            // (it doesn't get a pool-managed Python worker), so the
            // admission gate never observes this value in production.
            Self::Whisper | Self::WhisperHub | Self::WhisperX => WHISPER_LARGE_V3_RSS_MB,
            // Cloud HTTP clients with no local model.
            Self::RevAi
            | Self::WhisperOai
            | Self::HkTencent
            | Self::HkAliyun
            | Self::HkFunaudio => HTTP_CLIENT_BASELINE_RSS_MB,
        }
    }

    /// Whether this engine is the Rust-owned Rev.AI path.
    pub fn is_revai(&self) -> bool {
        matches!(self, Self::RevAi)
    }
}

impl Serialize for AsrEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for AsrEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed translation engine selector.
///
/// The wire format uses the lowercase tokens ``"google"``,
/// ``"seamless"``, ``"nllb"``, and ``"tencent"``; the Python worker's
/// ``resolve_translate_engine``
/// (``batchalign/worker/_model_loading/translation.py``) matches on
/// those exact strings. Any change here must be mirrored on the Python
/// side or dispatch breaks silently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TranslateEngineName {
    /// Public Google Translate via the ``googletrans`` library. Requires
    /// outbound reachability to ``translate.google.com`` — unsuitable
    /// behind the Great Firewall without a VPN.
    Google,
    /// Local Meta SeamlessM4T model, loaded from HuggingFace and run
    /// in-process in the Python worker. No outbound network at
    /// inference time. Retained for back-compat with BA2 callers;
    /// short-CJK quality is poor, prefer ``Nllb`` or ``Tencent`` for
    /// new work.
    Seamless,
    /// Local Meta NLLB-200-distilled-1.3B (~5 GB), text-MT-native.
    /// No outbound network at inference time. Self-hosted fallback
    /// that handles Cantonese first-class (Tencent does not).
    Nllb,
    /// Tencent Cloud TMT (Text Translation) — cloud-API engine.
    /// Best empirical quality on Mandarin (zh→en) verified
    /// 2026-05-26; does NOT support Cantonese (``yue``). Requires
    /// CAM credentials with ``tmt:TextTranslate`` permission in
    /// ``~/.batchalign.ini``. Free tier 5M chars/month.
    Tencent,
}

impl EngineBackend for TranslateEngineName {
    fn wire_name(&self) -> &str {
        match self {
            Self::Google => "google",
            Self::Seamless => "seamless",
            Self::Nllb => "nllb",
            Self::Tencent => "tencent",
        }
    }

    fn is_rust_owned(&self) -> bool {
        // All backends run in the Python worker. No Rust-owned
        // translate path exists today.
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "google" => Some(Self::Google),
            "seamless" => Some(Self::Seamless),
            "nllb" => Some(Self::Nllb),
            "tencent" => Some(Self::Tencent),
            _ => None,
        }
    }
}

impl TranslateEngineName {
    /// The override name used in worker pool keys for dispatch.
    ///
    /// Identical to ``wire_name`` — translate has no legacy alias
    /// divergence between dispatch and wire today. Provided for
    /// shape-parity with ``AsrEngineName`` and ``FaEngineName``.
    pub fn dispatch_override_name(&self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Seamless => "seamless",
            Self::Nllb => "nllb",
            Self::Tencent => "tencent",
        }
    }

    /// Parse one persisted wire-format token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: "translate",
        })
    }

    /// Borrow the wire-format token for JSON/SQLite.
    pub fn as_wire_name(&self) -> &str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process
    /// running this translate engine, in MB. Used by the admission
    /// gate to reserve enough headroom for engines whose actual RSS
    /// exceeds the default IO-profile reservation
    /// (``tier.io_startup_mb``: 2 GB Small/Medium, 4 GB Large/Fleet).
    /// The estimate is the observed model + tokenizer + Python
    /// runtime footprint with a modest margin; conservative on the
    /// side of over-reserving so the OS OOM killer isn't the fallback
    /// safety mechanism. Related but distinct from the *on-disk*
    /// model-size hints used by the Python progress events
    /// (``batchalign/worker/_progress.py::_HF_SIZE_HINTS_GB``).
    pub fn resident_memory_mb(&self) -> u64 {
        match self {
            // googletrans + Tencent TMT are both thin HTTP-client
            // engines with no local model loaded — same baseline.
            Self::Google | Self::Tencent => HTTP_CLIENT_BASELINE_RSS_MB,
            Self::Seamless => SEAMLESS_M4T_MEDIUM_RSS_MB,
            Self::Nllb => NLLB_200_DISTILLED_1_3B_RSS_MB,
        }
    }
}

/// Resident memory estimate for any worker that runs a thin HTTP-client
/// engine with no local model loaded — googletrans for translate, and
/// the cloud ASR engines (Rev.AI, WhisperOai, HkTencent, HkAliyun,
/// HkFunaudio). Baseline Python + worker scaffolding only.
pub(crate) const HTTP_CLIENT_BASELINE_RSS_MB: u64 = 200;

/// Resident memory estimate for a worker running the local
/// SeamlessM4T-medium model: ~2.4 GB weights + tokenizer + runtime,
/// with margin.
pub(crate) const SEAMLESS_M4T_MEDIUM_RSS_MB: u64 = 2_900;

/// Resident memory estimate for a worker running the local
/// NLLB-200-distilled-1.3B model: ~5 GB weights + tokenizer +
/// runtime, with margin.
pub(crate) const NLLB_200_DISTILLED_1_3B_RSS_MB: u64 = 5_500;

/// Resident memory estimate for a worker running the Whisper-large-v3
/// ASR model or the Whisper-large-v2 FA model (same shape). ~3 GB
/// weights + tokenizer + Python runtime + margin.
pub(crate) const WHISPER_LARGE_V3_RSS_MB: u64 = 3_500;

/// Resident memory estimate for a worker running an MMS / Wave2Vec
/// forced-alignment model (including the Cantonese variant): ~1.2 GB
/// torchaudio weights + runtime margin.
pub(crate) const WAVE2VEC_FA_RSS_MB: u64 = 1_800;

impl Serialize for TranslateEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for TranslateEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// EngineOverrides — typed engine override selection
// ---------------------------------------------------------------------------

/// Typed engine overrides for one job or worker spawn.
///
/// Replaces `BTreeMap<String, String>` in `CommonOptions.engine_overrides`.
/// Only populated fields are serialized; empty overrides produce `{}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct EngineOverrides {
    /// ASR engine override (e.g., `AsrEngineName::HkTencent`).
    pub asr: Option<AsrEngineName>,
    /// FA engine override (e.g., `FaEngineName::Wav2vecCanto`).
    pub fa: Option<FaEngineName>,
    /// Translate engine override (e.g., `TranslateEngineName::Seamless`).
    pub translate: Option<TranslateEngineName>,
}

impl EngineOverrides {
    /// Return `true` when no overrides are set.
    pub fn is_empty(&self) -> bool {
        self.asr.is_none() && self.fa.is_none() && self.translate.is_none()
    }

    /// Serialize to a JSON string for pool worker keying and CLI pass-through.
    ///
    /// Returns empty string when no overrides are set.
    pub fn to_json_string(&self) -> String {
        if self.is_empty() {
            String::new()
        } else {
            serde_json::to_string(self).unwrap_or_else(|e| format!("<serialization failed: {e}>"))
        }
    }
}

impl Serialize for EngineOverrides {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let count = self.asr.is_some() as usize
            + self.fa.is_some() as usize
            + self.translate.is_some() as usize;
        let mut map = serializer.serialize_map(Some(count))?;
        if let Some(ref asr) = self.asr {
            map.serialize_entry("asr", asr.as_wire_name())?;
        }
        if let Some(ref fa) = self.fa {
            map.serialize_entry("fa", fa.as_wire_name())?;
        }
        if let Some(ref translate) = self.translate {
            map.serialize_entry("translate", translate.as_wire_name())?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for EngineOverrides {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::deserialize(deserializer)?;
        let mut overrides = Self::default();
        for (key, value) in &map {
            match key.as_str() {
                "asr" => {
                    overrides.asr = Some(
                        AsrEngineName::from_wire_name(value).map_err(serde::de::Error::custom)?,
                    );
                }
                "fa" => {
                    overrides.fa = Some(
                        FaEngineName::from_wire_name(value).map_err(serde::de::Error::custom)?,
                    );
                }
                "translate" => {
                    overrides.translate = Some(
                        TranslateEngineName::from_wire_name(value)
                            .map_err(serde::de::Error::custom)?,
                    );
                }
                other => {
                    return Err(serde::de::Error::custom(format!(
                        "unknown engine override key: {other}"
                    )));
                }
            }
        }
        Ok(overrides)
    }
}

#[cfg(test)]
mod tests {
    //! Wire-name / dispatch-key roundtrip coverage for ``AsrEngineName``.
    //!
    //! The wire name is the single source of truth shared between
    //! Rust (``AsrEngineName`` here, ``AsrBackendV2`` in
    //! ``batchalign-types``), Python (``AsrEngine`` enum in
    //! ``batchalign/worker/_types.py``), the CLI flag parser, and SQLite
    //! job persistence. A mismatch in any one of those locations breaks
    //! dispatch silently. These tests pin the contract at the Rust
    //! entry point.
    use super::*;

    #[test]
    fn whisper_hub_wire_roundtrip() {
        assert_eq!(AsrEngineName::WhisperHub.wire_name(), "whisper_hub");
        assert_eq!(
            AsrEngineName::try_from_wire_name("whisper_hub"),
            Some(AsrEngineName::WhisperHub),
        );
    }

    #[test]
    fn whisper_hub_is_not_rust_owned() {
        // Rust-owned engines talk to providers directly from the server
        // (only Rev.AI today). whisper_hub runs in a Python worker like
        // stock Whisper / WhisperX / HK engines.
        assert!(!AsrEngineName::WhisperHub.is_rust_owned());
        assert!(!AsrEngineName::WhisperHub.is_revai());
    }

    #[test]
    fn whisper_hub_dispatch_override_name_matches_wire_name() {
        // Worker pool keys must match the wire name so the Python worker
        // bootstrap sees ``engine_overrides["asr"] == "whisper_hub"`` and
        // routes to the fine-tune loader in ``_model_loading/asr.py``.
        assert_eq!(
            AsrEngineName::WhisperHub.dispatch_override_name(),
            Some("whisper_hub"),
        );
    }

    // ---- TranslateEngineName ----
    //
    // Pinned because the Python worker's `resolve_translate_engine`
    // (`batchalign/worker/_model_loading/translation.py`) matches on the
    // exact strings "google" and "seamless". A typo here would
    // silently fall through to the default engine on the Python side.

    #[test]
    fn translate_engine_google_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Google.wire_name(), "google");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("google"),
            Some(TranslateEngineName::Google),
        );
    }

    #[test]
    fn translate_engine_seamless_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Seamless.wire_name(), "seamless");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("seamless"),
            Some(TranslateEngineName::Seamless),
        );
    }

    #[test]
    fn translate_engine_nllb_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Nllb.wire_name(), "nllb");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("nllb"),
            Some(TranslateEngineName::Nllb),
        );
    }

    #[test]
    fn translate_engine_tencent_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Tencent.wire_name(), "tencent");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("tencent"),
            Some(TranslateEngineName::Tencent),
        );
    }

    #[test]
    fn translate_engine_unknown_wire_name_is_rejected() {
        assert_eq!(TranslateEngineName::try_from_wire_name("gogle"), None);
        let err = TranslateEngineName::from_wire_name("gogle").unwrap_err();
        assert_eq!(err.category, "translate");
        assert_eq!(err.name, "gogle");
    }

    #[test]
    fn translate_engine_resident_memory_ordering() {
        // Pins the physical ordering — Google (HTTP client) <
        // Seamless (~2.4 GB) < NLLB (~5 GB) — that the admission-gate
        // engine-aware reservation
        // (``worker::pool::memory_gate::engine_aware_startup_reservation_mb``)
        // relies on. A typo here would silently re-introduce under-
        // reservation for the heavier engines.
        let google_mb = TranslateEngineName::Google.resident_memory_mb();
        let seamless_mb = TranslateEngineName::Seamless.resident_memory_mb();
        let nllb_mb = TranslateEngineName::Nllb.resident_memory_mb();
        assert!(
            google_mb < seamless_mb,
            "Google ({google_mb} MB) must be smaller than Seamless ({seamless_mb} MB)"
        );
        assert!(
            seamless_mb < nllb_mb,
            "Seamless ({seamless_mb} MB) must be smaller than NLLB ({nllb_mb} MB)"
        );
    }

    #[test]
    fn asr_engine_resident_memory_partitions_local_vs_cloud() {
        // Local Whisper variants must all match the heavy-model
        // footprint; cloud HTTP clients must all match the cheap
        // baseline. The admission gate's engine-aware reservation
        // depends on this partition being clean.
        assert_eq!(
            AsrEngineName::Whisper.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            AsrEngineName::WhisperHub.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            AsrEngineName::WhisperX.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        for cloud in [
            AsrEngineName::RevAi,
            AsrEngineName::WhisperOai,
            AsrEngineName::HkTencent,
            AsrEngineName::HkAliyun,
            AsrEngineName::HkFunaudio,
        ] {
            assert_eq!(
                cloud.resident_memory_mb(),
                HTTP_CLIENT_BASELINE_RSS_MB,
                "{cloud:?} should match the cloud HTTP-client baseline"
            );
        }
        assert!(HTTP_CLIENT_BASELINE_RSS_MB < WHISPER_LARGE_V3_RSS_MB);
    }

    #[test]
    fn fa_engine_resident_memory_separates_whisper_from_wave2vec() {
        assert_eq!(
            FaEngineName::Whisper.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            FaEngineName::Wave2Vec.resident_memory_mb(),
            WAVE2VEC_FA_RSS_MB
        );
        assert_eq!(
            FaEngineName::Wav2vecCanto.resident_memory_mb(),
            WAVE2VEC_FA_RSS_MB
        );
        assert!(WAVE2VEC_FA_RSS_MB < WHISPER_LARGE_V3_RSS_MB);
    }

    #[test]
    fn translate_engine_tencent_matches_http_client_baseline() {
        // Tencent TMT is a thin HTTP-client engine — no local model
        // loaded — so its resident footprint is the same as Google's
        // and Seamless's lightweight baseline. Pinned to prevent
        // accidental inflation (which would over-reserve memory and
        // refuse spawns on hosts that can comfortably run Tencent
        // translate workers).
        assert_eq!(
            TranslateEngineName::Tencent.resident_memory_mb(),
            HTTP_CLIENT_BASELINE_RSS_MB
        );
        assert_eq!(
            TranslateEngineName::Tencent.resident_memory_mb(),
            TranslateEngineName::Google.resident_memory_mb()
        );
    }

    #[test]
    fn translate_engine_no_variant_is_rust_owned() {
        // All backends run in the Python worker — none talk to a
        // provider directly from the Rust server.
        assert!(!TranslateEngineName::Google.is_rust_owned());
        assert!(!TranslateEngineName::Seamless.is_rust_owned());
        assert!(!TranslateEngineName::Nllb.is_rust_owned());
        assert!(!TranslateEngineName::Tencent.is_rust_owned());
    }

    #[test]
    fn translate_engine_serializes_as_wire_string() {
        let json = serde_json::to_string(&TranslateEngineName::Seamless).unwrap();
        assert_eq!(json, "\"seamless\"");
    }

    #[test]
    fn translate_engine_deserializes_from_wire_string() {
        let parsed: TranslateEngineName = serde_json::from_str("\"seamless\"").unwrap();
        assert_eq!(parsed, TranslateEngineName::Seamless);
    }

    #[test]
    fn translate_engine_deserialize_rejects_unknown_variant() {
        let err = serde_json::from_str::<TranslateEngineName>("\"gogle\"").unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("gogle"),
            "expected error to mention the bad name, got: {message}"
        );
    }

    // ---- EngineOverrides translate field ----

    #[test]
    fn engine_overrides_serializes_translate_field() {
        let overrides = EngineOverrides {
            asr: None,
            fa: None,
            translate: Some(TranslateEngineName::Seamless),
        };
        let json = overrides.to_json_string();
        assert_eq!(json, "{\"translate\":\"seamless\"}");
    }

    #[test]
    fn engine_overrides_deserializes_translate_field() {
        let parsed: EngineOverrides = serde_json::from_str("{\"translate\":\"seamless\"}").unwrap();
        assert_eq!(parsed.translate, Some(TranslateEngineName::Seamless));
        assert!(parsed.asr.is_none());
        assert!(parsed.fa.is_none());
    }

    #[test]
    fn engine_overrides_translate_only_is_not_empty() {
        let overrides = EngineOverrides {
            asr: None,
            fa: None,
            translate: Some(TranslateEngineName::Seamless),
        };
        assert!(!overrides.is_empty());
    }

    #[test]
    fn engine_overrides_all_none_is_still_empty() {
        let overrides = EngineOverrides::default();
        assert!(overrides.is_empty());
        assert_eq!(overrides.to_json_string(), "");
    }
}
