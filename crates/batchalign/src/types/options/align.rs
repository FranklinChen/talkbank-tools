//! Alignment options retain an admitted cwd-independent media root.

use super::*;

/// Options for the `align` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// FA engine selector (`wav2vec_fa`, `whisper_fa`, or plugin name).
    #[serde(default = "default_fa_engine")]
    pub fa_engine: FaEngineName,

    /// Utterance-timing-recovery selection and tuning.
    #[serde(flatten)]
    pub utr: AlignUtrOptions,

    /// Include pause durations in forced alignment.
    #[serde(default)]
    pub pauses: bool,

    /// Existing- and cross-utterance boundary projection policy.
    #[serde(flatten)]
    pub boundaries: AlignBoundaryOptions,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default = "default_wor_tier_include")]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Apply post-FA bullet repair to fix timing violations.
    ///
    /// Uses boundary averaging, gap filling, and selective removal instead
    /// of CLAN FIXBULLETS. Experimental.
    #[serde(default)]
    pub bullet_repair: bool,

    /// Review tier verbosity (none / low-confidence / all).
    #[serde(default)]
    pub review_level: crate::chat_ops::fa::ReviewLevel,

    /// Directory to search for media files (audio/video).
    /// When set, the aligner looks here in addition to the standard
    /// media resolution paths (alongside .cha file, server media roots).
    /// This is retained request metadata. Submission and dispatch admission
    /// must produce an `AbsoluteMediaRoot` before it can reach media search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_dir: Option<MediaRootDeclaration>,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::default(),
            fa_engine: default_fa_engine(),
            utr: AlignUtrOptions::default(),
            pauses: false,
            boundaries: AlignBoundaryOptions::default(),
            wor: default_wor_tier_include(),
            merge_abbrev: MergeAbbrevPolicy::default(),
            bullet_repair: false,
            review_level: crate::chat_ops::fa::ReviewLevel::default(),
            media_dir: None,
        }
    }
}

impl AlignOptions {
    /// Return the effective FA engine after applying any shared `fa` override.
    pub fn effective_fa_engine(&self) -> FaEngineName {
        self.common.engine_overrides.fa.unwrap_or(self.fa_engine)
    }

    /// Return the effective UTR engine after applying any shared `utr`
    /// override, or `None` when no UTR pass was asked for.
    ///
    /// The `None` here means "no UTR pass", which is a real answer. An override
    /// cannot conjure a pass that was not requested: `--engine-overrides
    /// '{"utr":...}'` says WHICH engine, not WHETHER, exactly as the `fa` and
    /// `asr` overrides do.
    pub fn effective_utr_engine(&self) -> Option<UtrEngine> {
        let requested = self.utr.engine.as_ref()?;
        Some(
            self.common
                .engine_overrides
                .utr
                .clone()
                .unwrap_or_else(|| requested.clone()),
        )
    }
}
