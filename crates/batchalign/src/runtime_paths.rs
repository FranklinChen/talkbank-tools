//! Shared runtime filesystem-path overrides.
//!
//! Centralizes environment-variable seams for Batchalign-owned cache paths so
//! the app runtime and the CLI `cache` command agree on where analysis/media
//! artifacts live.

use std::path::{Path, PathBuf};

/// Environment variable naming the one root of everything Batchalign owns.
pub const HOME_ENV: &str = "BATCHALIGN_HOME";

/// Where Batchalign keeps what it owns: `.batchalign.ini`, the
/// `.batchalign3` state directory, and its analysis and media caches.
///
/// Model libraries (Hugging Face, Stanza, torch) are NOT under it: they keep
/// resolving their caches from the process's real `HOME`. That separation is
/// the point. The test harness used to isolate Batchalign by faking `HOME`,
/// which relocated every library's cache too, so each live test re-downloaded
/// its models (23 GB for whisper-large-v3) into a temporary directory inside
/// its time budget. Setting `BATCHALIGN_HOME` instead isolates exactly what
/// Batchalign owns and nothing else, and needs no list of libraries.
///
/// The more specific overrides keep precedence over this root:
/// `BATCHALIGN_STATE_DIR` for state, [`ANALYSIS_CACHE_DIR_ENV`] and
/// [`MEDIA_CACHE_DIR_ENV`] for the caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchalignHome {
    /// The user's home: `~/.batchalign.ini`, `~/.batchalign3`, and the
    /// platform's cache and data directories.
    User(PathBuf),
    /// An explicit root (`BATCHALIGN_HOME`): everything Batchalign owns lives
    /// under it, the caches in `.cache` and `.local/share` whatever the
    /// platform.
    Rooted(PathBuf),
}

impl BatchalignHome {
    /// The root from `BATCHALIGN_HOME`, else the user's home (else the
    /// temporary directory, where a process with no home has always kept its
    /// state).
    pub fn from_env() -> Self {
        Self::from_sources(std::env::var(HOME_ENV).ok().as_deref(), dirs::home_dir())
    }

    /// The root from explicit sources.
    pub fn from_sources(root: Option<&str>, user_home: Option<PathBuf>) -> Self {
        match override_dir_from(root) {
            Some(root) => Self::Rooted(root),
            None => Self::User(user_home.unwrap_or_else(std::env::temp_dir)),
        }
    }

    /// The directory holding `.batchalign.ini` and `.batchalign3`.
    pub fn base(&self) -> &Path {
        match self {
            Self::User(home) | Self::Rooted(home) => home,
        }
    }

    /// `.batchalign.ini`: the default ASR engine and provider credentials.
    pub fn config_ini(&self) -> PathBuf {
        self.base().join(".batchalign.ini")
    }

    /// The state directory when `BATCHALIGN_STATE_DIR` does not name one.
    pub fn default_state_dir(&self) -> PathBuf {
        self.base().join(".batchalign3")
    }

    /// The base the analysis cache lives under, when its own override is unset.
    pub fn cache_base(&self) -> Option<PathBuf> {
        match self {
            Self::User(_) => dirs::cache_dir(),
            Self::Rooted(root) => Some(root.join(".cache")),
        }
    }

    /// The base the media cache lives under, when its own override is unset.
    pub fn data_base(&self) -> Option<PathBuf> {
        match self {
            Self::User(_) => dirs::data_dir(),
            Self::Rooted(root) => Some(root.join(".local").join("share")),
        }
    }
}

/// Environment variable that overrides the analysis-cache directory.
///
/// When set, the SQLite cache database lives at
/// `$BATCHALIGN_ANALYSIS_CACHE_DIR/cache.db`.
pub const ANALYSIS_CACHE_DIR_ENV: &str = "BATCHALIGN_ANALYSIS_CACHE_DIR";

/// Accepted spelling that follows the `batchalign3` application name.
pub const ANALYSIS_CACHE_DIR_ALIAS_ENV: &str = "BATCHALIGN3_ANALYSIS_CACHE_DIR";

/// Environment variable that overrides the media-conversion cache directory.
pub const MEDIA_CACHE_DIR_ENV: &str = "BATCHALIGN_MEDIA_CACHE_DIR";

fn override_dir_from(source: Option<&str>) -> Option<PathBuf> {
    source
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
}

/// Parse an explicit analysis-cache directory override from a caller-provided
/// source.
pub fn analysis_cache_dir_override_from(source: Option<&str>) -> Option<PathBuf> {
    override_dir_from(source)
}

/// Resolve the canonical analysis-cache setting and its application-name
/// alias. A nonempty canonical value wins if both are present.
pub fn analysis_cache_dir_override_from_sources(
    canonical: Option<&str>,
    alias: Option<&str>,
) -> Option<PathBuf> {
    analysis_cache_dir_override_from(canonical).or_else(|| analysis_cache_dir_override_from(alias))
}

/// Parse the analysis-cache directory override from the current process
/// environment.
pub fn analysis_cache_dir_override_from_env() -> Option<PathBuf> {
    analysis_cache_dir_override_from_sources(
        std::env::var(ANALYSIS_CACHE_DIR_ENV).ok().as_deref(),
        std::env::var(ANALYSIS_CACHE_DIR_ALIAS_ENV).ok().as_deref(),
    )
}

/// Parse an explicit media-cache directory override from a caller-provided
/// source.
pub fn media_cache_dir_override_from(source: Option<&str>) -> Option<PathBuf> {
    override_dir_from(source)
}

/// Parse the media-cache directory override from the current process
/// environment.
pub fn media_cache_dir_override_from_env() -> Option<PathBuf> {
    media_cache_dir_override_from(std::env::var(MEDIA_CACHE_DIR_ENV).ok().as_deref())
}

#[cfg(test)]
mod home_tests {
    use super::BatchalignHome;
    use std::path::PathBuf;

    #[test]
    fn an_explicit_root_holds_everything_batchalign_owns() {
        let home = BatchalignHome::from_sources(Some(" /srv/ba3 "), Some(PathBuf::from("/home/u")));
        assert_eq!(home, BatchalignHome::Rooted(PathBuf::from("/srv/ba3")));
        assert_eq!(home.config_ini(), PathBuf::from("/srv/ba3/.batchalign.ini"));
        assert_eq!(
            home.default_state_dir(),
            PathBuf::from("/srv/ba3/.batchalign3")
        );
        assert_eq!(home.cache_base(), Some(PathBuf::from("/srv/ba3/.cache")));
        assert_eq!(
            home.data_base(),
            Some(PathBuf::from("/srv/ba3/.local/share"))
        );
    }

    #[test]
    fn without_a_root_the_user_home_keeps_todays_layout() {
        let home = BatchalignHome::from_sources(Some("  "), Some(PathBuf::from("/home/u")));
        assert_eq!(home, BatchalignHome::User(PathBuf::from("/home/u")));
        assert_eq!(home.config_ini(), PathBuf::from("/home/u/.batchalign.ini"));
        assert_eq!(home.cache_base(), dirs::cache_dir());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        analysis_cache_dir_override_from, analysis_cache_dir_override_from_sources,
        media_cache_dir_override_from,
    };
    use std::path::PathBuf;

    #[test]
    fn analysis_override_trims_whitespace() {
        assert_eq!(
            analysis_cache_dir_override_from(Some("  /tmp/analysis-cache  ")),
            Some(PathBuf::from("/tmp/analysis-cache"))
        );
    }

    #[test]
    fn analysis_override_ignores_empty_values() {
        assert_eq!(analysis_cache_dir_override_from(Some("   ")), None);
        assert_eq!(analysis_cache_dir_override_from(None), None);
    }

    #[test]
    fn analysis_override_accepts_batchalign3_alias() {
        assert_eq!(
            analysis_cache_dir_override_from_sources(None, Some("/tmp/alias-cache")),
            Some(PathBuf::from("/tmp/alias-cache"))
        );
    }

    #[test]
    fn canonical_analysis_override_precedes_alias() {
        assert_eq!(
            analysis_cache_dir_override_from_sources(
                Some("/tmp/canonical-cache"),
                Some("/tmp/alias-cache"),
            ),
            Some(PathBuf::from("/tmp/canonical-cache"))
        );
    }

    #[test]
    fn media_override_trims_whitespace() {
        assert_eq!(
            media_cache_dir_override_from(Some("  /tmp/media-cache  ")),
            Some(PathBuf::from("/tmp/media-cache"))
        );
    }

    #[test]
    fn media_override_ignores_empty_values() {
        assert_eq!(media_cache_dir_override_from(Some("")), None);
        assert_eq!(media_cache_dir_override_from(None), None);
    }
}
