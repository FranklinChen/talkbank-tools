//! HK ASR provider accounts: read once, handed to workers.
//!
//! Rust owns credential discovery for worker-launched HK providers and injects
//! the resolved values into child env vars so Python stays on the provider SDK
//! boundary instead of reopening `~/.batchalign.ini`.
//!
//! The accounts are read ONCE, when the server builds its worker runtime
//! ([`ProviderAccounts::from_env`]), and every worker spawn hands over keys
//! from that one snapshot. A Tencent account is its keys and the engine it
//! bought, read together, so a running server can never hold one account's
//! keys beside another's engine; a configuration change takes effect on a
//! server restart, as a new binary does. Until 2026-11-05 the keys were re-read
//! at every spawn while the engine was fixed at start.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::PathBuf;

use tracing::warn;

const BATCHALIGN_TENCENT_ID: &str = "BATCHALIGN_TENCENT_ID";
const BATCHALIGN_TENCENT_KEY: &str = "BATCHALIGN_TENCENT_KEY";
const BATCHALIGN_TENCENT_REGION: &str = "BATCHALIGN_TENCENT_REGION";
const BATCHALIGN_TENCENT_BUCKET: &str = "BATCHALIGN_TENCENT_BUCKET";
const BATCHALIGN_ALIYUN_AK_ID: &str = "BATCHALIGN_ALIYUN_AK_ID";
const BATCHALIGN_ALIYUN_AK_SECRET: &str = "BATCHALIGN_ALIYUN_AK_SECRET";
const BATCHALIGN_ALIYUN_AK_APPKEY: &str = "BATCHALIGN_ALIYUN_AK_APPKEY";

const ENGINE_TENCENT_ID: &str = "engine.tencent.id";
const ENGINE_TENCENT_KEY: &str = "engine.tencent.key";
const ENGINE_TENCENT_REGION: &str = "engine.tencent.region";
const ENGINE_TENCENT_BUCKET: &str = "engine.tencent.bucket";
const ENGINE_ALIYUN_AK_ID: &str = "engine.aliyun.ak_id";
const ENGINE_ALIYUN_AK_SECRET: &str = "engine.aliyun.ak_secret";
const ENGINE_ALIYUN_AK_APPKEY: &str = "engine.aliyun.ak_appkey";

#[derive(Clone, PartialEq, Eq)]
struct TencentAsrCredentials {
    secret_id: String,
    secret_key: String,
    region: String,
    bucket: String,
}

impl TencentAsrCredentials {
    fn as_env(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (BATCHALIGN_TENCENT_ID.to_string(), self.secret_id.clone()),
            (BATCHALIGN_TENCENT_KEY.to_string(), self.secret_key.clone()),
            (BATCHALIGN_TENCENT_REGION.to_string(), self.region.clone()),
            (BATCHALIGN_TENCENT_BUCKET.to_string(), self.bucket.clone()),
        ])
    }
}

#[derive(Clone, PartialEq, Eq)]
struct AliyunAsrCredentials {
    ak_id: String,
    ak_secret: String,
    appkey: String,
}

impl AliyunAsrCredentials {
    fn as_env(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (BATCHALIGN_ALIYUN_AK_ID.to_string(), self.ak_id.clone()),
            (
                BATCHALIGN_ALIYUN_AK_SECRET.to_string(),
                self.ak_secret.clone(),
            ),
            (BATCHALIGN_ALIYUN_AK_APPKEY.to_string(), self.appkey.clone()),
        ])
    }
}

/// The HK ASR provider accounts this server runs as, read once.
///
/// `Debug` names which accounts are configured and the Tencent engine, never a
/// secret: it sits in [`crate::worker::handle::WorkerRuntimeConfig`], whose
/// `Debug` may reach a log.
#[derive(Clone, PartialEq, Eq)]
pub struct ProviderAccounts {
    revai: Option<crate::revai::RevAiApiKey>,
    tencent: Option<TencentAsrCredentials>,
    tencent_engine: crate::model_manifest::TencentEngineSetting,
    aliyun: Option<AliyunAsrCredentials>,
}

impl std::fmt::Debug for ProviderAccounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAccounts")
            .field("revai", &self.revai.as_ref().map(|_| "configured"))
            .field("tencent", &self.tencent.as_ref().map(|_| "configured"))
            .field("tencent_engine", &self.tencent_engine)
            .field("aliyun", &self.aliyun.as_ref().map(|_| "configured"))
            .finish()
    }
}

impl ProviderAccounts {
    /// Read the accounts from this process's environment and
    /// `~/.batchalign.ini`, once.
    pub(crate) fn from_env() -> Self {
        Self::read(&HkAsrCredentialSources::from_env())
    }

    /// Read the accounts from explicit sources, the file read once so the
    /// three accounts describe one state of it.
    pub(crate) fn read(sources: &HkAsrCredentialSources) -> Self {
        let section = sources.read_legacy_ini_asr_section();
        let section = section.as_ref();
        Self {
            // Rev.AI's key has its own reader (`revai::credentials`), shared
            // with the Rust Rev.AI client.
            revai: crate::revai::load_revai_api_key().ok(),
            tencent: sources.load_tencent(section),
            tencent_engine: HkAsrCredentialSources::tencent_engine_setting(section),
            aliyun: sources.load_aliyun(section),
        }
    }

    /// No provider account: what a test or an offline replay runs as, which
    /// reads no credentials from the machine running it.
    pub fn none() -> Self {
        Self {
            revai: None,
            tencent: None,
            tencent_engine: crate::model_manifest::TencentEngineSetting::Unset,
            aliyun: None,
        }
    }

    /// The Rev.AI key, handed to every worker.
    pub fn revai_api_key(&self) -> Option<&crate::revai::RevAiApiKey> {
        self.revai.as_ref()
    }

    /// No account but this Rev.AI key, for tests of what hands it over.
    #[cfg(test)]
    pub(crate) fn with_revai_api_key(key: Option<crate::revai::RevAiApiKey>) -> Self {
        Self {
            revai: key,
            ..Self::none()
        }
    }

    /// The Tencent engine the account chose.
    pub(crate) fn tencent_engine(&self) -> &crate::model_manifest::TencentEngineSetting {
        &self.tencent_engine
    }

    /// Worker env vars for the selected ASR engine, when its account is
    /// configured.
    pub(crate) fn worker_envs(&self, asr_override: Option<&str>) -> BTreeMap<String, String> {
        match asr_override {
            Some("tencent") => self
                .tencent
                .as_ref()
                .map_or_else(BTreeMap::new, TencentAsrCredentials::as_env),
            Some("aliyun") => self
                .aliyun
                .as_ref()
                .map_or_else(BTreeMap::new, AliyunAsrCredentials::as_env),
            _ => BTreeMap::new(),
        }
    }
}

/// Ambient sources Rust may use to resolve HK provider credentials before
/// spawning a Python worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HkAsrCredentialSources {
    legacy_ini_path: PathBuf,
    tencent_id: Option<String>,
    tencent_key: Option<String>,
    tencent_region: Option<String>,
    tencent_bucket: Option<String>,
    aliyun_ak_id: Option<String>,
    aliyun_ak_secret: Option<String>,
    aliyun_ak_appkey: Option<String>,
}

impl HkAsrCredentialSources {
    /// Resolve ambient worker-launch credential sources from the current process
    /// environment plus the default legacy config location.
    pub(crate) fn from_env() -> Self {
        let home = crate::runtime_paths::BatchalignHome::from_env();
        Self::from_sources(
            std::env::var(BATCHALIGN_TENCENT_ID).ok().as_deref(),
            std::env::var(BATCHALIGN_TENCENT_KEY).ok().as_deref(),
            std::env::var(BATCHALIGN_TENCENT_REGION).ok().as_deref(),
            std::env::var(BATCHALIGN_TENCENT_BUCKET).ok().as_deref(),
            std::env::var(BATCHALIGN_ALIYUN_AK_ID).ok().as_deref(),
            std::env::var(BATCHALIGN_ALIYUN_AK_SECRET).ok().as_deref(),
            std::env::var(BATCHALIGN_ALIYUN_AK_APPKEY).ok().as_deref(),
            home.base().to_str(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_sources(
        tencent_id: Option<&str>,
        tencent_key: Option<&str>,
        tencent_region: Option<&str>,
        tencent_bucket: Option<&str>,
        aliyun_ak_id: Option<&str>,
        aliyun_ak_secret: Option<&str>,
        aliyun_ak_appkey: Option<&str>,
        home_env: Option<&str>,
    ) -> Self {
        let home = home_env.unwrap_or(".");
        Self {
            legacy_ini_path: PathBuf::from(home).join(".batchalign.ini"),
            tencent_id: normalize_value(tencent_id),
            tencent_key: normalize_value(tencent_key),
            tencent_region: normalize_value(tencent_region),
            tencent_bucket: normalize_value(tencent_bucket),
            aliyun_ak_id: normalize_value(aliyun_ak_id),
            aliyun_ak_secret: normalize_value(aliyun_ak_secret),
            aliyun_ak_appkey: normalize_value(aliyun_ak_appkey),
        }
    }

    /// The account's Tencent engine choice, `engine.tencent.model` in the
    /// `[asr]` section of `~/.batchalign.ini`, beside the account's keys.
    ///
    /// Read from the file only, with no environment override: the keys have
    /// one because the variables are also how Rust hands them to a Python
    /// worker, while the engine reaches the worker as a pinned model in its
    /// spawn arguments and so needs no variable.
    fn tencent_engine_setting(
        section: Option<&BTreeMap<String, String>>,
    ) -> crate::model_manifest::TencentEngineSetting {
        crate::model_manifest::TencentEngineSetting::admit(
            section
                .and_then(|section| section.get(crate::model_manifest::TENCENT_ENGINE_INI_KEY))
                .map(String::as_str),
        )
    }

    /// The Tencent engine alone, for a reader that needs no keys (the
    /// evidence-cache inspection), so no secret is loaded into it.
    pub(crate) fn read_tencent_engine(&self) -> crate::model_manifest::TencentEngineSetting {
        Self::tencent_engine_setting(self.read_legacy_ini_asr_section().as_ref())
    }

    fn load_tencent(
        &self,
        section: Option<&BTreeMap<String, String>>,
    ) -> Option<TencentAsrCredentials> {
        complete_tencent(
            self.tencent_id.as_deref(),
            self.tencent_key.as_deref(),
            self.tencent_region.as_deref(),
            self.tencent_bucket.as_deref(),
        )
        .or_else(|| {
            let section = section?;
            complete_tencent(
                section.get(ENGINE_TENCENT_ID).map(String::as_str),
                section.get(ENGINE_TENCENT_KEY).map(String::as_str),
                section.get(ENGINE_TENCENT_REGION).map(String::as_str),
                section.get(ENGINE_TENCENT_BUCKET).map(String::as_str),
            )
        })
    }

    fn load_aliyun(
        &self,
        section: Option<&BTreeMap<String, String>>,
    ) -> Option<AliyunAsrCredentials> {
        complete_aliyun(
            self.aliyun_ak_id.as_deref(),
            self.aliyun_ak_secret.as_deref(),
            self.aliyun_ak_appkey.as_deref(),
        )
        .or_else(|| {
            let section = section?;
            complete_aliyun(
                section.get(ENGINE_ALIYUN_AK_ID).map(String::as_str),
                section.get(ENGINE_ALIYUN_AK_SECRET).map(String::as_str),
                section.get(ENGINE_ALIYUN_AK_APPKEY).map(String::as_str),
            )
        })
    }

    fn read_legacy_ini_asr_section(&self) -> Option<BTreeMap<String, String>> {
        let contents = match std::fs::read_to_string(&self.legacy_ini_path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == ErrorKind::NotFound => return None,
            Err(error) => {
                warn!(
                    path = %self.legacy_ini_path.display(),
                    error = %error,
                    "failed to read legacy HK credential config"
                );
                return None;
            }
        };
        let section = parse_legacy_ini_asr_section(&contents);
        (!section.is_empty()).then_some(section)
    }
}

fn normalize_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn complete_tencent(
    secret_id: Option<&str>,
    secret_key: Option<&str>,
    region: Option<&str>,
    bucket: Option<&str>,
) -> Option<TencentAsrCredentials> {
    Some(TencentAsrCredentials {
        secret_id: normalize_value(secret_id)?,
        secret_key: normalize_value(secret_key)?,
        region: normalize_value(region)?,
        bucket: normalize_value(bucket)?,
    })
}

fn complete_aliyun(
    ak_id: Option<&str>,
    ak_secret: Option<&str>,
    appkey: Option<&str>,
) -> Option<AliyunAsrCredentials> {
    Some(AliyunAsrCredentials {
        ak_id: normalize_value(ak_id)?,
        ak_secret: normalize_value(ak_secret)?,
        appkey: normalize_value(appkey)?,
    })
}

fn parse_legacy_ini_asr_section(contents: &str) -> BTreeMap<String, String> {
    let mut in_asr = false;
    let mut entries = BTreeMap::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if let Some(section) = trimmed
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            in_asr = section.trim().eq_ignore_ascii_case("asr");
            continue;
        }
        if !in_asr {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once('=') {
            entries.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::{
        BATCHALIGN_ALIYUN_AK_APPKEY, BATCHALIGN_ALIYUN_AK_ID, BATCHALIGN_ALIYUN_AK_SECRET,
        BATCHALIGN_TENCENT_BUCKET, BATCHALIGN_TENCENT_ID, BATCHALIGN_TENCENT_KEY,
        BATCHALIGN_TENCENT_REGION, HkAsrCredentialSources, ProviderAccounts,
    };

    #[test]
    fn provider_envs_prefer_explicit_tencent_env() {
        let sources = HkAsrCredentialSources::from_sources(
            Some(" env-id "),
            Some(" env-key "),
            Some(" ap-hongkong "),
            Some(" env-bucket "),
            None,
            None,
            None,
            Some("/tmp/unused-home"),
        );

        let envs = ProviderAccounts::read(&sources).worker_envs(Some("tencent"));

        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_ID).map(String::as_str),
            Some("env-id")
        );
        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_KEY).map(String::as_str),
            Some("env-key")
        );
        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_REGION).map(String::as_str),
            Some("ap-hongkong")
        );
        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_BUCKET).map(String::as_str),
            Some("env-bucket")
        );
    }

    /// The snapshot's `Debug` says which accounts are configured, never a
    /// secret: it lives in the worker runtime, whose `Debug` can reach a log.
    #[test]
    fn provider_accounts_debug_never_prints_a_secret() {
        let accounts = ProviderAccounts::read(&HkAsrCredentialSources::from_sources(
            Some("SECRET_ID"),
            Some("SECRET_KEY"),
            Some("ap-guangzhou"),
            Some("SECRET_BUCKET"),
            Some("ALIYUN_ID"),
            Some("ALIYUN_SECRET"),
            Some("ALIYUN_APP"),
            Some("/tmp/unused-home"),
        ));
        let rendered = format!("{accounts:?}");
        for secret in ["SECRET", "ALIYUN"] {
            assert!(!rendered.contains(secret), "{rendered}");
        }
        assert!(rendered.contains("configured"), "{rendered}");
    }

    /// The engine is read from the `[asr]` section beside the keys; a file
    /// without the line is `Unset`, never a default engine.
    #[test]
    fn the_tencent_engine_is_read_from_the_asr_section() {
        use crate::model_manifest::{TencentEngine, TencentEngineSetting};
        let tempdir = tempfile::tempdir().unwrap();
        let home = tempdir.path().to_str().unwrap();
        let sources = || {
            HkAsrCredentialSources::from_sources(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(home),
            )
        };
        std::fs::write(
            tempdir.path().join(".batchalign.ini"),
            "[asr]\nengine.tencent.id = id\n",
        )
        .unwrap();
        assert_eq!(sources().read_tencent_engine(), TencentEngineSetting::Unset);
        std::fs::write(
            tempdir.path().join(".batchalign.ini"),
            "[other]\nengine.tencent.model = 16k_yue\n[asr]\nengine.tencent.model = 16k_zh_en_2.0\n",
        )
        .unwrap();
        assert_eq!(
            sources().read_tencent_engine(),
            TencentEngineSetting::Chosen(TencentEngine::ZhEn2_0)
        );
    }

    #[test]
    fn provider_envs_fall_back_to_legacy_ini() {
        let tempdir = tempfile::tempdir().unwrap();
        std::fs::write(
            tempdir.path().join(".batchalign.ini"),
            "[asr]\nengine.aliyun.ak_id = legacy-id\nengine.aliyun.ak_secret = legacy-secret\nengine.aliyun.ak_appkey = legacy-appkey\n",
        )
        .unwrap();
        let sources = HkAsrCredentialSources::from_sources(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tempdir.path().to_str(),
        );

        let envs = ProviderAccounts::read(&sources).worker_envs(Some("aliyun"));

        assert_eq!(
            envs.get(BATCHALIGN_ALIYUN_AK_ID).map(String::as_str),
            Some("legacy-id")
        );
        assert_eq!(
            envs.get(BATCHALIGN_ALIYUN_AK_SECRET).map(String::as_str),
            Some("legacy-secret")
        );
        assert_eq!(
            envs.get(BATCHALIGN_ALIYUN_AK_APPKEY).map(String::as_str),
            Some("legacy-appkey")
        );
    }

    #[test]
    fn provider_envs_ignore_incomplete_env_and_other_overrides() {
        let tempdir = tempfile::tempdir().unwrap();
        std::fs::write(
            tempdir.path().join(".batchalign.ini"),
            "[asr]\nengine.tencent.id = legacy-id\nengine.tencent.key = legacy-key\nengine.tencent.region = ap-guangzhou\nengine.tencent.bucket = legacy-bucket\n",
        )
        .unwrap();
        let sources = HkAsrCredentialSources::from_sources(
            Some("partial-only"),
            None,
            None,
            None,
            None,
            None,
            None,
            tempdir.path().to_str(),
        );

        let envs = ProviderAccounts::read(&sources).worker_envs(Some("tencent"));
        let empty_envs = ProviderAccounts::read(&sources).worker_envs(Some("whisper"));

        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_ID).map(String::as_str),
            Some("legacy-id")
        );
        assert_eq!(
            envs.get(BATCHALIGN_TENCENT_BUCKET).map(String::as_str),
            Some("legacy-bucket")
        );
        assert!(empty_envs.is_empty());
    }
}
