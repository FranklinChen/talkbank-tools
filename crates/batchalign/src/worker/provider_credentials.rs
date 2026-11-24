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
        Self::admit_sources(
            sources,
            section.as_ref(),
            crate::revai::load_revai_api_key().ok(),
        )
    }

    fn admit_sources(
        sources: &HkAsrCredentialSources,
        section: Option<&BTreeMap<String, String>>,
        revai: Option<crate::revai::RevAiApiKey>,
    ) -> Self {
        Self {
            revai,
            tencent: sources.load_tencent(section),
            tencent_engine: HkAsrCredentialSources::tencent_engine_setting(section),
            aliyun: sources.load_aliyun(section),
        }
    }

    /// Admit an explicit synthetic account snapshot through the production readers,
    /// without consulting the process environment or any credential file.
    #[cfg(test)]
    pub(crate) fn from_explicit_hk_sources(sources: &HkAsrCredentialSources) -> Self {
        Self::admit_sources(sources, None, None)
    }

    /// No provider account: what a test or an offline replay runs as, which
    /// reads no credentials from the machine running it.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            revai: None,
            tencent: None,
            tencent_engine: crate::model_manifest::TencentEngineSetting::Unset,
            aliyun: None,
        }
    }

    /// The Rev.AI key, handed to every worker.
    #[must_use]
    pub const fn revai_api_key(&self) -> Option<&crate::revai::RevAiApiKey> {
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
    pub(crate) const fn tencent_engine(&self) -> &crate::model_manifest::TencentEngineSetting {
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
pub struct HkAsrCredentialSources {
    legacy_ini_path: PathBuf,
    tencent: Option<TencentAsrCredentials>,
    aliyun: Option<AliyunAsrCredentials>,
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
            tencent: complete_tencent(tencent_id, tencent_key, tencent_region, tencent_bucket),
            aliyun: complete_aliyun(aliyun_ak_id, aliyun_ak_secret, aliyun_ak_appkey),
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
        self.tencent.clone().or_else(|| {
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
        self.aliyun.clone().or_else(|| {
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
    use super::*;
    use crate::model_manifest::{TencentEngine, TencentEngineSetting};

    /// The fixture owns the parsed on-disk section used for every case.
    /// Each case admits synthetic environment values without reading real keys.
    struct IniAccountFixture(Option<BTreeMap<String, String>>);

    impl IniAccountFixture {
        fn observe(file: &str) -> Self {
            let home = tempfile::tempdir().unwrap();
            std::fs::write(home.path().join(".batchalign.ini"), file).unwrap();
            Self(Self::sources(None, None, home.path().to_str()).read_legacy_ini_asr_section())
        }

        fn sources(
            id: Option<&str>,
            key: Option<&str>,
            home: Option<&str>,
        ) -> HkAsrCredentialSources {
            HkAsrCredentialSources::from_sources(
                id,
                key,
                Some(" ap-hongkong "),
                Some(" env-bucket "),
                None,
                None,
                None,
                home,
            )
        }

        fn admit(&self, id: Option<&str>, key: Option<&str>) -> ProviderAccounts {
            ProviderAccounts::admit_sources(&Self::sources(id, key, None), self.0.as_ref(), None)
        }
    }

    /// Guards: A complete environment account takes precedence, an incomplete
    /// one falls back as a unit, and only the selected provider's keys cross
    /// the worker boundary. The engine comes from the INI file's ASR section.
    /// Kind: contract
    /// Not a type: External environment and INI bytes must normalize into
    /// the intended account, and their precedence remains a policy decision.
    /// State: A temporary INI and explicit fake accounts enter the production readers.
    /// Concurrency: All values are owned; no process environment or credentials are read.
    #[test]
    fn worker_provider_accounts_follow_source_and_provider_precedence() {
        let legacy = IniAccountFixture::observe(
            "[other]\nengine.tencent.model=16k_yue\n[asr]\n             engine.tencent.model=16k_zh_en_2.0\nengine.tencent.id=legacy-id\n             engine.tencent.key=legacy-key\nengine.tencent.region=ap-guangzhou\n             engine.tencent.bucket=legacy-bucket\nengine.aliyun.ak_id=legacy-aliyun\n             engine.aliyun.ak_secret=legacy-secret\nengine.aliyun.ak_appkey=legacy-app\n",
        );
        let explicit = legacy.admit(Some(" env-id "), Some(" env-key "));
        assert_eq!(
            explicit.worker_envs(Some("tencent")),
            BTreeMap::from([
                (BATCHALIGN_TENCENT_ID.into(), "env-id".into()),
                (BATCHALIGN_TENCENT_KEY.into(), "env-key".into()),
                (BATCHALIGN_TENCENT_REGION.into(), "ap-hongkong".into()),
                (BATCHALIGN_TENCENT_BUCKET.into(), "env-bucket".into()),
            ]),
        );
        let partial = legacy.admit(Some("partial-only"), None);
        assert_eq!(
            partial.worker_envs(Some("tencent")),
            BTreeMap::from([
                (BATCHALIGN_TENCENT_ID.into(), "legacy-id".into()),
                (BATCHALIGN_TENCENT_KEY.into(), "legacy-key".into()),
                (BATCHALIGN_TENCENT_REGION.into(), "ap-guangzhou".into()),
                (BATCHALIGN_TENCENT_BUCKET.into(), "legacy-bucket".into()),
            ]),
        );
        assert_eq!(
            partial.worker_envs(Some("aliyun")),
            BTreeMap::from([
                (BATCHALIGN_ALIYUN_AK_ID.into(), "legacy-aliyun".into()),
                (BATCHALIGN_ALIYUN_AK_SECRET.into(), "legacy-secret".into()),
                (BATCHALIGN_ALIYUN_AK_APPKEY.into(), "legacy-app".into()),
            ]),
        );
        assert!(partial.worker_envs(Some("whisper")).is_empty());
        assert_eq!(
            partial.tencent_engine(),
            &TencentEngineSetting::Chosen(TencentEngine::ZhEn2_0)
        );
        let unset = IniAccountFixture::observe("[asr]\nengine.tencent.id=id\n").admit(None, None);
        assert_eq!(unset.tencent_engine(), &TencentEngineSetting::Unset);
    }

    /// Guards: Account diagnostics report configuration status without secret text.
    /// Kind: contract
    /// Not a type: The runtime Debug formatter can expose any field it reads,
    /// so the real rendered text must exclude even complete credentials.
    /// State: Synthetic in-memory keys and the production Debug implementation.
    /// Concurrency: No environment or credential files are used.
    #[test]
    fn provider_accounts_debug_never_prints_a_secret() {
        let accounts =
            ProviderAccounts::from_explicit_hk_sources(&HkAsrCredentialSources::from_sources(
                Some("SECRET_ID"),
                Some("SECRET_KEY"),
                Some("ap-guangzhou"),
                Some("SECRET_BUCKET"),
                Some("ALIYUN_ID"),
                Some("ALIYUN_SECRET"),
                Some("ALIYUN_APP"),
                None,
            ));
        let rendered = format!("{accounts:?}");
        for secret in ["SECRET", "ALIYUN"] {
            assert!(!rendered.contains(secret), "{rendered}");
        }
        assert!(rendered.contains("configured"), "{rendered}");
    }
}
