//! Registry ownership and build admission retain their immutable source.
use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum DiscoveryDisposition<'source> {
    Accept(CompatibleEntry<'source>),
    SkipForeignOwner,
    ReapStaleOwned,
    RefuseForeignBuild(ForeignBuildDaemon),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CompatibleEntry<'source> {
    source: &'source RegistryEntry,
}
pub(super) struct PreparedEntry<'source> {
    source: &'source RegistryEntry,
    profile: WorkerProfile,
    language: WorkerLanguage,
}
impl<'source> CompatibleEntry<'source> {
    pub(super) fn prepare(self) -> Result<PreparedEntry<'source>, String> {
        let profile = self
            .source
            .worker_profile()
            .ok_or_else(|| format!("unknown worker profile {}", self.source.profile))?;
        let language = WorkerLanguage::parse_untrusted(&self.source.lang)
            .map_err(|error| error.to_string())?;
        Ok(PreparedEntry {
            source: self.source,
            profile,
            language,
        })
    }
}
impl<'source> PreparedEntry<'source> {
    pub(super) fn into_parts(self) -> (&'source RegistryEntry, WorkerProfile, WorkerLanguage) {
        (self.source, self.profile, self.language)
    }
}

pub(super) fn discovery_disposition<'source>(
    entry: &'source RegistryEntry,
    current: &str,
    build: &str,
) -> DiscoveryDisposition<'source> {
    classify(entry, current, build, |pid| {
        if process_alive(pid) {
            OwnerState::Alive
        } else {
            OwnerState::Exited
        }
    })
}

pub(super) enum OwnerState {
    Alive,
    Exited,
}
/// The external owner observation is admitted once before build comparison.
fn classify<'source>(
    entry: &'source RegistryEntry,
    current: &str,
    build: &str,
    owner: impl FnOnce(u32) -> OwnerState,
) -> DiscoveryDisposition<'source> {
    if entry.is_server_owned() {
        let Some(instance) = entry.owner_server_instance_id() else {
            return DiscoveryDisposition::ReapStaleOwned;
        };
        let Some(pid) = entry.owner_server_pid else {
            return DiscoveryDisposition::ReapStaleOwned;
        };
        if instance != current {
            return match owner(pid) {
                OwnerState::Alive => DiscoveryDisposition::SkipForeignOwner,
                OwnerState::Exited => DiscoveryDisposition::ReapStaleOwned,
            };
        }
    }
    let reported = match entry.build_identity.as_deref() {
        Some(reported) if reported == build => {
            return DiscoveryDisposition::Accept(CompatibleEntry { source: entry });
        }
        Some(reported) => ReportedBuild::Build(reported.to_owned()),
        None => ReportedBuild::Unreported,
    };
    DiscoveryDisposition::RefuseForeignBuild(ForeignBuildDaemon {
        pid: entry.pid,
        profile: entry.profile.clone(),
        lang: entry.lang.clone(),
        reported,
        ours: build.to_owned(),
    })
}

#[cfg(test)]
pub(super) fn classify_declared<'source>(
    entry: &'source RegistryEntry,
    current: &str,
    build: &str,
    owner: OwnerState,
) -> DiscoveryDisposition<'source> {
    classify(entry, current, build, |_| owner)
}
