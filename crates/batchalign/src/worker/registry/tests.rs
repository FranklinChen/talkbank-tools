use super::admission::{OwnerState, classify_declared};
use super::*;
const OUR_BUILD: &str = "our-build";
fn external_entry() -> RegistryEntry {
    serde_json::from_value(serde_json::json!({"pid":10,"host":"127.0.0.1","port":1234,"profile":"stanza","lang":"eng","build_identity":OUR_BUILD})).unwrap()
}
/// Guards: locked registry removal preserves a later writer's unrelated entry
/// and reports a second removal as absent without replacing the shared lock.
/// Kind: contract
/// Not a type: actual filesystem bytes and the shared Python/Rust lock protocol
/// can change after source admission and must be observed at the write boundary.
/// State: one private directory exercises the real registry read/locked-removal adapter.
/// Concurrency: the fixture owns all entries and paths, leaving ambient registries untouched.
#[test]
fn removal_keeps_entries_it_was_not_asked_to_remove() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("workers.json");
    let mut second = external_entry();
    second.pid = 11;
    second.port = 1235;
    std::fs::write(
        &path,
        serde_json::to_string(&[external_entry(), second]).expect("encode"),
    )
    .expect("seed");

    assert!(remove_stale_entry(&path, 10));
    let left = read_registry(&path);
    assert_eq!(left.len(), 1);
    assert!(left[0].is_same_daemon(&{
        let mut expected = external_entry();
        expected.pid = 11;
        expected.port = 1235;
        expected
    }));
    assert!(dir.path().join("workers.json.lock").exists());
    assert!(!remove_stale_entry(&path, 10), "nothing left to remove");
}
/// Guards: ownership precedes build admission, accepted metadata retains its
/// source, and shutdown authority belongs only to the matching server instance.
/// Kind: boundary
/// Not a type: registry JSON and the owner's external liveness select among
/// skip/reap/refuse/adopt policies before a CompatibleEntry can be issued.
/// State: decoded synthetic entries and declared owner observations replace live PID queries.
/// Concurrency: no process, registry file or environment is shared or changed.
#[test]
fn registry_source_admission_preserves_ownership_and_build_order() {
    enum EntryOwner {
        External,
        Ours,
        Foreign(OwnerState),
        MissingOwner,
    }
    #[derive(Debug)]
    enum Expected {
        Accept,
        Skip,
        Reap,
    }
    for (owner, expected) in [
        (EntryOwner::External, Expected::Accept),
        (EntryOwner::Ours, Expected::Accept),
        (EntryOwner::Foreign(OwnerState::Alive), Expected::Skip),
        (EntryOwner::Foreign(OwnerState::Exited), Expected::Reap),
        (EntryOwner::MissingOwner, Expected::Reap),
    ] {
        let mut entry = external_entry();
        let shutdown = matches!(&owner, EntryOwner::Ours);
        let observation = match owner {
            EntryOwner::External => OwnerState::Exited,
            EntryOwner::Ours => {
                entry.ownership = RegistryOwnership::ServerOwned;
                entry.owner_server_instance_id = Some("current-server".into());
                entry.owner_server_pid = Some(20);
                OwnerState::Alive
            }
            EntryOwner::Foreign(state) => {
                entry.ownership = RegistryOwnership::ServerOwned;
                entry.owner_server_instance_id = Some("other-server".into());
                entry.owner_server_pid = Some(20);
                state
            }
            EntryOwner::MissingOwner => {
                entry.ownership = RegistryOwnership::ServerOwned;
                OwnerState::Exited
            }
        };
        assert_eq!(should_shutdown_entry(&entry, "current-server"), shutdown);
        if matches!(expected, Expected::Skip) {
            entry.build_identity = Some("older-build".into());
        }
        match (
            classify_declared(&entry, "current-server", OUR_BUILD, observation),
            expected,
        ) {
            (DiscoveryDisposition::Accept(accepted), Expected::Accept) => {
                let (source, profile, language) = accepted.prepare().unwrap().into_parts();
                assert_eq!(source, &entry);
                assert_eq!(profile, WorkerProfile::Stanza);
                assert_eq!(
                    language,
                    WorkerLanguage::from(crate::api::LanguageCode3::eng())
                );
            }
            (DiscoveryDisposition::SkipForeignOwner, Expected::Skip)
            | (DiscoveryDisposition::ReapStaleOwned, Expected::Reap) => {}
            other => panic!("unexpected registry disposition {other:?}"),
        }
    }
    // Raw registry metadata still needs admission after ownership/build policy.
    for (profile, language) in [("unknown-profile", "eng"), ("stanza", "english")] {
        let mut entry = external_entry();
        entry.profile = profile.into();
        entry.lang = language.into();
        let DiscoveryDisposition::Accept(accepted) =
            classify_declared(&entry, "current-server", OUR_BUILD, OwnerState::Exited)
        else {
            panic!("compatible build reaches metadata admission");
        };
        assert!(accepted.prepare().is_err());
    }
}
/// Guards: foreign or absent build bytes retain the exact reported identity,
/// key and restart remedy through the real refusal presentation.
/// Kind: wire
/// Not a type: external build strings and the operator-visible remedy require
/// byte-level agreement after private compatibility admission excludes forged success.
/// State: one decoded source entry enters the actual classifier and refusal reader.
/// Concurrency: none
#[test]
fn refused_registry_bytes_retain_identity_and_restart_details() {
    for (build, reported) in [
        (Some("older-build".to_owned()), "build older-build"),
        (None, "an unreported build"),
    ] {
        let mut entry = external_entry();
        entry.build_identity = build;
        let DiscoveryDisposition::RefuseForeignBuild(refusal) =
            classify_declared(&entry, "current-server", OUR_BUILD, OwnerState::Exited)
        else {
            panic!("foreign build must be refused");
        };
        let rendered = refusal.to_string();
        assert!(rendered.contains(reported));
        assert!(rendered.contains("batchalign3 worker start --profile stanza --lang eng"));
        let displayed = refusal.to_refused_registry_worker();
        assert_eq!(displayed.worker_key, "profile:stanza:eng");
        assert_eq!(displayed.pid, 10);
        match displayed.reason {
            crate::api::RegistryWorkerRefusal::ForeignBuild {
                reported_build,
                server_build,
            } => {
                assert_eq!(reported_build, "older-build");
                assert_eq!(server_build, OUR_BUILD);
            }
            crate::api::RegistryWorkerRefusal::UnreportedBuild { server_build } => {
                assert_eq!(server_build, OUR_BUILD)
            }
        }
    }
}

/// Guards: Cancelling real health I/O closes its probe socket and releases its
/// claim. A subsequent probe can connect, and a foreign capability reply closes
/// that connection without publishing it or deleting the live registry entry.
/// Kind: contract
/// Not a type: The peer controls delivery of health and capability bytes. Socket
/// closure and availability to subsequent discovery depend on actual I/O and
/// future disposal.
/// State: A private registry and loopback listener exercise Rust discovery
/// without any model, process signalling or environment changes.
/// Concurrency: A timed test races two futures. The peer must read the first
/// request before discovery is dropped. Its second exchange announces readiness
/// through the actual socket.
#[tokio::test]
async fn probe_cancellation_and_foreign_reply_release_registry_admission() {
    use crate::types::worker_v2::TaskTimeoutOverrides;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let mut entry = external_entry();
    entry.port = listener.local_addr().unwrap().port();
    entry.build_identity = Some(crate::build_hash().to_owned());
    std::fs::write(&path, serde_json::to_vec(&[&entry]).unwrap()).unwrap();
    let admissions = RegistryAdmissions::default();
    let discover = || discover_workers(&path, TaskTimeoutOverrides::NONE, "test", &admissions);

    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let mut cancelled_peer = {
            let pending = discover();
            tokio::pin!(pending);
            tokio::select! {
                _ = &mut pending => panic!("probe finished before receiving a health response"),
                peer = async {
                    let mut peer = BufReader::new(listener.accept().await.unwrap().0);
                    let mut line = String::new();
                    peer.read_line(&mut line).await.unwrap();
                    assert_eq!(serde_json::from_str::<serde_json::Value>(&line).unwrap()["op"], "health");
                    peer
                } => peer,
            }
        };
        assert_eq!(cancelled_peer.read_line(&mut String::new()).await.unwrap(), 0);

        let health = concat!(
            r#"{"op":"health","response":{"status":"ok","command":"infer:morphosyntax","lang":"eng","pid":10,"uptime_s":0.0}}"#,
            "\n",
        );
        let (result, ()) = tokio::join!(discover(), async {
            let mut peer = BufReader::new(listener.accept().await.unwrap().0);
            for expected in ["health", "capabilities"] {
                let mut line = String::new();
                peer.read_line(&mut line).await.unwrap();
                assert_eq!(serde_json::from_str::<serde_json::Value>(&line).unwrap()["op"], expected);
                peer.get_mut().write_all(health.as_bytes()).await.unwrap();
            }
            assert_eq!(peer.read_line(&mut String::new()).await.unwrap(), 0);
        });
        assert!(result.workers.is_empty());
        assert_eq!(read_registry(&path), [entry]);
    })
    .await
    .expect("loopback discovery completes its owned exchanges");
}
