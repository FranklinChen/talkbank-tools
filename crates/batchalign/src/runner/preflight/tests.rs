use super::*;
use crate::api::JobId;
use crate::db::JobDB;
use crate::store::test_support::{
    PathsFixture, claim_runner, pending_files, queued_paths_morphotag_job_fixture,
};
use crate::store::{InputKind, JobInputs, JobStore, PathsFiles};
use std::sync::Arc;

fn media_pending(root: &PathsFixture, name: &str) -> Vec<PendingJobFile> {
    let files =
        PathsFiles::admit(vec![root.input(name, InputKind::Media)]).expect("admit file set");
    pending_files(&JobInputs::Paths {
        files,
        source_dir: None,
    })
}

/// Guards: a missing admitted source becomes a rejection naming that same path.
/// Kind: boundary
/// Not a type: path authority proves containment and identity, while the file
/// can be absent when Tokio later observes its metadata.
/// State: an isolated existing parent admits a missing child. The real metadata
/// call supplies NotFound, so a pure observation double would miss this adapter.
/// Concurrency: a unique directory prevents another test from supplying the file.
#[tokio::test]
async fn media_validate_missing_file() {
    let root = PathsFixture::create();
    let observed = ObservedFiles::observe(media_pending(&root, "missing.wav")).await;
    assert!(observed.accepted.is_empty());
    assert_eq!(observed.rejected.len(), 1);
    assert!(
        observed.rejected[0]
            .description()
            .contains("File not found:")
    );
    assert!(observed.rejected[0].description().contains("missing.wav"));
}

/// Guards: a real zero-length recording is refused before format admission.
/// Kind: boundary
/// Not a type: the private NonemptyMedia constructor must still classify the
/// zero length supplied by the actual filesystem adapter before format admission.
/// State: one empty .wav in an isolated directory exercises Tokio metadata.
/// A fabricated Nonempty observation would bypass the boundary being checked.
/// Concurrency: directory ownership scopes file lifetime and removes rename races.
#[tokio::test]
async fn media_validate_empty_file() {
    let root = PathsFixture::create();
    std::fs::write(root.path().join("empty.wav"), []).expect("write empty recording");
    let observed = ObservedFiles::observe(media_pending(&root, "empty.wav")).await;
    assert!(observed.accepted.is_empty());
    assert_eq!(observed.rejected.len(), 1);
    assert!(
        observed.rejected[0]
            .description()
            .contains("File is empty:")
    );
}

/// Guards: unknown and absent filename extensions produce their own refusal.
/// Kind: boundary
/// Not a type: format admission interprets actual path suffix bytes after the
/// nonempty metadata observation. A new type cannot prove that policy mapping.
/// State: two nonempty files exercise the observer and suffix-policy boundary.
/// Their bytes intentionally require no decoder, which preflight never invokes.
/// Concurrency: both paths live in a unique temporary directory.
#[tokio::test]
async fn media_validate_unknown_extension() {
    let root = PathsFixture::create();
    for (name, reason) in [
        ("test.xyz", "Unknown media format '.xyz':"),
        ("extensionless", "File has no extension:"),
    ] {
        std::fs::write(root.path().join(name), b"data").expect("write nonempty source");
        let observed = ObservedFiles::observe(media_pending(&root, name)).await;
        assert!(observed.accepted.is_empty());
        assert_eq!(observed.rejected.len(), 1);
        assert!(observed.rejected[0].description().contains(reason));
    }
}

/// Guards: every advertised media suffix passes preflight in original input order.
/// Kind: contract
/// Not a type: the recognized-format catalog and path suffix admission must
/// agree on concrete filename bytes, including each newly advertised format.
/// State: one isolated directory holds nonempty files for the actual metadata
/// adapter. No media decoder or worker is involved in this metadata contract.
/// Concurrency: the file set and directory belong solely to this batch.
#[tokio::test]
async fn media_validate_all_known_extensions() {
    let root = PathsFixture::create();
    let names: Vec<_> = MediaExtensions::candidates("test").collect();
    let files = names
        .iter()
        .map(|name| {
            std::fs::write(root.path().join(name), b"data").expect("write nonempty source");
            root.input(name, InputKind::Media)
        })
        .collect();
    let files = PathsFiles::admit(files).expect("admit full advertised file set");
    let pending = pending_files(&JobInputs::Paths {
        files,
        source_dir: None,
    });
    let observed = ObservedFiles::observe(pending).await;
    assert!(observed.rejected.is_empty());
    assert_eq!(
        observed
            .accepted
            .iter()
            .map(|file| file.filename().as_ref())
            .collect::<Vec<&str>>(),
        names.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

/// Controlled metadata refusal shared with the same consuming observation path.
struct DeniedMetadata;
impl MediaObserver for DeniedMetadata {
    async fn size(&self, _file: &InPlaceMedia) -> MediaSizeObservation {
        MediaSizeObservation::Unavailable(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "synthetic refusal",
        ))
    }
}

/// Guards: recording the source-bound preflight rejection persists one failed
/// FileSetup attempt and the corresponding file error with the actual reason.
/// Kind: boundary
/// Not a type: ownership fixes the source pairing and recording order, while
/// the SQLite adapter must still write these exact fields and the
/// controlled PermissionDenied observation must retain its diagnostic meaning.
/// State: one SQLite database and registry exercise real recording effects.
/// DeniedMetadata supplies the observation, so no media file or decoder is needed.
/// Concurrency: the temporary database, admitted paths and run claim are isolated.
#[tokio::test]
async fn preflight_media_failure_records_setup_attempt() {
    let root = PathsFixture::create();
    let db = Arc::new(JobDB::open(Some(root.path())).await.expect("open db"));
    let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::for_tests(
        crate::config::ServerConfig::default(),
        Some(db.clone()),
        tx,
        Arc::new(crate::clock::SystemClock),
    ));
    let files = PathsFiles::admit(vec![root.input("missing.wav", InputKind::Media)])
        .expect("admit file set");
    let mut job = queued_paths_morphotag_job_fixture(files);
    job.identity.job_id = "job-media-preflight".into();
    let id = JobId::from("job-media-preflight");
    store.submit(job).await.expect("submit job");
    let claim = claim_runner(&store, &id).await;
    let run = ClaimedDispatch::for_run(&claim);
    let observed = ObservedPreflight::observe_with(run, &DeniedMetadata).await;
    let ready = observed.record().await;
    assert!(ready.into_inputs().files().is_empty());
    let attempts = db
        .load_attempts_for_job("job-media-preflight")
        .await
        .expect("load attempts");
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].work_unit_kind,
        crate::scheduling::WorkUnitKind::FileSetup
    );
    assert_eq!(
        attempts[0].outcome,
        crate::scheduling::AttemptOutcome::Failed
    );
    assert_eq!(
        attempts[0].failure_category,
        Some(FailureCategory::Validation)
    );
    let detail = store.get_job_detail(&id).await.expect("job detail");
    let recorded_files = detail
        .file_statuses
        .into_iter()
        .map(|file| (file.status, file.error))
        .collect::<Vec<_>>();
    let reason = format!(
        "Cannot inspect media file {}: synthetic refusal",
        root.path().join("missing.wav").display(),
    );
    assert_eq!(
        recorded_files,
        vec![(crate::api::FileStatusKind::Error, Some(reason))]
    );
    claim.complete().await.expect("complete the observed run");
}
