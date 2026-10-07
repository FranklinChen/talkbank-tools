//! Recorded-artifact lookup and real streaming/file controls. Test payloads
//! exercise binary transport, not recording validity or acoustic model quality.

use super::*;
use crate::api::{ContentType, DisplayPath, JobStatus};
use crate::store::{FileResultEntry, JobDetail};

fn detail(root: &std::path::Path) -> JobDetail {
    JobDetail {
        command: crate::ReleasedCommand::Align, // command routing is not exercised
        options: crate::recipe_runner::runtime::test_options(crate::ReleasedCommand::Align),
        status: JobStatus::Completed,
        paths_mode: true,
        staging_dir: batchalign_types::paths::ServerPath::new(root),
        results: vec![FileResultEntry {
            filename: DisplayPath::from("nested/recording.wav"),
            content_type: ContentType::Wav,
            error: None,
        }],
        file_statuses: vec![],
    }
}

#[tokio::test]
async fn recorded_binary_stream_and_descriptor_preserve_non_utf8_bytes() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("output/nested")).unwrap();
    let bytes = (0..200_000).map(|i| (i % 256) as u8).collect::<Vec<_>>();
    std::fs::write(root.path().join("output/nested/recording.wav"), &bytes).unwrap();
    let detail = detail(root.path());
    let mut opened = open_artifact(&detail, "nested/recording.wav")
        .await
        .unwrap();
    let descriptor = opened.descriptor().await.unwrap();
    assert_eq!(descriptor.byte_len.get(), bytes.len() as u64);
    assert_eq!(
        descriptor.digest,
        crate::api::ArtifactDigest::from_hash(blake3::hash(&bytes))
    );
    let response = opened.into_response().unwrap();
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
    assert_eq!(response.headers()[header::CONTENT_LENGTH], "200000");
    assert_eq!(
        axum::body::to_bytes(response.into_body(), bytes.len())
            .await
            .unwrap()
            .as_ref(),
        bytes
    );
}

#[tokio::test]
async fn absent_failed_text_empty_and_nonregular_results_cannot_open_a_binary_stream() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("output/nested")).unwrap();
    std::fs::write(root.path().join("output/nested/recording.wav"), []).unwrap();
    let mut detail = detail(root.path());
    assert!(
        open_artifact(&detail, "nested/unrecorded.wav")
            .await
            .is_err()
    );
    assert!(
        open_artifact(&detail, "nested/recording.wav")
            .await
            .is_err()
    );
    std::fs::write(
        root.path().join("output/nested/recording.wav"),
        b"binary bytes",
    )
    .unwrap();
    detail.results[0].error = Some("producer failed".into());
    assert!(
        open_artifact(&detail, "nested/recording.wav")
            .await
            .is_err()
    );
    detail.results[0].error = None;
    detail.results[0].content_type = ContentType::Text;
    assert!(
        open_artifact(&detail, "nested/recording.wav")
            .await
            .is_err()
    );
    detail.results[0].content_type = ContentType::Wav;
    detail.results[0].filename = "nested".into();
    assert!(open_artifact(&detail, "nested").await.is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn recorded_leaf_and_parent_symlinks_cannot_escape_or_substitute_artifacts() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("output/nested")).unwrap();
    std::fs::write(root.path().join("retained.wav"), b"retained bytes").unwrap();
    std::os::unix::fs::symlink(
        root.path().join("retained.wav"),
        root.path().join("output/nested/recording.wav"),
    )
    .unwrap();
    let mut detail = detail(root.path());
    assert!(
        open_artifact(&detail, "nested/recording.wav")
            .await
            .is_err()
    );
    std::os::unix::fs::symlink(root.path(), root.path().join("output/escape")).unwrap();
    detail.results[0].filename = "escape/retained.wav".into();
    assert!(open_artifact(&detail, "escape/retained.wav").await.is_err());
    detail.results[0].filename = "../retained.wav".into();
    assert!(open_artifact(&detail, "../retained.wav").await.is_err());
    assert_eq!(
        std::fs::read(root.path().join("retained.wav")).unwrap(),
        b"retained bytes"
    );
}
