//! Actual HTTP/body and filesystem controls for the delivery boundary.
//! Payload bytes are protocol controls, not valid-media/acoustic gold.

use super::*;
use crate::api::{ContentType, FileProvenance, ResultContent};
use crate::cli::output::{PlannedOutputPath, ResultDestination};
use axum::{
    Router,
    body::Body,
    http::{HeaderValue, header},
    response::Response,
    routing::get,
};
use std::num::NonZeroU64;

struct BodyServer {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for BodyServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn server(bytes: Option<Vec<u8>>, kind: Option<&'static str>) -> BodyServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/bytes", listener.local_addr().unwrap());
    let router = Router::new().route(
        "/bytes",
        get(move || {
            let bytes = bytes.clone();
            async move {
                let body = match bytes {
                    Some(bytes) => Body::from(bytes),
                    None => Body::from_stream(futures::stream::pending::<
                        Result<Vec<u8>, std::io::Error>,
                    >()),
                };
                let mut response = Response::new(body);
                if let Some(kind) = kind {
                    response
                        .headers_mut()
                        .insert(header::CONTENT_TYPE, HeaderValue::from_static(kind));
                }
                response
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    BodyServer { url, task }
}

fn result(bytes: &[u8]) -> FileResult {
    FileResult {
        filename: "recording.wav".into(),
        content: ResultContent::Binary(BinaryResultDescriptor {
            byte_len: NonZeroU64::new(bytes.len() as u64).unwrap(),
            digest: ArtifactDigest::from_hash(blake3::hash(bytes)),
        }),
        content_type: ContentType::Wav,
        error: None,
        provenance: FileProvenance::NotRead,
    }
}

/// Construct the tested output-boundary fixture from real path/root admission.
/// This is not command discovery, export encoding or corpus coverage evidence.
fn destinations(root: &std::path::Path) -> ResultDestinations {
    ResultDestinations {
        artifacts: [(
            "recording.wav".into(),
            ResultDestination {
                path: PlannedOutputPath::already_planned(&root.join("recording.wav")).unwrap(),
                root: OutputRoot::admit(root).unwrap(),
                content_type: ContentType::Wav,
            },
        )]
        .into(),
    }
}

fn admit<'a>(result: &'a FileResult, destinations: &ResultDestinations) -> BinaryWritePlan<'a> {
    let ResultContent::Binary(descriptor) = &result.content else {
        panic!("binary fixture")
    };
    BinaryWritePlan::admit(result, descriptor, destinations).unwrap()
}

#[tokio::test]
async fn actual_non_utf8_http_bytes_publish_only_after_complete_verification() {
    let root = tempfile::tempdir().unwrap();
    let bytes = (0..200_000).map(|i| (i % 256) as u8).collect::<Vec<_>>();
    let result = result(&bytes);
    let destinations = destinations(root.path());
    let server = server(Some(bytes.clone()), Some("audio/wav")).await;
    let response = reqwest::get(&server.url).await.unwrap();
    let verified = admit(&result, &destinations)
        .receive(response)
        .await
        .unwrap();
    assert!(!root.path().join("recording.wav").exists());
    verified.publish().unwrap();
    assert_eq!(
        std::fs::read(root.path().join("recording.wav")).unwrap(),
        bytes
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}

#[tokio::test]
async fn length_digest_and_encoding_refusals_never_publish_or_leave_temporaries() {
    let expected = b"\0\xff\x80\0";
    for (bytes, kind) in [
        (b"\0\xff".to_vec(), Some("audio/wav")),
        (b"\0\xff\x80\0extra".to_vec(), Some("audio/wav")),
        (b"different"[..4].to_vec(), Some("audio/wav")),
        (expected.to_vec(), Some("audio/mpeg")),
        (expected.to_vec(), None),
    ] {
        let root = tempfile::tempdir().unwrap();
        let result = result(expected);
        let destinations = destinations(root.path());
        let server = server(Some(bytes), kind).await;
        let response = reqwest::get(&server.url).await.unwrap();
        assert!(matches!(
            admit(&result, &destinations).receive(response).await,
            Err(CliError::BinaryResult { .. })
        ));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn mp3_delivery_uses_the_admitted_mpeg_media_type() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"\0\xff\x80\0";
    let mut result = result(bytes);
    result.filename = "recording.mp3".into();
    result.content_type = ContentType::Mp3;
    let destinations = ResultDestinations {
        artifacts: [(
            "recording.mp3".into(),
            ResultDestination {
                path: PlannedOutputPath::already_planned(&root.path().join("recording.mp3"))
                    .unwrap(),
                root: OutputRoot::admit(root.path()).unwrap(),
                content_type: ContentType::Mp3,
            },
        )]
        .into(),
    };
    let server = server(Some(bytes.to_vec()), Some("audio/mpeg")).await;
    let verified = admit(&result, &destinations)
        .receive(reqwest::get(&server.url).await.unwrap())
        .await
        .unwrap();
    verified.publish().unwrap();
    assert_eq!(
        std::fs::read(root.path().join("recording.mp3")).unwrap(),
        bytes
    );
}

#[tokio::test]
async fn a_competing_writer_after_verification_is_retained_and_temp_is_removed() {
    let root = tempfile::tempdir().unwrap();
    let result = result(b"\0\xff\x80\0");
    let destinations = destinations(root.path());
    let server = server(Some(b"\0\xff\x80\0".to_vec()), Some("audio/wav")).await;
    let verified = admit(&result, &destinations)
        .receive(reqwest::get(&server.url).await.unwrap())
        .await
        .unwrap();
    std::fs::write(root.path().join("recording.wav"), b"other writer").unwrap();
    assert!(matches!(
        verified.publish(),
        Err(CliError::BinaryResult {
            refusal: BinaryResultRefusal::Existing(_),
            ..
        })
    ));
    assert_eq!(
        std::fs::read(root.path().join("recording.wav")).unwrap(),
        b"other writer"
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}

#[tokio::test]
async fn shared_filesystem_requires_identical_bytes_not_mere_existence() {
    let root = tempfile::tempdir().unwrap();
    let bytes = b"\0\xff\x80\0";
    let result = result(bytes);
    let destinations = destinations(root.path());
    std::fs::write(root.path().join("recording.wav"), bytes).unwrap();
    assert!(
        admit(&result, &destinations)
            .existing_matches()
            .await
            .unwrap()
    );
    std::fs::write(root.path().join("recording.wav"), b"abcd").unwrap();
    assert!(matches!(
        admit(&result, &destinations).existing_matches().await,
        Err(CliError::BinaryResult {
            refusal: BinaryResultRefusal::Existing(_),
            ..
        })
    ));
    assert_eq!(
        std::fs::read(root.path().join("recording.wav")).unwrap(),
        b"abcd"
    );
}

#[tokio::test]
async fn dropping_a_polled_pending_receive_removes_its_private_temp_without_timing() {
    let root = tempfile::tempdir().unwrap();
    let result = result(b"\0\xff\x80\0");
    let destinations = destinations(root.path());
    let server = server(None, Some("audio/wav")).await;
    let response = reqwest::get(&server.url).await.unwrap();
    let mut pending = Box::pin(admit(&result, &destinations).receive(response));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    drop(pending);
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn binary_destinations_refuse_leaf_symlinks_even_inside_the_root() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("retained.wav"), b"retained").unwrap();
    std::os::unix::fs::symlink(
        root.path().join("retained.wav"),
        root.path().join("recording.wav"),
    )
    .unwrap();
    let result = result(b"\0\xff\x80\0");
    let destinations = destinations(root.path());
    let ResultContent::Binary(descriptor) = &result.content else {
        unreachable!()
    };
    assert!(matches!(
        BinaryWritePlan::admit(&result, descriptor, &destinations),
        Err(CliError::BinaryResult { .. })
    ));
    assert_eq!(
        std::fs::read(root.path().join("retained.wav")).unwrap(),
        b"retained"
    );
}
