//! Fresh requests must refuse unadmitted media roots. Refusal before staging
//! needs no assertion: materializing a job requires the validation proof.

use super::*;
use crate::options::{AbsoluteMediaRoot, AlignOptions};

#[tokio::test]
async fn media_root_submission_refuses_relative_before_creating_job_directories() {
    let temp = tempfile::tempdir().unwrap();
    let mut submission = morphotag_submission(false);
    // Align is per-file like morphotag, so the language stays `PerFile`.
    submission.command = ReleasedCommand::Align;
    submission.files = vec![FilePayload {
        filename: "sample.cha".into(),
        content: "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|test|CHI|||||Child|||\n*CHI:\tI go .\n@End\n".into(),
    }];
    submission.options = serde_json::from_value(serde_json::json!({
        "command": "align", "media_dir": "media"
    }))
    .expect("wire declaration remains readable");
    let context = SubmissionContext {
        job_id: "root-admission".into(),
        correlation_id: "root-admission".into(),
        jobs_dir: temp.path().join("jobs"),
        submitter: crate::store::Submitter::client(
            std::net::Ipv4Addr::LOCALHOST.into(),
            "localhost".into(),
        ),
        submitted_at: crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0)),
        allowed_roots: batchalign_types::paths::AllowedRoots::none(),
    };
    assert!(
        submission.validate().is_err(),
        "a relative media root is refused"
    );

    submission.options = CommandOptions::Align(AlignOptions {
        media_dir: Some(
            AbsoluteMediaRoot::admit(temp.path().join("media"))
                .unwrap()
                .into(),
        ),
        ..Default::default()
    });
    let validated = submission.validate().expect("absolute-root control");
    let job = materialize_submission_job(&validated, &context)
        .await
        .expect("staged control");
    assert!(job.filesystem.staging_dir.as_path().exists());
}
