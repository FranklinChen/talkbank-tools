//! Fresh requests must refuse unadmitted media roots before staging mutations.

use super::*;
use crate::options::{AbsoluteMediaRoot, AlignOptions};

#[tokio::test]
async fn media_root_submission_refuses_relative_before_creating_job_directories() {
    let temp = tempfile::tempdir().unwrap();
    let mut submission = morphotag_submission(false);
    submission.command = ReleasedCommand::Align;
    submission.lang = LanguageSpec::Resolved(crate::api::LanguageCode3::eng());
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
    };
    assert!(submission.validate().is_err());
    assert!(
        materialize_submission_job(&submission, &context)
            .await
            .is_err()
    );
    assert!(
        !context.jobs_dir.exists(),
        "refusal must precede any filesystem mutation"
    );

    submission.options = CommandOptions::Align(AlignOptions {
        media_dir: Some(
            AbsoluteMediaRoot::admit(temp.path().join("media"))
                .unwrap()
                .into(),
        ),
        ..Default::default()
    });
    submission.validate().expect("absolute-root control");
    let job = materialize_submission_job(&submission, &context)
        .await
        .expect("staged control");
    assert!(job.filesystem.staging_dir.as_path().exists());
}
