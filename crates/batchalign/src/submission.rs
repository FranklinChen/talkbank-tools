//! Build internal jobs from validated API submissions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use batchalign_types::paths::ServerPath;

use crate::api::{CorrelationId, DisplayPath, JobId};
use crate::error::ServerError;
use crate::store::{
    FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
    JobRuntimeControl, JobScheduleState, JobSourceContext, Submitter,
};

/// Trusted host-side context needed to materialize one job from a submission.
pub(crate) struct SubmissionContext {
    /// Stable identifier for the new job.
    pub job_id: JobId,
    /// Correlation context attached to logs and responses.
    pub correlation_id: CorrelationId,
    /// Host-owned runtime jobs directory.
    pub jobs_dir: PathBuf,
    /// Who is submitting, used for conflict detection and display.
    pub submitter: Submitter,
    /// When the job was submitted, read from the store's clock: an
    /// [`EventTime`](crate::store::EventTime), so a caller cannot choose it.
    pub submitted_at: crate::store::EventTime,
    /// The folders this server lets paths-mode jobs use, from its config.
    pub allowed_roots: batchalign_types::paths::AllowedRoots,
}

async fn ensure_dir(path: &Path, context: &str) -> Result<(), ServerError> {
    tokio::fs::create_dir_all(path).await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("{context} {}: {error}", path.display()),
        ))
    })
}

/// Write one staged input, refusing to replace anything: the staging folder
/// is fresh, so a file already there is another submitted name that this
/// disk treats as the same file (`A.cha` and `a.cha` where case is
/// ignored), and writing over it would give one file the other's text.
async fn write_staged_file(path: &Path, content: &str) -> Result<(), ServerError> {
    use tokio::io::AsyncWriteExt;
    let open = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await;
    let mut file = match open {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(ServerError::Validation(format!(
                "two submitted files are one file on this server's disk (names that differ \
                 only in letter case or Unicode form): {}",
                path.display()
            )));
        }
        Err(error) => {
            return Err(ServerError::Io(std::io::Error::new(
                error.kind(),
                format!("staging input file {}: {error}", path.display()),
            )));
        }
    };
    let written = async {
        file.write_all(content.as_bytes()).await?;
        file.flush().await
    };
    written.await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("staging input file {}: {error}", path.display()),
        ))
    })
}

/// Materialize one internal job from a validated API submission. Taking the
/// validation proof means every refusal of the submission precedes staging:
/// what can still fail here is the filesystem itself. Content-mode files are
/// written only at their admitted [`StagedName`](crate::types::request::StagedName)s,
/// which cannot leave the staging directory or collide.
pub(crate) async fn materialize_submission_job(
    validated: &crate::types::request::ValidatedSubmission<'_>,
    context: &SubmissionContext,
) -> Result<Job, ServerError> {
    let submission = validated.submission();

    let staging_dir = ServerPath::from(context.jobs_dir.join(context.job_id.to_string()));
    use crate::store::{ContentInput, InputKind, JobInputs, PathsInput};
    use crate::types::request::Staging;
    let inputs = match validated.staging() {
        Staging::Paths(sources) => {
            // Every path is admitted, and the files checked as a set, before
            // anything is created on disk: a refused job leaves no job
            // directory behind. Admission resolves each path on disk, so it
            // runs on the blocking pool: a job of tens of thousands of files
            // on a network volume must not hold a runtime thread.
            let owned: Vec<_> = sources
                .iter()
                .map(|source| {
                    (
                        source.display.clone(),
                        source.source.clone(),
                        source.output.clone(),
                        source.before.cloned(),
                    )
                })
                .collect();
            let source_dir =
                (!submission.source_dir.is_empty()).then(|| submission.source_dir.clone());
            let roots = context.allowed_roots.clone();
            let job_id = context.job_id.clone();
            let (files, source_dir) = crate::blocking::spawn_in_span(move || {
                let files = owned
                    .into_iter()
                    .map(|(display, source, output, before)| {
                        Ok(PathsInput {
                            // Through the one predicate, shared with validation.
                            kind: InputKind::of_source_name(source.as_str()),
                            display,
                            source: roots.admit(&source)?,
                            output: roots.admit(&output)?,
                            before: before.map(|before| roots.admit(&before)).transpose()?,
                        })
                    })
                    .collect::<Result<Vec<_>, ServerError>>()?;
                let files = crate::store::PathsFiles::admit(files)?;
                // The submitted folder is only where the media search also
                // looks (each transcript's own folder is searched
                // regardless). One the folders do not cover, such as the
                // common parent of a file list spanning two allowed folders,
                // is not searched; it never refuses a job whose every file
                // is allowed.
                let source_dir = source_dir.and_then(|folder| match roots.admit_folder(&folder) {
                    Ok(folder) => Some(folder),
                    Err(refusal) => {
                        tracing::info!(%job_id, %refusal,
                            "submitted folder is outside the paths-mode folders; media search skips it");
                        None
                    }
                });
                Ok::<_, ServerError>((files, source_dir))
            })
            .await
            .map_err(|error| ServerError::Io(std::io::Error::other(format!("path admission task: {error}"))))??;
            ensure_dir(staging_dir.as_ref(), "creating paths-mode staging dir").await?;
            ensure_dir(
                staging_dir.join("output").as_ref(),
                "creating paths-mode staged output dir",
            )
            .await?;
            JobInputs::Paths { files, source_dir }
        }
        Staging::Content(staged) => {
            let input_dir = staging_dir.join("input");
            ensure_dir(input_dir.as_ref(), "creating content-mode input dir").await?;
            ensure_dir(
                staging_dir.join("output").as_ref(),
                "creating content-mode output dir",
            )
            .await?;
            let mut files = Vec::with_capacity(staged.len() + submission.media_files.len());
            for file in staged {
                let dest = input_dir.as_path().join(file.name.relative_path());
                if let Some(parent) = dest.parent() {
                    ensure_dir(parent, "creating staged input parent dir").await?;
                }
                write_staged_file(&dest, file.content).await?;
                files.push(ContentInput {
                    display: file.display.clone(),
                    kind: InputKind::Chat,
                });
            }
            files.extend(
                submission
                    .media_files
                    .iter()
                    .map(|media_name| ContentInput {
                        display: DisplayPath::from(media_name.as_str()),
                        kind: InputKind::Media,
                    }),
            );
            JobInputs::Content(files)
        }
    };

    let mut file_statuses = HashMap::new();
    for filename in inputs.displays() {
        file_statuses.insert(filename.to_string(), FileStatus::new(filename.clone()));
    }

    Ok(Job {
        identity: JobIdentity {
            job_id: context.job_id.clone(),
            correlation_id: context.correlation_id.clone(),
        },
        dispatch: JobDispatchConfig {
            command: submission.command,
            lang: submission.lang.clone(),
            num_speakers: submission.num_speakers,
            options: submission.options.clone(),
            runtime_state: std::collections::BTreeMap::new(),
            debug_traces: submission.debug_traces,
        },
        source: JobSourceContext {
            submitter: Some(context.submitter.clone()),
            source_dir: submission.source_dir.clone(),
        },
        filesystem: JobFilesystemConfig {
            staging_dir,
            media_mapping: submission.media_mapping.clone(),
            media_subdir: submission.media_subdir.clone(),
        },
        execution: JobExecutionState {
            // A new job is queued on the inputs just admitted.
            lifecycle: crate::store::JobLifecycle::Active {
                status: crate::api::ActiveStatus::Queued,
                inputs,
            },
            file_statuses,
            results: Vec::new(),
            error: None,
        },
        schedule: JobScheduleState {
            submitted_at: context.submitted_at.instant(),
            completed_at: None,
            next_eligible_at: None,
            num_workers: None,
            lease: None,
            last_cancel: None,
        },
        runtime: JobRuntimeControl::unclaimed(),
        execution_plan: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{SubmissionContext, materialize_submission_job};
    use crate::api::{
        DisplayPath, FilePayload, JobSubmission, LanguageSpec, NumSpeakers, ReleasedCommand,
    };
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};

    #[path = "media_root_tests.rs"]
    mod media_root_tests;

    fn morphotag_submission(paths_mode: bool) -> JobSubmission {
        // Morphotag has no job-level `--lang`; per-file resolution from
        // each file's `@Languages:` header is mandatory. Job-level lang
        // sentinels are rejected by `JobSubmission::validate` (see the
        // 2026-05-03 morphotag incident note there).
        JobSubmission {
            command: ReleasedCommand::Morphotag,
            lang: LanguageSpec::PerFile,
            num_speakers: NumSpeakers(1),
            files: Vec::new(),
            media_files: Vec::new(),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: "/tmp/input".into(),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
            paths_mode,
            source_paths: Vec::new(),
            output_paths: Vec::new(),
            display_names: Vec::new(),
            debug_traces: false,
            before_paths: Vec::new(),
        }
    }

    #[tokio::test]
    async fn materialize_paths_submission_uses_display_names() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        // The allowed root is the temporary folder: every submitted path
        // must lie inside it to be admitted.
        let root = tempdir.path().to_string_lossy().into_owned();
        let source = format!("{root}/source/sample.cha");
        let output = format!("{root}/output/sample.cha");
        let mut submission = morphotag_submission(true);
        submission.source_dir = format!("{root}/source").as_str().into();
        submission.source_paths = vec![source.as_str().into()];
        submission.output_paths = vec![output.as_str().into()];
        submission.display_names = vec!["nested/sample.cha".into()];

        let context = SubmissionContext {
            job_id: "job-paths".into(),
            correlation_id: "corr-paths".into(),
            jobs_dir: tempdir.path().join("jobs"),
            submitter: crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            ),
            submitted_at: crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0)),
            allowed_roots: batchalign_types::paths::AllowedRoots::admit_roots(&[tempdir
                .path()
                .to_path_buf()])
            .expect("the temporary folder"),
        };

        let job = materialize_submission_job(&submission.validate().expect("valid"), &context)
            .await
            .expect("materialize paths submission");

        let Some(crate::store::JobInputs::Paths { files, source_dir }) = job.admitted_inputs()
        else {
            panic!("a paths submission materializes paths inputs");
        };
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].display, DisplayPath::from("nested/sample.cha"));
        assert_eq!(files[0].source.client().as_str(), source);
        assert_eq!(files[0].output.client().as_str(), output);
        assert!(source_dir.is_some());
        assert!(job.filesystem.staging_dir.as_path().ends_with("job-paths"));
        assert!(job.filesystem.staging_dir.as_path().exists());

        // SURVIVES: boundary. Files are checked as a set on what their paths
        // name, before anything is created: these pairs are one file.
        let context_for = |job_id: String| SubmissionContext {
            job_id: job_id.into(),
            correlation_id: "corr-overlap".into(),
            jobs_dir: tempdir.path().join("jobs"),
            submitter: crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            ),
            submitted_at: crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0)),
            allowed_roots: batchalign_types::paths::AllowedRoots::admit_roots(&[tempdir
                .path()
                .to_path_buf()])
            .expect("the temporary folder"),
        };
        std::fs::create_dir_all(format!("{root}/output")).expect("output folder");
        #[cfg(unix)]
        std::os::unix::fs::symlink(format!("{root}/output"), format!("{root}/alias"))
            .expect("alias");
        let two = |outputs: [&str; 2], sources: [&str; 2]| {
            let mut pair = morphotag_submission(true);
            pair.source_dir = Default::default();
            pair.source_paths = sources
                .iter()
                .map(|name| format!("{root}/{name}").as_str().into())
                .collect();
            pair.output_paths = outputs
                .iter()
                .map(|name| format!("{root}/{name}").as_str().into())
                .collect();
            pair
        };
        // Two outputs differing only by case are one file exactly where the
        // volume ignores case (the macOS default), and two elsewhere.
        let same_user = batchalign_types::paths::AllowedRoots::same_user();
        let case_pair = |name: &str| {
            same_user
                .admit(&format!("{root}/output/{name}").as_str().into())
                .expect("an absolute path")
                .identity()
                .clone()
        };
        let case_is_one_file = case_pair("a.cha") == case_pair("A.cha");
        let case_outputs = two(["output/a.cha", "output/A.cha"], ["in/a.cha", "in/b.cha"]);
        let outcome = materialize_submission_job(
            &case_outputs.validate().expect("well formed"),
            &context_for("job-case".to_owned()),
        )
        .await;
        assert_eq!(
            matches!(outcome, Err(crate::error::ServerError::InputsOverlap(_))),
            case_is_one_file,
            "{:?}",
            outcome.as_ref().err()
        );
        let mut refused = vec![
            two(["in/b.cha", "output/y.cha"], ["in/a.cha", "in/b.cha"]),
            two(["output/x.cha", "in/a.cha"], ["in/a.cha", "in/b.cha"]),
        ];
        #[cfg(unix)]
        refused.push(two(
            ["output/a.cha", "alias/a.cha"],
            ["in/a.cha", "in/b.cha"],
        ));
        // The second file writes, in place, the prior the first file reads:
        // caught whichever file is listed first.
        for order in [[0, 1], [1, 0]] {
            let names = [("in/a.cha", "output/a.cha"), ("in/p.cha", "in/p.cha")];
            let mut pair = two(
                [names[order[0]].1, names[order[1]].1],
                [names[order[0]].0, names[order[1]].0],
            );
            pair.before_paths = order
                .iter()
                .map(|&index| (index == 0).then(|| format!("{root}/in/p.cha").as_str().into()))
                .collect();
            refused.push(pair);
        }
        for (index, pair) in refused.iter().enumerate() {
            let context = context_for(format!("job-overlap-{index}"));
            let outcome =
                materialize_submission_job(&pair.validate().expect("well formed"), &context).await;
            assert!(
                matches!(outcome, Err(crate::error::ServerError::InputsOverlap(_))),
                "pair {index}: {:?}",
                outcome.as_ref().err()
            );
            assert!(
                !tempdir
                    .path()
                    .join(format!("jobs/job-overlap-{index}"))
                    .exists()
            );
        }
        // A file may be written over its own source (in place).
        let in_place = two(["in/a.cha", "in/b.cha"], ["in/a.cha", "in/b.cha"]);
        let context = context_for("job-in-place".to_owned());
        materialize_submission_job(&in_place.validate().expect("well formed"), &context)
            .await
            .expect("in-place outputs");
    }

    /// SURVIVES: boundary. Content-mode file names arrive over the wire and
    /// become paths under the staging directory; `validate` is the one place
    /// they are admitted, so its refusals are asserted here and nowhere
    /// downstream can see an unadmitted name.
    #[test]
    fn a_content_file_name_is_admitted_only_as_a_plain_relative_path() {
        let with_names = |names: &[&str]| {
            let mut submission = morphotag_submission(false);
            submission.files = names
                .iter()
                .map(|name| FilePayload {
                    filename: DisplayPath::from(*name),
                    content: "@UTF8\n@Begin\n@End\n".into(),
                })
                .collect();
            submission
        };
        for refused in [
            &["/etc/target.cha"][..],
            &["../target.cha"],
            &["corpus/../../target.cha"],
            &["./a.cha"],
            &["a.cha", "a.cha"],
            &["a.cha", "a.cha/b.cha"],
        ] {
            assert!(with_names(refused).validate().is_err(), "{refused:?}");
        }
        let nested = with_names(&["PWA/TYO_a1.cha", "PWA/TYO_a2.cha"]);
        assert!(nested.validate().is_ok());
        // A media name is tracked by its name too: no repeat of a media name
        // or of a file's name.
        for media in [&["x.wav", "x.wav"][..], &["PWA/TYO_a1.cha"]] {
            let mut repeated = with_names(&["PWA/TYO_a1.cha"]);
            repeated.media_files = media.iter().map(|name| (*name).to_owned()).collect();
            let refusal = repeated.validate().err().map(|error| error.to_string());
            assert!(
                refusal
                    .as_deref()
                    .is_some_and(|text| text.contains("both named")),
                "{media:?}: {refusal:?}"
            );
        }
    }

    #[tokio::test]
    async fn materialize_content_submission_stages_chat_inputs() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mut submission = morphotag_submission(false);
        submission.files = vec![FilePayload {
            filename: DisplayPath::from("nested/sample.cha"),
            content: "@UTF8\n@Begin\n*CHI:\tI go .\n@End\n".into(),
        }];

        let context = SubmissionContext {
            job_id: "job-content".into(),
            correlation_id: "corr-content".into(),
            jobs_dir: tempdir.path().join("jobs"),
            submitter: crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            ),
            submitted_at: crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0)),
            allowed_roots: batchalign_types::paths::AllowedRoots::none(),
        };

        let job = materialize_submission_job(&submission.validate().expect("valid"), &context)
            .await
            .expect("materialize content submission");

        let staged_input = job.filesystem.staging_dir.join("input/nested/sample.cha");
        assert!(matches!(
            job.admitted_inputs(),
            Some(crate::store::JobInputs::Content(_))
        ));
        assert!(job.filesystem.staging_dir.join("output").as_path().exists());
        assert_eq!(
            tokio::fs::read_to_string(staged_input)
                .await
                .expect("read staged file"),
            "@UTF8\n@Begin\n*CHI:\tI go .\n@End\n"
        );
    }
}
