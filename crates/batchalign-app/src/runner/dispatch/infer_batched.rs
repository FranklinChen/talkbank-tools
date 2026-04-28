//! Batched text NLP dispatch (morphotag, utseg, translate, coref, compare).

use std::collections::HashMap;

use crate::api::{LanguageCode3, ReleasedCommand};
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::recipe_runner::runtime::{primary_output_artifact, write_text_output_artifact};
use crate::runner::DispatchHostContext;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
use tracing::warn;

use crate::store::{RunnerJobSnapshot, unix_now};

use super::super::util::{FileRunTracker, FileStage, set_file_progress};
use super::BatchedInferDispatchPlan;

/// Group files into windows bounded by total utterance count.
///
/// Each window gets at least 1 file. If a single file exceeds the budget,
/// it gets its own window. Returns `Vec<(start_idx, end_idx)>` ranges
/// into the file list.
fn chunk_by_utterance_budget(
    files: &[TextBatchFileInput],
    utterance_counts: &[usize],
    budget: usize,
) -> Vec<(usize, usize)> {
    let mut windows = Vec::new();
    let mut start = 0;
    let mut window_utts = 0;

    for (i, &count) in utterance_counts.iter().enumerate() {
        if window_utts + count > budget && i > start {
            // Current file would exceed budget — close this window
            windows.push((start, i));
            start = i;
            window_utts = 0;
        }
        window_utts += count;
    }
    // Final window
    if start < files.len() {
        windows.push((start, files.len()));
    }
    windows
}

/// Parse CHAT text, apply merge_abbreviations transform, re-serialize.
pub(in crate::runner) fn apply_merge_abbrev(chat_text: &str) -> String {
    let parser = crate::chat_parser();
    let (mut file, _) = batchalign_chat_ops::parse::parse_lenient(&parser, chat_text);
    batchalign_chat_ops::merge_abbrev::merge_abbreviations(&mut file);
    batchalign_chat_ops::serialize::to_chat_string(&file)
}

/// Dispatch files via the server-side infer path.
///
/// Reads all CHAT files, runs processing in Rust (parse ->
/// cache -> infer -> inject -> serialize), and records results per file.
pub(crate) async fn dispatch_batched_infer(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    services: PipelineServices<'_>,
    plan: BatchedInferDispatchPlan,
) {
    let BatchedInferDispatchPlan {
        kernel_plan,
        tokenization_mode,
        multilingual_policy,
        should_merge_abbrev,
        mwt,
        l2_morphotag,
        respect_pos_hints,
    } = plan;
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_list = &job.pending_files;
    let fallback_lang = LanguageCode3::eng();
    let lang: &LanguageCode3 = job.dispatch.lang.as_resolved().unwrap_or(&fallback_lang);
    let command = job.dispatch.command;
    debug_assert_eq!(kernel_plan.file_parallelism_hint, 1);

    let started_at = unix_now();
    let sink = host.sink().clone();

    let stage = FileStage::for_batch_command(command);

    // Mark all files as processing, open their batch attempts, and publish the
    // initial stage label.
    for file in file_list {
        let filename = file.filename.as_ref();
        FileRunTracker::new(sink.as_ref(), job_id, filename)
            .begin_first_attempt(WorkUnitKind::BatchInfer, started_at, stage)
            .await;
    }

    // Read all CHAT file contents (and optional "before" texts for incremental)
    let mut file_texts: Vec<TextBatchFileInput> = Vec::with_capacity(file_list.len());
    let mut before_texts: HashMap<String, String> = HashMap::new();
    let mut read_errors: Vec<(usize, String)> = Vec::new();

    for file in file_list {
        let file_index = file.file_index;
        let filename = file.filename.as_ref();
        let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);

        // Transition to Reading while doing I/O so the frontend shows activity.
        lifecycle.stage(FileStage::Reading).await;

        let read_path: std::path::PathBuf =
            if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
                job.filesystem.source_paths[file_index]
                    .assume_shared_filesystem()
                    .as_path()
                    .to_owned()
            } else {
                job.filesystem
                    .staging_dir
                    .join("input")
                    .join(filename)
                    .as_path()
                    .to_owned()
            };
        let before_path = if !job.filesystem.before_paths.is_empty()
            && file_index < job.filesystem.before_paths.len()
        {
            Some(job.filesystem.before_paths[file_index].assume_shared_filesystem())
        } else {
            None
        };
        match tokio::fs::read_to_string(&read_path).await {
            Ok(content) => {
                file_texts.push(TextBatchFileInput::new(filename.to_string(), content));
                // Read the corresponding "before" file if available
                if let Some(bp) = before_path
                    && let Ok(before_content) = tokio::fs::read_to_string(&bp).await
                {
                    before_texts.insert(filename.to_string(), before_content);
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to read input: {e}");
                lifecycle
                    .fail(&err_msg, FailureCategory::InputMissing, unix_now())
                    .await;
                read_errors.push((file_index, filename.to_string()));
            }
        }
    }

    if file_texts.is_empty() {
        return;
    }

    // Publish the batch total so frontends can display "0/N" while inference runs.
    let total_files = file_texts.len() as i64;
    for file in &file_texts {
        set_file_progress(
            sink.as_ref(),
            job_id,
            file.filename.as_ref(),
            stage,
            Some(0),
            Some(total_files),
        )
        .await;
    }

    // Run the appropriate server-side orchestrator
    let results = match command {
        ReleasedCommand::Morphotag => {
            let mor_params = MorphosyntaxParams {
                lang,
                tokenization_mode,
                multilingual_policy,
                mwt: &mwt,
                l2_morphotag,
                respect_pos_hints,
            };
            // Use incremental processing when before texts are available
            if !before_texts.is_empty() {
                let mut results: TextBatchFileResults = Vec::new();
                for file in &file_texts {
                    let filename = file.filename.as_ref();
                    let after_text = file.chat_text.as_ref();
                    let result = if let Some(before_text) = before_texts.get(filename) {
                        crate::morphosyntax::process_morphosyntax_incremental(
                            before_text,
                            after_text,
                            services,
                            &mor_params,
                        )
                        .await
                    } else {
                        crate::morphosyntax::process_morphosyntax(after_text, services, &mor_params)
                            .await
                    };
                    match result {
                        Ok(text) => {
                            results.push(TextBatchFileResult::ok(file.filename.clone(), text))
                        }
                        Err(e) => results.push(TextBatchFileResult::err(
                            file.filename.clone(),
                            e.to_string(),
                        )),
                    }
                }
                results
            } else {
                // Windowed batch dispatch: process files in windows bounded
                // by utterance count, writing results after each window.
                //
                // --batch-window controls the max utterance budget per window
                // (default 25 = ~2000 utterances from 25 average files).
                // The actual chunking counts utterances per file (by scanning
                // for *SPK: lines) and groups files until the budget is reached.
                // This prevents pathological windows where 25 CallHome files
                // produce 37K utterances while 25 child-language files produce
                // 500 utterances.
                //
                // 0 means all-in-one (no windowing).
                let configured_window = job.dispatch.options.common().batch_window;
                let batch_window_size = match configured_window {
                    0 => file_texts.len(),
                    1..=1000 => configured_window,
                    _ => {
                        tracing::warn!(
                            configured = configured_window,
                            "batch_window clamped to 1000 (was {configured_window})"
                        );
                        1000
                    }
                };

                // Estimate utterance count per file by counting *SPK: lines.
                // This is a lightweight scan (~1ms per file) that avoids full
                // tree-sitter parsing just for chunking decisions.
                let utterance_counts: Vec<usize> = file_texts
                    .iter()
                    .map(|f| {
                        f.chat_text
                            .as_ref()
                            .lines()
                            .filter(|line| line.starts_with('*'))
                            .count()
                    })
                    .collect();

                // Build windows bounded by utterance budget.
                // Each window gets at least 1 file (even if that one file
                // exceeds the budget). Budget = batch_window_size * 80
                // utterances (empirical average per file).
                let utterance_budget = if configured_window == 0 {
                    usize::MAX
                } else {
                    batch_window_size * 80
                };
                let windows =
                    chunk_by_utterance_budget(&file_texts, &utterance_counts, utterance_budget);

                // Create a progress channel for batch-level monitoring.
                // The drain task stays alive across all windows.
                let (progress_tx, mut progress_rx) =
                    tokio::sync::mpsc::channel::<crate::types::worker_v2::ProgressEventV2>(64);

                let progress_job_id = job_id.clone();
                let progress_sink = sink.clone();
                let drain_handle = tokio::spawn(async move {
                    use crate::runner::util::batch_progress::BatchInferProgress;
                    let mut progress = BatchInferProgress::new();
                    let mut last_log = tokio::time::Instant::now();

                    loop {
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(120),
                            progress_rx.recv(),
                        )
                        .await
                        {
                            Ok(Some(event)) => {
                                let lang = &event.stage;
                                if !progress.language_groups.contains_key(lang) {
                                    progress.register_group(lang, event.total as u64);
                                }
                                progress.update_group(lang, event.completed as u64);

                                let now = tokio::time::Instant::now();
                                if now.duration_since(last_log).as_secs() >= 2 {
                                    last_log = now;
                                    tracing::info!(
                                        job_id = %progress_job_id,
                                        summary = %progress.summary(),
                                        "Batch morphosyntax progress"
                                    );
                                    progress_sink
                                        .set_batch_progress(&progress_job_id, &progress)
                                        .await;
                                }
                            }
                            Ok(None) => break,
                            Err(_) => {
                                let incomplete: Vec<_> = progress.incomplete_groups();
                                tracing::warn!(
                                    job_id = %progress_job_id,
                                    summary = %progress.summary(),
                                    stalled_groups = ?incomplete,
                                    "No batch progress heartbeat for 120s — possible stuck worker"
                                );
                                progress_sink
                                    .set_batch_progress(&progress_job_id, &progress)
                                    .await;
                            }
                        }
                    }

                    tracing::info!(
                        job_id = %progress_job_id,
                        summary = %progress.summary(),
                        "Batch morphosyntax complete"
                    );
                    progress_sink
                        .set_batch_progress(&progress_job_id, &progress)
                        .await;
                });

                let group_timeout =
                    std::time::Duration::from_secs(host.config().audio_task_timeout_s.max(1800));

                let total_files = file_texts.len();
                let total_windows = windows.len();
                let mut global_written: usize = 0;

                for (window_idx, (start, end)) in windows.iter().enumerate() {
                    let window = &file_texts[*start..*end];
                    let window_utts: usize = utterance_counts[*start..*end].iter().sum();
                    tracing::info!(
                        job_id = %job_id,
                        window = window_idx + 1,
                        total_windows,
                        window_files = window.len(),
                        window_utterances = window_utts,
                        "Processing morphotag batch window"
                    );

                    // Signal the Parsing stage for each file in this window
                    // so the dashboard shows activity during parse/collect/cache.
                    for file in window {
                        set_file_progress(
                            sink.as_ref(),
                            job_id,
                            file.filename.as_ref(),
                            FileStage::Parsing,
                            Some(window_idx as i64 + 1),
                            Some(total_windows as i64),
                        )
                        .await;
                    }

                    // Window-level timeout covers the ENTIRE pipeline: parse,
                    // collect, cache, inference, injection, serialization.
                    // Without spawn_blocking/rayon, all phases are async and
                    // properly cancellable — dropping the future releases
                    // semaphore permits, progress channels, everything.
                    let pool_dispatcher = crate::morphosyntax::PoolDispatcher::new(services.pool);
                    let window_result = tokio::time::timeout(
                        group_timeout,
                        crate::morphosyntax::run_morphosyntax_batch_impl(
                            window,
                            services,
                            &pool_dispatcher,
                            &mor_params,
                            Some(progress_tx.clone()),
                            group_timeout,
                        ),
                    )
                    .await;

                    let window_results = match window_result {
                        Ok(results) => results,
                        Err(_) => {
                            tracing::error!(
                                job_id = %job_id,
                                window = window_idx + 1,
                                window_size = window.len(),
                                timeout_s = group_timeout.as_secs(),
                                "Batch window timed out — marking files as failed"
                            );
                            let finished_at = unix_now();
                            for file in window {
                                let lifecycle = FileRunTracker::new(
                                    sink.as_ref(),
                                    job_id,
                                    file.filename.as_ref(),
                                );
                                lifecycle
                                    .fail(
                                        &format!(
                                            "Batch window timed out after {}s",
                                            group_timeout.as_secs()
                                        ),
                                        FailureCategory::WorkerTimeout,
                                        finished_at,
                                    )
                                    .await;
                            }
                            continue;
                        }
                    };

                    // Write results for this window immediately.
                    let finished_at = unix_now();
                    for file_result in window_results {
                        let filename = file_result.filename;
                        let result = file_result.result;
                        let lifecycle =
                            FileRunTracker::new(sink.as_ref(), job_id, filename.as_ref());
                        let file_index = file_list
                            .iter()
                            .find(|file| file.filename == filename)
                            .map(|file| file.file_index)
                            .unwrap_or(0);

                        match result {
                            Ok(output_chat) => {
                                global_written += 1;
                                set_file_progress(
                                    sink.as_ref(),
                                    job_id,
                                    filename.as_ref(),
                                    FileStage::Writing,
                                    Some(global_written as i64),
                                    Some(total_files as i64),
                                )
                                .await;

                                let output_text = if should_merge_abbrev {
                                    apply_merge_abbrev(output_chat.as_ref())
                                } else {
                                    output_chat.into_string()
                                };

                                let primary_output = primary_output_artifact(command, &filename);

                                if let Err(e) = write_text_output_artifact(
                                    &job.filesystem,
                                    file_index,
                                    &primary_output.display_path,
                                    &output_text,
                                )
                                .await
                                {
                                    warn!(
                                        job_id = %job_id,
                                        filename = %filename,
                                        error = %e,
                                        "Failed to write infer output"
                                    );
                                }

                                lifecycle
                                    .complete_with_result(
                                        primary_output.display_path.clone(),
                                        primary_output.content_type,
                                        finished_at,
                                    )
                                    .await;
                            }
                            Err(err) => {
                                let err_msg = err.into_message();
                                lifecycle
                                    .fail(&err_msg, FailureCategory::ProviderTerminal, finished_at)
                                    .await;
                            }
                        }
                    }

                    tracing::info!(
                        job_id = %job_id,
                        window = window_idx + 1,
                        total_windows,
                        files_written = global_written,
                        "Batch window complete"
                    );
                }

                // Drop the sender so the drain task sees channel close.
                drop(progress_tx);
                let _ = drain_handle.await;

                // All files already written per-window — skip the shared
                // result-writing loop below.
                return;
            }
        }
        ReleasedCommand::Utseg => {
            crate::utseg::process_utseg_batch(
                &file_texts,
                lang,
                services.pool,
                services.cache,
                services.engine_version,
            )
            .await
        }
        ReleasedCommand::Translate => {
            crate::translate::process_translate_batch(
                &file_texts,
                lang,
                services.pool,
                services.cache,
                services.engine_version,
            )
            .await
        }
        ReleasedCommand::Coref => {
            crate::coref::process_coref_batch(&file_texts, lang, services.pool).await
        }
        _ => {
            warn!(command = %command, "Unsupported batched infer command");
            return;
        }
    };

    let finished_at = unix_now();

    // Record results per file, reporting Writing progress as each file completes.
    let total_results = results.len() as i64;
    for (result_idx, file_result) in results.into_iter().enumerate() {
        let filename = file_result.filename;
        let result = file_result.result;
        let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename.as_ref());
        // Find the file_index for this filename
        let file_index = file_list
            .iter()
            .find(|file| file.filename == filename)
            .map(|file| file.file_index)
            .unwrap_or(0);

        match result {
            Ok(output_chat) => {
                // Signal the Writing stage with a per-file counter.
                set_file_progress(
                    sink.as_ref(),
                    job_id,
                    filename.as_ref(),
                    FileStage::Writing,
                    Some(result_idx as i64 + 1),
                    Some(total_results),
                )
                .await;

                // Optionally merge abbreviations before writing
                let output_text = if should_merge_abbrev {
                    apply_merge_abbrev(output_chat.as_ref())
                } else {
                    output_chat.into_string()
                };

                // Write output
                let primary_output = primary_output_artifact(command, &filename);

                if let Err(e) = write_text_output_artifact(
                    &job.filesystem,
                    file_index,
                    &primary_output.display_path,
                    &output_text,
                )
                .await
                {
                    warn!(
                        job_id = %job_id,
                        correlation_id = %correlation_id,
                        filename = %filename,
                        error = %e,
                        "Failed to write infer output"
                    );
                }

                lifecycle
                    .complete_with_result(
                        primary_output.display_path.clone(),
                        primary_output.content_type,
                        finished_at,
                    )
                    .await;
            }
            Err(err) => {
                let err_msg = err.into_message();
                lifecycle
                    .fail(&err_msg, FailureCategory::ProviderTerminal, finished_at)
                    .await;
            }
        }
    }
    let _ = correlation_id; // mark used for non-compare paths
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_files(n: usize) -> Vec<TextBatchFileInput> {
        (0..n)
            .map(|i| TextBatchFileInput::new(format!("file{i}.cha"), String::new()))
            .collect()
    }

    #[test]
    fn chunk_uniform_files_under_budget() {
        let files = make_files(5);
        let counts = vec![100, 100, 100, 100, 100];
        let windows = chunk_by_utterance_budget(&files, &counts, 300);
        assert_eq!(windows, vec![(0, 3), (3, 5)]);
    }

    #[test]
    fn chunk_single_large_file_gets_own_window() {
        let files = make_files(3);
        let counts = vec![50, 5000, 50];
        let windows = chunk_by_utterance_budget(&files, &counts, 200);
        assert_eq!(windows, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn chunk_all_in_one_when_budget_is_max() {
        let files = make_files(10);
        let counts = vec![100; 10];
        let windows = chunk_by_utterance_budget(&files, &counts, usize::MAX);
        assert_eq!(windows, vec![(0, 10)]);
    }

    #[test]
    fn chunk_each_file_separate_when_budget_is_one() {
        let files = make_files(3);
        let counts = vec![10, 20, 30];
        let windows = chunk_by_utterance_budget(&files, &counts, 1);
        assert_eq!(windows, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn chunk_empty_files() {
        let files: Vec<TextBatchFileInput> = vec![];
        let counts: Vec<usize> = vec![];
        let windows = chunk_by_utterance_budget(&files, &counts, 100);
        assert!(windows.is_empty());
    }

    #[test]
    fn chunk_variable_sizes() {
        let files = make_files(6);
        let counts = vec![10, 20, 30, 500, 10, 10];
        // budget=100: [10+20+30=60], [500], [10+10=20]
        let windows = chunk_by_utterance_budget(&files, &counts, 100);
        assert_eq!(windows, vec![(0, 3), (3, 4), (4, 6)]);
    }
}
