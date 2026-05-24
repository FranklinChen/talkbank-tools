//! [`AnalysisRunner`] -- file loading, filtering, and command dispatch.
//!
//! The runner is the orchestrator that loads CHAT files, applies
//! [`FilterConfig`] criteria, and feeds matching utterances to the command's
//! [`process_utterance()`](super::AnalysisCommand::process_utterance) method.
//! It replaces CUTT's main loop from the original CLAN C framework.
//!
//! Two execution modes are supported:
//! - [`run()`](AnalysisRunner::run) -- aggregated mode: all files share one
//!   state, producing a single combined output
//! - [`run_per_file()`](AnalysisRunner::run_per_file) -- per-file mode: each
//!   file gets its own fresh state and produces independent output
//!
//! Parse/validation failures for individual files are logged as warnings and
//! skipped (non-fatal), following CLAN's behavior of continuing through file
//! errors.

use std::collections::HashMap;
use std::path::PathBuf;

use talkbank_model::validation::ValidationState;
use talkbank_model::{ChatFile, IDHeader, ParseValidateOptions, SpeakerCode};
use talkbank_transform::{PipelineError, parse_file_and_validate};
use tracing::{debug, warn};

use super::command::{AnalysisCommand, FileContext};
use super::filter::{FilterConfig, update_active_gems};
use super::id_filter::IdFilter;

/// Error type for analysis runner operations.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// Failed to parse or validate a CHAT file.
    #[error("Failed to process {path}: {source}")]
    Pipeline {
        /// Path to the file that failed.
        path: PathBuf,
        /// The underlying pipeline error.
        source: PipelineError,
    },
    /// No input files were provided.
    #[error("No input files provided")]
    NoFiles,
}

/// Orchestrates file loading, filtering, and command execution.
///
/// The runner handles:
/// 1. Loading and parsing CHAT files via talkbank-transform
/// 2. Tracking @BG/@EG gem boundaries
/// 3. Applying filter criteria (speakers, gems, etc.)
/// 4. Dispatching matching utterances to the command
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use talkbank_clan::framework::{AnalysisRunner, FilterConfig};
/// use talkbank_clan::commands::freq::FreqCommand;
///
/// let runner = AnalysisRunner::with_filter(FilterConfig::default());
/// let command = FreqCommand::default();
/// let result = runner.run(&command, &[PathBuf::from("file.cha")]);
/// ```
pub struct AnalysisRunner {
    /// Filter criteria applied to each utterance
    filter: FilterConfig,
}

impl AnalysisRunner {
    /// Create a runner with default (no-op) filtering.
    pub fn new() -> Self {
        Self {
            filter: FilterConfig::default(),
        }
    }

    /// Create a runner with the given filter configuration.
    pub fn with_filter(filter: FilterConfig) -> Self {
        Self { filter }
    }

    /// Run an analysis command across the given files, aggregating results.
    ///
    /// # Lifecycle
    ///
    /// For each file:
    ///   1. Parse and validate the file
    ///   2. For each utterance, update gem tracking from preceding headers
    ///   3. Apply filter criteria (speaker, gem, word, utterance range)
    ///   4. Call `command.process_utterance()` for matching utterances
    ///   5. Call `command.end_file()` after all utterances
    ///
    /// After all files: call `command.finalize()` to produce output.
    ///
    /// # Errors
    ///
    /// Returns `RunnerError::NoFiles` if no files are provided.
    /// Parse/validation failures for individual files are logged as warnings
    /// and skipped (non-fatal), following CLAN's behavior of continuing
    /// through file errors.
    pub fn run<C: AnalysisCommand>(
        &self,
        command: &C,
        files: &[PathBuf],
    ) -> Result<C::Output, RunnerError> {
        if files.is_empty() {
            return Err(RunnerError::NoFiles);
        }

        let mut state = C::State::default();
        self.process_files(command, files, &mut state);
        Ok(command.finalize(state))
    }

    /// Run an analysis command per file, returning separate results for each.
    ///
    /// Each file gets its own fresh `State`, processed independently, and
    /// finalized into its own `Output`. This corresponds to CLAN's per-file
    /// output mode.
    ///
    /// # Errors
    ///
    /// Returns `RunnerError::NoFiles` if no files are provided.
    pub fn run_per_file<C: AnalysisCommand>(
        &self,
        command: &C,
        files: &[PathBuf],
    ) -> Result<Vec<(PathBuf, C::Output)>, RunnerError> {
        if files.is_empty() {
            return Err(RunnerError::NoFiles);
        }

        let mut results = Vec::new();
        for path in files {
            let mut state = C::State::default();
            self.process_files(command, std::slice::from_ref(path), &mut state);
            results.push((path.clone(), command.finalize(state)));
        }
        Ok(results)
    }

    /// Process files into accumulator state, applying filters and dispatching
    /// matching utterances to the command.
    fn process_files<C: AnalysisCommand>(
        &self,
        command: &C,
        files: &[PathBuf],
        state: &mut C::State,
    ) {
        let options = ParseValidateOptions::default().with_validation();

        for path in files {
            debug!(path = %path.display(), "Processing file");

            let chat_file = match parse_file_and_validate(path, options.clone()) {
                Ok(f) => f,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Skipping file due to parse error");
                    continue;
                }
            };

            // Single `@ID:` header pass builds the speaker-lookup map
            // when either `--id-filter` or `--role` is active, and
            // applies the `--id-filter` prefilter when present.
            // When neither flag is set, the map stays empty and the
            // per-utterance role / id checks short-circuit on
            // `role_include.is_empty()` / `filter.is_none()`.
            let want_id_map =
                self.filter.id_filter.is_some() || !self.filter.roles.include.is_empty();
            let id_by_speaker: HashMap<SpeakerCode, &IDHeader> = if want_id_map {
                let (admits, map) = scan_id_headers(&chat_file, self.filter.id_filter.as_ref());
                if !admits {
                    if let Some(filter) = self.filter.id_filter.as_ref() {
                        debug!(path = %path.display(), pattern = %filter, "Skipping file: no @ID matches --id-filter");
                    }
                    continue;
                }
                map
            } else {
                HashMap::new()
            };

            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let file_ctx = FileContext {
                path,
                chat_file: &chat_file,
                filename,
                line_map: chat_file.line_map.as_ref(),
            };

            // Track @BG/@EG boundaries and utterance index per file
            let mut active_gems: Vec<String> = Vec::new();
            let mut utterance_index: usize = 0;

            for utterance in chat_file.utterances() {
                // Update gem tracking from preceding headers
                update_active_gems(&utterance.preceding_headers, &mut active_gems);

                // 1-based utterance index for range filtering
                utterance_index += 1;

                // Apply filters (speaker, gem, word, range)
                if !self
                    .filter
                    .matches(utterance, &active_gems, utterance_index)
                {
                    continue;
                }

                // --id-filter speaker-level filter (only when the flag
                // was set; otherwise no work).
                if let Some(id_filter) = self.filter.id_filter.as_ref()
                    && !speaker_passes_id_filter(id_filter, &id_by_speaker, &utterance.main.speaker)
                {
                    continue;
                }

                // --role speaker-level filter (only when at least one
                // role was requested; otherwise no work). Files
                // without `@ID:` headers pass through — see
                // `speaker_passes_role_filter` for rationale.
                if !speaker_passes_role_filter(
                    &self.filter.roles.include,
                    &id_by_speaker,
                    &utterance.main.speaker,
                ) {
                    continue;
                }

                command.process_utterance(utterance, &file_ctx, state);
            }

            command.end_file(&file_ctx, state);
        }
    }
}

/// Single pass over the file's `@ID` headers, returning both:
///
/// - whether the file passes the optional `--id-filter` prefilter
///   (at least one `@ID` matches the pattern, or `true` when no
///   filter is supplied), and
/// - a `SpeakerCode → &IDHeader` lookup for the per-utterance
///   speaker filter (`--id-filter` or `--role`). First `@ID:` row
///   per speaker code wins.
///
/// A file with no `@ID` headers fails the `id-filter` prefilter
/// (absence cannot satisfy a positive match), but passes the
/// no-filter case (empty map, all speakers permitted).
fn scan_id_headers<'a, S: ValidationState>(
    chat_file: &'a ChatFile<S>,
    filter: Option<&IdFilter>,
) -> (bool, HashMap<SpeakerCode, &'a IDHeader>) {
    let mut admits = filter.is_none();
    let mut map: HashMap<SpeakerCode, &'a IDHeader> = HashMap::new();
    for id in chat_file.id_headers() {
        if let Some(filter) = filter
            && filter.matches(id)
        {
            admits = true;
        }
        map.entry(id.speaker.clone()).or_insert(id);
    }
    (admits, map)
}

/// Whether the speaker's `@ID` row matches the filter.
///
/// If the file has no `@ID` row for this speaker, the speaker fails the
/// filter (no evidence to admit). This is conservative and matches the
/// file-prefilter rule: absence cannot satisfy a positive match.
fn speaker_passes_id_filter(
    filter: &IdFilter,
    id_by_speaker: &HashMap<SpeakerCode, &IDHeader>,
    speaker: &SpeakerCode,
) -> bool {
    match id_by_speaker.get(speaker) {
        Some(id_header) => filter.matches(id_header),
        None => false,
    }
}

/// Decide whether a speaker's `@ID:` role is in the user's role-
/// include list (case-insensitive). Returns `true` when the filter
/// is empty (pass-through), when the speaker has no `@ID` (we can't
/// determine the role, so don't drop — matches CLAN's behaviour on
/// files without `@ID` headers), or when the role matches one of
/// the included names.
fn speaker_passes_role_filter(
    role_include: &[String],
    id_by_speaker: &HashMap<SpeakerCode, &IDHeader>,
    speaker: &SpeakerCode,
) -> bool {
    if role_include.is_empty() {
        return true;
    }
    let Some(id_header) = id_by_speaker.get(speaker) else {
        // No `@ID:` for this speaker — role unknown. Pass through to
        // avoid silently dropping utterances when role data is just
        // missing.
        return true;
    };
    let role_str = id_header.role.as_str();
    role_include
        .iter()
        .any(|wanted| wanted.eq_ignore_ascii_case(role_str))
}

impl Default for AnalysisRunner {
    /// Construct a runner with pass-through filtering (`FilterConfig::default()`).
    fn default() -> Self {
        Self::new()
    }
}
