//! Inference progress for the batched-text commands.
//!
//! # What this exists to answer
//!
//! "How far into this file's inference are we." A file's utterances go to a text
//! backend as a batch, so without this the file carries a stage
//! (`FileStage::Analyzing`) and no numbers for the whole of its inference, which
//! was the largest remaining violation of the crate's time-transparency rule
//! (see `book/src/batchalign/architecture/time-transparency.md`): every audio
//! command publishes counts, and the batched-text path did not.
//!
//! The backend already produces the numbers.
//! `batchalign/worker/_text_v2.py` installs an `_on_progress` callback that
//! emits a `progress_v2` event at most once per second per request, a
//! `request`-scoped event carrying `completed` / `total` utterances
//! ([`UnitProgressV2`]). This module is the accounting that turns that stream
//! into one honest per-file count; only request progress reaches it (see
//! `execution::morphotag::progress::BackendProgressPort::report`).
//!
//! # One projection, per input file
//!
//! ```text
//! Python backend ──progress_v2──▶ worker handle ──▶ BackendProgress
//!                                                        │
//!                                            BatchProgressLedger
//!                                                        │
//!                                                        ▼
//!                                                 SourceProgress
//!                                       (per-file counts on the existing
//!                                        file-progress channel)
//! ```
//!
//! # Why there is no job-level per-language aggregate any more
//!
//! There was one (`BatchInferProgress`), with a REST field, a dashboard panel
//! and a CLI summary line, and it was **retired on 2026-07-30 because it could
//! not be made honest** under per-file processing.
//!
//! A denominator only enters this ledger when a file's payloads are collected
//! and dispatched, and files are dispatched `num_workers` at a time. So a
//! job-wide "1250/1500 utterances (83%)" was really "83% of the handful of
//! files currently in flight", displayed next to `0/740 files` and drifting up
//! toward truth over hours. Under the pooled batching this replaced, the same
//! field was honest: everything was pooled up front, so the denominator really
//! was the job. Making it honest again would mean parsing every file before
//! dispatch, which is exactly the up-front work that was deliberately removed
//! for interfering with per-file processing.
//!
//! A file's own total, by contrast, is exact the moment its payloads exist. If
//! a job-level view is ever wanted, the honest form is a rate or ETA computed
//! from COMPLETED files, not a mid-flight utterance percentage.
//!
//! # Why the key is (source, group, chunk)
//!
//! This is the correction that makes the feature trustworthy, and it is worth
//! reading before changing anything here.
//!
//! `morphosyntax::worker::infer_batch_homogeneous` splits one language group
//! into up to `max_workers_per_key` chunks whenever it holds `2 *
//! MIN_CHUNK_SIZE` items or more, which on any multi-worker host is always.
//! Each chunk is a separate backend request reporting its OWN `completed` /
//! `total`. The 2026-04 ledger keyed on language alone, took the first
//! event's `total` as the denominator, and let each later event overwrite
//! `completed`. With four chunks of 274 items that displayed `453/274`, i.e.
//! 165%, which is exactly the overflow recorded in this module's history and
//! observed live before the feature was removed on 2026-05-03.
//!
//! Keying on the CHUNK rather than the request id is deliberate:
//! `dispatch_execute_v2_with_retry_and_progress` retries a chunk under a
//! fresh request id, so a request-keyed ledger would count a retried chunk's
//! utterances twice and keep the abandoned attempt's stale count forever. The
//! chunk is the stable unit of work; the request is one attempt at it.
//!
//! Completions are therefore SUMMED across chunks and the latest report per
//! chunk wins, which is correct under both chunking and retry. The DENOMINATOR
//! is not summed from chunk reports at all: it is declared up front (see
//! [`ProgressReport::GroupTotal`]), because a file whose first chunk finished
//! would otherwise look complete while its other chunks had not started, and the
//! publisher would skip it as done. That bug shipped briefly on 2026-07-30 and
//! is pinned by `one_finished_chunk_does_not_make_a_file_look_complete`.

use std::collections::BTreeMap;

use crate::api::{DisplayPath, LanguageCode3, UtteranceCount};
use crate::types::worker_v2::UnitProgressV2;

/// Index of one chunk within a single (source, language-group) batch.
///
/// Not a request id: a chunk retried after a worker failure keeps its index
/// and gets a new request id, and the ledger must treat that as the same unit
/// of work. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BatchChunkIndex(pub(crate) u32);

/// One report on the progress channel.
///
/// A group DECLARES its item count before dispatch and then reports chunk
/// completions. The declaration is what makes a denominator trustworthy: a
/// file's total cannot be inferred from the chunks that happen to have reported,
/// because a chunk that finished first would make the file look complete while
/// three others had not started. That is the same scope error as the retired
/// job-level aggregate, one level smaller, and it is why this is an enum rather
/// than a single message type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProgressReport {
    /// This (source, group) will process exactly this many utterances.
    GroupTotal {
        /// Which input file.
        source_id: DisplayPath,
        /// Which homogeneous batch group.
        group: LanguageCode3,
        /// Utterances the group will process in total.
        total: UtteranceCount,
    },
    /// One chunk of a group has progressed.
    Chunk(BackendProgress),
}

/// Progress reported by one backend for one chunk of work.
///
/// Named `BackendProgress` after the same concept in the upstream fork whose
/// mechanisms this crate is adopting: where that fork already names a concept
/// being introduced here, its name is taken, so the two codebases stay
/// comparable at a glance. `source_id` likewise borrows its vocabulary for
/// input provenance, typed with our own `DisplayPath` rather than a bare
/// string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendProgress {
    /// Which input file this work came from.
    pub(crate) source_id: DisplayPath,
    /// The homogeneous batch group, i.e. the language the backend is running.
    pub(crate) group: LanguageCode3,
    /// Which chunk of that group's items this report covers.
    pub(crate) chunk: BatchChunkIndex,
    /// Utterances finished in this chunk so far.
    pub(crate) completed: UtteranceCount,
    /// Utterances in this chunk in total.
    ///
    /// Kept for the coherence check in tests and for tracing; the DENOMINATOR a
    /// UI sees comes from [`ProgressReport::GroupTotal`], never from this.
    pub(crate) total: UtteranceCount,
}

impl BackendProgress {
    /// Lift one request's admitted item counts into a domain message.
    ///
    /// Takes the counts, not the whole event: a load's progress is not request
    /// progress, and the caller has already matched it away, so no other
    /// measure's number can be read as utterances; and `completed <= total`
    /// was admitted at the wire. The event's `stage` is not taken either. The 2026-04
    /// producer smuggled the language code through it by rewriting
    /// `event.stage` in flight, so a field named "stage" carried a language by
    /// a convention no type enforced. Provenance now arrives as typed fields
    /// supplied by the dispatch site, which knows them for certain.
    pub(crate) fn from_units(
        source_id: DisplayPath,
        group: LanguageCode3,
        chunk: BatchChunkIndex,
        units: UnitProgressV2,
    ) -> Self {
        Self {
            source_id,
            group,
            chunk,
            completed: UtteranceCount(u64::from(units.completed())),
            total: UtteranceCount(u64::from(units.total())),
        }
    }
}

/// Identity of one homogeneous group within one input file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    source_id: DisplayPath,
    group: LanguageCode3,
}

/// Identity of one unit of batched work within a job.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ChunkKey {
    source_id: DisplayPath,
    group: LanguageCode3,
    chunk: BatchChunkIndex,
}

/// The latest report for one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkProgress {
    completed: UtteranceCount,
    total: UtteranceCount,
}

/// Running account of a job's batched inference.
///
/// Owned by the reporter task (`execution::morphotag::progress`), never
/// shared: every mutation arrives as a `BackendProgress` on a channel, so
/// there is no lock here by construction.
#[derive(Debug, Default)]
pub(crate) struct BatchProgressLedger {
    /// Declared work per (source, group): the authoritative denominator.
    declared: BTreeMap<GroupKey, UtteranceCount>,
    /// Latest completion report per chunk: the numerator.
    chunks: BTreeMap<ChunkKey, ChunkProgress>,
}

impl BatchProgressLedger {
    /// Create an empty ledger.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record one report.
    ///
    /// Returns whether this changed the ledger, so the caller can skip
    /// republishing identical numbers. A chunk's later report replaces its
    /// earlier one; a retried chunk replaces its own abandoned attempt.
    pub(crate) fn record(&mut self, report: ProgressReport) -> bool {
        match report {
            ProgressReport::GroupTotal {
                source_id,
                group,
                total,
            } => {
                let key = GroupKey { source_id, group };
                self.declared.insert(key, total) != Some(total)
            }
            ProgressReport::Chunk(progress) => {
                let key = ChunkKey {
                    source_id: progress.source_id,
                    group: progress.group,
                    chunk: progress.chunk,
                };
                let next = ChunkProgress {
                    completed: progress.completed,
                    total: progress.total,
                };
                match self.chunks.insert(key, next) {
                    Some(previous) => previous != next,
                    None => true,
                }
            }
        }
    }

    /// Project the ledger onto per-source counts, for the file-progress channel
    /// every other command already publishes on.
    ///
    /// The total comes from the DECLARED group work, so a file shows its full
    /// denominator from the moment its groups are declared, before any chunk has
    /// finished. Completions are summed across chunks.
    pub(crate) fn source_progress(&self) -> Vec<SourceProgress> {
        let mut per_source: BTreeMap<&DisplayPath, SourceProgress> = BTreeMap::new();
        for (key, declared) in &self.declared {
            let entry = per_source
                .entry(&key.source_id)
                .or_insert_with(|| SourceProgress {
                    source_id: key.source_id.clone(),
                    completed: UtteranceCount(0),
                    total: UtteranceCount(0),
                });
            entry.total = UtteranceCount(entry.total.0 + declared.0);
        }
        for (key, chunk) in &self.chunks {
            let entry = per_source
                .entry(&key.source_id)
                .or_insert_with(|| SourceProgress {
                    source_id: key.source_id.clone(),
                    completed: UtteranceCount(0),
                    total: UtteranceCount(0),
                });
            entry.completed = UtteranceCount(entry.completed.0 + chunk.completed.0);
        }
        per_source.into_values().collect()
    }
}

/// Aggregate utterance counts for one input file.
///
/// A named struct rather than a `(DisplayPath, u64, u64)` tuple: two counts of
/// the same primitive type next to each other is exactly the seam where
/// completed and total get swapped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceProgress {
    /// The input file these counts belong to.
    pub(crate) source_id: DisplayPath,
    /// Utterances finished across all of this file's chunks.
    pub(crate) completed: UtteranceCount,
    /// Utterances in total across all of this file's chunks.
    pub(crate) total: UtteranceCount,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lang(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("test language code must be a valid ISO 639-3 code")
    }

    fn declare(ledger: &mut BatchProgressLedger, source: &str, group: &str, total: u64) -> bool {
        ledger.record(ProgressReport::GroupTotal {
            source_id: DisplayPath::from(source),
            group: lang(group),
            total: UtteranceCount(total),
        })
    }

    fn report(
        ledger: &mut BatchProgressLedger,
        source: &str,
        group: &str,
        chunk: u32,
        completed: u32,
        total: u32,
    ) -> bool {
        ledger.record(ProgressReport::Chunk(BackendProgress::from_units(
            DisplayPath::from(source),
            lang(group),
            BatchChunkIndex(chunk),
            UnitProgressV2::new(completed, total).expect("a test count within its total"),
        )))
    }

    /// THE REGRESSION THIS MODULE EXISTS FOR.
    ///
    /// One language group split across four chunks of 274 items each. The
    /// 2026-04 ledger keyed on language alone, so the denominator was the
    /// first chunk's 274 and the numerator summed every chunk's completions:
    /// `453/274`, 165%, observed live. Summing per chunk gives 1,096 as the
    /// denominator, and completed can never exceed it.
    #[test]
    fn chunked_group_totals_sum_instead_of_overwriting() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "a.cha", "eng", 1_096);
        for chunk in 0..4 {
            report(&mut ledger, "a.cha", "eng", chunk, 0, 274);
        }
        report(&mut ledger, "a.cha", "eng", 0, 274, 274);
        report(&mut ledger, "a.cha", "eng", 1, 179, 274);

        let sources = ledger.source_progress();
        let file = sources.first().expect("a.cha must be present");
        assert_eq!(file.total, UtteranceCount(1_096));
        assert_eq!(file.completed, UtteranceCount(453));
        assert!(
            file.completed.0 <= file.total.0,
            "completed must never exceed total: {file:?}"
        );
    }

    /// A retried chunk must replace its own abandoned attempt, not add to it.
    ///
    /// `dispatch_execute_v2_with_retry_and_progress` reissues a failed chunk
    /// under a fresh request id. Keying the ledger on the request id would
    /// double the denominator and strand the dead attempt's count; keying on
    /// the chunk index makes the retry overwrite.
    #[test]
    fn retried_chunk_replaces_its_abandoned_attempt() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "a.cha", "eng", 300);
        report(&mut ledger, "a.cha", "eng", 0, 90, 300);
        // Same chunk, new attempt, counts restart from zero.
        report(&mut ledger, "a.cha", "eng", 0, 10, 300);

        let sources = ledger.source_progress();
        let file = sources.first().expect("a.cha must be present");
        assert_eq!(file.total, UtteranceCount(300));
        assert_eq!(file.completed, UtteranceCount(10));
    }

    /// The wire-protocol `stage` label cannot merge two languages, and cannot
    /// be mistaken for one.
    ///
    /// The group is still part of the ledger's key even though nothing displays
    /// it (the batcher will want it), so a file that code-switches keeps its
    /// per-language chunks distinct rather than overwriting them.
    #[test]
    fn the_shared_stage_label_can_never_become_a_group() {
        assert!(
            LanguageCode3::try_new("stanza_processing").is_err(),
            "the wire stage label must not be constructible as a language group"
        );

        let mut ledger = BatchProgressLedger::new();
        // One file, two languages (the L2 code-switch path), same chunk index.
        declare(&mut ledger, "mixed.cha", "eng", 100);
        declare(&mut ledger, "mixed.cha", "spa", 60);
        report(&mut ledger, "mixed.cha", "eng", 0, 40, 100);
        report(&mut ledger, "mixed.cha", "spa", 0, 10, 60);

        let sources = ledger.source_progress();
        assert_eq!(sources.len(), 1, "one file, one row");
        let file = sources.first().expect("mixed.cha must be present");
        assert_eq!(
            file.total,
            UtteranceCount(160),
            "both language groups of one file count toward that file"
        );
        assert_eq!(file.completed, UtteranceCount(50));
    }

    /// The same events must also answer "how far into THIS file are we",
    /// which is the projection the 2026-04 design never fed.
    #[test]
    fn per_source_projection_sums_that_file_only() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "a.cha", "eng", 200);
        declare(&mut ledger, "b.cha", "spa", 40);
        report(&mut ledger, "a.cha", "eng", 0, 50, 100);
        report(&mut ledger, "a.cha", "eng", 1, 25, 100);
        report(&mut ledger, "b.cha", "spa", 0, 10, 40);

        let sources = ledger.source_progress();
        assert_eq!(sources.len(), 2);
        let a = sources
            .iter()
            .find(|s| &*s.source_id == "a.cha")
            .expect("a.cha must be present");
        assert_eq!(a.completed, UtteranceCount(75));
        assert_eq!(a.total, UtteranceCount(200));
        let b = sources
            .iter()
            .find(|s| &*s.source_id == "b.cha")
            .expect("b.cha must be present");
        assert_eq!(b.completed, UtteranceCount(10));
        assert_eq!(b.total, UtteranceCount(40));
    }

    /// Two files in the same language stay two rows.
    ///
    /// They used to be summed into one job-level language bar. That aggregate
    /// was retired: see the module docs on why its denominator lied.
    #[test]
    fn two_files_in_one_language_stay_two_rows() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "a.cha", "eng", 100);
        declare(&mut ledger, "b.cha", "eng", 100);
        report(&mut ledger, "a.cha", "eng", 0, 10, 100);
        report(&mut ledger, "b.cha", "eng", 0, 40, 100);

        let sources = ledger.source_progress();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].completed, UtteranceCount(10));
        assert_eq!(sources[1].completed, UtteranceCount(40));
    }

    /// THE BUG A LIVE RUN CAUGHT, and the reason the denominator is declared.
    ///
    /// A file's first chunk finishing must not make the whole file look
    /// complete. With the total inferred from the chunks that had reported, one
    /// finished chunk of four read as 375/375, the publisher skipped the file as
    /// done, and no per-file update was ever sent: the common case was silently
    /// suppressed while every unit test passed.
    #[test]
    fn one_finished_chunk_does_not_make_a_file_look_complete() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "long.cha", "eng", 1_500);
        // Chunk 0 finished; chunks 1..3 have not reported at all.
        report(&mut ledger, "long.cha", "eng", 0, 375, 375);

        let sources = ledger.source_progress();
        let file = sources.first().expect("long.cha must be present");
        assert_eq!(
            file.total,
            UtteranceCount(1_500),
            "the denominator is declared work, not reported work"
        );
        assert_eq!(file.completed, UtteranceCount(375));
        assert!(
            file.completed.0 < file.total.0,
            "a file with three chunks unreported is not complete"
        );
    }

    /// A declaration alone gives a file its denominator, before any inference.
    #[test]
    fn a_declared_group_shows_zero_of_its_total() {
        let mut ledger = BatchProgressLedger::new();
        declare(&mut ledger, "a.cha", "eng", 800);

        let sources = ledger.source_progress();
        let file = sources.first().expect("a.cha must be present");
        assert_eq!(file.completed, UtteranceCount(0));
        assert_eq!(file.total, UtteranceCount(800));
    }

    /// Republishing identical numbers is wasted work, so `record` reports
    /// whether it changed anything.
    #[test]
    fn record_reports_whether_the_ledger_changed() {
        let mut ledger = BatchProgressLedger::new();
        assert!(declare(&mut ledger, "a.cha", "eng", 100));
        assert!(!declare(&mut ledger, "a.cha", "eng", 100));
        assert!(report(&mut ledger, "a.cha", "eng", 0, 10, 100));
        assert!(!report(&mut ledger, "a.cha", "eng", 0, 10, 100));
        assert!(report(&mut ledger, "a.cha", "eng", 0, 11, 100));
    }

    #[test]
    fn an_empty_ledger_has_no_rows() {
        assert!(BatchProgressLedger::new().source_progress().is_empty());
    }
}
