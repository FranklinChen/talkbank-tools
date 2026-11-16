//! What a worker holds in memory, per load: the residency table.
//!
//! A worker starts with no model. It holds its process baseline (the Python
//! runtime, the worker package and the libraries its loads import) from the
//! moment it starts, and each model from the `ensure_task` that loads it.
//! Each of those is reserved with the host-memory coordinator by its own
//! lease, sized from this table: the baseline by the spawn's startup lease
//! ([`WORKER_BASELINE`]), each model by its load's lease
//! ([`Residency::of`]). This module is the one place those sizes live.
//!
//! # Where the figures come from
//!
//! Each footprint says whether it was measured or estimated ([`Basis`]).
//!
//! - **Measured** with `scripts/measure_model_residency.py` on an Apple
//!   Silicon Mac (macOS, CPU), in a process that has already imported the
//!   worker's libraries (the baseline, measured the same way): what building
//!   the model adds, and the most the process holds over the baseline while
//!   the model runs a request. The two differ by far more than activations:
//!   building a model leaves most of its weights unread, and they become
//!   resident only when the first request reads them (the coreference
//!   pipeline: +519 MiB built, +2568 MiB in use). A lease is sized from the
//!   in-use figure.
//! - **Estimated** where no measurement exists. The ASR, FA and translation
//!   engines reuse the per-engine `resident_memory_mb` the engine tables
//!   already declare (`crate::types::engines`), which were themselves
//!   estimates of a whole worker running that engine, Python runtime
//!   included, so they over-reserve each load by about the baseline. The
//!   Stanza loads are built from the measured English morphosyntax pipeline.
//!
//! # The activation margin, and what a settled lease keeps
//!
//! A load's lease reserves its in-use footprint plus
//! [`ACTIVATION_MARGIN_PERCENT`] of it: each measurement ran one input on one
//! machine, and does not see a larger input or another platform's allocator
//! or device runtime. A quarter is the stated estimate.
//!
//! Once the worker confirms the load, the lease is settled to what the
//! host's own reading of used memory does not yet show
//! ([`Footprint::unread`]: the in-use figure less the built one) plus the
//! margin. Settling to the margin alone would under-reserve the first
//! request by every weight it has yet to read. After that request the
//! reading shows them too, so the settled lease over-reserves by the same
//! amount: the safe direction.

use crate::api::MemoryMb;
use crate::types::engines::{EngineOverrides, SpeakerEngineName, TranslateEngineName};
use crate::types::worker_v2::SpeakerEmbeddingBackendV2;
use crate::worker::InferTask;

/// How far over a load's own footprint its lease reserves, in percent: the
/// estimate for inference activations and for what one measurement on one
/// machine does not see. Stated, not measured; see the module docs.
pub const ACTIVATION_MARGIN_PERCENT: u64 = 25;

/// How a footprint is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Basis {
    /// Measured with `scripts/measure_model_residency.py`.
    Measured,
    /// An estimate: no measurement exists for this load.
    Estimated,
}

/// What one load holds in a worker, and how that is known.
///
/// Built only from the constants in this module, so every size in the table
/// states its basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Footprint(FootprintKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FootprintKind {
    /// Measured with `scripts/measure_model_residency.py`: what building the
    /// model adds over the baseline, and the most it holds over the
    /// baseline while running a request (never less than `built`).
    Measured {
        /// Added by building the model, before any request.
        built: MemoryMb,
        /// The peak while running a request.
        in_use: MemoryMb,
    },
    /// No measurement exists: one figure, taken as both built and in use.
    Estimated(MemoryMb),
}

impl Footprint {
    /// A measured footprint. A const constructor, so a constant whose
    /// in-use figure is below its built one fails to compile.
    const fn measured(built: u64, in_use: u64) -> Self {
        assert!(
            in_use >= built,
            "a model holds at least what building it added"
        );
        Self(FootprintKind::Measured {
            built: MemoryMb(built),
            in_use: MemoryMb(in_use),
        })
    }

    const fn estimated(mb: u64) -> Self {
        Self(FootprintKind::Estimated(MemoryMb(mb)))
    }

    /// The most the load holds while running a request.
    pub const fn in_use(self) -> MemoryMb {
        match self.0 {
            FootprintKind::Measured { in_use, .. } | FootprintKind::Estimated(in_use) => in_use,
        }
    }

    /// The part of [`Self::in_use`] that building the model leaves unread,
    /// so the host's reading of used memory does not show it until the first
    /// request: nothing for an estimate, which has no built figure apart.
    pub const fn unread(self) -> MemoryMb {
        match self.0 {
            FootprintKind::Measured { built, in_use } => MemoryMb(in_use.0 - built.0),
            FootprintKind::Estimated(_) => MemoryMb(0),
        }
    }

    /// How the footprint is known.
    pub const fn basis(self) -> Basis {
        match self.0 {
            FootprintKind::Measured { .. } => Basis::Measured,
            FootprintKind::Estimated(_) => Basis::Estimated,
        }
    }

    /// The activation margin: [`ACTIVATION_MARGIN_PERCENT`] of the in-use
    /// figure, rounded up so a small load still reserves something.
    pub const fn margin(self) -> MemoryMb {
        MemoryMb((self.in_use().0 * ACTIVATION_MARGIN_PERCENT).div_ceil(100))
    }

    /// What a lease for this load reserves before the model is resident:
    /// the in-use figure and its margin.
    pub const fn reserved(self) -> MemoryMb {
        MemoryMb(self.in_use().0 + self.margin().0)
    }

    /// What the lease keeps once the worker holds the model: the unread
    /// part and the margin (see the module docs).
    pub const fn settled(self) -> MemoryMb {
        MemoryMb(self.unread().0 + self.margin().0)
    }
}

/// A worker process before any load: the Python runtime with the worker
/// package and the libraries its loads import (torch, Stanza, transformers).
/// Measured: 275 MiB resident after those imports (built and in use alike:
/// nothing here waits for a request). The same for every profile, since no
/// model loads at startup any more; the startup lease reserves it, margin
/// included, until the worker reports ready.
pub const WORKER_BASELINE: Footprint = Footprint::measured(275, 275);

/// One English Stanza morphosyntax pipeline (tokenize, mwt, pos, lemma,
/// depparse) through the worker's loader, in use. Measured: +736 MiB built,
/// +752 MiB at its peak annotating a short English document.
const STANZA_MORPHOSYNTAX_PIPELINE_MB: u64 = 752;

/// How many Stanza pipelines one worker keeps resident: the worker's bounded
/// cache (`STANZA_PIPELINE_CAPACITY` in `batchalign/worker/_pipeline_cache.py`).
/// A morphosyntax load holds at most this many languages at once, building
/// each on its first use; MUST stay in step with the Python constant.
const STANZA_PIPELINE_CAPACITY: u64 = 2;

/// The morphosyntax load: up to [`STANZA_PIPELINE_CAPACITY`] pipelines, each
/// sized as the measured English one in use. An estimate: other languages' packs
/// differ in size, and the cache is shared with utseg's pipeline, so a worker
/// running both tasks is over-reserved by up to one pipeline.
const MORPHOSYNTAX: Footprint =
    Footprint::estimated(STANZA_PIPELINE_CAPACITY * STANZA_MORPHOSYNTAX_PIPELINE_MB);

/// The utseg load: one Stanza constituency pipeline (estimated at the
/// measured morphosyntax pipeline's in-use size) and the pinned BERT-base
/// boundary model (estimated at 460 MiB, its float32 weights with
/// tokenizer). Not measured: the boundary model was not cached where the
/// measurements ran, and they run with downloads off.
const UTSEG: Footprint = Footprint::estimated(STANZA_MORPHOSYNTAX_PIPELINE_MB + 460);

/// The Stanza English coreference pipeline (tokenize with the RoBERTa-large
/// LoRA coreference package), built once by the load and again by every
/// batch. Measured: +519 MiB built; +2568 MiB in use resolving a 27-word
/// document (about the same over three batches on one build); a release
/// after use returns about 1.6 GB.
///
/// UNDERSIZED for real documents, and known to be: the in-use peak grows
/// faster than the document (on one paragraph repeated: about 4.0 GB at
/// 540 words, 10.7 GB at 2,700 and 39 GB at 10,800, 2026-11-06), and a coref
/// batch is one whole transcript. No fixed figure can cover that; how coref
/// is bounded is an open design decision recorded in the memory-safety
/// chapter, and this figure stays the small-document measurement until it
/// is made.
const COREF: Footprint = Footprint::measured(519, 2568);

/// The local pyannote diarization pipeline. Measured: +378 MiB built, and at
/// its peak diarizing one hour of audio +2274 MiB (30 seconds: +1053; ten
/// minutes: +1843); a release returns nothing.
const PYANNOTE_DIARIZATION: Footprint = Footprint::measured(378, 2274);

/// The speaker-embedding model. Measured: +381 MiB built, +514 MiB at its
/// peak embedding a 30-second window; a release returns nothing.
const SPEAKER_EMBEDDING: Footprint = Footprint::measured(381, 514);

/// What one on-demand load keeps resident in a worker, and for how long.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Residency {
    /// The load builds no model in the worker: the hosted pyannoteAI engine
    /// (a cloud ASR provider is not this: its worker keeps a small client,
    /// sized by its engine table), an ASR load whose engine runs in Rust, a
    /// model-free analysis (openSMILE,
    /// AVQI, word segmentation, whose libraries the baseline covers), or an
    /// FA load naming no engine, which the worker refuses. Takes no lease.
    Nothing,
    /// A model the worker builds once and keeps until it exits.
    WorkerLife(Footprint),
    /// A model the worker builds for each batch and releases after it
    /// (coreference): the load builds it once, and every batch builds it
    /// again under its own lease.
    PerBatch(Footprint),
}

impl Residency {
    /// The residency of the load `task` needs, given the engines its load
    /// reads (`EngineSelection::task_keys`): the residency table.
    ///
    /// Exhaustive over [`InferTask`] and over each engine family, with no
    /// catch-all, so a new task or engine breaks compilation here until its
    /// author says what it holds.
    pub fn of(task: InferTask, engines: &EngineOverrides) -> Self {
        match task {
            InferTask::Morphosyntax => Self::WorkerLife(MORPHOSYNTAX),
            InferTask::Utseg => Self::WorkerLife(UTSEG),
            InferTask::Coref => Self::PerBatch(COREF),
            InferTask::Translate => {
                // The worker loads Google when no engine is named
                // (`resolve_translate_engine`), so an unnamed engine is sized
                // as Google, not as nothing.
                let engine = engines
                    .translate
                    .as_ref()
                    .unwrap_or(&TranslateEngineName::Google);
                Self::WorkerLife(Footprint::estimated(engine.resident_memory_mb()))
            }
            InferTask::Fa => match &engines.fa {
                Some(engine) => Self::WorkerLife(Footprint::estimated(engine.resident_memory_mb())),
                None => Self::Nothing,
            },
            InferTask::Asr => match &engines.asr {
                // Only an engine a Python worker hosts loads anything there;
                // the others (Rev.AI, whisper.cpp, cloud APIs) run in Rust.
                Some(engine) => match engine.dispatch_override_name() {
                    Some(_) => Self::WorkerLife(Footprint::estimated(engine.resident_memory_mb())),
                    None => Self::Nothing,
                },
                None => Self::Nothing,
            },
            InferTask::Speaker => match (&engines.speaker, &engines.speaker_embedding) {
                (Some(SpeakerEngineName::Pyannote), _) => Self::WorkerLife(PYANNOTE_DIARIZATION),
                (
                    Some(SpeakerEngineName::PyannoteAi) | None,
                    Some(SpeakerEmbeddingBackendV2::Pyannote),
                ) => Self::WorkerLife(SPEAKER_EMBEDDING),
                (Some(SpeakerEngineName::PyannoteAi) | None, None) => Self::Nothing,
            },
            InferTask::Opensmile | InferTask::Avqi | InferTask::WordSegmentation => Self::Nothing,
        }
    }

    /// What the load's lease reserves while the load runs, `None` when it
    /// takes no lease. A per-batch model is built once by its load, so the
    /// load reserves one build.
    pub const fn loading(self) -> Option<MemoryMb> {
        match self {
            Self::Nothing => None,
            Self::WorkerLife(footprint) | Self::PerBatch(footprint) => Some(footprint.reserved()),
        }
    }

    /// What the load's lease still reserves once the worker holds the
    /// model: a model kept for the worker's life keeps its settled figure
    /// ([`Footprint::settled`]), and a per-batch model keeps nothing, since
    /// its batches reserve their own use.
    pub const fn resident(self) -> MemoryMb {
        match self {
            Self::Nothing | Self::PerBatch(_) => MemoryMb(0),
            Self::WorkerLife(footprint) => footprint.settled(),
        }
    }

    /// What one batch's lease reserves, for a per-batch model only.
    pub const fn batch(self) -> Option<MemoryMb> {
        match self {
            Self::Nothing | Self::WorkerLife(_) => None,
            Self::PerBatch(footprint) => Some(footprint.reserved()),
        }
    }
}

#[cfg(test)]
mod tests;
