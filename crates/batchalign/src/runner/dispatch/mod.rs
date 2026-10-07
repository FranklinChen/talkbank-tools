//! Dispatch functions for released infer processing.
//!
//! This module is the "traffic director" between the job runner and
//! command-specific orchestrators.
//!
//! # Three dispatch shapes, all audio-bound
//!
//! - [`dispatch_fa_infer`] handles forced alignment (`align`), which is audio-
//!   bound and therefore processed per file (with per-group batching inside each
//!   file).
//! - [`dispatch_transcribe_infer`] handles transcription (`transcribe`), which
//!   takes audio input and produces CHAT output through a multi-step pipeline
//!   (ASR -> post-processing -> CHAT assembly -> optional utseg/morphosyntax).
//! - [`dispatch_benchmark_infer`] handles benchmarking (`benchmark`), which
//!   composes the Rust transcribe and compare pipelines for audio-plus-gold
//!   evaluation.
//!
//! # Why this split exists
//!
//! FA, transcribe and benchmark all require per-file audio, so the top-level
//! loop stays per-file to keep file/audio provenance and failure handling
//! deterministic.
//!
//! # This module is legacy, and text no longer lives here
//!
//! Every command that operates on CHAT payloads only (`morphotag`, `utseg`,
//! `translate`, `coref`, `compare`) executes on the recipe-owned stack in
//! `crate::execution`, reached by a name-matched arm in
//! `crate::runner::routing`. The batched-text dispatch module that used to sit
//! here was retired once the last of those arms landed and nothing could reach
//! it; `routing`'s test module pins that no released batched-text command can
//! fall through to this side again.
//!
//! What remains here is the audio ML orchestration (FA, ASR, media analysis,
//! benchmark), which has no counterpart on the recipe stack yet. Porting it is
//! its own piece of work; do not add new commands to this module.
//!
//! # Related modules
//!
//! - `crate::runner::mod` decides whether infer mode is enabled.
//! - `plan` translates store-owned job snapshots into typed command-family
//!   plans before orchestration begins.
//! - `crate::fa` implements forced-alignment orchestration.
//! - `crate::transcribe` implements the multi-step transcribe orchestrator.
//! - `crate::execution` implements the recipe-owned text orchestrators.

mod asr_media;
mod audio_output;
mod audio_task;
mod benchmark_pipeline;
mod diarize_chat;
pub(crate) mod diarize_turns;
mod fa_pipeline;
mod kernel_plan;
mod media_analysis_v2;
mod media_search;
mod options;
mod plan;
pub(crate) mod speaker_identity_pipeline;
mod transcribe_pipeline;
mod utr;

pub(crate) use benchmark_pipeline::{BenchmarkDispatchRuntime, dispatch_benchmark_infer};
pub(crate) use fa_pipeline::*;
pub(crate) use media_analysis_v2::{MediaAnalysisDispatchRuntime, dispatch_media_analysis_v2};
pub(crate) use plan::*;
pub(crate) use transcribe_pipeline::{TranscribeDispatchRuntime, dispatch_transcribe_infer};
