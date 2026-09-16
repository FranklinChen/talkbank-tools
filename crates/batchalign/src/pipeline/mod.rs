//! Internal pipeline helpers for command-local orchestration.
//!
//! This module intentionally stays private to `batchalign-server`. It is not a
//! general executor; it is a small sequential stage runner used to make
//! per-command orchestration explicit.

use crate::cache::UtteranceCache;
use crate::worker::pool::WorkerPool;

pub(crate) mod morphosyntax;
pub(crate) mod plan;
pub(crate) mod post_validate;
pub(crate) mod text_infer;
pub(crate) mod transcribe;

/// Shared services used by pipeline helpers.
///
/// Deliberately carries no engine identity. It used to carry one engine
/// version for the whole pipeline, which inside a multi-engine command (the
/// ASR, utseg and morphotag stages of `transcribe`) could only be right for one
/// stage and was stamped onto the others. Each stage now names its own engine:
/// morphotag, utseg, translate and coref from the engine each applied result
/// names (never from a capability report); transcribe from the ASR identity
/// admitted with its plan; forced alignment from the `FaCacheNamespace` read
/// out of the selected worker's report after FA loaded, which
/// `crate::fa::FaServices` carries; and UTR ASR through its engine's own cache
/// namespace.
///
/// `TreeSitterParser` is `!Send + !Sync` (uses `RefCell` internally), so it
/// cannot be stored here, `PipelineServices` is carried across async task
/// boundaries. Callers that need a parser create one locally via
/// `TreeSitterParser::new()`.
#[derive(Clone, Copy)]
pub(crate) struct PipelineServices<'a> {
    /// Worker pool for inference.
    pub pool: &'a WorkerPool,
    /// Shared utterance cache.
    pub cache: &'a UtteranceCache,
}

impl<'a> PipelineServices<'a> {
    /// Create services.
    pub fn new(pool: &'a WorkerPool, cache: &'a UtteranceCache) -> Self {
        Self { pool, cache }
    }
}
