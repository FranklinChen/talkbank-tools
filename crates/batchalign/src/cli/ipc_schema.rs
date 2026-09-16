//! Generate JSON Schema for all Rust types that cross the Python IPC boundary.
//!
//! This module is the Rust half of the Rust→Python type sync pipeline:
//!
//! 1. Rust types derive `schemars::JsonSchema`
//! 2. This module emits JSON Schema for all IPC-boundary types
//! 3. `scripts/generate_ipc_types.sh` writes that schema into `ipc-schema/`
//! 4. `scripts/check_ipc_type_drift.sh` fails when a Rust type has changed
//!    without the schema being regenerated
//!
//! The hand-written Python models are then checked against the schema by
//! `batchalign/tests/test_ipc_type_conformance.py`. They are not generated;
//! see the "Rust to Python IPC Type Sync" developer page for why.
//!
//! ## Adding new IPC types
//!
//! When you add a new Rust type that will be serialized to JSON and
//! deserialized by Python Pydantic models:
//!
//! 1. Add `schemars::JsonSchema` to the derive block
//! 2. Add a `register!()` call in [`generate_ipc_schema()`] below
//! 3. Run `bash scripts/generate_ipc_types.sh`

use std::collections::BTreeMap;
use std::path::Path;

use schemars::Schema;

/// Helper macro to reduce boilerplate when registering types.
///
/// Uses the schema title (type name without module path) as the map key,
/// falling back to the last segment of `stringify!($ty)` if no title is set.
macro_rules! register {
    ($map:expr, $($ty:ty),+ $(,)?) => {
        $(
            let schema = schemars::schema_for!($ty);
            let name = schema
                .get("title")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    let full = stringify!($ty);
                    full.rsplit("::").next().unwrap_or(full).to_string()
                });
            $map.insert(name, schema);
        )+
    };
}

/// Generate JSON Schema for all IPC-boundary types, grouped by protocol layer.
///
/// The `worker_v2` layer name is intentional: it matches the live typed worker
/// namespace across the Rust module and `ipc-schema/worker_v2`, while the
/// frozen V1 worker surface still exists beside it.
///
/// Returns a map of `layer_name → { type_name → Schema }`.
pub fn generate_ipc_schema() -> BTreeMap<String, BTreeMap<String, Schema>> {
    let mut layers: BTreeMap<String, BTreeMap<String, Schema>> = BTreeMap::new();

    // --- Live typed worker protocol types (layer name retained while V1 remains) ---
    {
        use crate::types::worker_v2::*;

        let mut v2 = BTreeMap::new();
        register!(
            v2,
            // Envelope
            ExecuteRequestV2,
            ExecuteResponseV2,
            ExecuteOutcomeV2,
            // Task requests (discriminated union)
            TaskRequestV2,
            ForcedAlignmentRequestV2,
            AsrRequestV2,
            SpeakerRequestV2,
            SpeakerEmbeddingRequestV2,
            MorphosyntaxRequestV2,
            UtsegRequestV2,
            TranslateRequestV2,
            CorefRequestV2,
            OpenSmileRequestV2,
            AvqiRequestV2,
            // Task results (discriminated union)
            TaskResultV2,
            IndexedWordTimingResultV2,
            MonologueAsrResultV2,
            WhisperChunkResultV2,
            WhisperTokenTimingResultV2,
            SpeakerResultV2,
            MorphosyntaxResultV2,
            MorphosyntaxItemResultV2,
            MorphosyntaxModelIdentityV2,
            // The relation repairs an analyzed item carries. Registered as
            // named types, not left to appear only as `$defs`, for the same
            // reason as the model identity: the Python conformance test reads
            // these FILES, so a type with no file of its own is a contract
            // Python is never checked against.
            UdRelationRepairV2,
            UdRelationRepairKindV2,
            // Pinned ASR model identity. Registered as named types rather than
            // left to appear only as `$defs` inside the request and result
            // schemas, because the Python conformance test reads these FILES:
            // an identity type with no file of its own is a contract Python is
            // never checked against.
            AsrRequestedModelsV2,
            AsrModelIdentityV2,
            RequestedModelV2,
            LoadedModelV2,
            RequestedRevisionV2,
            ObservedRevisionV2,
            UtsegResultV2,
            UtsegItemResultV2,
            TranslationResultV2,
            TranslationItemResultV2,
            CorefResultV2,
            CorefItemResultV2,
            CorefAnnotationV2,
            CorefChainRefV2,
            SpeakerEmbeddingResultV2,
            SpeakerEmbeddingSpanResultV2,
            SpeakerEmbeddingOutcomeV2,
            OpenSmileResultV2,
            AvqiResultV2,
            // Prepared batches
            InlineJsonRefV2,
            PreparedAudioRefV2,
            PreparedTextRefV2,
            PreparedAudioInputV2,
            SpeakerPreparedAudioInputV2,
            SpeakerEmbeddingSpanV2,
            // Inputs
            AsrInputV2,
            SpeakerInputV2,
            ProviderMediaInputV2,
            SubmittedJobInputV2,
            // Artifacts
            ArtifactRefV2,
            // Sub-types
            AsrElementV2,
            AsrElementKindV2,
            AsrMonologueV2,
            // Speaker attribution: a provider's own label, or an explicit
            // "this engine separates nobody". Registered as named types so the
            // Python conformance test reads them as files, the same reason the
            // model-identity types are registered.
            SpeakerAttributionV2,
            ProviderSpeakerLabelV2,
            ProviderDiarizationV2,
            IndexedWordTimingV2,
            WhisperChunkSpanV2,
            WhisperTokenTimingV2,
            SpeakerSegmentV2,
            ProgressEventV2,
            // Enums
            WorkerKindV2,
            InferenceTaskV2,
            AsrBackendV2,
            FaBackendV2,
            FaTextModeV2,
            SpeakerBackendV2,
            SpeakerEmbeddingBackendV2,
            PreparedAudioEncodingV2,
            PreparedTextEncodingV2,
            ProtocolErrorCodeV2,
            WorkerAttachmentKindV2,
            // Handshake
            HelloRequestV2,
            HelloResponseV2,
            ShutdownRequestV2,
            WorkerRuntimeInfoV2,
        );
        layers.insert("worker_v2".into(), v2);
    }

    // --- Batch item types (chat-ops) ---
    {
        let mut batch = BTreeMap::new();
        register!(
            batch,
            crate::chat_ops::morphosyntax_ops::MorphosyntaxBatchItem,
            batchalign_transform::utseg::UtsegBatchItem,
            batchalign_transform::translate::TranslateBatchItem,
            batchalign_transform::coref::CorefBatchItem,
            batchalign_transform::coref::CorefAnnotation,
            batchalign_transform::coref::ChainRef,
        );
        layers.insert("batch_items".into(), batch);
    }

    layers
}

/// Write IPC schemas to a directory structure: `{dir}/{layer}/{type}.json`.
pub fn write_ipc_schema(
    schemas: &BTreeMap<String, BTreeMap<String, Schema>>,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let base = Path::new(dir);
    for (layer, types) in schemas {
        let layer_dir = base.join(layer);
        std::fs::create_dir_all(&layer_dir)?;
        for (type_name, schema) in types {
            let path = layer_dir.join(format!("{type_name}.json"));
            let json = serde_json::to_string_pretty(schema)?;
            std::fs::write(&path, json)?;
        }
    }
    let total: usize = schemas.values().map(|v| v.len()).sum();
    eprintln!(
        "Wrote {} schemas across {} layers to {}",
        total,
        schemas.len(),
        dir
    );
    Ok(())
}

/// Check that existing schema files match the current Rust types.
pub fn check_ipc_schema(
    schemas: &BTreeMap<String, BTreeMap<String, Schema>>,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let base = Path::new(dir);
    let mut drift = Vec::new();

    for (layer, types) in schemas {
        let layer_dir = base.join(layer);
        for (type_name, schema) in types {
            let path = layer_dir.join(format!("{type_name}.json"));
            let expected = serde_json::to_string_pretty(schema)?;
            match std::fs::read_to_string(&path) {
                Ok(actual) if actual.trim() == expected.trim() => {}
                Ok(_) => drift.push(format!("  CHANGED: {}", path.display())),
                Err(_) => drift.push(format!("  MISSING: {}", path.display())),
            }
        }

        // A schema file with no Rust type behind it. Renaming a type used to
        // leave its old file on disk forever: the generator never prunes, and
        // checking only the types we know about cannot see what is left over.
        // That matters because the Python conformance test reads these FILES,
        // so an orphan means Python is being held to a contract Rust no longer
        // has, with both gates green.
        let Ok(entries) = std::fs::read_dir(&layer_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let is_known = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| types.contains_key(stem));
            if !is_known {
                drift.push(format!("  ORPHANED: {}", path.display()));
            }
        }
    }

    if drift.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "IPC schema drift detected ({} files):\n{}\n\nRun: bash scripts/generate_ipc_types.sh",
            drift.len(),
            drift.join("\n")
        )
        .into())
    }
}
