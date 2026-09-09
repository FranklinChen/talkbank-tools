//! Audit wide Rust structs so field-bag growth stays explicit and reviewed.
//!
//! Scans all `.rs` files under the audit roots for structs with ≥10
//! named fields and ensures each is registered in `WIDE_STRUCT_ALLOWANCES`
//! with a reviewed field cap and classification.
//!
//! Run it with `cargo run -q -p xtask -- lint-wide-structs`. It is wired into
//! `make ci-local` and `make batchalign-ci-rust`, so CI fails on drift.
//!
//! The line here used to name a second entrypoint: a nextest-invoked proxy
//! test in a `talkbank-tools` package. All three parts of that were stale:
//! nextest is banned and uninstalled in this workspace, that package no longer
//! exists (the workspace is virtual), and no `wide_struct_audit` test target
//! has ever existed here. Only CI keeps this honest, which is why it now runs
//! there.

use std::path::Path;

use crate::Result;
use crate::rust_scan::{brace_delta, bracket_delta, rust_scan_roots, walkdir};

const WIDE_STRUCT_THRESHOLD: usize = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WideStructDisposition {
    BoundaryShim,
    TransportRecord,
    RealAggregate,
    RefactorTarget,
}

impl WideStructDisposition {
    fn label(self) -> &'static str {
        match self {
            Self::BoundaryShim => "boundary shim",
            Self::TransportRecord => "transport record",
            Self::RealAggregate => "real aggregate",
            Self::RefactorTarget => "refactor target",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WideStructAllowance {
    path: &'static str,
    struct_name: &'static str,
    max_fields: usize,
    max_bool_fields: usize,
    disposition: WideStructDisposition,
    reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NamedStructInfo {
    path: String,
    line: usize,
    struct_name: String,
    field_count: usize,
    bool_field_count: usize,
}

/// Every struct in this workspace with at least [`WIDE_STRUCT_THRESHOLD`] named
/// fields, with the field cap a human actually reviewed and the verdict they
/// reached.
///
/// Re-adjudicated wholesale on 2026-07-30, because the table had been failing
/// since the chatter split and nobody was reading its output. Two thirds of its
/// entries named crates that left this repo in 2026-05/06 (`talkbank-clan`,
/// `talkbank-model`, `talkbank-parser-*`, `talkbank-cli`, `spec/tools`,
/// `src/test_dashboard`); those are gone. Ten more named batchalign paths that
/// moved under `cli/`, so the audit reported the same struct as both a stale
/// entry and an unregistered one. `ServeStartArgs` was dropped rather than
/// repointed: it is under the threshold at its new location.
///
/// A `RefactorTarget` entry is a recorded verdict, not a plan: the cap stops the
/// struct growing further while the reason names the specific fix, so whoever
/// picks it up does not have to re-derive it.
const WIDE_STRUCT_ALLOWANCES: &[WideStructAllowance] = &[
    WideStructAllowance {
        path: "crates/batchalign/src/chat_ops/speaker_identity/evidence.rs",
        struct_name: "SpeakerIdentityProvenance",
        max_fields: 13,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "the complete recipe for one speaker-identification run, written beside its own verdicts: schema version, the agreement-not-accuracy caveat, the transcript and media it read, the prepared decode's sample rate, the embedding backend, model revision, vector width and minimum span the worker reported, the threshold the caller stated, the tiers scored, the enrolled spans, and the build that wrote it; narrowing it would move a fact out of the artifact and into a runbook, which is what makes a generated file read as unreproducible",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/handle/mod.rs",
        struct_name: "WorkerHandle",
        max_fields: 10,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RealAggregate,
        reason: "one live child process: its config, handle, pid, runtime identity, the two pipes, the activity clock, the loaded-task set, and the request-flight state the checkout's Drop decides from; the tenth field is that state, added so a cancelled attempt's worker is discarded rather than requeued",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/engines.rs",
        struct_name: "FaEngineSpec",
        max_fields: 11,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RealAggregate,
        reason: "one row of the forced-alignment engine declaration table, and its width is the design rather than sprawl: adding an engine used to be an edit in nineteen places, so every fact an engine decides now lives in one required field (variant, wire name, selection name, aliases, the name the Python worker resolves, timing resolution, longest dispatch window, resident memory, language support, fallback policy, worker backend) and the struct having no Default is what makes a new field break every row until its author states it; the cap is still doing its job, since a twelfth field is a fact nobody has reviewed",
    },
    WideStructAllowance {
        path: "crates/batchalign-transform/src/morphosyntax/ud_types.rs",
        struct_name: "UdWord",
        max_fields: 10,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RealAggregate,
        reason: "the ten CoNLL-U columns; the format fixes the width, so it cannot grow",
    },
    WideStructAllowance {
        path: "crates/batchalign-types/src/worker_v2/responses.rs",
        struct_name: "AvqiResultV2",
        max_fields: 11,
        max_bool_fields: 1,
        disposition: WideStructDisposition::TransportRecord,
        reason: "worker protocol response payload for AVQI scoring",
    },
    WideStructAllowance {
        path: "crates/batchalign-whisper-pilot/src/decoder.rs",
        struct_name: "Decoder",
        max_fields: 14,
        max_bool_fields: 1,
        disposition: WideStructDisposition::RealAggregate,
        reason: "whisper decoding state; eight of the fourteen are special-token ids, and \
                 gathering those into one SpecialTokens struct would take it to seven",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/commands.rs",
        struct_name: "AlignArgs",
        max_fields: 16,
        max_bool_fields: 8,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "UTR selection/tuning and boundary policy now have owned flattened types; the \
                 remaining --flag/--no-flag pairs (wor, merge_abbrev) and BA2 compatibility \
                 switches still make the outer clap boundary a refactor target",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/commands.rs",
        struct_name: "BenchmarkArgs",
        max_fields: 14,
        max_bool_fields: 7,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "same two --flag/--no-flag pairs as AlignArgs (wor, merge_abbrev); fix them \
                 together",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/commands.rs",
        struct_name: "MorphotagArgs",
        max_fields: 12,
        max_bool_fields: 8,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "two --flag/--no-flag pairs (merge_abbrev, and skipmultilang against multilang, \
                 which is the same pattern under an unrelated pair of names)",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/commands.rs",
        struct_name: "TranscribeArgs",
        max_fields: 17,
        max_bool_fields: 11,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "three --flag/--no-flag pairs (wor, merge_abbrev, diarize); eleven bools on one \
                 command is where a flat clap struct stops being a boundary and starts being \
                 the design",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/commands.rs",
        struct_name: "TranscribeReplayRunArgs",
        max_fields: 11,
        max_bool_fields: 4,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "offline experiment CLI boundary: four independent switches plus explicit policy \
                 and pass-topology enums; the mutually constrained segmentation states are \
                 lowered immediately into TranscribeUtsegExecution typestate",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/args/global_opts.rs",
        struct_name: "GlobalOpts",
        max_fields: 17,
        max_bool_fields: 9,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "three --flag/--no-flag pairs (server, tui, open_dashboard); tui against no_tui \
                 is the literal example the repo's rule 4 gives",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/dispatch/mod.rs",
        struct_name: "DispatchRequest",
        max_fields: 21,
        max_bool_fields: 6,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "re-spreads nine GlobalOpts fields one by one instead of holding the struct, \
                 which is the seam shape corrected elsewhere in this crate by passing \
                 &MorphosyntaxParams rather than five of its fields",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/eval_cmd/l2_morphotag/report.rs",
        struct_name: "PairAggregate",
        max_fields: 10,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "evaluation report aggregate row",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/eval_cmd/transcribe_replay.rs",
        struct_name: "ReplayRunReceipt",
        max_fields: 12,
        max_bool_fields: 3,
        disposition: WideStructDisposition::TransportRecord,
        reason: "versioned offline replay receipt that records executable, engine, policy \
                 topology, presentation switches, and output identities for reproducibility",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/cli/tui/app.rs",
        struct_name: "FileState",
        max_fields: 11,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "TUI file row state; the width is one progress triple plus one error pair",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/db/insert.rs",
        struct_name: "NewJobRecord",
        max_fields: 19,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "database insert row for newly submitted jobs",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/db/query.rs",
        struct_name: "CancellationRow",
        max_fields: 10,
        max_bool_fields: 1,
        disposition: WideStructDisposition::TransportRecord,
        reason: "database query row for cancellation history",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/db/schema.rs",
        struct_name: "AttemptRow",
        max_fields: 12,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "database row describing one execution attempt",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/db/schema.rs",
        struct_name: "JobRow",
        max_fields: 31,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "database row for persisted job state",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/host_facts/mod.rs",
        struct_name: "HostFacts",
        max_fields: 11,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "host capability snapshot rendered into fleet/runtime decisions",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/runner/dispatch/fa_pipeline.rs",
        struct_name: "AlignAudioTask",
        max_fields: 22,
        max_bool_fields: 4,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "forced-alignment execution bag still mixes file context, engine controls, and \
                 output policy; it is inside the 4b audio port, so do not touch it separately",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/runner/dispatch/fa_pipeline.rs",
        struct_name: "FaFileContext",
        max_fields: 13,
        max_bool_fields: 1,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "forced-alignment per-file context still bundles several workflow concerns; \
                 same 4b caveat as AlignAudioTask",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/store/job/types.rs",
        struct_name: "JobFilesystemConfig",
        max_fields: 10,
        max_bool_fields: 2,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "filesystem/layout configuration boundary for job storage",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/store/mod.rs",
        struct_name: "FileStatus",
        max_fields: 14,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "stored per-file processing status record",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/transcribe/types.rs",
        struct_name: "TranscribeOptions",
        max_fields: 13,
        max_bool_fields: 6,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "transcription option bag crossing CLI/server/runtime boundaries; the six bools \
                 are independent switches rather than flag/no-flag pairs, which is why this \
                 stays a shim while the clap structs above do not",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/cancellation.rs",
        struct_name: "CancellationRecord",
        max_fields: 10,
        max_bool_fields: 1,
        disposition: WideStructDisposition::TransportRecord,
        reason: "API/runtime record for a cancellation event",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/config/server.rs",
        struct_name: "ServerConfig",
        max_fields: 47,
        max_bool_fields: 3,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "server configuration boundary intentionally mirrors a broad operator-facing \
                 config file; its width is the width of server.yaml",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/options.rs",
        struct_name: "AlignOptions",
        max_fields: 10,
        max_bool_fields: 2,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "alignment option boundary retains its flat JSON contract while typed UTR and \
                 boundary sub-capabilities own the related Rust fields",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/request.rs",
        struct_name: "JobSubmission",
        max_fields: 15,
        max_bool_fields: 2,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "public request payload for job submission",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/response.rs",
        struct_name: "FileStatusEntry",
        max_fields: 14,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "API response record for one file's processing status",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/response.rs",
        struct_name: "HealthResponse",
        max_fields: 33,
        max_bool_fields: 1,
        disposition: WideStructDisposition::TransportRecord,
        reason: "health/status API response aggregates independent runtime metrics; the newest \
                 field is the content-addressed worker-runtime identity collection required to \
                 prove which Python code actually executed",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/response.rs",
        struct_name: "JobInfo",
        max_fields: 26,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "API response record for a full job snapshot",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/response.rs",
        struct_name: "JobListItem",
        max_fields: 18,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "API response row for job list summaries",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/scheduling.rs",
        struct_name: "AttemptRecord",
        max_fields: 12,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "runtime scheduling record for one attempt",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/types/traces.rs",
        struct_name: "FaTimelineTrace",
        max_fields: 14,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "versioned forced-alignment evidence transport preserving independent grouping, \
                 cache, timing, decision, violation, and fallback facts for offline analysis; \
                 schema 4 adds the flat dropped_word_timings section derived from the decisions",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/handle/config.rs",
        struct_name: "WorkerConfig",
        max_fields: 13,
        max_bool_fields: 1,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "worker configuration boundary for runtime startup and tuning",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/handle/config.rs",
        struct_name: "WorkerRuntimeConfig",
        max_fields: 10,
        max_bool_fields: 2,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "the host-and-process half of WorkerConfig above, same boundary, same verdict",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/pool/mod.rs",
        struct_name: "PoolConfig",
        max_fields: 16,
        max_bool_fields: 1,
        disposition: WideStructDisposition::BoundaryShim,
        reason: "worker-pool configuration boundary",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/pool/mod.rs",
        struct_name: "WorkerPool",
        max_fields: 17,
        max_bool_fields: 0,
        disposition: WideStructDisposition::RefactorTarget,
        reason: "grew 12 to 17, and all five additions are monotonic AtomicU64 admission \
                 rejection counters read only through metrics_snapshot; collecting them into \
                 one AdmissionRejectionCounters takes the pool back to 13 without touching the \
                 registry/scheduling/lifecycle mixing that earned this verdict originally",
    },
    WideStructAllowance {
        path: "crates/batchalign/src/worker/registry.rs",
        struct_name: "RegistryEntry",
        max_fields: 10,
        max_bool_fields: 0,
        disposition: WideStructDisposition::TransportRecord,
        reason: "worker registry snapshot entry",
    },
    WideStructAllowance {
        path: "crates/batchalign/tests/common/test_worker_pool.rs",
        struct_name: "ConfigKey",
        max_fields: 14,
        max_bool_fields: 2,
        disposition: WideStructDisposition::TransportRecord,
        reason: "test-only cache key; its width is the width of the WorkerConfig it keys on, so \
                 it tracks that struct rather than growing on its own",
    },
];

fn scan_named_structs(root: &Path) -> Result<Vec<NamedStructInfo>> {
    let mut structs = Vec::new();
    for base in rust_scan_roots(root) {
        if !base.exists() {
            continue;
        }
        for path in walkdir(&base) {
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(_) => continue,
            };
            structs.extend(parse_named_structs_in_file(&relative, &text));
        }
    }
    structs.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.struct_name.cmp(&right.struct_name))
    });
    Ok(structs)
}

fn parse_named_structs_in_file(relative_path: &str, text: &str) -> Vec<NamedStructInfo> {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        let Some(struct_name) = struct_name_from_declaration(line) else {
            index += 1;
            continue;
        };
        let mut depth = brace_delta(line);
        let mut field_count = 0;
        let mut bool_field_count = 0;
        let start_line = index + 1;
        index += 1;
        // Bracket depth of an attribute that spans several lines. A
        // continuation line of `#[arg(...)]` is not a field, and some of them
        // contain a colon (`default_value = some::path::CONST`), which is the
        // only thing `is_named_field` looks for. Counting those inflated the
        // field count and failed the audit on a struct that had not grown.
        let mut attribute_depth: isize = 0;
        while index < lines.len() && depth > 0 {
            let current = lines[index];
            let trimmed = current.trim();
            if attribute_depth > 0 || trimmed.starts_with("#[") {
                attribute_depth += bracket_delta(current);
            } else if depth == 1 && is_named_field(trimmed) {
                field_count += 1;
                if field_type(trimmed).is_some_and(|value| value.contains("bool")) {
                    bool_field_count += 1;
                }
            }
            depth += brace_delta(current);
            index += 1;
        }
        result.push(NamedStructInfo {
            path: relative_path.to_string(),
            line: start_line,
            struct_name,
            field_count,
            bool_field_count,
        });
    }
    result
}

fn struct_name_from_declaration(line: &str) -> Option<String> {
    let declaration = line
        .strip_prefix("pub struct ")
        .or_else(|| line.strip_prefix("struct "))?;
    if !declaration.contains('{') {
        return None;
    }
    let name = declaration.split('{').next()?.trim();
    let name = name.split('<').next()?.trim();
    let name = name.split_whitespace().next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn is_named_field(line: &str) -> bool {
    if line.is_empty()
        || line.starts_with("//")
        || line.starts_with("///")
        || line.starts_with("#[")
        || line.starts_with("pub use ")
    {
        return false;
    }
    line.contains(':') && !line.starts_with("fn ") && !line.starts_with("where ")
}

fn field_type(line: &str) -> Option<&str> {
    let (_, ty) = line.split_once(':')?;
    Some(ty.trim().trim_end_matches(','))
}

pub fn run(root: &Path) -> Result<()> {
    let wide_structs: Vec<NamedStructInfo> = scan_named_structs(root)?
        .into_iter()
        .filter(|info| info.field_count >= WIDE_STRUCT_THRESHOLD)
        .collect();

    let mut failures = Vec::new();

    for info in &wide_structs {
        let Some(allowance) = WIDE_STRUCT_ALLOWANCES
            .iter()
            .find(|entry| entry.path == info.path && entry.struct_name == info.struct_name)
        else {
            failures.push(format!(
                "{}:{}: {} has {} fields and {} bool fields but no audit entry",
                info.path, info.line, info.struct_name, info.field_count, info.bool_field_count
            ));
            continue;
        };
        if info.field_count > allowance.max_fields {
            failures.push(format!(
                "{}:{}: {} grew from reviewed max {} fields to {} ({}, {})",
                info.path,
                info.line,
                info.struct_name,
                allowance.max_fields,
                info.field_count,
                allowance.disposition.label(),
                allowance.reason
            ));
        }
        if info.bool_field_count > allowance.max_bool_fields {
            failures.push(format!(
                "{}:{}: {} grew from reviewed max {} bool fields to {} ({}, {})",
                info.path,
                info.line,
                info.struct_name,
                allowance.max_bool_fields,
                info.bool_field_count,
                allowance.disposition.label(),
                allowance.reason
            ));
        }
    }

    for allowance in WIDE_STRUCT_ALLOWANCES {
        let registered = wide_structs
            .iter()
            .any(|info| info.path == allowance.path && info.struct_name == allowance.struct_name);
        if !registered {
            failures.push(format!(
                "{}: stale audit entry for {} ({}, {})",
                allowance.path,
                allowance.struct_name,
                allowance.disposition.label(),
                allowance.reason
            ));
        }
    }

    // There is deliberately no third pass over `actual_by_key` here. One used to
    // exist, reporting "unexpected wide struct audit state" for any scanned
    // struct missing from the allowances, which is the identical condition the
    // first loop already reports with the field counts attached. Every
    // unregistered struct therefore appeared twice, once usefully and once as
    // noise, and the failure count read double. On 2026-07-30 that meant 70
    // lines where 32 were information.

    if failures.is_empty() {
        println!("wide struct audit: OK");
        Ok(())
    } else {
        Err(format!("wide struct audit failures:\n- {}", failures.join("\n- ")).into())
    }
}

#[cfg(test)]
mod tests {
    use super::parse_named_structs_in_file;

    /// A multi-line attribute is not a field, even when it contains a colon.
    ///
    /// `is_named_field` recognises a field by the presence of `:`, and an
    /// attribute continuation line such as `default_value = a::b::C,` has one.
    /// Parentheses do not change brace depth, so those lines sat at depth 1 and
    /// were counted, inflating the struct by one field per multi-line attribute
    /// and failing the audit on a struct that had not gained a field at all.
    #[test]
    fn a_multi_line_attribute_is_not_counted_as_a_field() {
        let source = r#"
pub struct Example {
    /// A real field.
    #[arg(
        long,
        default_value = crate::types::engines::DEFAULT_NAME,
        value_parser = some_parser(),
    )]
    pub engine: String,

    /// Another real field.
    #[arg(long)]
    pub count: u32,
}
"#;
        let found = parse_named_structs_in_file("example.rs", source);
        let example = found
            .iter()
            .find(|s| s.struct_name == "Example")
            .expect("Example must be found");
        assert_eq!(
            example.field_count, 2,
            "only `engine` and `count` are fields; the attribute lines are not"
        );
    }

    /// Bool counting still works, and is likewise not fooled by attributes.
    #[test]
    fn bool_fields_are_counted_without_attribute_noise() {
        let source = r#"
pub struct Flags {
    #[arg(
        long,
        help = "see mod::path::THING",
    )]
    pub verbose: bool,
    pub quiet: bool,
    pub name: String,
}
"#;
        let found = parse_named_structs_in_file("flags.rs", source);
        let flags = found
            .iter()
            .find(|s| s.struct_name == "Flags")
            .expect("Flags must be found");
        assert_eq!(flags.field_count, 3);
        assert_eq!(flags.bool_field_count, 2);
    }
}
