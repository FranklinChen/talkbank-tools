/** Generated from openapi.json. Regenerate through npm run generate:decoders. */

import * as Schema from "effect/Schema";
export type MachineTime = string;
export const MachineTime: Schema.Codec<MachineTime> = Schema.String.annotate({
  "format": "date-time",
  "identifier": "MachineTime"
});
export type NodeId = string;
export const NodeId: Schema.Codec<NodeId> = Schema.String.annotate({
  "identifier": "NodeId"
});
export type ReleasedCommand = "align" | "transcribe" | "transcribe_s" | "translate" | "morphotag" | "coref" | "utseg" | "benchmark" | "opensmile" | "compare" | "avqi" | "diarize" | "speaker_identify" | "convert";
export const ReleasedCommand: Schema.Codec<ReleasedCommand> = Schema.Literals(["align", "transcribe", "transcribe_s", "translate", "morphotag", "coref", "utseg", "benchmark", "opensmile", "compare", "avqi", "diarize", "speaker_identify", "convert"]).annotate({
  "identifier": "ReleasedCommand"
});
export type JobControlPlaneBackendKind = "local";
export const JobControlPlaneBackendKind: Schema.Codec<JobControlPlaneBackendKind> = Schema.Literal("local").annotate({
  "identifier": "JobControlPlaneBackendKind"
});
export type NonNegativeSeconds = number;
export const NonNegativeSeconds: Schema.Codec<NonNegativeSeconds> = Schema.Number.annotate({
  "format": "double"
}).check(Schema.isFinite().annotate({
  "expected": "a finite number"
})).check(Schema.isGreaterThanOrEqualTo(0).annotate({
  "expected": "a value greater than or equal to 0",
  "identifier": "NonNegativeSeconds"
}));
export type ExecutionMode = "local" | "staged_remote" | "explicit_remote";
export const ExecutionMode: Schema.Codec<ExecutionMode> = Schema.Literals(["local", "staged_remote", "explicit_remote"]).annotate({
  "identifier": "ExecutionMode"
});
export type JobId = string;
export const JobId: Schema.Codec<JobId> = Schema.String.annotate({
  "identifier": "JobId"
});
export type ExecutionStage = "staging" | "executing" | "copying_back" | "done" | "failed";
export const ExecutionStage: Schema.Codec<ExecutionStage> = Schema.Literals(["staging", "executing", "copying_back", "done", "failed"]).annotate({
  "identifier": "ExecutionStage"
});
export type JudgementBar = "construction" | "preservation";
export const JudgementBar: Schema.Codec<JudgementBar> = Schema.Literals(["construction", "preservation"]).annotate({
  "identifier": "JudgementBar"
});
export type FindingCodeCount = {
  readonly "code"?: string | null;
  readonly "count": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const FindingCodeCount: Schema.Codec<FindingCodeCount> = Schema.StructWithRest(Schema.Struct({
  "code": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "count": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "FindingCodeCount"
});
export type FindingLevel = "parseable" | "structurally_complete" | "main_tier_valid" | "construction";
export const FindingLevel: Schema.Codec<FindingLevel> = Schema.Literals(["parseable", "structurally_complete", "main_tier_valid", "construction"]).annotate({
  "identifier": "FindingLevel"
});
export type FullFindings = {
  readonly "kind": "inline";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "sidecar";
  readonly "path": string;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "error": string;
  readonly "kind": "unwritten";
  readonly "path": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const FullFindings: Schema.Codec<FullFindings> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("inline")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("sidecar"),
  "path": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "error": Schema.String,
  "kind": Schema.Literal("unwritten"),
  "path": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "FullFindings"
});
export type FailureCategory = "validation" | "parse_error" | "input_missing" | "evidence_unavailable" | "analysis_unavailable" | "worker_crash" | "worker_timeout" | "worker_protocol" | "worker_bootstrap" | "local_runtime" | "provider_transient" | "provider_terminal" | "provider_quota_exhausted" | "provider_account_refused" | "memory_pressure" | "cancelled" | "system" | "model_access_denied" | "model_output_invalid";
export const FailureCategory: Schema.Codec<FailureCategory> = Schema.Literals(["validation", "parse_error", "input_missing", "evidence_unavailable", "analysis_unavailable", "worker_crash", "worker_timeout", "worker_protocol", "worker_bootstrap", "local_runtime", "provider_transient", "provider_terminal", "provider_quota_exhausted", "provider_account_refused", "memory_pressure", "cancelled", "system", "model_access_denied", "model_output_invalid"]).annotate({
  "identifier": "FailureCategory"
});
export type UtrPass = "initial" | "fallback";
export const UtrPass: Schema.Codec<UtrPass> = Schema.Literals(["initial", "fallback"]).annotate({
  "identifier": "UtrPass"
});
export type SecondaryExtractionFailure = "language_unresolved" | "alignment" | "tokenization_disagrees" | "split_word" | "unmapped_word";
export const SecondaryExtractionFailure: Schema.Codec<SecondaryExtractionFailure> = Schema.Literals(["language_unresolved", "alignment", "tokenization_disagrees", "split_word", "unmapped_word"]).annotate({
  "identifier": "SecondaryExtractionFailure"
});
export type PrimaryMorphologyDeclineCause = "no_analysis" | "mapping_failed" | "misaligned" | "retokenization_failed" | "worker_failed";
export const PrimaryMorphologyDeclineCause: Schema.Codec<PrimaryMorphologyDeclineCause> = Schema.Literals(["no_analysis", "mapping_failed", "misaligned", "retokenization_failed", "worker_failed"]).annotate({
  "identifier": "PrimaryMorphologyDeclineCause"
});
export type SecondaryMorphologyFailureKind = "invalid_language" | "worker" | "response_count" | "empty_analysis" | "merge" | "splice";
export const SecondaryMorphologyFailureKind: Schema.Codec<SecondaryMorphologyFailureKind> = Schema.Literals(["invalid_language", "worker", "response_count", "empty_analysis", "merge", "splice"]).annotate({
  "identifier": "SecondaryMorphologyFailureKind"
});
export type RetiredSecondaryMorphologyFailureKind = "registry_unavailable";
export const RetiredSecondaryMorphologyFailureKind: Schema.Codec<RetiredSecondaryMorphologyFailureKind> = Schema.Literal("registry_unavailable").annotate({
  "identifier": "RetiredSecondaryMorphologyFailureKind"
});
export type SecondaryMorphologySpan = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "first_word": number;
  readonly "language": string;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const SecondaryMorphologySpan: Schema.Codec<SecondaryMorphologySpan> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "first_word": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "language": Schema.String,
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "SecondaryMorphologySpan"
});
export type OptionalStage = "utterance_segmentation" | "morphosyntax";
export const OptionalStage: Schema.Codec<OptionalStage> = Schema.Literals(["utterance_segmentation", "morphosyntax"]).annotate({
  "identifier": "OptionalStage"
});
export type StoredUtteranceLocation = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const StoredUtteranceLocation: Schema.Codec<StoredUtteranceLocation> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "StoredUtteranceLocation"
});
export type WindowSeamRecord = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "dropped_from_window": number;
  readonly "seam_ms": number;
  readonly "text": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const WindowSeamRecord: Schema.Codec<WindowSeamRecord> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "dropped_from_window": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "seam_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })),
  "text": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "WindowSeamRecord"
});
export type DecodedAudioShortfallV2 = {
  readonly "kind": "no_bytes";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "decoded_bytes": number;
  readonly "kind": "partial_sample";
  readonly "sample_bytes": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const DecodedAudioShortfallV2: Schema.Codec<DecodedAudioShortfallV2> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("no_bytes")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "decoded_bytes": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(1).annotate({
    "expected": "a value greater than or equal to 1"
  })),
  "kind": Schema.Literal("partial_sample"),
  "sample_bytes": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(1).annotate({
    "expected": "a value greater than or equal to 1"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "DecodedAudioShortfallV2"
});
export type FaConstraintUnitV2 = "audio_samples" | "encoded_frames" | "text_labels";
export const FaConstraintUnitV2: Schema.Codec<FaConstraintUnitV2> = Schema.Literals(["audio_samples", "encoded_frames", "text_labels"]).annotate({
  "identifier": "FaConstraintUnitV2"
});
export type UnusableAnchorsTrace = "no_reliable_anchors" | "anchors_refused" | "anchors_describe_other_words" | "no_interior_cut";
export const UnusableAnchorsTrace: Schema.Codec<UnusableAnchorsTrace> = Schema.Literals(["no_reliable_anchors", "anchors_refused", "anchors_describe_other_words", "no_interior_cut"]).annotate({
  "identifier": "UnusableAnchorsTrace"
});
export type WordTierTrace = "main_tier" | "wor";
export const WordTierTrace: Schema.Codec<WordTierTrace> = Schema.Literals(["main_tier", "wor"]).annotate({
  "identifier": "WordTierTrace"
});
export type OffRecordPostcode = "diary";
export const OffRecordPostcode: Schema.Codec<OffRecordPostcode> = Schema.Literal("diary").annotate({
  "identifier": "OffRecordPostcode"
});
export type DisplayPath = string;
export const DisplayPath: Schema.Codec<DisplayPath> = Schema.String.annotate({
  "identifier": "DisplayPath"
});
export type FileProgressStage = "processing" | "reading" | "resolving_audio" | "recovering_utterance_timing" | "recovering_timing_fallback" | "preparing_alignment" | "aligning" | "transcribing" | "benchmarking" | "checking_cache" | "applying_results" | "post_processing" | "building_chat" | "segmenting_utterances" | "analyzing_morphosyntax" | "finalizing" | "writing" | "parsing" | "analyzing" | "segmenting" | "translating" | "resolving_coreference" | "comparing" | "retry_scheduled" | "waiting_for_worker" | "waiting_for_host_memory" | "preparing_worker" | "downloading_model";
export const FileProgressStage: Schema.Codec<FileProgressStage> = Schema.Literals(["processing", "reading", "resolving_audio", "recovering_utterance_timing", "recovering_timing_fallback", "preparing_alignment", "aligning", "transcribing", "benchmarking", "checking_cache", "applying_results", "post_processing", "building_chat", "segmenting_utterances", "analyzing_morphosyntax", "finalizing", "writing", "parsing", "analyzing", "segmenting", "translating", "resolving_coreference", "comparing", "retry_scheduled", "waiting_for_worker", "waiting_for_host_memory", "preparing_worker", "downloading_model"]).annotate({
  "identifier": "FileProgressStage"
});
export type ReviewBlock = {
  readonly "headline": string;
  readonly "lines"?: ReadonlyArray<string>;
} & {
  readonly [x: string]: Schema.Json;
};
export const ReviewBlock: Schema.Codec<ReviewBlock> = Schema.StructWithRest(Schema.Struct({
  "headline": Schema.String,
  "lines": Schema.optionalKey(Schema.Array(Schema.String))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "ReviewBlock"
});
export type FileStampOutcome = {
  readonly "kind": "unrecorded";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "command": string;
  readonly "kind": "stamped";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "command": string;
  readonly "kind": "not_stamped";
  readonly "reason": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const FileStampOutcome: Schema.Codec<FileStampOutcome> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("unrecorded")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "command": Schema.String,
  "kind": Schema.Literal("stamped")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "command": Schema.String,
  "kind": Schema.Literal("not_stamped"),
  "reason": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "FileStampOutcome"
});
export type FileStatusKind = "queued" | "processing" | "done" | "diagnosed" | "error" | "interrupted";
export const FileStatusKind: Schema.Codec<FileStatusKind> = Schema.Literals(["queued", "processing", "done", "diagnosed", "error", "interrupted"]).annotate({
  "identifier": "FileStatusKind"
});
export type LanguageSpec = string;
export const LanguageSpec: Schema.Codec<LanguageSpec> = Schema.String.annotate({
  "identifier": "LanguageSpec"
});
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" | "writeback_failed";
export const JobStatus: Schema.Codec<JobStatus> = Schema.Literals(["queued", "running", "completed", "failed", "cancelled", "interrupted", "writeback_failed"]).annotate({
  "identifier": "JobStatus"
});
export type HostMemoryPressureLevel = "healthy" | "guarded" | "constrained" | "critical";
export const HostMemoryPressureLevel: Schema.Codec<HostMemoryPressureLevel> = Schema.Literals(["healthy", "guarded", "constrained", "critical"]).annotate({
  "identifier": "HostMemoryPressureLevel"
});
export type MemoryMb = number;
export const MemoryMb: Schema.Codec<MemoryMb> = Schema.Number.annotate({
  "format": "int64"
}).check(Schema.isInt().annotate({
  "expected": "an integer"
})).check(Schema.isGreaterThanOrEqualTo(0).annotate({
  "expected": "a value greater than or equal to 0",
  "identifier": "MemoryMb"
}));
export type RegistryWorkerRefusal = {
  readonly "kind": "foreign_build";
  readonly "reported_build": string;
  readonly "server_build": string;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "unreported_build";
  readonly "server_build": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const RegistryWorkerRefusal: Schema.Codec<RegistryWorkerRefusal> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("foreign_build"),
  "reported_build": Schema.String,
  "server_build": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("unreported_build"),
  "server_build": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "RegistryWorkerRefusal"
});
export type HealthStatus = "ok";
export const HealthStatus: Schema.Codec<HealthStatus> = Schema.Literal("ok").annotate({
  "identifier": "HealthStatus"
});
export type EngineReportAdmissionError = {
  readonly "kind": "missing_engine_report";
  readonly "task": string;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "unadvertised_engine_report";
  readonly "task": string;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "engine_named_for_non_fa_task";
  readonly "task": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const EngineReportAdmissionError: Schema.Codec<EngineReportAdmissionError> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("missing_engine_report"),
  "task": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("unadvertised_engine_report"),
  "task": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("engine_named_for_non_fa_task"),
  "task": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "EngineReportAdmissionError"
});
export type Sha256Digest = string;
export const Sha256Digest: Schema.Codec<Sha256Digest> = Schema.String.annotate({
  "identifier": "Sha256Digest"
});
export type StatusMessageResponse = {
  readonly "message": string;
  readonly "status": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const StatusMessageResponse: Schema.Codec<StatusMessageResponse> = Schema.StructWithRest(Schema.Struct({
  "message": Schema.String,
  "status": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "StatusMessageResponse"
});
export type LeaseRecord = {
  readonly "expires_at": MachineTime;
  readonly "heartbeat_at": MachineTime;
  readonly "leased_by_node": NodeId;
} & {
  readonly [x: string]: Schema.Json;
};
export const LeaseRecord: Schema.Codec<LeaseRecord> = Schema.StructWithRest(Schema.Struct({
  "expires_at": MachineTime,
  "heartbeat_at": MachineTime,
  "leased_by_node": NodeId
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "LeaseRecord"
});
export type JobControlPlaneInfo = {
  readonly "backend": JobControlPlaneBackendKind;
} & {
  readonly [x: string]: Schema.Json;
};
export const JobControlPlaneInfo: Schema.Codec<JobControlPlaneInfo> = Schema.StructWithRest(Schema.Struct({
  "backend": JobControlPlaneBackendKind
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "JobControlPlaneInfo"
});
export type ExecutionPlan = {
  readonly "execution_host": string;
  readonly "mode": ExecutionMode;
  readonly "remote_job_id"?: JobId | null;
  readonly "stage": ExecutionStage;
} & {
  readonly [x: string]: Schema.Json;
};
export const ExecutionPlan: Schema.Codec<ExecutionPlan> = Schema.StructWithRest(Schema.Struct({
  "execution_host": Schema.String,
  "mode": ExecutionMode,
  "remote_job_id": Schema.optionalKey(Schema.Union([JobId, Schema.Null], {
    mode: "oneOf"
  })),
  "stage": ExecutionStage
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "ExecutionPlan"
});
export type OutputFindingRecord = {
  readonly "code"?: string | null;
  readonly "level": FindingLevel;
  readonly "message": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const OutputFindingRecord: Schema.Codec<OutputFindingRecord> = Schema.StructWithRest(Schema.Struct({
  "code": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "level": FindingLevel,
  "message": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "OutputFindingRecord"
});
export type SecondaryMorphologyUnplacedUtterance = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "cause": SecondaryExtractionFailure;
  readonly "detail": string;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const SecondaryMorphologyUnplacedUtterance: Schema.Codec<SecondaryMorphologyUnplacedUtterance> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": SecondaryExtractionFailure,
  "detail": Schema.String,
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "SecondaryMorphologyUnplacedUtterance"
});
export type PrimaryMorphologyDeclinedUtterance = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "cause": PrimaryMorphologyDeclineCause;
  readonly "detail": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const PrimaryMorphologyDeclinedUtterance: Schema.Codec<PrimaryMorphologyDeclinedUtterance> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": PrimaryMorphologyDeclineCause,
  "detail": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "PrimaryMorphologyDeclinedUtterance"
});
export type StoredSecondaryMorphologyFailureKind = SecondaryMorphologyFailureKind | RetiredSecondaryMorphologyFailureKind;
export const StoredSecondaryMorphologyFailureKind: Schema.Codec<StoredSecondaryMorphologyFailureKind> = Schema.Union([SecondaryMorphologyFailureKind, RetiredSecondaryMorphologyFailureKind], {
  mode: "oneOf"
}).annotate({
  "identifier": "StoredSecondaryMorphologyFailureKind"
});
export type FaPreparationRefusalV2 = {
  readonly "engine": string;
  readonly "kind": "model_constraint";
  readonly "maximum": number;
  readonly "provided": number;
  readonly "unit": FaConstraintUnitV2;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "engine": string;
  readonly "kind": "insufficient_input";
  readonly "minimum": number;
  readonly "provided": number;
  readonly "unit": FaConstraintUnitV2;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "projection_unavailable";
  readonly "message": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const FaPreparationRefusalV2: Schema.Codec<FaPreparationRefusalV2> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "engine": Schema.String,
  "kind": Schema.Literal("model_constraint"),
  "maximum": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "provided": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "unit": FaConstraintUnitV2
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "engine": Schema.String,
  "kind": Schema.Literal("insufficient_input"),
  "minimum": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "provided": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "unit": FaConstraintUnitV2
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("projection_unavailable"),
  "message": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "FaPreparationRefusalV2"
});
export type TrimmedWordTimingTrace = {
  readonly "end_ms": number;
  readonly "measured_end_ms": number;
  readonly "measured_start_ms": number;
  readonly "start_ms": number;
  readonly "tier": WordTierTrace;
  readonly "word_index": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const TrimmedWordTimingTrace: Schema.Codec<TrimmedWordTimingTrace> = Schema.StructWithRest(Schema.Struct({
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "measured_end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "measured_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "tier": WordTierTrace,
  "word_index": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "TrimmedWordTimingTrace"
});
export type DroppedWordTimingTrace = {
  readonly "end_ms": number;
  readonly "start_ms": number;
  readonly "tier": WordTierTrace;
  readonly "word_index": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const DroppedWordTimingTrace: Schema.Codec<DroppedWordTimingTrace> = Schema.StructWithRest(Schema.Struct({
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "tier": WordTierTrace,
  "word_index": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "DroppedWordTimingTrace"
});
export type RetiredUntimedCause = {
  readonly "kind": "no_usable_timing";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "not_in_recording";
  readonly "postcode": OffRecordPostcode;
} & {
  readonly [x: string]: Schema.Json;
};
export const RetiredUntimedCause: Schema.Codec<RetiredUntimedCause> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("no_usable_timing")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("not_in_recording"),
  "postcode": OffRecordPostcode
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "RetiredUntimedCause"
});
export type ExcludedUtteranceRecord = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "postcode": OffRecordPostcode;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const ExcludedUtteranceRecord: Schema.Codec<ExcludedUtteranceRecord> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "postcode": OffRecordPostcode,
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "ExcludedUtteranceRecord"
});
export type FileReview = {
  readonly "left_out"?: ReviewBlock | null;
  readonly "needs_review"?: ReviewBlock | null;
} & {
  readonly [x: string]: Schema.Json;
};
export const FileReview: Schema.Codec<FileReview> = Schema.StructWithRest(Schema.Struct({
  "left_out": Schema.optionalKey(Schema.Union([ReviewBlock, Schema.Null], {
    mode: "oneOf"
  })),
  "needs_review": Schema.optionalKey(Schema.Union([ReviewBlock, Schema.Null], {
    mode: "oneOf"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "FileReview"
});
export type SystemMemoryReading = {
  readonly "available_mb": MemoryMb;
  readonly "total_mb": MemoryMb;
  readonly "used_mb": MemoryMb;
} & {
  readonly [x: string]: Schema.Json;
};
export const SystemMemoryReading: Schema.Codec<SystemMemoryReading> = Schema.StructWithRest(Schema.Struct({
  "available_mb": MemoryMb,
  "total_mb": MemoryMb,
  "used_mb": MemoryMb
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "SystemMemoryReading"
});
export type RefusedRegistryWorker = {
  readonly "pid": number;
  readonly "reason": RegistryWorkerRefusal;
  readonly "worker_key": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const RefusedRegistryWorker: Schema.Codec<RefusedRegistryWorker> = Schema.StructWithRest(Schema.Struct({
  "pid": Schema.Number.annotate({
    "format": "int32"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "reason": RegistryWorkerRefusal,
  "worker_key": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "RefusedRegistryWorker"
});
export type CapabilityAdmissionOutcome = {
  readonly "infer_tasks": ReadonlyArray<string>;
  readonly "kind": "admitted";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "refused";
  readonly "reason": EngineReportAdmissionError;
} & {
  readonly [x: string]: Schema.Json;
};
export const CapabilityAdmissionOutcome: Schema.Codec<CapabilityAdmissionOutcome> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "infer_tasks": Schema.Array(Schema.String),
  "kind": Schema.Literal("admitted")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("refused"),
  "reason": EngineReportAdmissionError
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "CapabilityAdmissionOutcome"
});
export type WorkerRuntimeIdentity = {
  readonly "batchalign_core_extension_sha256": Sha256Digest;
  readonly "batchalign_package_tree_sha256": Sha256Digest;
  readonly "distribution_inventory_sha256": Sha256Digest;
  readonly "python_executable_sha256": Sha256Digest;
  readonly "python_version": string;
  readonly "schema_version": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const WorkerRuntimeIdentity: Schema.Codec<WorkerRuntimeIdentity> = Schema.StructWithRest(Schema.Struct({
  "batchalign_core_extension_sha256": Sha256Digest,
  "batchalign_package_tree_sha256": Sha256Digest,
  "distribution_inventory_sha256": Sha256Digest,
  "python_executable_sha256": Sha256Digest,
  "python_version": Schema.String,
  "schema_version": Schema.Number.annotate({
    "format": "int32"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "WorkerRuntimeIdentity"
});
export type JobListItem = {
  readonly "active_lease"?: LeaseRecord | null;
  readonly "command": ReleasedCommand;
  readonly "completed_at"?: MachineTime | null;
  readonly "completed_files": number;
  readonly "control_plane"?: JobControlPlaneInfo | null;
  readonly "diagnosed_files"?: number;
  readonly "duration_s"?: NonNegativeSeconds | null;
  readonly "error"?: string | null;
  readonly "error_files"?: number;
  readonly "job_id": JobId;
  readonly "lang": LanguageSpec;
  readonly "next_eligible_at"?: MachineTime | null;
  readonly "num_workers"?: number | null;
  readonly "source_dir"?: string;
  readonly "status": JobStatus;
  readonly "submitted_at": MachineTime;
  readonly "submitted_by"?: string | null;
  readonly "submitted_by_name"?: string | null;
  readonly "total_files": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const JobListItem: Schema.Codec<JobListItem> = Schema.StructWithRest(Schema.Struct({
  "active_lease": Schema.optionalKey(Schema.Union([LeaseRecord, Schema.Null], {
    mode: "oneOf"
  })),
  "command": ReleasedCommand,
  "completed_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "completed_files": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })),
  "control_plane": Schema.optionalKey(Schema.Union([JobControlPlaneInfo, Schema.Null], {
    mode: "oneOf"
  })),
  "diagnosed_files": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "duration_s": Schema.optionalKey(Schema.Union([NonNegativeSeconds, Schema.Null], {
    mode: "oneOf"
  })),
  "error": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "error_files": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "job_id": JobId,
  "lang": LanguageSpec,
  "next_eligible_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "num_workers": Schema.optionalKey(Schema.Union([Schema.Int, Schema.Null]).annotate({
    "format": "int64"
  })),
  "source_dir": Schema.optionalKey(Schema.String),
  "status": JobStatus,
  "submitted_at": MachineTime,
  "submitted_by": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "submitted_by_name": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "total_files": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "JobListItem"
});
export type JudgedFindingsRecord = {
  readonly "bar": JudgementBar;
  readonly "finding_count": number;
  readonly "findings_by_code"?: ReadonlyArray<FindingCodeCount>;
  readonly "first_findings"?: ReadonlyArray<OutputFindingRecord>;
  readonly "full_findings": FullFindings;
} & {
  readonly [x: string]: Schema.Json;
};
export const JudgedFindingsRecord: Schema.Codec<JudgedFindingsRecord> = Schema.StructWithRest(Schema.Struct({
  "bar": JudgementBar,
  "finding_count": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "findings_by_code": Schema.optionalKey(Schema.Array(FindingCodeCount)),
  "first_findings": Schema.optionalKey(Schema.Array(OutputFindingRecord)),
  "full_findings": FullFindings
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "JudgedFindingsRecord"
});
export type StageRefusalRecord = {
  readonly "bar": JudgementBar;
  readonly "finding_count": number;
  readonly "findings_by_code"?: ReadonlyArray<FindingCodeCount>;
  readonly "first_findings": ReadonlyArray<OutputFindingRecord>;
  readonly "kind": "judged";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "unestablished";
  readonly "reason": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const StageRefusalRecord: Schema.Codec<StageRefusalRecord> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "bar": JudgementBar,
  "finding_count": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "findings_by_code": Schema.optionalKey(Schema.Array(FindingCodeCount)),
  "first_findings": Schema.Array(OutputFindingRecord),
  "kind": Schema.Literal("judged")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("unestablished"),
  "reason": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "StageRefusalRecord"
});
export type SecondaryMorphologyFailure = {
  readonly "cause": StoredSecondaryMorphologyFailureKind;
  readonly "detail": string;
  readonly "span": SecondaryMorphologySpan;
} & {
  readonly [x: string]: Schema.Json;
};
export const SecondaryMorphologyFailure: Schema.Codec<SecondaryMorphologyFailure> = Schema.StructWithRest(Schema.Struct({
  "cause": StoredSecondaryMorphologyFailureKind,
  "detail": Schema.String,
  "span": SecondaryMorphologySpan
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "SecondaryMorphologyFailure"
});
export type FaResourceConstraintV2 = {
  readonly "kind": "empty_decoded_audio";
  readonly "shortfall": DecodedAudioShortfallV2;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "cause": FaPreparationRefusalV2;
  readonly "kind": "model";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "ceiling_mb": number;
  readonly "engine": string;
  readonly "kind": "host_memory";
  readonly "required_mb": number;
  readonly "reserve_mb": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const FaResourceConstraintV2: Schema.Codec<FaResourceConstraintV2> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("empty_decoded_audio"),
  "shortfall": DecodedAudioShortfallV2
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "cause": FaPreparationRefusalV2,
  "kind": Schema.Literal("model")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "ceiling_mb": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "engine": Schema.String,
  "kind": Schema.Literal("host_memory"),
  "required_mb": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "reserve_mb": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "FaResourceConstraintV2"
});
export type RemovalEffectTrace = {
  readonly "following_line_idx": number;
  readonly "following_speaker": string;
  readonly "following_start_ms": number;
  readonly "kind": "start_ahead_stripped";
  readonly "utterance_start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "start_regression_stripped";
  readonly "preceding_line_idx": number;
  readonly "preceding_speaker": string;
  readonly "preceding_start_ms": number;
  readonly "utterance_start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "start_regression_raised";
  readonly "preceding_line_idx": number;
  readonly "preceding_speaker": string;
  readonly "preceding_start_ms": number;
  readonly "utterance_start_ms": number;
  readonly "words_trimmed": ReadonlyArray<TrimmedWordTimingTrace>;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "zero_duration_clamp_stripped";
  readonly "next_line_idx": number;
  readonly "next_start_ms": number;
  readonly "utterance_start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "bound_ms": number;
  readonly "kind": "end_clamped_interleaved_words";
  readonly "next_line_idx": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kept_boundary_ms": number;
  readonly "kept_line_idx": number;
  readonly "kept_speaker": string;
  readonly "kind": "yielded_to_kept_bullet_stripped";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kept_end_ms": number;
  readonly "kept_line_idx": number;
  readonly "kept_speaker": string;
  readonly "kind": "yielded_to_kept_end_leading_words_dropped";
} & {
  readonly [x: string]: Schema.Json;
};
export const RemovalEffectTrace: Schema.Codec<RemovalEffectTrace> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "following_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "following_speaker": Schema.String,
  "following_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kind": Schema.Literal("start_ahead_stripped"),
  "utterance_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("start_regression_stripped"),
  "preceding_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "preceding_speaker": Schema.String,
  "preceding_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "utterance_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("start_regression_raised"),
  "preceding_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "preceding_speaker": Schema.String,
  "preceding_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "utterance_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "words_trimmed": Schema.Array(TrimmedWordTimingTrace)
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("zero_duration_clamp_stripped"),
  "next_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "next_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "utterance_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "bound_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kind": Schema.Literal("end_clamped_interleaved_words"),
  "next_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kept_boundary_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kept_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kept_speaker": Schema.String,
  "kind": Schema.Literal("yielded_to_kept_bullet_stripped")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kept_end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kept_line_idx": Schema.Int.check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kept_speaker": Schema.String,
  "kind": Schema.Literal("yielded_to_kept_end_leading_words_dropped")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "RemovalEffectTrace"
});
export type OutputExclusionRecord = {
  readonly "first_spans": ReadonlyArray<SecondaryMorphologySpan>;
  readonly "kind": "unsupported_secondary_morphology";
  readonly "spans": number;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "excluded_utterances": number;
  readonly "excluded_words": number;
  readonly "first_excluded": ReadonlyArray<ExcludedUtteranceRecord>;
  readonly "kind": "not_in_recording";
} & {
  readonly [x: string]: Schema.Json;
};
export const OutputExclusionRecord: Schema.Codec<OutputExclusionRecord> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "first_spans": Schema.Array(SecondaryMorphologySpan),
  "kind": Schema.Literal("unsupported_secondary_morphology"),
  "spans": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "excluded_utterances": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "excluded_words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "first_excluded": Schema.Array(ExcludedUtteranceRecord),
  "kind": Schema.Literal("not_in_recording")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "OutputExclusionRecord"
});
export type WorkerCapabilityAdmission = {
  readonly "outcome": CapabilityAdmissionOutcome;
  readonly "worker_key": string;
} & {
  readonly [x: string]: Schema.Json;
};
export const WorkerCapabilityAdmission: Schema.Codec<WorkerCapabilityAdmission> = Schema.StructWithRest(Schema.Struct({
  "outcome": CapabilityAdmissionOutcome,
  "worker_key": Schema.String
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "WorkerCapabilityAdmission"
});
export type RefusedWindowTrace = {
  readonly "cause": "resource";
  readonly "constraint": FaResourceConstraintV2;
  readonly "end_ms": number;
  readonly "start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "budget_ms": number;
  readonly "cause": "over_budget";
  readonly "end_ms": number;
  readonly "start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "at_ms": number;
  readonly "cause": "empty";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "cause": "inverted";
  readonly "end_ms": number;
  readonly "start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "cause": "past_recording";
  readonly "end_ms": number;
  readonly "exceeds_by_ms": number;
  readonly "start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "budget_ms": number;
  readonly "cause": "anchor_gap";
  readonly "end_ms": number;
  readonly "gap_end_ms": number;
  readonly "gap_start_ms": number;
  readonly "start_ms": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "budget_ms": number;
  readonly "cause": "anchors_unusable";
  readonly "end_ms": number;
  readonly "start_ms": number;
  readonly "unusable": UnusableAnchorsTrace;
} & {
  readonly [x: string]: Schema.Json;
};
export const RefusedWindowTrace: Schema.Codec<RefusedWindowTrace> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "cause": Schema.Literal("resource"),
  "constraint": FaResourceConstraintV2,
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "budget_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": Schema.Literal("over_budget"),
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "at_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": Schema.Literal("empty")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "cause": Schema.Literal("inverted"),
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "cause": Schema.Literal("past_recording"),
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "exceeds_by_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "budget_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": Schema.Literal("anchor_gap"),
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "gap_end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "gap_start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "budget_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": Schema.Literal("anchors_unusable"),
  "end_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "start_ms": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "unusable": UnusableAnchorsTrace
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "RefusedWindowTrace"
});
export type WordTimingRemovalTrace = {
  readonly "effect": RemovalEffectTrace;
  readonly "measured": ReadonlyArray<DroppedWordTimingTrace>;
} & {
  readonly [x: string]: Schema.Json;
};
export const WordTimingRemovalTrace: Schema.Codec<WordTimingRemovalTrace> = Schema.StructWithRest(Schema.Struct({
  "effect": RemovalEffectTrace,
  "measured": Schema.Array(DroppedWordTimingTrace)
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "WordTimingRemovalTrace"
});
export type HealthResponse = {
  readonly "active_jobs"?: number;
  readonly "attempts_retried"?: number;
  readonly "attempts_started"?: number;
  readonly "build_hash"?: string;
  readonly "cache_backend"?: string;
  readonly "capabilities"?: ReadonlyArray<string>;
  readonly "deferred_work_units"?: number;
  readonly "forced_terminal_errors"?: number;
  readonly "free_threaded"?: boolean;
  readonly "host_memory_active_leases"?: ReadonlyArray<string>;
  readonly "host_memory_error"?: string | null;
  readonly "host_memory_job_leases"?: number;
  readonly "host_memory_ml_test_locks"?: number;
  readonly "host_memory_pressure"?: HostMemoryPressureLevel;
  readonly "host_memory_reserved_mb"?: MemoryMb;
  readonly "host_memory_startup_leases"?: number;
  readonly "job_slots_available"?: number;
  readonly "live_worker_keys"?: ReadonlyArray<string>;
  readonly "live_workers"?: number;
  readonly "loaded_pipelines"?: ReadonlyArray<string>;
  readonly "media_mapping_keys"?: ReadonlyArray<string>;
  readonly "media_roots"?: ReadonlyArray<string>;
  readonly "memory_gate_aborts"?: number;
  readonly "memory_gate_threshold_mb"?: MemoryMb;
  readonly "node_id"?: NodeId;
  readonly "refused_registry_workers"?: ReadonlyArray<RefusedRegistryWorker>;
  readonly "status": HealthStatus;
  readonly "system_memory"?: SystemMemoryReading | null;
  readonly "version": string;
  readonly "worker_capability_admissions"?: ReadonlyArray<WorkerCapabilityAdmission>;
  readonly "worker_crashes"?: number;
  readonly "worker_runtime_identities"?: ReadonlyArray<WorkerRuntimeIdentity>;
  readonly "workers_available"?: number;
} & {
  readonly [x: string]: Schema.Json;
};
export const HealthResponse: Schema.Codec<HealthResponse> = Schema.StructWithRest(Schema.Struct({
  "active_jobs": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "attempts_retried": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "attempts_started": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "build_hash": Schema.optionalKey(Schema.String),
  "cache_backend": Schema.optionalKey(Schema.String),
  "capabilities": Schema.optionalKey(Schema.Array(Schema.String)),
  "deferred_work_units": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "forced_terminal_errors": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "free_threaded": Schema.optionalKey(Schema.Boolean),
  "host_memory_active_leases": Schema.optionalKey(Schema.Array(Schema.String)),
  "host_memory_error": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "host_memory_job_leases": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "host_memory_ml_test_locks": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "host_memory_pressure": Schema.optionalKey(HostMemoryPressureLevel),
  "host_memory_reserved_mb": Schema.optionalKey(MemoryMb),
  "host_memory_startup_leases": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "job_slots_available": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "live_worker_keys": Schema.optionalKey(Schema.Array(Schema.String)),
  "live_workers": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "loaded_pipelines": Schema.optionalKey(Schema.Array(Schema.String)),
  "media_mapping_keys": Schema.optionalKey(Schema.Array(Schema.String)),
  "media_roots": Schema.optionalKey(Schema.Array(Schema.String)),
  "memory_gate_aborts": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "memory_gate_threshold_mb": Schema.optionalKey(MemoryMb),
  "node_id": Schema.optionalKey(NodeId),
  "refused_registry_workers": Schema.optionalKey(Schema.Array(RefusedRegistryWorker)),
  "status": HealthStatus,
  "system_memory": Schema.optionalKey(Schema.Union([SystemMemoryReading, Schema.Null], {
    mode: "oneOf"
  })),
  "version": Schema.String,
  "worker_capability_admissions": Schema.optionalKey(Schema.Array(WorkerCapabilityAdmission)),
  "worker_crashes": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))),
  "worker_runtime_identities": Schema.optionalKey(Schema.Array(WorkerRuntimeIdentity)),
  "workers_available": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "HealthResponse"
});
export type UntimedCauseRecord = {
  readonly "kind": "window_refused";
  readonly "window": RefusedWindowTrace;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "not_placed";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "localization_unavailable";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "answered": number;
  readonly "kind": "requested";
  readonly "refused"?: number;
  readonly "unaligned": number;
  readonly "unrequested": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "removed";
  readonly "removals": ReadonlyArray<WordTimingRemovalTrace>;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "needs_recording";
  readonly "removals": ReadonlyArray<WordTimingRemovalTrace>;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "unrecorded";
} & {
  readonly [x: string]: Schema.Json;
};
export const UntimedCauseRecord: Schema.Codec<UntimedCauseRecord> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("window_refused"),
  "window": RefusedWindowTrace
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("not_placed")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("localization_unavailable")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "answered": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kind": Schema.Literal("requested"),
  "refused": Schema.optionalKey(Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))),
  "unaligned": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "unrequested": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("removed"),
  "removals": Schema.Array(WordTimingRemovalTrace)
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("needs_recording"),
  "removals": Schema.Array(WordTimingRemovalTrace)
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("unrecorded")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "UntimedCauseRecord"
});
export type StoredUntimedCause = UntimedCauseRecord | RetiredUntimedCause;
export const StoredUntimedCause: Schema.Codec<StoredUntimedCause> = Schema.Union([UntimedCauseRecord, RetiredUntimedCause], {
  mode: "oneOf"
}).annotate({
  "identifier": "StoredUntimedCause"
});
export type UntimedUtteranceRecord = {
  readonly "main_tier"?: string | null;
  readonly "utterance": number;
  readonly "cause": StoredUntimedCause;
  readonly "untimed_words": number;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const UntimedUtteranceRecord: Schema.Codec<UntimedUtteranceRecord> = Schema.StructWithRest(Schema.Struct({
  "main_tier": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "utterance": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "cause": StoredUntimedCause,
  "untimed_words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "UntimedUtteranceRecord"
});
export type OutputShortfallRecord = {
  readonly "category": FailureCategory;
  readonly "detail": string;
  readonly "engine": string;
  readonly "kind": "utr_unavailable";
  readonly "pass": UtrPass;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "detail": string;
  readonly "kind": "morphology_provenance_incomplete";
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_utterances": ReadonlyArray<SecondaryMorphologyUnplacedUtterance>;
  readonly "kind": "secondary_morphology_unplaced";
  readonly "utterances": number;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_utterances": ReadonlyArray<PrimaryMorphologyDeclinedUtterance>;
  readonly "kind": "primary_morphology_incomplete";
  readonly "utterances": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_spans": ReadonlyArray<SecondaryMorphologyFailure>;
  readonly "kind": "secondary_morphology_incomplete";
  readonly "spans": number;
  readonly "words": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "stage_skipped";
  readonly "stage": OptionalStage;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "kind": "stage_not_applied";
  readonly "refusal": StageRefusalRecord;
  readonly "stage": OptionalStage;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_held_out": ReadonlyArray<StoredUtteranceLocation>;
  readonly "held_out_utterances": number;
  readonly "kind": "stage_held_out";
  readonly "stage": OptionalStage;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_seams": ReadonlyArray<WindowSeamRecord>;
  readonly "kind": "window_seam_ambiguous";
  readonly "seams": number;
} & {
  readonly [x: string]: Schema.Json;
} | {
  readonly "first_untimed": ReadonlyArray<UntimedUtteranceRecord>;
  readonly "kind": "timing_incomplete";
  readonly "required_words": number;
  readonly "untimed_utterances": number;
  readonly "untimed_words": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const OutputShortfallRecord: Schema.Codec<OutputShortfallRecord> = Schema.Union([Schema.StructWithRest(Schema.Struct({
  "category": FailureCategory,
  "detail": Schema.String,
  "engine": Schema.String,
  "kind": Schema.Literal("utr_unavailable"),
  "pass": UtrPass
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "detail": Schema.String,
  "kind": Schema.Literal("morphology_provenance_incomplete")
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_utterances": Schema.Array(SecondaryMorphologyUnplacedUtterance),
  "kind": Schema.Literal("secondary_morphology_unplaced"),
  "utterances": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_utterances": Schema.Array(PrimaryMorphologyDeclinedUtterance),
  "kind": Schema.Literal("primary_morphology_incomplete"),
  "utterances": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_spans": Schema.Array(SecondaryMorphologyFailure),
  "kind": Schema.Literal("secondary_morphology_incomplete"),
  "spans": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("stage_skipped"),
  "stage": OptionalStage
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "kind": Schema.Literal("stage_not_applied"),
  "refusal": StageRefusalRecord,
  "stage": OptionalStage
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_held_out": Schema.Array(StoredUtteranceLocation),
  "held_out_utterances": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "kind": Schema.Literal("stage_held_out"),
  "stage": OptionalStage
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_seams": Schema.Array(WindowSeamRecord),
  "kind": Schema.Literal("window_seam_ambiguous"),
  "seams": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]), Schema.StructWithRest(Schema.Struct({
  "first_untimed": Schema.Array(UntimedUtteranceRecord),
  "kind": Schema.Literal("timing_incomplete"),
  "required_words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "untimed_utterances": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  })),
  "untimed_words": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })).check(Schema.isGreaterThanOrEqualTo(0).annotate({
    "expected": "a value greater than or equal to 0"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))])], {
  mode: "oneOf"
}).annotate({
  "identifier": "OutputShortfallRecord"
});
export type FileOutputDiagnostics = {
  readonly "findings"?: JudgedFindingsRecord | null;
  readonly "shortfalls"?: ReadonlyArray<OutputShortfallRecord>;
} & {
  readonly [x: string]: Schema.Json;
};
export const FileOutputDiagnostics: Schema.Codec<FileOutputDiagnostics> = Schema.StructWithRest(Schema.Struct({
  "findings": Schema.optionalKey(Schema.Union([JudgedFindingsRecord, Schema.Null], {
    mode: "oneOf"
  })),
  "shortfalls": Schema.optionalKey(Schema.Array(OutputShortfallRecord))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "FileOutputDiagnostics"
});
export type FileStatusEntry = {
  readonly "diagnostics"?: FileOutputDiagnostics | null;
  readonly "duration_s"?: NonNegativeSeconds | null;
  readonly "error"?: string | null;
  readonly "error_category"?: FailureCategory | null;
  readonly "exclusions"?: ReadonlyArray<OutputExclusionRecord>;
  readonly "filename": DisplayPath;
  readonly "finished_at"?: MachineTime | null;
  readonly "next_eligible_at"?: MachineTime | null;
  readonly "progress_current"?: number | null;
  readonly "progress_label"?: string | null;
  readonly "progress_stage"?: FileProgressStage | null;
  readonly "progress_total"?: number | null;
  readonly "review"?: FileReview;
  readonly "stamp"?: FileStampOutcome;
  readonly "started_at"?: MachineTime | null;
  readonly "status": FileStatusKind;
} & {
  readonly [x: string]: Schema.Json;
};
export const FileStatusEntry: Schema.Codec<FileStatusEntry> = Schema.StructWithRest(Schema.Struct({
  "diagnostics": Schema.optionalKey(Schema.Union([FileOutputDiagnostics, Schema.Null], {
    mode: "oneOf"
  })),
  "duration_s": Schema.optionalKey(Schema.Union([NonNegativeSeconds, Schema.Null], {
    mode: "oneOf"
  })),
  "error": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "error_category": Schema.optionalKey(Schema.Union([FailureCategory, Schema.Null], {
    mode: "oneOf"
  })),
  "exclusions": Schema.optionalKey(Schema.Array(OutputExclusionRecord)),
  "filename": DisplayPath,
  "finished_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "next_eligible_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "progress_current": Schema.optionalKey(Schema.Union([Schema.Int, Schema.Null]).annotate({
    "format": "int64"
  })),
  "progress_label": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "progress_stage": Schema.optionalKey(Schema.Union([FileProgressStage, Schema.Null], {
    mode: "oneOf"
  })),
  "progress_total": Schema.optionalKey(Schema.Union([Schema.Int, Schema.Null]).annotate({
    "format": "int64"
  })),
  "review": Schema.optionalKey(FileReview),
  "stamp": Schema.optionalKey(FileStampOutcome),
  "started_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "status": FileStatusKind
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "FileStatusEntry"
});
export type JobInfo = {
  readonly "active_lease"?: LeaseRecord | null;
  readonly "command": ReleasedCommand;
  readonly "completed_at"?: MachineTime | null;
  readonly "completed_files": number;
  readonly "control_plane"?: JobControlPlaneInfo | null;
  readonly "current_file"?: string | null;
  readonly "duration_s"?: NonNegativeSeconds | null;
  readonly "error"?: string | null;
  readonly "execution_plan"?: ExecutionPlan | null;
  readonly "file_statuses"?: ReadonlyArray<FileStatusEntry>;
  readonly "job_id": JobId;
  readonly "lang": LanguageSpec;
  readonly "last_cancelled_at"?: MachineTime | null;
  readonly "last_cancelled_host"?: string | null;
  readonly "last_cancelled_reason"?: string | null;
  readonly "last_cancelled_source"?: string | null;
  readonly "next_eligible_at"?: MachineTime | null;
  readonly "num_workers"?: number | null;
  readonly "options": Schema.Json;
  readonly "source_dir"?: string;
  readonly "status": JobStatus;
  readonly "submitted_at": MachineTime;
  readonly "submitted_by"?: string | null;
  readonly "submitted_by_name"?: string | null;
  readonly "total_files": number;
} & {
  readonly [x: string]: Schema.Json;
};
export const JobInfo: Schema.Codec<JobInfo> = Schema.StructWithRest(Schema.Struct({
  "active_lease": Schema.optionalKey(Schema.Union([LeaseRecord, Schema.Null], {
    mode: "oneOf"
  })),
  "command": ReleasedCommand,
  "completed_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "completed_files": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  })),
  "control_plane": Schema.optionalKey(Schema.Union([JobControlPlaneInfo, Schema.Null], {
    mode: "oneOf"
  })),
  "current_file": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "duration_s": Schema.optionalKey(Schema.Union([NonNegativeSeconds, Schema.Null], {
    mode: "oneOf"
  })),
  "error": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "execution_plan": Schema.optionalKey(Schema.Union([ExecutionPlan, Schema.Null], {
    mode: "oneOf"
  })),
  "file_statuses": Schema.optionalKey(Schema.Array(FileStatusEntry)),
  "job_id": JobId,
  "lang": LanguageSpec,
  "last_cancelled_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "last_cancelled_host": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "last_cancelled_reason": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "last_cancelled_source": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "next_eligible_at": Schema.optionalKey(Schema.Union([MachineTime, Schema.Null], {
    mode: "oneOf"
  })),
  "num_workers": Schema.optionalKey(Schema.Union([Schema.Int, Schema.Null]).annotate({
    "format": "int64"
  })),
  "options": Schema.Json.annotate({
    "expected": "JSON value"
  }),
  "source_dir": Schema.optionalKey(Schema.String),
  "status": JobStatus,
  "submitted_at": MachineTime,
  "submitted_by": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "submitted_by_name": Schema.optionalKey(Schema.Union([Schema.String, Schema.Null])),
  "total_files": Schema.Number.annotate({
    "format": "int64"
  }).check(Schema.isInt().annotate({
    "expected": "an integer"
  }))
}), [Schema.Record(Schema.String, Schema.Json.annotate({
  "expected": "JSON value"
}))]).annotate({
  "identifier": "JobInfo"
});
