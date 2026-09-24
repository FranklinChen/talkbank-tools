# Evidence, Replay, and Experiment Topology

**Status:** Current
**Last updated:** 2026-09-23 23:31 EDT

This chapter is the visual map for BA3's evidence architecture. Version 0.3.0
has raw-evidence caching and FA evidence schema 2. Version 0.4.0 additionally
has the experimental
`rebuild-from-evidence` and `preserve-cross-speaker` projections plus schema 3
stable utterance ordinals; those additions are not claims about the deployed
v0.3 service. Detailed
contracts remain in [Audio-Task Cache](../../architecture/runtime/audio-task-cache.md),
[Observability](observability.md), and the developer references for
[transcribe](../developer/commands/transcribe.md) and
[align](../developer/commands/align.md).

The central design rule is that acquiring model evidence, projecting that
evidence through local algorithms, and judging transcript quality are separate
operations. BA3 now constrains the first two. Human or corpus-specific
adjudication remains an experiment-layer responsibility.

For forced-alignment reruns, `--existing-wor-boundaries` is one such local
projection dimension. It is intentionally downstream of raw evidence and
absent from the cache key. `preserve` is the compatibility default;
`rebuild-from-evidence` keeps fresh word extents and reconstructs main-tier
coverage from their hull. A valid or structurally self-contained result still
requires acoustic adjudication, especially when adjacent utterances overlap
and the later monotonicity pass intervenes.

`--end-overlap-policy` is an independent local projection dimension. Its
compatibility default clamps all adjacent end overlap. The experimental
`preserve-cross-speaker` arm keeps overlap only when adjacent speaker codes
differ; same-speaker clamps and start-regression stripping remain unchanged.
Both policies travel inside one `FaProjectionPolicy`, so full, incremental,
all-reusable, and empty-group paths cannot silently apply different
combinations. `--main-bullets {derive,keep}` is a third dimension in the same
value: `keep` leaves every utterance bullet the input carried unchanged and fits
word timings inside it, again without touching the cache key. A second phase type, `FaFinalized`, requires optional bullet
repair to run before that policy's monotonicity projection on every path.

A cache-only ten-file development experiment held every other typed option
fixed and changed only this policy. All 547 FA groups replayed
from cache. Nine normalized outputs were identical; the tenth retained exactly
one cross-speaker overlap that the compatibility arm had clamped. The ten
same-speaker clamps and seven start-regression removals were unchanged. This
proves causal isolation of the local projection, not that either boundary is
acoustically preferable; a sealed listening experiment remains the promotion
gate.

## Implemented evidence lanes

```mermaid
flowchart TB
    MEDIA["Media bytes"]
    CHATIN["Existing CHAT"]

    subgraph PAID["Remote or model inference boundaries"]
        REV["Rev raw ASR evidence"]
        SPK["Raw speaker evidence"]
        FAW["Raw FA worker evidence"]
        UTW["Boundary-model evidence"]
    end

    subgraph LOCAL["Versioned local projections"]
        ASRP["ASR cleanup and timed chunks"]
        SPKP["Normalized turns and speaker projection"]
        FAP["Word timings, %wor policy,<br/>and typed end-overlap projection"]
        UTP["Pre-CHAT and post-CHAT boundaries"]
    end

    subgraph OUTPUTS["Durable experiment products"]
        CACHE["Content-addressed raw/derived cache"]
        SIDE["Causal evidence sidecars<br/>schema 3: line + utterance identity<br/>utseg evidence: schema 4"]
        REPLAY["Fingerprint-admitted replay bundle"]
        OUT["Validated CHAT"]
    end

    MEDIA --> REV --> ASRP
    MEDIA --> SPK --> SPKP
    MEDIA --> FAW --> FAP
    CHATIN --> FAP
    ASRP --> UTW --> UTP
    ASRP --> SPKP --> UTP --> OUT
    FAP --> OUT

    REV -.-> CACHE
    SPK -.-> CACHE
    FAW -.-> CACHE
    REV -.-> SIDE
    SPK -.-> SIDE
    FAW -.-> SIDE
    UTW -.-> SIDE
    ASRP -.-> REPLAY
    SPKP -.-> REPLAY
    OUT -.-> REPLAY
```

Solid arrows are semantic processing. Dashed arrows are retained evidence or
replay products. Sidecars are files, not CHAT dependent tiers: current BA3 does
not generate `%xalign` or `%xrev`.

Utterance-segmentation sidecars carry their own version, and this build writes
schema 4, in which the boundary model's pinned revision is a required part of
the evidence. `eval utseg-replay` admits only that version and refuses any
other by name rather than migrating it, so every utseg sidecar retained before
this build is schema 3 and cannot be replayed; regenerating it with the current
build is the remedy.

## Inference authorization is a state transition

A cache miss is not permission to call a provider. The resolver must consume
the miss into a single-use authorization, and successful evidence must be
validated and committed durably before projection succeeds.

```mermaid
stateDiagram-v2
    [*] --> WorkerRecipe: derive exact engine recipe
    WorkerRecipe --> SelectedWorker: select/load recipe-specific worker
    SelectedWorker --> RequestIdentity: obtain that worker's live engine identity
    RequestIdentity --> CompletedEvidence: admitted durable hit
    RequestIdentity --> CacheMiss: absent or deliberate refresh
    RequestIdentity --> Refused: corrupt or incompatible evidence
    CacheMiss --> Refused: RequireCache
    CacheMiss --> AuthorizedRun: UseCache or SkipCache
    AuthorizedRun --> VerifiedRun: media digest reverified
    VerifiedRun --> CompletedEvidence: provider/model result validated and committed
    VerifiedRun --> Refused: media drift, invalid result, or commit failure
    CompletedEvidence --> CurrentProjection
    CurrentProjection --> [*]
    Refused --> [*]
```

This is a typestate boundary. Provider adapters cannot manufacture
`AuthorizedRun`, and offline replay cannot accidentally acquire provider-call
capability. A per-job cache key uses the capability of the exact worker selected
for that request. A process-wide availability snapshot is not evidence of which
engine handled a job and cannot enter cache identity. Lazy workers are keyed by
their engine recipe, so requests for different ASR or forced-alignment engines
cannot reuse one process and silently inherit whichever model loaded first.

## Replay has two deliberately different meanings

```mermaid
flowchart LR
    subgraph CACHE_REPLAY["Raw-evidence replay"]
        CR["Validated raw cache envelope"] --> CA["Re-admit against current request"]
        CA --> CP["Run current Rust projection"]
    end

    subgraph BUNDLE_REPLAY["Offline transcribe replay"]
        BM["Immutable manifest"] --> BF{"Verify media and artifact fingerprints"}
        BF -->|match| BA["Admitted projected ASR and turns"]
        BF -->|drift| BX["Refuse before output/model load"]
        BA --> BP["Run current downstream CHAT logic"]
    end
```

Raw-evidence replay can test a changed local normalizer or aligner projection.
The current offline transcribe bundle begins from retained projected ASR and
turn artifacts, so it tests downstream speaker projection, segmentation, CHAT
construction, and postprocessing without claiming that those artifacts are raw
Rev or raw pyannote evidence.

FA schema 3 keeps the input-AST line index for debugging and adds an ordinal
among utterances only for every structured monotonicity effect and its
neighbour. Command provenance can insert an `@Comment` before final
serialization, so a line index alone is not a stable final-output address.
Experiment admission cross-checks the recorded ordinal against the exact input
CHAT and then resolves the same speaker/token identity in output CHAT; a
header-only rewrite succeeds, while an utterance insertion, deletion, reorder,
or lexical drift refuses.

Global UTR has a third, narrower offline replay seam. The
`eval utr-alignment` action consumes an exact clean CHAT document and retained
UTR timing tokens, then emits the typed global word-to-token plan without
inference or CHAT mutation. The plan keeps proposals for already timed lines
even though current production projection preserves their bullets. This makes
joint-boundary and word-prior research possible without confusing observed
alignment evidence with a production policy.

```mermaid
flowchart LR
    UCHAT["Fingerprint UTR input CHAT"]
    UTOK["Fingerprint UTR token JSON"]
    UDP["Global monotone alignment"]
    UM["Typed word matches"]
    UP["All-line timing proposals"]
    UR["Immutable JSON report"]
    POLICY["Separate research policy"]

    UCHAT --> UDP
    UTOK --> UDP
    UDP --> UM --> UP --> UR --> POLICY
```

The replay report is not raw provider evidence and does not establish final
`%wor` timing quality. A downstream experiment must admit input identity,
coverage, lexical relation, and proposal validity before comparing policies.
Reports are serialized completely before a destination is touched, staged in
the destination directory, fsynced, and atomically published without
replacing an existing evidence artifact.

## Utterance segmentation has a reproduction seam

The `eval utseg-replay` action is a fourth seam, and the only one that
reproduces rather than explores. It reapplies the boundary evidence a run
retained and asks whether the current build still produces the document that
run wrote. The evidence and the output are both retained artifacts, so the
answer isolates the local segmentation projection: nothing infers, and the
boundary model never loads.

```mermaid
flowchart LR
    subgraph RETAINED["One run's retained artifacts"]
        UEV["Utseg evidence sidecar<br/>pre-CHAT or post-CHAT"]
        USRC["Input CHAT, or retained ASR response"]
        UOUT["Output CHAT the run wrote"]
    end

    UEV --> UADM{"Admit: schema, pass,<br/>per-item invariants"}
    USRC --> UCOL["Collect requests with<br/>the current build"]
    UADM --> UBIND{"Bind one-to-one"}
    UCOL --> UBIND
    UBIND --> UAPP["Reapply boundaries<br/>through the production transform"]
    UAPP --> UCMP["Compare CHAT semantics,<br/>generated comments set aside"]
    UOUT --> UCMP
    UCMP --> UVERD["Reproduced, or a typed difference"]
```

Two properties make the verdict mean something. The evidence passes the same
admission a live worker result passes, so a sidecar is reapplied only while it
still describes applicable work, and binding proves the retained items describe
the very requests this build collects rather than some other population. What
a run generates to say a run happened, its stamp and the unchecked-ASR warning,
is recognized through the provenance codec that writes it and set aside on both
sides: a timestamp can never match by equality, and comparing it would report
every replay as a difference.

A difference is an outcome, not an error, and what it implicates depends on the
pass. The post-CHAT pass holds everything else fixed: the document is given, the
boundaries are retained, and only the local boundary-application projection runs,
so a difference there does say that projection changed between the build that
wrote the artifact and the build replaying it. The pre-ASR pass rebuilds the
document from the retained response, so ASR post-processing, utterance
retokenization and CHAT construction all run again, and a difference there
implicates any of them until a post-CHAT replay or a narrower probe separates
them. Neither is by itself a verdict about which segmentation is better.

Both passes compare one basis, the AST of the serialized CHAT text, so a
difference always refers to what a run would have written rather than to an
in-memory shape no reader sees.

## Reproducible comparative experiment loop

```mermaid
flowchart TD
    Q["Narrow quality question"] --> C["Frozen troublesome clips<br/>and reference annotations"]
    C --> B["Capture baseline identities,<br/>requests, raw outputs, and CHAT"]
    B --> M["Fingerprint manifest"]
    M --> V1["Projection/policy variant A"]
    M --> V2["Projection/policy variant B"]
    V1 --> CMP["Machine comparison:<br/>words, speakers, boundaries, timings"]
    V2 --> CMP
    CMP --> H["Blind human adjudication<br/>with uncertainty/notes"]
    H --> D{"Evidence supports change?"}
    D -->|yes| T["Regression test + implementation + docs"]
    D -->|no| N["Record negative or inconclusive result"]
    T --> R["Code review and exact-commit gates"]
    R --> M2["New versioned projection identity"]
    M2 --> Q
    N --> Q
```

This loop is the basis for precise comparisons with the upstream BA3 fork and for
segmentation, diarization, Rev-media, and `%wor` studies. A system-level claim
requires the whole chain; a few plausible transcripts do not establish
universal superiority.

## Boundary with IISRP and MichiganChild merge work

The following is the intended downstream research topology, not a feature the
BA3 v0.3 CLI currently performs:

```mermaid
flowchart LR
    MAN["Imperfect child-only<br/>manual CHAT"]
    FULL["BA3 full-audio candidates<br/>words, speakers, boundaries, timings"]
    ACOU["Acoustic signals<br/>pitch, overlap, pauses"]
    SEM["Semantic signals<br/>fuzzy match, echo, lexical context"]
    REC["Typed reconciliation candidates"]
    SCORE["Auditable holistic scoring"]
    AUTO{"Confidence / ambiguity state"}
    MERGE["Merged transcript candidate"]
    REVIEW["Targeted human review"]
    FINAL["Validated delivery CHAT"]

    MAN --> REC
    FULL --> REC
    ACOU --> SCORE
    SEM --> SCORE
    REC --> SCORE --> AUTO
    AUTO -->|high confidence| MERGE
    AUTO -->|ambiguous| REVIEW --> MERGE
    MERGE --> FINAL
```

The manual child transcript is evidence, not an oracle: it may contain `xxx`,
miss adult interruptions, or choose different but defensible utterance
boundaries. The merge layer should therefore preserve competing candidates and
ambiguity until a typed decision is made. BA3 supplies replayable full-audio
evidence; project tooling performs the corpus-specific reconciliation.

## Diagram maintenance rule

When a new cache state, evidence artifact, inference capability, or projection
revision is added, update the smallest detailed diagram and this overview in
the same change. A diagram is part of the contract: if it cannot distinguish
raw evidence from a derived artifact or implemented behavior from planned
research, it is misleading and must not be marked current.
