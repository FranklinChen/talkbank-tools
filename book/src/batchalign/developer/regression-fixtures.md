# Regression Fixtures

**Status:** Current
**Last updated:** 2026-10-06 15:51 EDT

This page describes the per-command regression-fixture system: how it is
laid out, how to add a new fixture when a user reports a bug, and how the
runner verifies it. The intent is to monotonically grow batchalign3's
real-world test surface, every bug a user reports becomes a permanent
regression that catches future drift.

## Why this exists

Batchalign3 has rich unit tests, ML golden tests against fixed audio, and
TUI parity checks against legacy behavior. None of those capture the kind
of bug a real user finds when running the production CLI on a real corpus
file. Alignment quality issues, transcribe glitches, and CJK segmentation
regressions have historically lived in scattered emails and ad-hoc
`/tmp/` experiment folders, and the bug came back as soon as the model or
the surrounding code shifted.

The regression-fixture system fixes that. Each reported bug becomes a
small command-shaped fixture directory under
`test-fixtures/<command>/regressions/<bug-name>/`. A Rust integration
test runs each fixture through the same in-process direct host the
production CLI uses and asserts a structural invariant on the output.
The bug then **cannot** silently regress: any future change that
re-introduces the failure mode will fail the test in CI.

**Note on privacy:** real-user bug reports usually involve real-corpus
audio and transcripts that belong to restricted-access corpora and
cannot be committed to this public repository. The runner in this repo
is generic scaffolding; the fixture content lives in a separate
private repository
[`TalkBank/<private-fixtures>`](https://github.com/TalkBank/<private-fixtures>)
that maintainers clone locally and expose to the runner via the
`BATCHALIGN3_PRIVATE_FIXTURES_DIR` environment variable. The in-tree
`test-fixtures/<command>/regressions/` directories are gitignored
below the per-command README level specifically so private material
cannot land in this public repo by accident. Contributors without
access to the private fixture repository will see the regression
tests skip gracefully rather than fail.

## Directory layout

### Distinguish timing hints, request crops and completed output

Joint lexical recovery has two different search capabilities. A complete
candidate envelope covers all optimal lexical placements. When source words
can be missing, a borrowed partial-order corridor instead uses obligatory words
outside every permitted overlap episode containing that turn, plus the
preserved preceding/following words on the same speaker's chain. The joint matcher
owns the episode policy; adjacent document-order turns are not automatically
temporal bounds. All candidate locations for each obligatory boundary word are
retained, including its full provider interval; an ambiguous boundary does not
become a word anchor. The corridor retains the exact source word census and may end
at the actual recording edge when no following bound is proved. It grants no
word timing, main-tier bullet or output permission. Request containment, engine
budgets and final complete timing remain separate admissions; contradictory or
over-budget search bounds are refused rather than clipped into convenient crops.

The `utr-debug` research report is schema 8. Existing complete candidate
envelopes retain their previous shape. Partial-order envelopes explicitly carry
`floor_ms`, nullable `ceiling_ms` and `scope.kind = "order_corridor"`; a null
ceiling is a recording-bound search obligation, not a made-up timestamp.

UTR hint coverage is an intermediate observation, not the alignment output
contract. A withheld hint must not prevent FA from searching the words' actual
acoustic region. Keep selected lexical matches, common correspondences,
candidate search ranges and producer-admitted complete output distinct.

The deterministic opt-in
`joint_recovery_request_windows_retain_the_complete_source_population` control
checks every planned FA request, including anchored pieces, against the synthetic
fixture producer's word intervals. It retains missing requests, duplicate
requests and intervals excluded by crops as separate failures; scheduling every
word is not proof that its audio was searched. Run it explicitly in a planned
recovery batch rather than adding repeated long probes to routine editing.
An ignored control is unverified, not a pass. Its retained red result identifies
remaining overlap-window truncation; it does not certify acoustic-model accuracy.

Actual managed model outputs still require complete timing, checked CHAT output
admission and independent timing assessment. Synthetic request-window proofs,
test doubles and acoustic-model controls establish different facts. Batch
compatible controls on one reviewed build; reuse admitted source/media and
retained results until a specific contrary fact reopens them.

```text
batchalign3/test-fixtures/
├── README.md                     # convention overview + JSON schema
├── align/
│   ├── README.md
│   └── regressions/              # fixture subdirs are gitignored
│       └── <bug-name>/           # staged locally from a private mirror
│           ├── README.md
│           ├── input.cha         # optional CHAT input for CHAT-first commands
│           ├── input.<ext>       # audio input or sidecar audio, depending on command
│           ├── actual.cha        # current buggy output (reference)
│           └── source.json       # typed manifest
├── transcribe/regressions/       # one per-command root each
├── morphotag/regressions/
├── utseg/regressions/
├── translate/regressions/
└── coref/regressions/
```

Per-bug directory naming convention: opaque slot names such as
`align-regression-004/`, `transcribe-regression-001/`, and so on.
The public test function names and the private fixture directory names
must match so the runner can resolve fixtures without exposing reporter
identity or corpus details in public code. Put the human-readable date,
reporter, and bug context inside the private fixture's `README.md` and
`source.json`, not in the directory name.

## The `source.json` schema

`source.json` is parsed into the typed
`crate::common::regression_manifest::FixtureManifest` struct in the test
binary. Every field has a domain newtype where it earns one. Sample:

```json
{
  "command": "transcribe",
  "language": "eng",
  "audio": "input.mp3",
  "transcribe": {
    "asr_engine": "rev_ai",
    "wor": "omit"
  },
  "source": {
    "report": "<opaque ref to private email thread>",
    "original_chat": "<opaque ref to private source file>",
    "trimmed_utterance_range": [60, 64],
    "trimmed_audio_offset_ms": 362695
  },
  "bug": {
    "summary": "<short plain-English description of the failure mode>",
    "class": "transcribe_regression_harness",
    "affected_main_tier_index": 0
  },
  "assertions": [
    {
      "kind": "no_zero_duration_wor_words",
      "main_tier_index": 0
    }
  ]
}
```

`input_chat` is required for CHAT-first commands and optional for audio-first
commands such as `transcribe`.

`transcribe` is optional and currently carries transcribe-local fixture
overrides such as ASR engine choice, `%wor` policy, and `diarize=true`.

`assertions` is a list of typed checks. Adding a new variant requires
defining it in `regression_manifest.rs`, implementing it in
`regression_fixtures::run_one_assertion`, and documenting it here. Do
not pre-build assertion variants; add only the ones a real fixture
needs. Utterance-scoped assertions carry `main_tier_index`; whole-output
assertions operate on the parsed `ChatFile` directly and do not.

### Currently supported assertions

| `kind` | Catches |
|--------|---------|
| `no_zero_duration_wor_words` | FA emits all words in `%wor` but with `start_ms == end_ms`, or omits per-word bullets entirely. |
| `min_wor_word_duration_ms` | DP collapse to end: the tail of the word sequence crammed into 40-100 ms per word. Threshold is per-fixture. |
| `min_last_wor_word_duration_ms` | Last-word cutoff: the closing word of the utterance gets squished into a sliver. Threshold is per-fixture. |
| `max_wor_word_duration_proportion` | First-word dominance: one word eats >N% of the utterance bullet. |
| `max_main_tier_lead_before_first_wor_ms` | Stale utterance start: the main-tier bullet begins far before the first timed `%wor` word, often because an inherited parent start was preserved. Threshold is per-fixture. |
| `max_last_wor_overrun_past_main_end_ms` | Main-tier cutoff/overrun mismatch: the last timed `%wor` word ends far past the utterance bullet end. Threshold is per-fixture. |
| `min_main_tier_utterance_count` | Whole-output under-segmentation: `transcribe` or `utseg` collapses the clip into too few main-tier utterances. Threshold is per-fixture. |
| `max_first_main_tier_word_count` | Front-loaded segmentation regression: the first emitted utterance grows implausibly large instead of splitting earlier. Threshold is per-fixture. |
| `no_wor_tiers_present` | `%wor` policy regression: a fixture that intentionally requests `wor=Omit` still materializes `%wor` tiers. |
| `min_distinct_main_tier_speaker_count` | Diarization regression: a fixture that requests `diarize=true` collapses back to too few distinct speaker labels. Threshold is per-fixture. |
| `media_header_matches_input_basename` | Output-contract regression: the serialized `@Media` header stops preserving the input media basename and leaks a temporary/cached filename instead. |

## When you find a bug: the workflow

1. **Get a small, reproducible input.** For CHAT-first commands, use a
   structured CHAT/audio trim tool that preserves timing bullets, rewrites the
   `@Media` header, and rebases word timings to the trimmed audio. For
   audio-first commands such as `transcribe`, stage the minimal audio clip the
   bug needs. Do **not** hand-roll a clip with `ffmpeg`, `head`, `tail`, or
   any other improvised pipeline, the trim helper handles CHAT header
   preservation, timing-bullet rebasing, `@Media` rewriting, and audio
   re-encoding fallback, and reinventing any of that produces fixtures that
   look right but silently mis-time alignment.

2. **Stage the command input into a new directory** under
   `test-fixtures/<command>/regressions/<command>-regression-NNN/`
   in your local checkout. Use `input.cha` for CHAT-first fixtures and
   `input.<ext>` for the required audio file. If the command consumes CHAT,
   rewrite the `@Media` line so the staged audio resolves locally.

3. **Run the command in the production CLI** to capture the buggy
   output as `actual.cha` for documentation:

   ```bash
    # CHAT-first example
    batchalign3 --no-open-dashboard align input.cha --no-server --workers 1
    cp input.cha actual.cha

    # audio-first example
    batchalign3 --no-open-dashboard transcribe input.mp3 --no-server -o out/
    cp out/input.cha actual.cha
   ```

4. **Identify the bug class** from the actual output and pick the
   assertion that captures it. If none of the existing variants fits,
   add a new one in `regression_manifest.rs` and the
   `regression_fixtures::harness` assertion runner, and document it in the
   table above.

5. **Write `source.json`** with the manifest fields and `README.md`
   describing what is wrong and how to reproduce. Redact anything
   about the reporter's identity or the private corpus path: use an
   opaque reference that the private fixture store can resolve, not a
   real name or home directory.

6. **Add a command-local test function to `<command>/regressions.rs`:**

   ```rust,ignore
    #[tokio::test]
     async fn transcribe_regression_001() {
         run_fixture("transcribe", "transcribe-regression-001").await
     }
    ```

   The helper `run_fixture` does discovery, staging, dispatch, and
   assertion checking. If the fixture directory is not present locally
   the test skips cleanly.

7. **Run the test, confirm RED:**

   ```bash
   cargo test -p batchalign --features ml-golden --test ml_golden \
         -E 'test(transcribe::regressions::transcribe_regression_001)'
   ```

8. **Commit the test function + the assertion logic** to this public
   repo. **Do NOT commit the staged fixture files (`input.cha`,
   `input.<ext>`, `actual.cha`, `README.md`, or `source.json`)**: those stay
   in the private fixture mirror. The fixture subdir is gitignored precisely
   so this cannot happen by accident.

## Running the regression suite

The regression-fixture tests live in the `ml_golden` test binary so they
share its warmed worker pool with the other ML golden tests. They are
gated behind the `ml-golden` cargo feature and will not run on a normal
`cargo test` or `make test` invocation.

```bash
# Run every align regression fixture
cargo test -p batchalign --features ml-golden --test ml_golden \
    -E 'test(align::regressions::)' --no-fail-fast

# Run every transcribe regression fixture
cargo test -p batchalign --features ml-golden --test ml_golden \
    -E 'test(transcribe::regressions::)' --no-fail-fast

# Run a single fixture
cargo test -p batchalign --features ml-golden --test ml_golden \
    -E 'test(transcribe::regressions::transcribe_regression_001)'
```

Tests skip cleanly when the corresponding fixture directory is missing
locally.

## How the runner works

The command-local `tests/ml_golden/<command>/regressions.rs` modules call
`run_fixture`, which does this for each fixture:

1. Resolves the fixture directory in this order:
   a. `$BATCHALIGN3_PRIVATE_FIXTURES_DIR/<command>/regressions/<bug>/`
     , the recommended path. Point this env var at your local clone
      of
      [`TalkBank/<private-fixtures>`](https://github.com/TalkBank/<private-fixtures>).
   b. `<batchalign3-repo>/test-fixtures/<command>/regressions/<bug>/`
     , the in-tree fallback, used only for fixtures whose content is
      verifiably safe to ship in the public repo. This path is
      gitignored below the per-command README level so private
      material cannot land here by accident.
2. Loads `source.json` into the typed `FixtureManifest`. If neither
   location has a `source.json` for the requested `(command, bug)`
   pair, the test reports `SKIP` rather than `FAIL`.
3. Acquires a `LiveDirectSession` from the shared `ml_golden` worker
   pool. Skips cleanly if the relevant `InferTask` is unavailable.
4. Stages the command's primary input into the session's state directory:
   `input.cha` for CHAT-first commands, the audio file for audio-first
   commands like `transcribe`. When a staged CHAT also has sidecar audio,
   the runner copies that alongside it so `@Media` resolves locally.
5. Runs the command via `submit_paths_and_complete_direct` with the
   manifest's language and a `CommandOptions` constructed from the
   command type.
6. Parses the output CHAT via
   `talkbank_transform::parse::parse_lenient`
   (at `../chatter/crates/talkbank-transform/src/parse.rs:17`) into a typed
   `ChatFile` AST. Asserts no parse errors.
7. Walks every assertion in the manifest, running each one against the
   typed AST. Some assertions target one main-tier utterance; others
   inspect the whole parsed output. Failures are collected and reported
   together so the human reviewer sees all violations at once, not just
   the first.

The runner does no string hacking. All assertions operate on the typed
`WorTier` / `Word` / `Bullet` AST exposed by `talkbank-model`.
