# Long-Term Reliability Program

**Status:** Current
**Last updated:** 2026-04-02 07:37 EDT

This document defines the ongoing reliability practices for batchalign3.
It covers corpus-level regression testing, failure tracking, stress/recovery
testing, and a release readiness scorecard. These practices ensure that
regressions are caught early, failures are classified and root-caused, and
every release meets a minimum quality bar.

---

## 1. Corpus-Level Regression Runs (T130)

Schedule regular regression runs against the full TalkBank corpus data.

### What to run

- `batchalign3 morphotag` against reference CHAT files (no audio needed).
- Compare output against golden baselines stored in version control.
- Track error rates, timing, and output drift across runs.

### Cadence

| Trigger | Scope | Where |
|---------|-------|-------|
| After every release (mandatory) | Full corpus | `net` (256 GB RAM) |
| Weekly on `main` (future scheduled CI) | Full corpus | `net` |
| On demand for significant changes | Targeted subset | `net` or developer machine (small subset only) |

### Infrastructure

- **Execution host:** `net` (Mac Studio, M3 Ultra, 256 GB RAM) for full
  corpus runs. Developer machines may run small targeted subsets but must
  never attempt the full corpus (OOM risk).
- **Golden baselines:** stored in `batchalign3/tests/golden/`. Each baseline
  is a deterministic snapshot of pipeline output for a fixed set of input
  files.
- **Comparison tool:** diff golden output against new run output. Any
  difference is either an intentional change (update the baseline with a
  commit message explaining why) or a regression (file a bug).

### Baseline update policy

1. Run the regression suite and capture output.
2. Diff against the committed baseline.
3. If differences exist and are intentional, update the baseline in a
   dedicated commit with a clear explanation of what changed and why.
4. If differences are unintentional, file a GitHub Issue, classify per
   Section 2, and fix before releasing.

---

## 2. Failure Classification Tracking (T131)

Track failure classes reported by researchers in a structured format so that
patterns become visible and root causes are addressed systematically.

### Categories

| Category | Label | Examples |
|----------|-------|----------|
| **ASR accuracy** | `failure/asr` | Wrong words, missing segments, language detection failures |
| **Alignment** | `failure/alignment` | Timing drift, bullet placement errors, silence misattribution |
| **Morphosyntax** | `failure/morphosyntax` | Wrong POS tag, missing lemma, MWT expansion failure |
| **Infrastructure** | `failure/infra` | Crashes, hangs, OOM, worker spawn failures, deploy errors |
| **CHAT format** | `failure/chat-format` | Parse failures, serialization bugs, roundtrip drift |

### Process

1. Each reported failure gets a **GitHub Issue** with the appropriate
   `failure/*` label.
2. The issue body includes: reproduction steps, input file (or trimmed
   fixture), observed vs. expected output, and the failure category.
3. Root cause is documented in the issue before closing.
4. Every fix includes a **regression test** that reproduces the original
   failure (red/green TDD: write the failing test first, then fix).
5. Periodically review open failure issues to identify systemic patterns
   (e.g., recurring alignment drift in a specific language family).

### Metrics to track

- **Mean time to root-cause** (from report to documented root cause).
- **Failure category distribution** over time (are ASR failures trending
  down? are infra failures trending up?).
- **Recurrence rate** (failures in categories that already have fixes — indicates
  insufficient regression coverage).

---

## 3. Stress and Recovery Test Suites (T132)

These tests verify that the system degrades gracefully under load and
recovers correctly from crashes. They are resource-intensive and run only
on the server or on explicit request.

### Worker stress tests

| Test | What it verifies |
|------|-----------------|
| Concurrent spawn/kill cycles | Workers start and stop cleanly under rapid cycling |
| Crash during inference (simulated via signal) | Server detects worker death and reports error, does not hang |
| Worker timeout with pending results | Timeout fires, partial results are discarded, job is marked failed |
| Memory pressure during model loading | OOM is caught and reported, server remains operational |
| Multiple simultaneous jobs contending for workers | Job queue drains correctly, no deadlock or starvation |

### Recovery tests

| Test | What it verifies |
|------|-----------------|
| Server restart with in-flight jobs | Jobs are marked as failed or retried; no silent data loss |
| Worker crash recovery | Server spawns replacement worker, next job succeeds |
| Cache corruption recovery | Corrupted cache entries are detected and evicted, not served |
| Network interruption during remote execution | Client receives a clear error; server-side state is consistent |

### Implementation

- Use `cargo nextest` with `--profile stress` for test isolation and
  parallelism control.
- **Not part of default CI.** Triggered manually or as part of the release
  process.
- **Run on the server only** — these tests consume significant memory and CPU.
  Never run on developer machines or production workers.
- Each stress test must have a timeout (no infinite hangs) and must clean
  up all spawned processes on completion or failure.

---

## 4. Release Readiness Scorecard (T133)

A living scorecard reviewed before every release. All blockers must be
resolved before tagging a version.

### Scorecard dimensions

| Dimension | Metric | Target | Notes |
|-----------|--------|--------|-------|
| **CI green rate** | % of `main` branch builds passing | 100% | Any red build blocks release |
| **Clippy clean** | Warnings on `cargo clippy --all-targets` | 0 | Includes all workspace members |
| **Wheel smoke** | Clean `pip install` + `batchalign3 --help` passes | Pass | Covers sdist and wheel paths |
| **Cross-platform** | Smoke tests on macOS (ARM) + Linux (x86_64) | 2/2 | CI matrix or manual verification |
| **Version consistency** | `Cargo.toml`, `pyproject.toml`, CLI `--version` agree | Match | Checked by release script |
| **License consistency** | All surfaces show BSD-3-Clause | Match | Cargo.toml, pyproject.toml, LICENSE |
| **Audit clean** | `pip-audit` + `cargo-deny` report no critical vulns | 0 critical | Advisory-only items documented |
| **Test coverage** | Integration + unit + golden tests pass | >80% line coverage | Measured via `cargo llvm-cov` |
| **Regression baseline** | Corpus regression run shows no unintentional drift | No drift | Per Section 1 |
| **Open blockers** | GitHub Issues labeled `release-blocker` | 0 | All must be closed or deferred with justification |

### Process

1. **Before tagging:** run the scorecard checks (automated where possible,
   manual where not yet automated). Record results.
2. **Blockers:** any dimension marked as failing blocks the release.
   Resolve or explicitly defer with a documented justification and a
   follow-up issue.
3. **Historical tracking:** record each release's scorecard in the release
   notes or a dedicated `docs/release-scores/` directory so that trends
   are visible over time.
4. **Automation goal:** progressively move scorecard checks into CI so that
   the scorecard is computed automatically on every push to `main`, not
   just at release time.
