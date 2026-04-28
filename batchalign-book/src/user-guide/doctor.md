# Doctor

**Status:** Current
**Last updated:** 2026-04-25 21:54 EDT

`batchalign3 doctor` is the diagnostic surface for a batchalign3
deployment. It runs in two modes:

- **Default** — runs the worker pipeline test (Python availability,
  Stanza imports, test-echo round-trip, morphotag pipeline,
  available memory) and prints a host-facts summary.
- **`--check`** — host-facts only. Skips the Python pipeline entirely
  for fast config-sanity verification.

This page covers the operator workflow. Implementation details for
contributors live in `developer/host-facts.md` (when added).

## Common workflows

### "Is my server.yaml going to deploy cleanly?"

```bash
batchalign3 doctor --check
```

Loads the deployed `server.yaml`, detects host facts (OS/arch, RAM,
GPU presence), resolves operator overrides against the host-facts
recommendations, and reports any contradictions. Exits 0 if
validation is clean, non-zero if any error fires.

Sample output on a clean Apple Silicon host:

```
Host facts (snapshot at startup):
  os/arch:           MacOs/Arm64 (12 logical cores, 8 physical)
  ram:               65536 MB total, 32768 MB available
  gpu:               AppleMps { functional_for_batchalign: false, ... }

Effective config (after operator-override + recommendation merge):
  gpu_thread_pool_size: 1
  force_cpu:           true
  max_total_workers:   10
  max_concurrent_jobs: 4
  memory_gate_mb:      8000
  max_workers_per_key: gpu=4 stanza=5 io=1

Validation:           OK (no override contradicts detected facts)

Validation passed cleanly.
```

### "Why does this knob have this value?"

```bash
batchalign3 doctor --explain gpu_thread_pool_size
```

Traces one resolved value end-to-end: the resolved number, whether it
came from an operator override or the host-facts recommendation, the
recommendation rule, and the relevant detected facts.

Useful when `--check` warns about a knob and you want to see the
recommendation's reasoning without grepping the source.

Valid knob names:

- `gpu_thread_pool_size`
- `force_cpu`
- `max_total_workers`
- `max_concurrent_jobs`
- `max_workers_per_key`
- `memory_gate_mb`

### "Is this machine ready to run workloads?"

```bash
batchalign3 doctor
```

Default mode. Spawns a test worker, runs the morphotag pipeline
end-to-end, validates every word has the expected fields, and
reports timing per check. Catches machine-specific issues (stale
Stanza models, missing processors, MWT quirks) before they surface
during real jobs.

Slower than `--check` because it loads ML models. Use it after
software updates, model refreshes, or when a new fleet host comes
online — not as a per-deploy gate.

## CI gate: zero-warning deployments

```bash
batchalign3 doctor --check --warnings-as-errors
```

Default `--check` exits non-zero only on errors. Add
`--warnings-as-errors` for the strict CI posture: any contradiction
between an operator override and a host-facts recommendation
becomes fatal.

Use when you want a server.yaml change to fail review before reaching
production. Skip when operators legitimately need to override the
recommendation (e.g., simulating constrained memory on a large host
via `memory_tier: small`).

## Output formats

### Human (default)

Rendered as label/value pairs without color or boxes; composes
cleanly with surrounding shell output.

### JSON: `--format json`

For machine consumers (CI scripts, monitoring dashboards). The schema
is stable — fields can be added but renames or removals require a
deliberate version bump.

```bash
batchalign3 doctor --check --format json | jq '.validation.warnings[]'
batchalign3 doctor --explain force_cpu --format json | jq '.source'
```

Top-level keys in `--check` mode:

- `detected` — the `HostFacts` snapshot (OS, arch, RAM, GPU, etc.)
- `effective` — resolved knob values (operator overrides merged with
  recommendations)
- `validation` — `{ warnings: [string], errors: [string] }`

Top-level keys in `--explain` mode:

- `knob` — the requested knob name
- `resolved_value` — the value the runtime will use
- `source` — `"operator_override"` or `"recommendation"`
- `recommendation` — what the recommendation function returned
  (always present, so operators can compare)
- `rule` — narrative description of the recommendation rule
- `facts_used` — narrative description of the relevant detected facts

In default mode (the worker-pipeline path), the payload is
`{ "checks": [CheckResult], "host_facts": HostFactsReport }`.

## When validation fires

The validator reports two kinds of finding:

- **Warnings** — the override is suboptimal but the server can still
  run. Surfaced as `tracing::warn!` lines at server startup. Default
  exit policy: non-fatal.
- **Errors** — the override would deterministically crash or produce
  wrong output. The server refuses to start; `doctor --check` exits
  non-zero with the recommendation in the message.

Today's warning variants:

| Variant | Triggered by |
|---|---|
| `GpuThreadPoolSizeAboveOneOnCpu` | `gpu_thread_pool_size > 1` on a host with no functional GPU. The configured threads contend for one CPU-bound model process without parallelism gain. |
| `MaxConcurrentJobsAboveRamBudget` | `max_concurrent_jobs` higher than the recommendation derived from `ram_total_mb` and CPU availability. Risks memory-pressure stalls and worker OOMs under load. |
| `MaxTotalWorkersAboveRamBudget` | `max_total_workers` higher than `clamp(ram_total_mb / 6 GB, 2, 32)`. Risks OOMs under sustained load. |
| `ForceCpuFalseOnNonFunctionalGpu` | `force_cpu: false` set on a host whose GPU is not functional for batchalign. The asserted intent is wrong; common cause is a CUDA-host server.yaml copied to an Apple Silicon machine. |

Today's error variants:

| Variant | Triggered by |
|---|---|
| `MaxConcurrentJobsWouldDeterministicallyOom` | `max_concurrent_jobs * worst_case_per_job_peak_ram_mb > ram_total_mb`. Worst case = the heaviest worker profile (GPU at 16 GB). Even if every job uses the heaviest profile, no scheduling outcome fits — the server refuses to start. Drop `max_concurrent_jobs` from `server.yaml` (the host-facts recommendation is by construction safe) or set a value that satisfies `n * 16384 <= ram_total_mb`. |

Conservative-vs-recommendation cases (operator under-eager) are
intentionally silent. The operator knows their host better than
`recommend()` does.

## Exit codes

`batchalign3 doctor` follows the standard CLI exit codes documented in
`exit-codes.md`:

| Code | Meaning |
|---|---|
| 0 | All checks passed. |
| 2 | Usage error (unknown `--explain` knob name, etc.). |
| Non-zero | Validation found errors (or warnings under `--warnings-as-errors`); or the worker pipeline failed any check. |
