# Chapter 11: Remediation Roadmap

## Phase 0: Immediate Stop-Ship Corrections

Target: 1-2 weeks.

### Must Finish Before Any Public Release Attempt

- Fix the wheel packaging path so the installed CLI actually runs.
- Add required artifact smoke tests to CI.
- Turn on CI for pull requests and normal pushes.
- Resolve the license metadata conflict.
- Publish one authoritative release-state document.
- Decide the `talkbank-tools` release boundary.

## Phase 1: Release Engineering and Test Re-Tiering

Target: 2-6 weeks.

- Create a formal internal release checklist.
- Add cross-platform smoke lanes for supported artifacts.
- Add mandatory integration tests for:
  - CLI startup
  - server startup
  - worker startup
  - one real inference smoke per released command family
- Re-tier the suite:
  - fast required
  - medium required
  - heavy scheduled

## Phase 2: Control-Plane Decomposition

Target: 1-3 months.

- Split worker lifecycle from worker transport.
- Split runner planning from runner execution and persistence.
- Reduce oversized state/status modules.
- Remove low-value dead scripts and dormant workflow leftovers.
- Improve evidence preservation in failure paths.

## Phase 3: Cross-Repo and Cross-Platform Hardening

Target: 2-4 months.

- Complete the `talkbank-tools` release or vendoring decision.
- Add explicit compatibility versioning between repos.
- Prove Windows and macOS install/startup paths in CI.
- Tighten release artifact metadata and bills of materials.

## Phase 4: Long-Term Reliability Program

Target: ongoing.

- schedule corpus-level regression runs
- track failure classes found by researchers
- add stress and recovery suites for worker/runtime behavior
- maintain a living release-readiness scorecard

## Suggested Ownership

- Release engineering owner:
  packaging, versioning, artifact smoke, metadata, CI triggers
- Runtime owner:
  worker pool, process lifecycle, concurrency, shutdown, observability
- Test strategy owner:
  lane design, coverage priorities, regression policy
- Platform owner:
  Windows/macOS/Linux support matrix
- Cross-repo owner:
  `talkbank-tools` contract and joint release hygiene
