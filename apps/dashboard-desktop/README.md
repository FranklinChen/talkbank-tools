# BA3 Runner desktop shell

**Last modified:** 2026-11-01 15:44 EST

BA3 Runner is a small native launcher for the installed `batchalign3`
CLI and its managed local server. It offers command selection, folder input,
explicit output selection, job progress and diagnostics. It is experimental;
local tests alone do not establish a qualified distributable bundle.

Its scope is six one-command workflows over the CLI: transcription, morphology,
alignment, translation, segmentation and accuracy scoring. Usability means
making those workflows and their failures understandable from the app.

Its installed bundle is `BA3 Runner.app`, executable `ba3-runner`, identifier
`com.franklinchen.ba3-runner`. It has a separate identity and installation path
from other Batchalign apps; installing it must not replace them.

## Runtime ownership

The app asks `batchalign3 serve status --format json` for a structured server
observation. Only an identified loopback endpoint is admitted. A stopped service
can be started through `batchalign3 serve start`; an incompatible or unreachable
service is reported with an actionable error. The CLI checks its own build
against the running service. The app never retains a service PID, stops the
service on exit, or replaces an incompatible running service.

Discovery requests have a 15-second deadline; a service-start request has a
60-second deadline. A stalled CLI produces a retryable error, not a guessed
service state. Cancelling or timing out a request kills only that owned CLI
probe, never the managed service. Check for operating-system file-access
prompts before retrying when configuration lives on a protected volume.

`DesktopBootstrap` completes discovery before mounting server consumers. The
native IPC inventory is `frontend/src/desktop/protocol.ts`; components use the
focused capabilities in `DesktopContext.tsx`. Configuration writes share the
CLI's credential-preserving preferences writer. Unrelated keys, provider
credentials and comments in `~/.batchalign.ini` are retained; read or write
failures are visible.

The native adapter admits reads as `PreferenceDraft`, writes only validated
`PreferenceEdit` and returns `SavedPreferences` only after a complete native
acknowledgement. Settings cannot be edited before a successful read; failed
reads offer retry and unsupported engines require an explicit choice. Reopening
retires the old read owner. Stored keys are not copied into the password field:
retain and replace are different intents, and successful replacement retires
into retain before another save. The first-run wizard owns credential fields
only in its Rev.AI states and cannot show Done while saving or after failure.

A `PreparedJob` captures immutable choices before native filesystem admission
can produce the private `JobPlan` accepted by submission. Native admission
resolves directory aliases, checks file identities and derives the output paths.
Separate input/output trees must be disjoint. Outputs may not overwrite selected
inputs or collide through hard links, case or Unicode normalization; explicit
in-place CHAT processing remains supported. Discovery visits regular files and
directories, skipping links inside the chosen folder and special files.
Admission is read-only observation, not a reservation against concurrent changes.
Submission has one synchronous owner throughout admission and preference reads.
The form also owns folder dialogs and discovery as distinct pending states.
Pending selection disables editing and submission. Dialog cancellation retains
the earlier explicit choice; failed input discovery clears it. Back withdraws
the interaction so late results cannot alter another configuration.
Discovery has distinct empty and nonempty outcomes. Only immutable
`SelectedInputs` can reach job preparation. A keyboard-accessible review lists
the discovered full paths in pages of 25; paging never changes job membership.
Command metadata owns both discovery extensions and empty-folder guidance.
Benchmarking requires a
separate output folder and matching gold `.cha` files beside its audio inputs.
Job progress comes from authoritative `GET /jobs/{id}` observations. Disconnects
are not completion and never cause automatic resubmission. Diagnosed,
interrupted and writeback-failed jobs remain distinct from clean completion.
Opening output is a native directory-only command, not a generic shell opener.

Cancellation has a job-bound owner above reconnecting progress views. Available,
requesting, acknowledged and failed states have different capabilities. The
owner closes synchronously before HTTP; acknowledgement does not re-enable
Cancel or manufacture a complete job record. A failure permits explicit retry
and retires its error on the next attempt. An already-finished response retains
the service's actual outcome while file results are refreshed.

Submission distinguishes a definite refusal, a confirmed `SubmittedJob` identity
and an unconfirmed result. Transport loss, server timeouts/failures or unusable
acknowledgements never return the captured plan to a retryable form. The
unconfirmed view retains the task and folders, offers task-list inspection and
requires deliberate new configuration before another job. No mutation retry is
automatic. This is protection within the launcher, not server-side deduplication.

## Development and local build

Install the CLI first, then install npm dependencies in both `frontend/` and
`apps/dashboard-desktop/`. From this directory:

```bash
npm run dev
npm run build
```

The dev webview uses port 1420. Backend discovery uses the installed CLI's
configuration; the app does not start a private fixed-port server. The release
bundle is under the Cargo workspace's `target/release/bundle/`.

The icon source is `app-icon.svg`. Run `npm run tauri -- icon app-icon.svg`
here to regenerate the platform icons; retain the desktop icon files. The
build admits every PNG named in the bundle configuration as a complete square
RGBA8 image before Tauri embeds it. An unsupported 16-bit image is a build error,
rather than an application that packages successfully and aborts on launch.

## Coordinated CLI and app deployment

Qualify the CLI wheel and native app from the same clean source commit. Preserve
both exact artifacts and their digests, platform/architecture and source
identity. A GUI bundle alone does not install the Python worker environment or
CLI. Conversely, installing a CLI wheel does not install the app.

Before describing a deployment as ready, verify both installed artifacts and
exercise the packaged native app: startup discovery, a real managed-server job,
diagnostics, output files and opening their directory. Closing a window must
leave the service and its jobs intact. Configuration must survive an upgrade.
For macOS distribution, use a Developer ID signed and notarized application;
verify the actual signed artifact, not just rejection fixtures. Unsigned local
builds are development evidence only. No public release is implied by the
experimental artifact workflow.

## Focused verification

```bash
cargo test -p batchalign-dashboard-desktop --bin batchalign-dashboard-desktop
cargo test -p batchalign-types --lib asr_preferences
cargo test -p batchalign --test cli_integration_suite serve_status
```

Frontend seam tests live in `frontend/e2e/tests/mock-server.spec.mjs`. They use a
fake native transport and do not qualify Tauri permissions, Finder/Explorer,
model execution, signing or installation.
