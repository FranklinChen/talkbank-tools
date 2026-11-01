# BA3 Runner (Experimental)

**Last modified:** 2026-11-01 19:40 EST

BA3 Runner is a small native interface to the installed `batchalign3`
CLI and its managed local server. It remains experimental until a packaged
application has been qualified for deployment alongside its matching CLI.
See [Installation](installation.md) for the supported CLI installation path.

The application is named **BA3 Runner**, with its own bundle identity. It can
coexist with other Batchalign applications and does not replace them.

## Getting started

Install the matching CLI and desktop app. When evaluating from source, install
npm dependencies in `frontend/` and `apps/dashboard-desktop/`, then run
`npm run dev` from the latter directory.

On launch the app locates the CLI and discovers its managed local service. It
starts a missing service through the CLI, or reports why a service cannot be
used. Closing the window leaves the service and running jobs intact. A build
mismatch must be resolved by updating the CLI and managed service; the app does
not silently replace a running service.

On first launch, choose Rev.AI (a paid cloud service, requiring a key and uploading
audio) or Whisper (local inference, requiring model downloads). Setup and
Settings edit the same `~/.batchalign.ini` as the CLI. Other provider keys and
settings are preserved. Configuration errors are shown rather than replaced
with defaults.

## Processing a folder

1. Choose Transcribe, Add Grammar, Align, Translate, Segment Utterances or Score
   Accuracy. Segmentation finds utterance boundaries; it does not identify
   speaker turns. Alignment infers timing, which still needs review.
2. Choose an input folder. The app recursively discovers matching regular files.
   Links inside that folder and special files are skipped. To include linked
   content, choose its original folder.
   The supported file types appear below the folder button. If none are found,
   check those types or choose another folder. Expand **Review input files** to
   inspect full paths, 25 at a time; Previous/Next changes only the displayed
   page. The job uses every discovered input file, including the other pages.
3. Choose a separate output folder explicitly, or select Modify in place for
   commands that permit it. Neither separate folder may contain the other.
   In-place processing replaces input files: keep
   backups. Score Accuracy requires separate output and a matching gold `.cha`
   beside each input audio file.
4. Where shown, set the ISO 639-3 language code (for example `eng`) and the
   positive whole-number speaker count. Add Grammar and Translate read language
   from each transcript's `@Languages` header. Transcribe and Score Accuracy use the
   Rev.AI or Whisper preference selected in Settings.
5. Start once. The app submits one job and displays its observed state.

While choosing a folder or scanning its files, configuration and Start Processing
are disabled. Canceling the folder dialog retains your earlier choice. If input
selection or scanning fails, choose the input folder again; the previous folder
cannot be submitted accidentally. Back abandons the pending choice, and a late
dialog or scan result cannot change the next configuration.

Before submission, the native app checks actual file identities and resolves
folder links. An output that aliases a selected input is refused, except for
explicit in-place CHAT processing. Multiple inputs may not share an output,
including names differing only in case or Unicode normalization. This portable
naming policy also applies on case-sensitive filesystems. Rename conflicting
inputs or select a disjoint output folder as the diagnostic directs.
Nested relative paths are preserved. Transcription writes `.cha` output rather
than overwriting input audio. Existing unrelated outputs can be replaced; keep
backups. The check does not reserve files against changes by another program.

## Progress and diagnostics

If the service refuses a submission, correct the reported problem before
starting again. If submission **could not be confirmed**, the service may
already have started it. The app preserves the task and folders and does not
send it again. Check Recent Tasks or open the task list to inspect the job.
An absent entry does not prove rejection. After checking, use **I checked the
tasks; configure another job** only when you intend a new submission.

While **Checking input and output files** or **Reading ASR preferences** is
shown, leaving Process abandons that unsent launch. Delayed replies cannot start
it after you leave or change a later configuration. Once **Submitting job** is
shown, the request has been attempted: navigating away does not cancel or resend
it. Use Recent Tasks to inspect it, and Cancel explicitly if needed.

Progress is refreshed from the server's job record. If connectivity is lost,
the app reports that observation failed: the job may still be running. Retry
reconnects to the same job; it does not submit the work again.

Clean completion, completion with diagnostics, failure, cancellation,
interruption and output-write failure have different outcomes. Read the file
messages and any listed utterances before using diagnosed output. Missing-timing
diagnostics do not certify the correctness of timing that was produced.

Open Output Folder reveals the job's actual output directory. After reopening
the app, select a job under Recent Tasks: its detail page offers the folders
recorded by the local service, even though the original submission screen is
gone. Jobs with outputs in several directories list each folder separately.
These are recorded destinations, not a claim that every file succeeded or
that the folder still exists. Missing or moved folders produce a visible error
without changing the job result. Uploaded-content jobs and records without
local output paths explain why no local folder is available. A remote job's
folders must be opened on its processing computer.

Cancel sends one request. While
it is pending or acknowledged, it cannot be sent again. The app waits for the
service's final job record, including when progress disconnects and reconnects.
If cancellation cannot be confirmed, read the explanation and use Retry
cancellation explicitly. A job can finish before cancellation arrives; the app
preserves that outcome. Completed output may remain on disk.

## Settings and troubleshooting

Settings can select Rev.AI or Whisper and update a Rev.AI key. Switching to
Whisper does not delete a stored Rev.AI key. Other engines and advanced options
remain available through the CLI; an unsupported configured engine is reported
rather than silently substituted.

Settings first reads your current preferences. If that fails, use **Retry
reading settings** after resolving the reported problem. An unsupported engine
stays visible until you choose Rev.AI or Whisper explicitly. Stored API keys are
not displayed: leave the Rev.AI key field blank to keep the stored key, or enter
a replacement to change it. After a confirmed save, the field returns to blank
and later saves keep the current stored key. Switching to Whisper also keeps it.

**Reload settings** reads the file again and discards unsaved edits. While a
save is pending, editing and closing the settings dialog are disabled. A failed
or unconfirmed save shows its explanation; reload to check the file. Setup
likewise keeps failed saves on their original screen and offers completion only
after confirmation. If the initial configuration check fails, use **Retry
configuration check**; the app does not treat the failure as a missing file.

If the CLI is missing, install it using the [installation guide](installation.md).
The macOS shell also checks the usual `~/.local/bin` and Homebrew locations.
For connection failures, use `batchalign3 serve status` and follow its diagnostic.
If a connection attempt times out, check for a macOS file-access prompt for
BA3 Runner. The app needs access to its configuration and chosen folders;
retry after resolving the prompt. A timed-out probe does not mean the service
or its jobs stopped.
For missing inputs, check the selected command's supported extensions and access
to the folder. First-use model downloads can delay processing.

For command behavior and diagnostic remedies, see [Troubleshooting](troubleshooting.md).
