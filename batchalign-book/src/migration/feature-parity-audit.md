# Feature Parity Audit: Live Legacy CLI Comparisons

**Status:** Current
**Last updated:** 2026-04-07 06:13 EDT

This page is the running record of **real live CLI parity runs** against the
canonical preserved legacy baselines. It is the place to record what we
actually ran, which baseline we used, how the old runner was invoked, and what
the current measured delta is.

This page is supplemental to the migration overview in
[`index.md`](index.md). The migration book explains policy; this audit records
the live comparison evidence.

## Canonical baseline policy

The parity program uses a **dual-baseline** policy:

| Scope | Canonical baseline |
| --- | --- |
| core / non-HK behavior | Jan 9 2026 `batchalign2-master` commit `84ad500b09e52a82aca982c41a8ccd46b01f4f2c` |
| HK / Cantonese behavior | Jan 9 2026 `~/BatchalignHK` commit `84ad500b09e52a82aca982c41a8ccd46b01f4f2c` |

Secondary tracking lines still matter:

- later `batchalign2-master` development remains an active comparison target
- later `BatchalignHK` development remains an active comparison target

But those later lines do **not** replace the Jan 9 anchors for migration work.

later Python operational packages, fleet wheels, and other deployment
packaging are useful for operations, but they are **not** the canonical
migration baseline.

## Comparison discipline

All live comparison work should follow these rules:

- current `batchalign3` runs with the modern CLI surface
- preserved Jan 9 legacy runners use their native historical
  `command inputfolder outputfolder` syntax
- HK material uses the separate historical `batchalignhk` command, not stock
  `batchalign`
- `scripts/stock_batchalign_harness.py` owns curated benchmark comparisons
- when an older legacy runner predates modern `.compare.csv` output, the harness
  rescoring path feeds its emitted `.asr.cha` through current
  `batchalign3 compare` so both sides use the same metric surface
- runs are isolated by `HOME` and `BATCHALIGN_STATE_DIR`, and the harness kills
  exact daemon PIDs recorded in isolated state files after each run
- credentialed parity runs should copy a real local config explicitly via
  `BATCHALIGN_STOCK_CONFIG_SOURCE`; whisper-only smoke cases may use the
  harness's seeded minimal config

## Current curated live cases

| Case | Baseline | Status | Result |
| --- | --- | --- | --- |
| `hk-05b-clip-whisper` | Jan 9 `batchalignhk` (`84ad500...`) | complete | current `batchalign3` beats legacy baseline |
| `fra-chloe-whisper-probe` | Jan 9 `batchalign` (`84ad500...`) plus later `batchalign2-master` tracking | complete | current `batchalign3`, Jan 9 stock, and later master all score the same good Whisper result after restoring wrapper-local torchaudio/MPS compatibility |
| `wallet-trimmed-whisper-probe` | Jan 9 `batchalign` (`84ad500...`) plus later `batchalign2-master` tracking | complete | current `batchalign3` beats both the Jan 9 anchor and later master on a second tiny non-HK Whisper benchmark case; later master still matches the poor Jan 9 rescored legacy output (`wer=0.2766`) |
| `fra-chloe-rev-probe` | Jan 9 `batchalign` (`84ad500...`) plus later `batchalign2-master` tracking | complete | current `batchalign3`, Jan 9 stock, and later master all score the same poor Rev result on Chloe; this is legacy-equivalent behavior, not a new current regression |

## Current-only priority verification

These runs began as the current release-readiness verification wave for the
highest-priority command surfaces. Several now also include targeted Jan 9 /
later-master follow-up when that clarified whether a result was current-only or
legacy-equivalent. They still record important live behavior and one real
current-side CLI bug.

| Case | Command | Status | Result |
| --- | --- | --- | --- |
| `morphotag-jpn-retok-smoke` | `morphotag` | complete | tiny Japanese retokenize fixture processed successfully; main tier stayed stable and `%mor` / `%gra` were added |
| `morphotag-jpn-retok-hamasaki-20319` | `morphotag` | complete | first real Japanese stress case: current, Jan 9 stock, and later master all succeed on `Hamasaki/20319.cha`; later master is not byte-identical to Jan 9, but both stock lines still differ from current mainly in retokenization, `%gra` root convention, and Japanese analysis details; an explicit current `--server` rerun is byte-identical to local current |
| `align-401home-whisper-control` | `align` | caution | isolated local `whisper` UTR run produced `0/13` bulleted utterances; useful only as a lower-quality smoke path, not as the internal-quality control |
| `align-401home-rev-control` | `align` | complete | with copied real `.batchalign.ini` and `server.yaml`, current, Jan 9 stock, and later master all recover `13/13` bulleted utterances on `401home-1`, matching archived good output; an explicit current `--server` rerun against an isolated real server also recovers `13/13` and proves remote content mode can resolve the staged local media via `media_roots`; a fresh-home current rerun also exposed and closed a real FA ready-signal pollution bug from MMS model-download chatter |
| `align-407-rev` | `align` | complete | fresh-home current rerun with copied real config recovered `134/135` bulleted utterances on `407-1`; an explicit current `--server` rerun against an isolated real server also recovered `134/135`; Jan 9 stock and later master both match the archived good `131/135`, while current additionally aligns three short `*INV: alright .` turns and leaves only the leading `*PAR: www .` unaligned |
| `transcribe-fra-chloe-rev` | `transcribe` | complete | current credentialed French Rev.AI run scored `wer=0.6538 accuracy=0.3462`; direct Jan 9 and later-master reruns match exactly, so this poor score is legacy-equivalent rather than a current regression |
| `transcribe-fra-chloe-whisper-fixed` | `transcribe` | fixed | direct current, Jan 9 stock, and later master `transcribe` Whisper runs now all score `wer=0.2308 accuracy=0.7692` on Chloe; current also restores the legacy/documented `@Comment` header |
| `transcribe-wallet-whisper-parity` | `transcribe` | complete | direct current `transcribe` on `wallet-trimmed.mp4` matches the benchmark story exactly: current scores `wer=0.0851 accuracy=0.9149`, while Jan 9 stock and later master both stay at `wer=0.2766 accuracy=0.7234` |
| `benchmark-05b-tencent-current` | `benchmark` | complete | current `benchmark --engine-overrides '{"asr":"tencent"}'` on the tiny `05b` fixture completed cleanly and matched the fixed transcribe Tencent score: `wer=0.1923 accuracy=0.8077` |
| `transcribe-05b-funaudio-current` | `transcribe` | complete | after syncing the repo venv to the current built-in HK dependency set and suppressing FunASR stdout chatter inside the worker host, current `--asr-engine-custom funaudio` transcribed the tiny `05b_clip` fixture and scored `wer=0.3462 accuracy=0.6538` |
| `transcribe-05b-tencent-current-fixed` | `transcribe` | fixed | documented `--engine-overrides '{"asr":"tencent"}'` initially fell back to Rev.AI; after the override/preflight fix, the same tiny `05b_clip` run emitted `ASR Engine tencent` and scored `wer=0.1923 accuracy=0.8077` |
| `transcribe-05b-aliyun-current` | `transcribe` | complete | current `--engine-overrides '{"asr":"aliyun"}'` completed cleanly on the tiny `05b_clip.wav` fixture, emitted `ASR Engine aliyun`, and scored `wer=0.5385 accuracy=0.4615` |
| `compare-gold-companion-filter` | `compare` | fixed | current CLI initially submitted `*.gold.cha` companions as primary inputs; after the CLI filter fix and rebuild, the same live rerun dropped from `Found 2 file(s)` to `Found 1 file(s)` |

## Commands exercised so far

The parity effort has exercised these real command surfaces so far:

- harness driver:
  - `uv run python scripts/stock_batchalign_harness.py ...`
  - `python3 scripts/compare_stock_batchalign.py --baseline-bin ... --common-arg=--retokenize .../20319.cha`
- current `batchalign3`:
  - `benchmark --lang yue --whisper -n 1 --output ... 05b_clip.mp3`
  - `benchmark --lang yue -n 1 --engine-overrides '{"asr":"tencent"}' --output ... input_dir`
    (with copied real `.batchalign.ini`, `server.yaml`, and benchmark input dir
    containing `05b_clip.mp3` plus gold `05b_clip.cha`)
  - `benchmark --lang eng --whisper -n 1 --output ... input_dir`
    (with staged `wallet-trimmed.mp4` plus gold `wallet-trimmed.cha`)
  - `benchmark --lang eng --whisper -n 1 --output ... test.mp3`
    (probe only; this exposed the prepared-audio language-name bug before we
    moved to a better non-HK fixture)
  - `benchmark --lang fra --whisper -n 1 --output ... chloe-trimmed.wav`
  - `morphotag --force-cpu --no-tui -o ... retok_jpn_fu_su.cha`
  - `morphotag --no-tui --retokenize --output ... data/childes-data/Japanese/Hamasaki/20319.cha`
  - `morphotag --force-cpu --no-tui --retokenize --output ... input_dir --server http://127.0.0.1:<port>`
  - `align --force-cpu --no-tui --utr-engine whisper -o ... 401home-1.cha`
  - `align --force-cpu --no-tui -o ... 401home-1.cha`
    (with copied real `.batchalign.ini` and `server.yaml`)
  - `align --force-cpu --no-tui -o ... 401home-1.cha --server http://127.0.0.1:<port>`
    (isolated real server with `media_roots` pointed at the staged `401home-1.mp3`)
  - `align --force-cpu --no-tui -o ... 407-1.cha`
    (with copied real `.batchalign.ini` and `server.yaml`)
  - `align --force-cpu --no-tui -o ... 407-1.cha --server http://127.0.0.1:<port>`
    (isolated real server with `media_roots` pointed at the staged `407-1.mp3`)
  - `compare --output ... input_dir`
    (used to rescore legacy transcript-only outputs)
  - `compare --output ... input_dir`
    (used to rescore the Jan 9 `wallet-trimmed.asr.cha` output against a staged
    `wallet-trimmed.gold.cha`)
  - `compare --force-cpu --no-tui -o ... transcribed_dir`
     (live validation that `compare` now skips `*.gold.cha` companions as
     primary inputs)
  - `transcribe --lang eng --asr-engine whisper --no-tui input_dir output_dir`
    (with staged `wallet-trimmed.mp4`, rescored against the staged gold CHAT
    transcript through current `compare`)
  - `transcribe --lang yue --whisper ...`
     (isolated live validation of the repaired HK current path)
  - `transcribe --force-cpu --no-tui --lang fra -o ... chloe-trimmed.wav`
     (with copied real `.batchalign.ini` and `server.yaml`)
  - `transcribe --lang fra --asr-engine whisper --no-tui input_dir output_dir`
    (with copied real `.batchalign.ini` and `server.yaml`)
  - `transcribe --lang yue --asr-engine-custom funaudio --no-tui input_dir output_dir`
    (with copied real `.batchalign.ini`, `server.yaml`, and the repo venv synced
    with the current built-in HK dependency set)
  - `transcribe --lang yue --engine-overrides '{"asr":"tencent"}' --no-tui input_dir output_dir`
    (with copied real `.batchalign.ini`, `server.yaml`, and the repo venv synced
    with the current built-in HK dependency set)
  - `transcribe --lang yue --engine-overrides '{"asr":"aliyun"}' --no-tui input_dir output_dir`
    (with copied real `.batchalign.ini`, `server.yaml`, and the repo-managed
    environment containing the built-in Cantonese/provider stack)
- preserved legacy Jan 9 runners:
  - `batchalignhk benchmark --lang yue --whisper -n 1 input_dir output_dir`
  - `batchalign align input_dir output_dir`
  - `batchalign benchmark --lang fra --whisper -n 1 input_dir output_dir`
  - `batchalign morphotag --retokenize input_dir output_dir`
  - `batchalign transcribe input_dir output_dir --whisper --lang eng -n 1`
  - `batchalign transcribe input_dir output_dir --whisper --lang fra -n 1`
  - `batchalign transcribe input_dir output_dir --rev --lang fra -n 1`
- preserved later stock runner:
  - `batchalign-master-fd816d4 align input_dir output_dir`
  - `batchalign-master-fd816d4 benchmark --lang eng --whisper -n 1 input_dir output_dir`
  - `batchalign-master-fd816d4 benchmark --lang fra --whisper -n 1 input_dir output_dir`
  - `batchalign-master-fd816d4 morphotag --retokenize input_dir output_dir`
  - `batchalign-master-fd816d4 transcribe input_dir output_dir --whisper --lang eng -n 1`
  - `batchalign-master-fd816d4 transcribe input_dir output_dir --whisper --lang fra -n 1`
  - `batchalign-master-fd816d4 transcribe input_dir output_dir --rev --lang fra -n 1`

So the real CLI surfaces exercised to date are:

- current `batchalign3`: `benchmark`, `morphotag`, `align`, `compare`, `transcribe`
- legacy stock `batchalign`: `benchmark`, `morphotag`, `align`, `transcribe`
- legacy HK `batchalignhk`: `benchmark`
- later `batchalign2-master`: `benchmark`, `morphotag`, `align`, `transcribe`

## Current `batchalign3` bug ledger

This count excludes harness-only improvements and legacy baseline/runtime-drift
problems. As of this audit, parity work has surfaced **15 actual
`batchalign3` bugs**: **15 fixed** and **0 currently open**.

| # | Status | Area | Summary |
| --- | --- | --- | --- |
| 1 | fixed | benchmark dispatch | `benchmark` used the wrong source audio path in the app runner |
| 2 | fixed | benchmark CLI routing | explicit `--server` routing sent `benchmark` down the wrong path |
| 3 | fixed | CHAT serialization | `@Media` names were not normalized before serialization |
| 4 | fixed | ASR postprocess timing | zero/invalid timestamps could emit invalid `0_0` bullets |
| 5 | fixed | ASR postprocess tokenization | single-chunk ASR output with embedded punctuation was not split before CHAT assembly |
| 6 | fixed | Cantonese ASR postprocess | Han-script `yue` chunks stayed as giant tokens instead of character-tokenized output |
| 7 | fixed | prepared-audio Whisper path | local benchmark/transcribe prepared-audio inference passed ISO-639-3 codes like `eng` instead of Whisper language names like `english` |
| 8 | fixed | CLI failure reporting | failed terminal jobs now propagate as non-zero CLI server errors instead of exiting `0` after printing a summary |
| 9 | fixed | compare CLI discovery | `compare` submitted `*.gold.cha` reference companions as normal input files instead of skipping them during command-specific filtering |
| 10 | fixed | V2 worker backend routing | worker-pool keys ignored per-request engine overrides, so `transcribe --asr-engine whisper` on Rev-configured machines could reuse an ASR worker booted for the wrong backend; forced-alignment workers had the same structural risk |
| 11 | fixed | local-daemon CLI dispatch | `dispatch/mod.rs` still swallowed local-daemon startup/processing failures in the auto-daemon and local-audio fallback paths, causing real failed jobs to exit `0` |
| 12 | fixed | transcribe CHAT headers | current transcribe output omitted the legacy/documented `@Comment` metadata line with Batchalign version and ASR engine name |
| 13 | fixed | FunAudio worker protocol | third-party FunASR emitted a raw version line to stdout during model import/init/inference, corrupting the JSON-lines worker protocol and making `--asr-engine-custom funaudio` fail with decode errors |
| 14 | fixed | global ASR override routing | documented `--engine-overrides '{"asr":"..."}'` values were ignored for transcribe/benchmark Rev preflight and dispatch because those paths only read the serialized `asr_engine` field, causing HK cloud-engine requests to fall back to Rev.AI |
| 15 | fixed | Rev diarization parity | `transcribe --rev --diarize` skipped the dedicated post-ASR speaker relabeling stage whenever Rev already returned speaker labels, diverging from Jan 9 BA2's audited `transcribe_s = asr,speaker` pipeline |

### `hk-05b-clip-whisper`

**Fixture**

- audio: `batchalign/tests/hk/fixtures/05b_clip.mp3`
- gold: `batchalign/tests/hk/fixtures/benchmark/05b_clip.cha`
- baseline executable: maintainer-local pinned wrapper such as
  `files/bin/batchalignhk-84ad500`

**Live command shape**

```bash
# current
./target/debug/batchalign3 benchmark --lang yue --whisper -n 1 \
  --output .../current/output \
  .../current/input/05b_clip.mp3

# Jan 9 HK baseline
batchalignhk-84ad500 benchmark --lang yue --whisper -n 1 \
  .../baseline/input \
  .../baseline/output
```

**Measured result**

| Metric | current `batchalign3` | Jan 9 HK baseline |
| --- | ---: | ---: |
| WER | 0.2692 | 0.4615 |
| accuracy | 0.7308 | 0.5385 |
| matches | 22 | 20 |
| insertions | 3 | 6 |
| deletions | 4 | 6 |
| total gold words | 26 | 26 |

**Interpretation**

- This is the first honest HK current-vs-legacy live comparison in the repo.
- The legacy Jan 9 HK benchmark does **not** emit modern `.compare.csv`; it
  emits `.wer.txt`, `.diff`, and `.asr.cha`.
- The harness now normalizes that legacy output by rescoring the emitted
  `.asr.cha` with current `batchalign3 compare`.
- Current `batchalign3` initially lost badly on this case because Rust `yue`
  ASR post-processing kept whisper chunks as giant Cantonese strings instead of
  character tokens.
- After fixing Cantonese Han-token splitting in
  `crates/batchalign-chat-ops/src/asr_postprocess/mod.rs`, current `batchalign3`
  beats the Jan 9 HK baseline on this tiny whisper case.
- No allowlist entry was added for this case; the improvement is currently
  treated as a real current-side gain, not a known legacy bug.

### `fra-chloe-whisper-probe`

**Fixture**

- audio: session-local trimmed corpus fixture
  `files/live-fixtures/chloe/trimmed/chloe-trimmed.wav`
- gold: session-local trimmed corpus fixture
  `files/live-fixtures/chloe/trimmed/chloe-trimmed.cha`
- baseline executables:
  - Jan 9 pinned wrapper `batchalign-jan84ad500`
  - later-master tracking wrapper `batchalign-master-fd816d4`

**Live command shape**

```bash
# current
./target/debug/batchalign3 benchmark --lang fra --whisper -n 1 \
  --output .../current/output \
  .../current/input/chloe-trimmed.wav

# Jan 9 stock baseline
batchalign-jan84ad500 benchmark --lang fra --whisper -n 1 \
  .../baseline/input \
  .../baseline/output

# later master tracking baseline
batchalign-master-fd816d4 benchmark --lang fra --whisper -n 1 \
  .../baseline/input \
  .../baseline/output
```

**Measured result**

| Metric | current `batchalign3` | Jan 9 stock baseline | later `batchalign2-master` |
| --- | ---: | ---: | ---: |
| WER | 0.2308 | 0.2308 | 0.2308 |
| accuracy | 0.7692 | 0.7692 | 0.7692 |
| matches | 23 | 23 | 23 |
| insertions | 3 | 3 | 3 |
| deletions | 3 | 3 | 3 |
| total gold words | 26 | 26 | 26 |

**Runtime compatibility repair**

- The original current-side `benchmark --lang eng --whisper` prepared-audio path
  bug was fixed first: current now maps ISO-639-3 codes like `eng` to the
  Whisper language names expected by the prepared-audio runtime.
- The first honest non-HK probe then moved to the small French
  `chloe-trimmed` fixture because the tracked `test.mp3` sample was too large
  and lacked a trustworthy gold transcript.
- The later `batchalign2-master` tracking wrapper now reproduces the same good
  score as both the Jan 9 anchor and current `batchalign3`, so this French
  Whisper path stayed fixed on the later stock line.
- The present-day pinned Jan 9 wrapper initially failed under modern
  `torchaudio` 2.10 with:

  ```text
  ERROR on file chloe-trimmed.wav: TorchCodec is required for
  load_with_torchcodec. Please install torchcodec to use this function.
  ```

- The maintainer-local pinned wrapper now uses narrow runtime-only compatibility
  shims under its existing `stubs/` path:
  - `pkg_resources.py` restores the small setuptools surface that the legacy
    praatio stack imports
  - `sitecustomize.py` restores `torchaudio.load()` / `torchaudio.info()` via
    `soundfile`
  - the same `sitecustomize.py` disables the legacy MPS path by default unless
    `BATCHALIGN_LEGACY_ENABLE_MPS=1` is set
- That MPS guard matters on this machine: under the modern tool environment
  (`transformers` 5.3.0), the frozen legacy Whisper configuration produced only
  giant `!!!!!!!!!!!!!!!!` chunks on MPS, while the exact same frozen code on
  CPU produced the expected French transcript.
- `scripts/stock_batchalign_harness.py` still keeps the copied-gold `.cha`
  silent-failure detection, so the harness continues to reject fake-success
  legacy outputs when the wrapper regresses.
- Later `batchalign2-master` commit `eaa32e8`
  (`feat: upgrade to torch 2.10, soundfile I/O, drop Python 3.9-3.10`)
  already repaired the torchaudio I/O side of this drift with the new
  `batchalign/models/audio_io.py` layer, but the MPS punctuation failure also
  reproduces there under the same present-day tool environment.

**Interpretation**

- This clears the earlier non-HK Jan 9 stock local-whisper runtime blocker for
  curated live comparisons on this maintainer machine.
- On this tiny French probe, current `batchalign3` and the Jan 9 stock baseline
  now tie exactly.
- The earlier `torchcodec` crash and punctuation-only MPS output were
  runtime-compatibility problems, not proven current-vs-stock transcript-quality
  regressions.
- The next step is to expand the non-HK live comparison set now that the stock
  whisper path can transcribe again, then add tiny credentialed Rev.AI cases.

### `wallet-trimmed-whisper-probe`

**Fixture**

- audio: session-local trimmed corpus fixture
  `files/live-fixtures/wallet/trimmed/wallet-trimmed.mp4`
- gold: session-local trimmed corpus fixture
  `files/live-fixtures/wallet/trimmed/wallet-trimmed.cha`
- baseline executables:
  - Jan 9 pinned wrapper `batchalign-jan84ad500`
  - later-master tracking wrapper `batchalign-master-fd816d4`

**Live command shape**

```bash
# current
./target/debug/batchalign3 --force-cpu benchmark --lang eng --whisper -n 1 \
  --output .../current/output \
  .../current/input

# Jan 9 stock baseline
batchalign-jan84ad500 benchmark --lang eng --whisper -n 1 \
  .../baseline/input \
  .../baseline/output

# later master tracking baseline
batchalign-master-fd816d4 benchmark --lang eng --whisper -n 1 \
  .../baseline/input \
  .../baseline/output
```

**Measured result**

| Metric | current `batchalign3` | Jan 9 stock baseline | later `batchalign2-master` |
| --- | ---: | ---: | ---: |
| WER | 0.0851 | 0.2766 | 0.2766 |
| accuracy | 0.9149 | 0.7234 | 0.7234 |
| matches | 46 | 45 | 45 |
| insertions | 3 | 11 | 11 |
| deletions | 1 | 2 | 2 |
| total gold words | 47 | 47 | 47 |
| total main words | 49 | 56 | 56 |

**Interpretation**

- This is the second real tiny non-HK local-Whisper benchmark case after
  `fra-chloe-whisper-probe`.
- The current run completed directly and produced the modern
  `wallet-trimmed.compare.csv` score surface.
- The Jan 9 runner still emitted its older transcript-oriented output shape
  (`wallet-trimmed.asr.cha` plus `wallet-trimmed.wer.txt`), so the baseline side
  was normalized through current `batchalign3 compare` against the same staged
  gold transcript to produce directly comparable metrics.
- The later `batchalign2-master` tracking wrapper produced the same rescored
  metrics as the Jan 9 anchor, so this particular English Whisper gap was not
  fixed later on the stock line.
- Current `batchalign3` materially beat the legacy baseline on this clip, mostly
  by avoiding the older runner's insertion-heavy collapse across the middle of
  the story prompt.

### `fra-chloe-rev-probe`

**Fixture**

- audio: session-local trimmed corpus fixture
  `files/live-fixtures/chloe/trimmed/chloe-trimmed.wav`
- gold: session-local trimmed corpus fixture
  `files/live-fixtures/chloe/trimmed/chloe-trimmed.cha`
- baseline executables:
  - Jan 9 pinned wrapper `batchalign-jan84ad500`
  - later-master tracking wrapper `batchalign-master-fd816d4`

**Live command shape**

```bash
# current
./target/debug/batchalign3 transcribe input_dir output_dir \
  --lang fra --no-tui

# Jan 9 stock baseline
batchalign-jan84ad500 transcribe input_dir output_dir \
  --rev --lang fra -n 1

# later master tracking baseline
batchalign-master-fd816d4 transcribe input_dir output_dir \
  --rev --lang fra -n 1
```

**Measured result**

| Metric | current `batchalign3` | Jan 9 stock baseline | later `batchalign2-master` |
| --- | ---: | ---: | ---: |
| WER | 0.6538 | 0.6538 | 0.6538 |
| accuracy | 0.3462 | 0.3462 | 0.3462 |
| matches | 16 | 16 | 16 |
| insertions | 7 | 7 | 7 |
| deletions | 10 | 10 | 10 |
| total gold words | 26 | 26 | 26 |

**Interpretation**

- The poor Rev.AI score on this tiny French clip is **not** a new current
  regression; current reproduces both the Jan 9 anchor and the later stock line
  exactly.
- On the same Chloe fixture, current local Whisper remains materially better
  (`wer=0.2308`) than the legacy-equivalent Rev path.
- The legacy Jan 9 and later-master outputs both include an `@Comment` line of
  the form:

  ```text
  @Comment:	Batchalign <version>, ASR Engine rev. Unchecked output of ASR model.
  ```

- Current `batchalign3` had temporarily lost that header during this wave; the
  omission was a real current-side transcribe regression and is now fixed.

### `transcribe-05b-funaudio-current`

**Fixture**

- audio: repo fixture `batchalign/tests/hk/fixtures/05b_clip.mp3`
- gold: repo benchmark fixture `batchalign/tests/hk/fixtures/benchmark/05b_clip.cha`

**Live command shape**

```bash
./target/debug/batchalign3 transcribe input_dir output_dir \
  --lang yue \
  --asr-engine-custom funaudio \
  --no-tui
```

**Measured result**

| Metric | current `batchalign3` |
| --- | ---: |
| WER | 0.3462 |
| accuracy | 0.6538 |
| matches | 22 |
| insertions | 5 |
| deletions | 4 |
| total gold words | 26 |
| total main words | 27 |

**Interpretation**

- The first live attempt failed honestly because the local project venv did not
  yet include the then-missing FunASR dependency set; syncing the repo venv
  with the current built-in HK dependency set cleared the environment blocker.
- The next live attempt then exposed a real current-side bug: FunASR printed a
  raw `funasr version: ...` banner to stdout, which polluted the JSON-lines
  worker protocol stream and caused `failed to decode response` errors.
- Current `batchalign3` now suppresses third-party FunASR stdout during import,
  model initialization, and `generate()` calls inside the FunAudio wrapper, and
  the same tiny Cantonese clip now completes successfully.

### `morphotag-jpn-retok-hamasaki-20319`

**Fixture**

- primary stress-case candidate: `data/childes-data/Japanese/Hamasaki/20319.cha`
- rejected probe: `data/ca-data/CallHome/jpn/1690.cha`

**Live command shape**

```bash
# current vs Jan 9 raw diff
python3 scripts/compare_stock_batchalign.py \
  --baseline-bin batchalign-jan84ad500 \
  --common-arg=--retokenize \
  data/childes-data/Japanese/Hamasaki/20319.cha
```

**Measured result**

- `CallHome/jpn/1690.cha` was not a usable parity case for current `morphotag`:
  Jan 9 accepted it, but current failed pre-validation because the transcript
  contains many main-tier utterances without terminators. That matches the
  current `MainTierValid` contract and was not counted as a new regression.
- `Hamasaki/20319.cha` was a valid real stress case:
  - current `batchalign3`: success
  - Jan 9 baseline: success
  - later master baseline: success
  - explicit current `--server` rerun: success, byte-identical to local current
  - raw diff status: `unexpected_difference` (manual classification required)

**Interpretation**

- Spot-checking the raw diff showed several recurring classes of divergence that
  look like expected current-side improvements rather than release blockers:
  - current generated `%gra` roots consistently use the documented `head=0`
    convention, while Jan 9 often kept `head=self`
  - current retokenization better matches the Japanese `%ort` forms in several
    visible utterances (for example, collapsing over-split main-tier fragments
    such as `あ そん` → `あそん`)
  - current Japanese morphosyntax analysis differs on auxiliaries/disfluencies,
    which is consistent with the live Japanese override path and the fact that
    Japanese remains one of the most customized non-English morphotag surfaces
- Later master is not byte-identical to Jan 9 on this file: it changes some
  root attachments and Japanese analyses relative to Jan 9, but still lands in
  the same broad divergence class versus current rather than converging to an
  exact match.
- The current output was byte-identical across the Jan 9 and later-master reruns,
  so the observed movement is on the stock side, not a drifting current result.
- Result: this case should be tracked as a successful real Japanese retokenize
  comparison with expected output divergence, not as a current regression.

### `transcribe-05b-tencent-current-fixed`

**Fixture**

- audio: repo fixture `batchalign/tests/hk/fixtures/05b_clip.mp3`
- gold: repo benchmark fixture `batchalign/tests/hk/fixtures/benchmark/05b_clip.cha`

**Live command shape**

```bash
./target/debug/batchalign3 transcribe input_dir output_dir \
  --lang yue \
  --engine-overrides '{"asr":"tencent"}' \
  --no-tui
```

**Measured result**

| Metric | current `batchalign3` |
| --- | ---: |
| WER | 0.1923 |
| accuracy | 0.8077 |
| matches | 25 |
| insertions | 4 |
| deletions | 1 |
| total gold words | 26 |
| total main words | 29 |

**Interpretation**

- The first live Tencent attempt exposed a real current-side bug instead of a
  Tencent credential problem:
  - the documented global `--engine-overrides '{"asr":"tencent"}'` flag was
    accepted by the CLI but ignored by transcribe preflight and dispatch
  - because `asr_engine` still held its default `rev`, the job fell back to
    Rev.AI and failed with `"'yu' is not a supported ISO language code"` on the
    tiny Cantonese clip
- Current `batchalign3` now resolves the effective ASR engine from
  `common.engine_overrides["asr"]` for both:
  - Rev preflight selection
  - transcribe / benchmark dispatch planning
- Focused Rust regression tests now cover:
  - effective ASR engine helpers on transcribe / benchmark option types
  - transcribe / benchmark dispatch extraction preferring the shared override
  - Rev preflight disabling when an `asr` override selects a non-Rev engine
- After that fix and rebuild, the same tiny current Tencent run completed
  successfully, emitted
  `@Comment: Batchalign 0.1.0, ASR Engine tencent. Unchecked output of ASR model.`
  and scored better than the current local `funaudio` run on this fixture.

### `benchmark-05b-tencent-current`

**Fixture**

- input dir contents:
  - repo fixture `batchalign/tests/hk/fixtures/05b_clip.mp3`
  - repo gold `batchalign/tests/hk/fixtures/benchmark/05b_clip.cha`

**Live command shape**

```bash
./target/debug/batchalign3 benchmark \
  --lang yue \
  -n 1 \
  --engine-overrides '{"asr":"tencent"}' \
  --no-tui \
  --output output_dir \
  input_dir
```

**Measured result**

| Metric | current `batchalign3` |
| --- | ---: |
| WER | 0.1923 |
| accuracy | 0.8077 |
| matches | 25 |
| insertions | 4 |
| deletions | 1 |
| total gold words | 26 |
| total main words | 29 |

**Interpretation**

- This validates that the shared global-ASR-override fix was not merely a
  `transcribe` repair. The same tiny Cantonese fixture now runs through
  `benchmark` cleanly under the documented `--engine-overrides` surface.
- The benchmark output CHAT again emits
  `@Comment: Batchalign 0.1.0, ASR Engine tencent. Unchecked output of ASR model.`
  and matches the fixed current Tencent transcribe score on the same clip.

### `transcribe-05b-aliyun-current`

**Fixture**

- audio: repo fixture `batchalign/tests/hk/fixtures/05b_clip.wav`
- gold: repo benchmark fixture `batchalign/tests/hk/fixtures/benchmark/05b_clip.cha`

**Live command shape**

```bash
./target/debug/batchalign3 transcribe input_dir output_dir \
  --lang yue \
  --engine-overrides '{"asr":"aliyun"}' \
  --no-tui
```

**Measured result**

| Metric | current `batchalign3` |
| --- | ---: |
| WER | 0.5385 |
| accuracy | 0.4615 |
| matches | 19 |
| insertions | 7 |
| deletions | 7 |
| total gold words | 26 |
| total main words | 26 |

**Interpretation**

- The current Aliyun path now completes cleanly on the same tiny Cantonese
  fixture under the documented `--engine-overrides` surface and emits the
  expected `ASR Engine aliyun` comment metadata.
- This score is materially worse than the current Tencent and current FunAudio
  results on the same clip, but the run itself did not surface a current-side
  crash, routing error, or protocol bug.
- Taken together with the verified FunAudio and Tencent runs, the tiny current
  HK ASR engine sweep for `05b` is now complete across:
  - `funaudio`
  - `tencent`
  - `aliyun`

## Open fronts

- continue adding core / non-HK live cases now that the stock baseline can
  again produce real transcript output on the pinned compatibility wrapper
- decide whether to anchor the tiny HK cloud-provider runs against Jan 9
  `batchalignhk` baselines as well, or keep them as current-only provider
  verification because cloud-ASR output can drift over time
- continue documenting the `batchalign` vs `batchalignhk` command history in the
  user/developer migration pages
## Rerun recipe

```bash
cd batchalign3
uv run python scripts/stock_batchalign_harness.py \
  --case hk-05b-clip-whisper \
  --current-executable ./target/debug/batchalign3 \
  --baseline-executable /path/to/pinned/batchalignhk-84ad500 \
  --run-root /tmp/hk-comparison-run
```

For credentialed runs, export:

```bash
export BATCHALIGN_STOCK_CONFIG_SOURCE=/path/to/real/.batchalign.ini
```
