#!/usr/bin/env python3
"""Run a tiny, honest batchalign3-vs-Jan-9 legacy benchmark comparison.

This repo-local harness owns one narrow comparison seam:
- curated benchmark cases are declared in JSON
- the current batchalign3 CLI runs each case in an isolated runtime root
- an optional Jan 9 2026 legacy `batchalign` or `batchalignhk` baseline
  runner runs the same case using its native `command input_dir output_dir`
  syntax
- the repo's existing Rust compare pipeline remains the scoring authority:
  modern runners emit `.compare.csv` directly, while older legacy runners that
  only emit `.asr.cha` are rescored through the current `compare` command

The harness classifies only the top-line metric deltas that matter for a first
nightly slice:
- allowlisted baseline bugs (current beats stock in a known way)
- unallowlisted improvements (current beats stock unexpectedly)
- unexpected regressions (current loses to stock)
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST: Final[Path] = (
    REPO_ROOT / "test-fixtures" / "stock_batchalign" / "manifest.json"
)
DEFAULT_ALLOWLIST: Final[Path] = (
    REPO_ROOT / "test-fixtures" / "stock_batchalign" / "allowlist.json"
)
JAN9_BASELINE_CANDIDATES: Final[tuple[Path, ...]] = (
    Path.home() / "bin" / "batchalign-jan84ad500",
    Path.home() / "bin" / "batchalign2-jan84ad500" / "batchalign",
    Path.home()
    / "bin"
    / "batchalign-pins"
    / "repos"
    / "batchalign2-jan84ad500"
    / "batchalign",
)
CONFIG_SOURCE_ENV: Final[str] = "BATCHALIGN_STOCK_CONFIG_SOURCE"

ClassifiedMetricName = Literal["wer", "matches", "insertions", "deletions"]
MetricName = Literal[
    "wer",
    "accuracy",
    "matches",
    "insertions",
    "deletions",
    "total_gold_words",
    "total_main_words",
]
CaseStatus = Literal[
    "planned",
    "completed",
    "baseline_skipped",
    "current_failed",
    "baseline_failed",
]
ImprovementDirection = Literal["current_better", "current_worse"]
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

CLASSIFIED_METRICS: Final[tuple[ClassifiedMetricName, ...]] = (
    "wer",
    "matches",
    "insertions",
    "deletions",
)
DEFAULT_TOLERANCES: Final[dict[ClassifiedMetricName, float]] = {
    "wer": 0.0001,
    "matches": 0.0,
    "insertions": 0.0,
    "deletions": 0.0,
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One curated Jan 9 legacy comparison case rooted in repo fixtures."""

    id: str
    description: str
    command: str
    audio_path: Path
    gold_path: Path
    argv: tuple[str, ...]
    tolerances: dict[ClassifiedMetricName, float]

    def output_stem(self) -> str:
        """Return the benchmark output stem derived from the input audio path."""
        return self.audio_path.stem


@dataclass(frozen=True)
class AllowlistEntry:
    """One expected Jan 9 legacy baseline bug for a specific case and metric."""

    case_id: str
    metric: ClassifiedMetricName
    reason: str
    max_delta: float | None = None


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Parsed `.compare.csv` metrics emitted by the benchmark command."""

    wer: float
    accuracy: float
    matches: int
    insertions: int
    deletions: int
    total_gold_words: int
    total_main_words: int

    def value(self, metric: MetricName) -> float:
        """Return one metric as a float for comparison and formatting."""
        return float(getattr(self, metric))

    def summary(self) -> str:
        """Render the core metrics in one stable human-readable line."""
        return (
            f"wer={self.wer:.4f} accuracy={self.accuracy:.4f} "
            f"matches={self.matches} insertions={self.insertions} "
            f"deletions={self.deletions} total_gold={self.total_gold_words}"
        )


class CliInvocationStyle(str, Enum):
    """How one runner expects its input and output paths on the CLI."""

    MODERN = "modern"
    LEGACY_DIRECTORY_IO = "legacy_directory_io"


@dataclass(frozen=True)
class PreparedBenchmarkCase:
    """Filesystem layout for one prepared benchmark invocation."""

    input_dir: Path
    output_dir: Path
    copied_input: Path

    def metrics_path(self, case: BenchmarkCase) -> Path:
        """Return the expected `.compare.csv` path for this prepared case."""
        return self.output_dir / f"{case.output_stem()}.compare.csv"


@dataclass(frozen=True)
class MetricDifference:
    """One observed current-vs-Jan-9 legacy metric delta for reporting."""

    metric: ClassifiedMetricName
    direction: ImprovementDirection
    delta: float
    current_value: float
    baseline_value: float
    reason: str | None = None

    def summary(self) -> str:
        """Render the observed delta in a stable report-friendly format."""
        verb = "improved" if self.direction == "current_better" else "regressed"
        detail = (
            f"{self.metric} {verb} by {self.delta:.4f} "
            f"(current={self.current_value:.4f}, baseline={self.baseline_value:.4f})"
        )
        if self.reason is None:
            return detail
        return f"{detail}: {self.reason}"


@dataclass(frozen=True)
class CommandExecution:
    """Captured subprocess result for one runner invocation."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def failed(self) -> bool:
        """Return whether the subprocess exited non-zero."""
        return self.returncode != 0

    def summary(self, label: str) -> str:
        """Summarize one subprocess with enough context to debug quickly."""
        return (
            f"{label} exited with exit code {self.returncode}: {shlex.join(self.argv)}\n"
            f"stdout:\n{self.stdout or '<empty>'}\n"
            f"stderr:\n{self.stderr or '<empty>'}"
        )

    def failure_summary(self, label: str) -> str:
        """Summarize a failed subprocess with enough context to debug quickly."""
        return self.summary(label)


@dataclass
class CaseReport:
    """Full report for one curated comparison case."""

    case: BenchmarkCase
    status: CaseStatus
    run_root: Path
    current_command: tuple[str, ...]
    baseline_command: tuple[str, ...] | None
    current_metrics: BenchmarkMetrics | None = None
    baseline_metrics: BenchmarkMetrics | None = None
    current_execution: CommandExecution | None = None
    baseline_execution: CommandExecution | None = None
    current_error: str | None = None
    baseline_error: str | None = None
    allowlisted_differences: list[MetricDifference] = field(default_factory=list)
    improvements: list[MetricDifference] = field(default_factory=list)
    unexpected_regressions: list[MetricDifference] = field(default_factory=list)


@dataclass
class HarnessReport:
    """Aggregate harness result across all requested cases."""

    cases: list[CaseReport]
    run_root: Path
    dry_run: bool

    def has_failures(self) -> bool:
        """Return whether any case failed or regressed unexpectedly."""
        return any(
            case.status in {"current_failed", "baseline_failed"}
            or bool(case.unexpected_regressions)
            for case in self.cases
        )


class HarnessConfigError(Exception):
    """Raised when the harness configuration or fixture manifest is invalid."""


class HarnessExecutionError(Exception):
    """Raised when the harness cannot resolve the requested runner command."""


def _parse_metric_name(raw: str, context: str) -> ClassifiedMetricName:
    """Parse one classified metric name with explicit schema validation."""
    if raw == "wer":
        return "wer"
    if raw == "matches":
        return "matches"
    if raw == "insertions":
        return "insertions"
    if raw == "deletions":
        return "deletions"
    raise HarnessConfigError(f"Unsupported metric {raw!r} in {context}")


def _resolve_input_path(raw: str, base_dir: Path) -> Path:
    """Resolve manifest paths relative to the manifest directory when needed."""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _expect_dict(value: JSONValue | None, context: str) -> dict[str, JSONValue]:
    """Require a JSON dictionary at one schema boundary."""
    if not isinstance(value, dict):
        raise HarnessConfigError(f"Expected {context} to be an object")
    return value


def _expect_list(value: JSONValue | None, context: str) -> list[JSONValue]:
    """Require a JSON list at one schema boundary."""
    if not isinstance(value, list):
        raise HarnessConfigError(f"Expected {context} to be a list")
    return value


def _expect_str(value: JSONValue | None, context: str) -> str:
    """Require a JSON string at one schema boundary."""
    if not isinstance(value, str):
        raise HarnessConfigError(f"Expected {context} to be a string")
    return value


def _expect_number(value: JSONValue | None, context: str) -> float:
    """Require a JSON number at one schema boundary."""
    if isinstance(value, bool) or not (
        isinstance(value, int) or isinstance(value, float)
    ):
        raise HarnessConfigError(f"Expected {context} to be a number")
    return float(value)


def _expect_optional_number(value: JSONValue | None, context: str) -> float | None:
    """Accept a JSON number or null for optional numeric fields."""
    if value is None:
        return None
    return _expect_number(value, context)


def _expect_optional_list_of_str(
    value: JSONValue | None, context: str
) -> tuple[str, ...]:
    """Accept a JSON string list or null for argv-like fields."""
    if value is None:
        return ()
    items = _expect_list(value, context)
    return tuple(_expect_str(item, f"{context}[]") for item in items)


def _parse_case_tolerances(
    raw: JSONValue | None,
    case_id: str,
) -> dict[ClassifiedMetricName, float]:
    """Load per-case tolerance overrides while keeping the default metric set stable."""
    tolerances = dict(DEFAULT_TOLERANCES)
    if raw is None:
        return tolerances

    parsed = _expect_dict(raw, f"case {case_id} tolerances")
    for key, value in parsed.items():
        metric = _parse_metric_name(key, f"case {case_id} tolerances")
        tolerances[metric] = _expect_number(value, f"case {case_id} tolerance {metric}")
    return tolerances


def load_cases(manifest_path: Path) -> list[BenchmarkCase]:
    """Load curated benchmark cases from the repo-local manifest."""
    root = _expect_dict(json.loads(manifest_path.read_text()), "manifest")
    schema_version = _expect_number(
        root.get("schema_version"), "manifest schema_version"
    )
    if int(schema_version) != 1:
        raise HarnessConfigError(
            f"Unsupported manifest schema_version {schema_version} in {manifest_path}"
        )

    base_dir = manifest_path.parent
    cases: list[BenchmarkCase] = []
    for index, raw_case in enumerate(
        _expect_list(root.get("cases"), "manifest cases"), start=1
    ):
        case_dict = _expect_dict(raw_case, f"case #{index}")
        case_id = _expect_str(case_dict.get("id"), f"case #{index} id")
        audio_path = _resolve_input_path(
            _expect_str(case_dict.get("audio_path"), f"case {case_id} audio_path"),
            base_dir,
        )
        gold_path = _resolve_input_path(
            _expect_str(case_dict.get("gold_path"), f"case {case_id} gold_path"),
            base_dir,
        )
        if not audio_path.is_file():
            raise HarnessConfigError(
                f"Case {case_id} audio fixture does not exist: {audio_path}"
            )
        if not gold_path.is_file():
            raise HarnessConfigError(
                f"Case {case_id} gold fixture does not exist: {gold_path}"
            )
        if audio_path.stem != gold_path.stem:
            raise HarnessConfigError(
                f"Case {case_id} expects matching audio/gold stems, got {audio_path.name} and {gold_path.name}"
            )
        cases.append(
            BenchmarkCase(
                id=case_id,
                description=_expect_str(
                    case_dict.get("description"), f"case {case_id} description"
                ),
                command=_expect_str(
                    case_dict.get("command"), f"case {case_id} command"
                ),
                audio_path=audio_path,
                gold_path=gold_path,
                argv=_expect_optional_list_of_str(
                    case_dict.get("argv"), f"case {case_id} argv"
                ),
                tolerances=_parse_case_tolerances(case_dict.get("tolerances"), case_id),
            )
        )
    return cases


def load_allowlist(allowlist_path: Path) -> list[AllowlistEntry]:
    """Load known Jan 9 legacy baseline bug entries from JSON."""
    root = _expect_dict(json.loads(allowlist_path.read_text()), "allowlist")
    schema_version = _expect_number(
        root.get("schema_version"), "allowlist schema_version"
    )
    if int(schema_version) != 1:
        raise HarnessConfigError(
            f"Unsupported allowlist schema_version {schema_version} in {allowlist_path}"
        )

    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(
        _expect_list(root.get("entries"), "allowlist entries"), start=1
    ):
        entry_dict = _expect_dict(raw_entry, f"allowlist entry #{index}")
        metric = _parse_metric_name(
            _expect_str(entry_dict.get("metric"), f"allowlist entry #{index} metric"),
            f"allowlist entry #{index}",
        )
        entries.append(
            AllowlistEntry(
                case_id=_expect_str(
                    entry_dict.get("case_id"), f"allowlist entry #{index} case_id"
                ),
                metric=metric,
                reason=_expect_str(
                    entry_dict.get("reason"), f"allowlist entry #{index} reason"
                ),
                max_delta=_expect_optional_number(
                    entry_dict.get("max_delta"),
                    f"allowlist entry #{index} max_delta",
                ),
            )
        )
    return entries


def _resolve_explicit_executable(raw: str) -> str:
    """Resolve one explicitly requested executable path or PATH lookup."""
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    discovered = shutil.which(raw)
    if discovered is not None:
        return discovered
    raise HarnessExecutionError(f"Could not resolve executable {raw!r}")


def resolve_current_runner(explicit: str | None) -> list[str]:
    """Resolve the current batchalign3 runner without changing public CLI surfaces."""
    if explicit is not None:
        return [_resolve_explicit_executable(explicit)]

    binary_name = "batchalign3.exe" if os.name == "nt" else "batchalign3"
    built_binary = REPO_ROOT / "target" / "debug" / binary_name
    if built_binary.is_file():
        return [str(built_binary)]

    discovered = shutil.which("batchalign3")
    if discovered is not None:
        return [discovered]

    cargo = shutil.which("cargo")
    if cargo is not None:
        return [cargo, "run", "-q", "-p", "batchalign", "--"]

    raise HarnessExecutionError(
        "Could not resolve the current batchalign3 runner. Build target/debug/batchalign3, put batchalign3 on PATH, or pass --current-executable."
    )


def resolve_baseline_runner(explicit: str | None) -> list[str] | None:
    """Resolve the optional Jan 9 2026 legacy baseline executable."""
    if explicit is not None:
        return [_resolve_explicit_executable(explicit)]

    for env_name in (
        "BATCHALIGN2_BASELINE_EXECUTABLE",
        "BATCHALIGN_BASELINE_EXECUTABLE",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            return [_resolve_explicit_executable(env_value)]

    for candidate in JAN9_BASELINE_CANDIDATES:
        if candidate.exists():
            return [str(candidate)]

    return None


def plan_benchmark_case(case: BenchmarkCase, case_root: Path) -> PreparedBenchmarkCase:
    """Compute one benchmark case layout without touching the filesystem."""
    input_dir = case_root / "input"
    output_dir = case_root / "output"
    copied_input = input_dir / case.audio_path.name
    return PreparedBenchmarkCase(
        input_dir=input_dir, output_dir=output_dir, copied_input=copied_input
    )


def prepare_benchmark_case(
    case: BenchmarkCase, case_root: Path
) -> PreparedBenchmarkCase:
    """Copy one benchmark fixture pair into an isolated case root."""
    prepared = plan_benchmark_case(case, case_root)
    prepared.input_dir.mkdir(parents=True, exist_ok=True)
    prepared.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case.audio_path, prepared.copied_input)
    shutil.copy2(case.gold_path, prepared.input_dir / f"{case.audio_path.stem}.cha")
    return prepared


def plan_compare_case(case: BenchmarkCase, case_root: Path) -> PreparedBenchmarkCase:
    """Compute one compare-case layout for rescoring a legacy `.asr.cha` output."""
    input_dir = case_root / "input"
    output_dir = case_root / "output"
    copied_input = input_dir / f"{case.output_stem()}.cha"
    return PreparedBenchmarkCase(
        input_dir=input_dir, output_dir=output_dir, copied_input=copied_input
    )


def prepare_compare_case(
    case: BenchmarkCase,
    case_root: Path,
    main_chat_path: Path,
) -> PreparedBenchmarkCase:
    """Copy one legacy ASR transcript plus its gold CHAT for current compare scoring."""
    prepared = plan_compare_case(case, case_root)
    prepared.input_dir.mkdir(parents=True, exist_ok=True)
    prepared.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_chat_path, prepared.copied_input)
    shutil.copy2(case.gold_path, prepared.input_dir / f"{case.output_stem()}.gold.cha")
    return prepared


def build_case_command(
    runner: list[str],
    case: BenchmarkCase,
    prepared: PreparedBenchmarkCase,
    cli_style: CliInvocationStyle,
) -> tuple[str, ...]:
    """Build one benchmark invocation for a specific runner and CLI surface."""
    argv = [*runner, case.command, *case.argv]
    if cli_style is CliInvocationStyle.MODERN:
        argv.extend(["--output", str(prepared.output_dir), str(prepared.copied_input)])
    else:
        argv.extend([str(prepared.input_dir), str(prepared.output_dir)])
    return tuple(argv)


def build_compare_command(
    runner: list[str], prepared: PreparedBenchmarkCase
) -> tuple[str, ...]:
    """Build one current `compare` invocation for rescoring a legacy transcript."""
    return tuple(
        [
            *runner,
            "compare",
            "--output",
            str(prepared.output_dir),
            str(prepared.input_dir),
        ]
    )


def _seed_runner_config(home_dir: Path) -> None:
    """Seed one isolated home with a usable config for noninteractive comparisons."""
    config_path = home_dir / ".batchalign.ini"
    config_source = os.environ.get(CONFIG_SOURCE_ENV)
    if config_source:
        source_path = Path(config_source).expanduser()
        if not source_path.is_file():
            raise HarnessExecutionError(
                f"{CONFIG_SOURCE_ENV} points at a missing file: {source_path}"
            )
        shutil.copy2(source_path, config_path)
        return
    if config_path.is_file():
        return
    config_path.write_text("[asr]\nengine = whisper\n", encoding="utf-8")


def _prepare_runner_env(case_root: Path, label: str) -> dict[str, str]:
    """Isolate runner HOME/state roots so live caches do not bleed across runs."""
    env = dict(os.environ)
    home_dir = case_root / label / "home"
    state_dir = case_root / label / "state"
    home_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _seed_runner_config(home_dir)
    env.pop("BATCHALIGN_SERVER", None)
    # Strip PYTHON_GIL: if the caller runs under a free-threaded Python 3.14t
    # build with PYTHON_GIL=0, that env var would be inherited by legacy-runner
    # subprocesses that use #!/usr/bin/env python3 (non-freethreaded), causing
    # "Disabling the GIL is not supported by this build" crashes.
    env.pop("PYTHON_GIL", None)
    env["BATCHALIGN_NO_BROWSER"] = "1"
    env["HOME"] = str(home_dir)
    env["BATCHALIGN_STATE_DIR"] = str(state_dir)
    repo_python = (
        REPO_ROOT
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    )
    if repo_python.is_file():
        env["BATCHALIGN_PYTHON"] = str(repo_python)
        env.setdefault("BATCHALIGN_SIDECAR_PYTHON", str(repo_python))
    if os.name == "nt":
        env["USERPROFILE"] = str(home_dir)
    return env


def _kill_exact_pid(pid: int) -> None:
    """Terminate one exact daemon PID without name-based matching."""
    if pid <= 0:
        return
    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _cleanup_runner_daemons(case_root: Path, label: str) -> None:
    """Kill any exact daemon PIDs left behind by one isolated runner root."""
    state_dir = case_root / label / "state"
    for name in ("daemon.json", "sidecar-daemon.json"):
        info_path = state_dir / name
        if not info_path.is_file():
            continue
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        pid = payload.get("pid")
        if isinstance(pid, int):
            _kill_exact_pid(pid)


def _execute_command(argv: tuple[str, ...], env: dict[str, str]) -> CommandExecution:
    """Run one harness subprocess and capture stdout/stderr for reporting."""
    completed = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandExecution(
        argv=argv,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_metrics_csv(path: Path) -> BenchmarkMetrics:
    """Parse the benchmark `.compare.csv` emitted by batchalign's compare pipeline."""
    if not path.is_file():
        raise HarnessExecutionError(
            f"Expected benchmark metrics file was not written: {path}"
        )

    metrics: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("metric")
            value = row.get("value")
            if metric is None or value is None:
                raise HarnessExecutionError(f"Malformed metrics row in {path}: {row}")
            metrics[metric] = value

    required = {
        "wer",
        "accuracy",
        "matches",
        "insertions",
        "deletions",
        "total_gold_words",
        "total_main_words",
    }
    missing = required.difference(metrics)
    if missing:
        raise HarnessExecutionError(
            f"Missing metrics {sorted(missing)} in benchmark CSV {path}"
        )

    return BenchmarkMetrics(
        wer=float(metrics["wer"]),
        accuracy=float(metrics["accuracy"]),
        matches=int(metrics["matches"]),
        insertions=int(metrics["insertions"]),
        deletions=int(metrics["deletions"]),
        total_gold_words=int(metrics["total_gold_words"]),
        total_main_words=int(metrics["total_main_words"]),
    )


def _available_output_artifacts(output_dir: Path) -> str:
    """Summarize the files currently present in one runner output directory."""
    artifacts = (
        sorted(path.name for path in output_dir.iterdir())
        if output_dir.is_dir()
        else []
    )
    return ", ".join(artifacts) if artifacts else "<none>"


def _rescore_legacy_asr_with_current_compare(
    current_runner: list[str],
    case: BenchmarkCase,
    case_root: Path,
    baseline_prepared: PreparedBenchmarkCase,
    baseline_execution: CommandExecution,
) -> BenchmarkMetrics:
    """Normalize a legacy transcript-only output through the current compare command."""
    legacy_asr_path = baseline_prepared.output_dir / f"{case.output_stem()}.asr.cha"
    legacy_chat_path = baseline_prepared.output_dir / f"{case.output_stem()}.cha"
    if legacy_asr_path.is_file():
        transcript_path = legacy_asr_path
    elif legacy_chat_path.is_file():
        if filecmp.cmp(legacy_chat_path, case.gold_path, shallow=False):
            raise HarnessExecutionError(
                "Legacy baseline wrote no modern `.compare.csv` and only left a `.cha` "
                "identical to the gold input, which indicates the benchmark failed "
                "before producing ASR output.\n"
                f"Available artifacts: {_available_output_artifacts(baseline_prepared.output_dir)}\n"
                + baseline_execution.summary("baseline runner")
            )
        transcript_path = legacy_chat_path
    else:
        raise HarnessExecutionError(
            "Legacy baseline wrote no modern `.compare.csv`, no legacy `.asr.cha`, "
            f"and no transcript `.cha` to rescore. Expected one of: {legacy_asr_path} "
            f"or {legacy_chat_path}. "
            f"Available artifacts: {_available_output_artifacts(baseline_prepared.output_dir)}\n"
            + baseline_execution.summary("baseline runner")
        )

    rescore_root = case_root / "baseline-compare"
    shutil.rmtree(rescore_root, ignore_errors=True)
    prepared = prepare_compare_case(case, rescore_root, transcript_path)
    compare_command = build_compare_command(current_runner, prepared)
    compare_env = _prepare_runner_env(case_root, "baseline-compare")
    try:
        compare_execution = _execute_command(compare_command, compare_env)
    finally:
        _cleanup_runner_daemons(case_root, "baseline-compare")

    if compare_execution.failed():
        raise HarnessExecutionError(
            "Failed to rescore legacy baseline output with the current `compare` command:\n"
            + compare_execution.failure_summary("baseline compare scorer")
        )

    metrics = _parse_metrics_csv(prepared.metrics_path(case))
    shutil.copy2(prepared.metrics_path(case), baseline_prepared.metrics_path(case))
    return metrics


def _match_allowlist(
    case: BenchmarkCase,
    metric: ClassifiedMetricName,
    delta: float,
    entries: list[AllowlistEntry],
) -> AllowlistEntry | None:
    """Find one allowlist entry that explains a current-beats-baseline delta."""
    for entry in entries:
        if entry.case_id != case.id or entry.metric != metric:
            continue
        if entry.max_delta is not None and delta > entry.max_delta + 0.0001:
            continue
        return entry
    return None


def _classify_metric(
    case: BenchmarkCase,
    metric: ClassifiedMetricName,
    current: BenchmarkMetrics,
    baseline: BenchmarkMetrics,
    allowlist_entries: list[AllowlistEntry],
) -> MetricDifference | None:
    """Classify one metric delta as improvement, regression, or no-op."""
    tolerance = case.tolerances[metric]
    current_value = current.value(metric)
    baseline_value = baseline.value(metric)

    if metric in {"wer", "insertions", "deletions"}:
        worse_gap = current_value - baseline_value
        if worse_gap > tolerance:
            return MetricDifference(
                metric=metric,
                direction="current_worse",
                delta=worse_gap,
                current_value=current_value,
                baseline_value=baseline_value,
            )
        improvement_gap = baseline_value - current_value
        if improvement_gap > tolerance:
            matched = _match_allowlist(case, metric, improvement_gap, allowlist_entries)
            return MetricDifference(
                metric=metric,
                direction="current_better",
                delta=improvement_gap,
                current_value=current_value,
                baseline_value=baseline_value,
                reason=None if matched is None else matched.reason,
            )
        return None

    worse_gap = baseline_value - current_value
    if worse_gap > tolerance:
        return MetricDifference(
            metric=metric,
            direction="current_worse",
            delta=worse_gap,
            current_value=current_value,
            baseline_value=baseline_value,
        )

    improvement_gap = current_value - baseline_value
    if improvement_gap > tolerance:
        matched = _match_allowlist(case, metric, improvement_gap, allowlist_entries)
        return MetricDifference(
            metric=metric,
            direction="current_better",
            delta=improvement_gap,
            current_value=current_value,
            baseline_value=baseline_value,
            reason=None if matched is None else matched.reason,
        )
    return None


def classify_case(
    case: BenchmarkCase,
    current: BenchmarkMetrics,
    baseline: BenchmarkMetrics,
    allowlist_entries: list[AllowlistEntry],
) -> tuple[list[MetricDifference], list[MetricDifference], list[MetricDifference]]:
    """Split metric deltas into allowlisted baseline bugs, improvements, and regressions."""
    allowlisted: list[MetricDifference] = []
    improvements: list[MetricDifference] = []
    regressions: list[MetricDifference] = []

    for metric in CLASSIFIED_METRICS:
        difference = _classify_metric(
            case, metric, current, baseline, allowlist_entries
        )
        if difference is None:
            continue
        if difference.direction == "current_worse":
            regressions.append(difference)
        elif difference.reason is not None:
            allowlisted.append(difference)
        else:
            improvements.append(difference)

    return allowlisted, improvements, regressions


def run_harness(
    cases: list[BenchmarkCase],
    allowlist_entries: list[AllowlistEntry],
    current_runner: list[str],
    baseline_runner: list[str] | None,
    run_root: Path,
    dry_run: bool,
) -> HarnessReport:
    """Execute the requested comparison cases under isolated runner roots."""
    reports: list[CaseReport] = []
    run_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_root = run_root / case.id
        current_case_root = case_root / "current"
        baseline_case_root = case_root / "baseline"
        current_command = build_case_command(
            current_runner,
            case,
            plan_benchmark_case(case, current_case_root),
            CliInvocationStyle.MODERN,
        )
        baseline_command = (
            None
            if baseline_runner is None
            else build_case_command(
                baseline_runner,
                case,
                plan_benchmark_case(case, baseline_case_root),
                CliInvocationStyle.LEGACY_DIRECTORY_IO,
            )
        )
        report = CaseReport(
            case=case,
            status="planned" if dry_run else "baseline_skipped",
            run_root=case_root,
            current_command=current_command,
            baseline_command=baseline_command,
        )

        if dry_run:
            reports.append(report)
            continue

        current_env = _prepare_runner_env(case_root, "current")
        current_prepared = prepare_benchmark_case(case, current_case_root)
        try:
            current_execution = _execute_command(current_command, current_env)
        finally:
            _cleanup_runner_daemons(case_root, "current")
        report.current_execution = current_execution
        if current_execution.failed():
            report.status = "current_failed"
            report.current_error = current_execution.failure_summary("current runner")
            reports.append(report)
            continue

        try:
            report.current_metrics = _parse_metrics_csv(
                current_prepared.metrics_path(case)
            )
        except HarnessExecutionError as exc:
            report.status = "current_failed"
            report.current_error = str(exc)
            reports.append(report)
            continue

        if baseline_command is None:
            report.status = "baseline_skipped"
            reports.append(report)
            continue

        baseline_env = _prepare_runner_env(case_root, "baseline")
        baseline_prepared = prepare_benchmark_case(case, baseline_case_root)
        try:
            baseline_execution = _execute_command(baseline_command, baseline_env)
        finally:
            _cleanup_runner_daemons(case_root, "baseline")
        report.baseline_execution = baseline_execution
        if baseline_execution.failed():
            report.status = "baseline_failed"
            report.baseline_error = baseline_execution.failure_summary(
                "baseline runner"
            )
            reports.append(report)
            continue

        try:
            baseline_metrics_path = baseline_prepared.metrics_path(case)
            if baseline_metrics_path.is_file():
                report.baseline_metrics = _parse_metrics_csv(baseline_metrics_path)
            else:
                report.baseline_metrics = _rescore_legacy_asr_with_current_compare(
                    current_runner,
                    case,
                    case_root,
                    baseline_prepared,
                    baseline_execution,
                )
        except HarnessExecutionError as exc:
            report.status = "baseline_failed"
            report.baseline_error = str(exc)
            reports.append(report)
            continue

        allowlisted, improvements, regressions = classify_case(
            case,
            report.current_metrics,
            report.baseline_metrics,
            allowlist_entries,
        )
        report.allowlisted_differences = allowlisted
        report.improvements = improvements
        report.unexpected_regressions = regressions
        report.status = "completed"
        reports.append(report)

    return HarnessReport(cases=reports, run_root=run_root, dry_run=dry_run)


def _print_case_listing(cases: list[BenchmarkCase]) -> None:
    """Print the curated comparison cases without executing anything."""
    for case in cases:
        args = " ".join(case.argv) if case.argv else "<none>"
        print(f"{case.id}: {case.description}")
        print(f"  command : {case.command}")
        print(f"  audio   : {case.audio_path}")
        print(f"  gold    : {case.gold_path}")
        print(f"  argv    : {args}")


def _print_report(report: HarnessReport, baseline_runner: list[str] | None) -> None:
    """Render the full harness result in a concise CLI-friendly format."""
    print(f"Run root: {report.run_root}")
    if baseline_runner is None:
        print(
            "Baseline runner: skipped (set BATCHALIGN2_BASELINE_EXECUTABLE or "
            "--baseline-executable to a Jan 9 2026 legacy batchalign/batchalignhk runner)"
        )
    else:
        print(f"Baseline runner: {shlex.join(baseline_runner)}")
    if report.dry_run:
        print("Dry run only; commands were planned but not executed.")

    for case in report.cases:
        print(f"\nCase {case.case.id}: {case.case.description}")
        print(f"  current cmd : {shlex.join(case.current_command)}")
        if case.baseline_command is None:
            print("  baseline cmd: <skipped>")
        else:
            print(f"  baseline cmd: {shlex.join(case.baseline_command)}")

        if case.current_metrics is not None:
            print(f"  current     : {case.current_metrics.summary()}")
        if case.baseline_metrics is not None:
            print(f"  baseline    : {case.baseline_metrics.summary()}")

        if case.status == "baseline_skipped":
            print("  status      : baseline skipped")
        elif case.status == "planned":
            print("  status      : planned")
        elif case.status == "completed":
            print("  status      : completed")
        elif case.status == "current_failed":
            print("  status      : current runner failed")
            if case.current_error is not None:
                print(case.current_error)
        elif case.status == "baseline_failed":
            print("  status      : baseline runner failed")
            if case.baseline_error is not None:
                print(case.baseline_error)

        if case.allowlisted_differences:
            print("  allowlisted baseline differences:")
            for difference in case.allowlisted_differences:
                print(f"    - {difference.summary()}")
        if case.improvements:
            print("  unallowlisted improvements:")
            for difference in case.improvements:
                print(f"    - {difference.summary()}")
        if case.unexpected_regressions:
            print("  unexpected regressions:")
            for difference in case.unexpected_regressions:
                print(f"    - {difference.summary()}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI flags for the Jan 9 legacy comparison harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the curated benchmark manifest JSON.",
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="Path to the known Jan 9 legacy baseline-bug allowlist JSON.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="case_ids",
        help="Run only the named case id (repeatable).",
    )
    parser.add_argument(
        "--current-executable",
        help="Path to the current batchalign3 executable. Defaults to target/debug/batchalign3, PATH, or cargo run.",
    )
    parser.add_argument(
        "--baseline-executable",
        help=(
            "Path to a Jan 9 2026 legacy `batchalign` or `batchalignhk` baseline executable. "
            "Defaults to BATCHALIGN2_BASELINE_EXECUTABLE / "
            "BATCHALIGN_BASELINE_EXECUTABLE or known maintainer-local pinned "
            "wrappers if present."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not auto-discover or execute any Jan 9 legacy baseline runner.",
    )
    parser.add_argument(
        "--run-root",
        help="Directory where runner state and outputs should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve cases and planned commands without executing them.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print curated case metadata and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Jan 9 legacy comparison harness."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    allowlist_path = Path(args.allowlist).expanduser().resolve()

    try:
        cases = load_cases(manifest_path)
        allowlist_entries = load_allowlist(allowlist_path)
    except (OSError, json.JSONDecodeError, HarnessConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case.id in requested]
        missing = sorted(requested.difference({case.id for case in cases}))
        if missing:
            print(f"Unknown case ids: {', '.join(missing)}", file=sys.stderr)
            return 1

    if args.list_cases:
        _print_case_listing(cases)
        return 0

    try:
        current_runner = resolve_current_runner(args.current_executable)
        baseline_runner = (
            None
            if args.skip_baseline
            else resolve_baseline_runner(args.baseline_executable)
        )
    except HarnessExecutionError as exc:
        print(f"Runner resolution error: {exc}", file=sys.stderr)
        return 1

    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root is not None
        else Path(tempfile.mkdtemp(prefix="stock-batchalign-compare-"))
    )

    report = run_harness(
        cases=cases,
        allowlist_entries=allowlist_entries,
        current_runner=current_runner,
        baseline_runner=baseline_runner,
        run_root=run_root,
        dry_run=args.dry_run,
    )
    _print_report(report, baseline_runner)
    return 1 if report.has_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
