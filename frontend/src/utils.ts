import type { components } from "./generated/api";

type FileProgressStage = components["schemas"]["FileProgressStage"];
type LanguageSpec = components["schemas"]["LanguageSpec"];
type FileOutputDiagnostics = components["schemas"]["FileOutputDiagnostics"];
type FullFindings = components["schemas"]["FullFindings"];
type JudgementBar = components["schemas"]["JudgementBar"];
type StageRefusalRecord = components["schemas"]["StageRefusalRecord"];
type OutputShortfallRecord = components["schemas"]["OutputShortfallRecord"];
type OptionalStage = components["schemas"]["OptionalStage"];
type UntimedCauseRecord = components["schemas"]["UntimedCauseRecord"];
type RefusedWindowTrace = components["schemas"]["RefusedWindowTrace"];

/** Formatting helpers. */

/**
 * Display string for a job's language.
 *
 * `LanguageSpec` is one string on the wire: `auto`, `per-file`, a 3-letter code
 * such as `eng`, or a code-switched pair such as `eng,spa`, which displays as
 * written. An absent language displays as `eng`, the submission default.
 *
 * This used to test for `"Auto"` and `{ Resolved: ... }`, the shapes a derived
 * OpenAPI schema claimed and the server never sent.
 */
export function displayLang(spec: LanguageSpec | undefined): string {
  return spec ?? "eng";
}

/**
 * Whether a LanguageSpec represents the default (eng), used to hide
 * the language badge when it would just say "eng".
 */
export function isDefaultLang(spec: LanguageSpec | undefined): boolean {
  return displayLang(spec) === "eng";
}

/**
 * Canonical dashboard-side labels for typed file progress stages.
 *
 * The server also derives `progress_label`, but the dashboard prefers the
 * stable stage code when present so UI logic is not coupled to free-form text.
 */
const PROGRESS_STAGE_LABELS: Record<FileProgressStage, string> = {
  processing: "Processing",
  reading: "Reading",
  resolving_audio: "Resolving audio",
  recovering_utterance_timing: "Recovering utterance timing",
  recovering_timing_fallback: "Recovering timing (fallback)",
  aligning: "Aligning",
  transcribing: "Transcribing",
  benchmarking: "Benchmarking",
  checking_cache: "Checking cache",
  applying_results: "Applying results",
  post_processing: "Post-processing",
  building_chat: "Building CHAT",
  segmenting_utterances: "Segmenting utterances",
  analyzing_morphosyntax: "Analyzing morphosyntax",
  finalizing: "Finalizing",
  writing: "Writing",
  parsing: "Parsing",
  analyzing: "Analyzing",
  segmenting: "Segmenting",
  translating: "Translating",
  resolving_coreference: "Resolving coreference",
  comparing: "Comparing",
  retry_scheduled: "Retry scheduled",
  waiting_for_worker: "Waiting for a worker",
};

/**
 * Newest first, for two server times. Server times are RFC 3339 UTC with
 * exactly three fractional digits (`MachineTime`), so string order is time
 * order and a plain comparison suffices.
 */
export function compareTimesNewestFirst(a: string, b: string): number {
  return a < b ? 1 : a > b ? -1 : 0;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m`;
  }
  return `${m}m ${s}s`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const now = Date.now();
  const then = new Date(iso).getTime();
  const diffS = Math.floor((now - then) / 1000);
  if (diffS < 10) return "just now";
  if (diffS < 60) return `${diffS}s ago`;
  const diffM = Math.floor(diffS / 60);
  if (diffM < 60) return `${diffM}m ago`;
  const diffH = Math.floor(diffM / 60);
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.floor(diffH / 24);
  return `${diffD}d ago`;
}

export function statusColor(status: string): string {
  switch (status) {
    case "queued":
      return "bg-gray-100 text-gray-700";
    case "running":
    case "processing":
      return "bg-blue-100 text-blue-700";
    case "completed":
    case "done":
      return "bg-green-100 text-green-700";
    // Output written with admission diagnostics: neither green (a clean
    // success) nor red (a failure).
    case "diagnosed":
      return "bg-yellow-100 text-yellow-800";
    case "failed":
    case "error":
      return "bg-red-100 text-red-700";
    case "cancelled":
      return "bg-amber-100 text-amber-700";
    case "interrupted":
      return "bg-orange-100 text-orange-700";
    case "writeback_failed":
      return "bg-amber-100 text-amber-700";
    default:
      return "bg-gray-100 text-gray-700";
  }
}

/** Status dot color (Tailwind bg-* class). */
export function statusDotColor(status: string): string {
  switch (status) {
    case "queued":
      return "bg-amber-400";
    case "running":
    case "processing":
      return "bg-blue-500";
    case "completed":
    case "done":
      return "bg-emerald-500";
    case "diagnosed":
      return "bg-yellow-400";
    case "failed":
    case "error":
      return "bg-red-500";
    case "cancelled":
      return "bg-gray-400";
    case "interrupted":
      return "bg-orange-400";
    case "writeback_failed":
      return "bg-amber-500";
    default:
      return "bg-gray-400";
  }
}

/** Command badge styling: [bgClass, textClass]. */
export function commandStyle(cmd: string): [string, string] {
  switch (cmd) {
    case "align":
      return ["bg-indigo-100", "text-indigo-700"];
    case "morphotag":
      return ["bg-violet-100", "text-violet-700"];
    case "transcribe":
    case "transcribe_s":
      return ["bg-emerald-100", "text-emerald-700"];
    case "benchmark":
      return ["bg-amber-100", "text-amber-700"];
    case "translate":
      return ["bg-teal-100", "text-teal-700"];
    case "opensmile":
      return ["bg-rose-100", "text-rose-700"];
    case "utseg":
      return ["bg-sky-100", "text-sky-700"];
    case "coref":
      return ["bg-orange-100", "text-orange-700"];
    default:
      return ["bg-gray-100", "text-gray-600"];
  }
}

/** Compact display for source_dir: show last 2-3 path components. */
export function shortPath(p: string | null | undefined): string {
  if (!p) return "";
  const parts = p.replace(/\/+$/, "").split("/").filter(Boolean);
  if (parts.length <= 3) return parts.join("/");
  return "\u2026/" + parts.slice(-3).join("/");
}

export function progressPercent(completed: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((completed / total) * 100);
}

/** Friendly display name for the submitter. */
export function submitterName(
  byName: string | null | undefined,
  byIp: string | null | undefined,
): string {
  if (byName && byName !== byIp) return byName;
  if (byIp) return byIp;
  return "";
}

/** Pretty-print JSON-ish API values for read-only dashboard display. */
export function formatJsonDisplay(value: unknown): string {
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/**
 * Resolve the operator-facing progress label for one file.
 *
 * The typed `progress_stage` is preferred when present because it is the
 * stable contract. The older `progress_label` remains as a display fallback.
 */
export function displayProgressLabel(
  stage: FileProgressStage | null | undefined,
  label: string | null | undefined,
): string {
  if (stage) {
    return PROGRESS_STAGE_LABELS[stage] ?? label ?? "";
  }
  return label ?? "";
}

/**
 * One line for a file whose output was written with admission diagnostics
 * (status `diagnosed`): the output is on disk, and this says how many
 * findings it carries. Never phrased as a success or as an error.
 */
export function diagnosedSummary(
  diagnostics: FileOutputDiagnostics | null | undefined,
): string {
  if (!diagnostics) return "written, diagnostics not recorded";
  const n = diagnostics.findings?.finding_count ?? 0;
  let line = `written, ${n} ${n === 1 ? "diagnostic" : "diagnostics"}`;
  const shortfalls = diagnostics.shortfalls ?? [];
  const stages = shortfalls.filter(
    (shortfall) => shortfall.kind === "stage_skipped" || shortfall.kind === "stage_not_applied",
  ).length;
  if (stages > 0) {
    line += `, ${stages} requested stage${stages === 1 ? "" : "s"} not applied`;
  }
  for (const shortfall of shortfalls) {
    if (shortfall.kind === "timing_incomplete") {
      line += `, ${shortfall.untimed_words} of ${shortfall.required_words} words untimed`;
    } else if (shortfall.kind === "stage_held_out") {
      line += `, ${stageName(shortfall.stage)} left out of ${shortfall.held_out_utterances} utterance${shortfall.held_out_utterances === 1 ? "" : "s"}`;
    }
  }
  return line;
}

/**
 * The lines a diagnosed file reports: the bar and its first findings (with
 * the code), how many more there are and where the full list is, then each
 * shortfall. The same lines the CLI prints (`FileOutputDiagnostics::lines`).
 */
export function diagnosticLines(diagnostics: FileOutputDiagnostics): string[] {
  const lines: string[] = [];
  const findings = diagnostics.findings;
  if (findings) {
    lines.push(`judged against: ${judgementBarText(findings.bar)}`);
    const first = findings.first_findings ?? [];
    for (const finding of first) {
      lines.push(finding.code ? `${finding.code} ${finding.message}` : finding.message);
    }
    if (findings.finding_count > first.length) {
      lines.push(moreFindingsLine(findings.finding_count - first.length, findings.full_findings));
    }
  }
  for (const shortfall of diagnostics.shortfalls ?? []) {
    lines.push(shortfallText(shortfall));
  }
  return lines;
}

/** One shortfall as a line: the server's `OutputShortfallRecord` Display. */
function shortfallText(shortfall: OutputShortfallRecord): string {
  switch (shortfall.kind) {
    case "stage_skipped":
      return `skipped ${stageName(shortfall.stage)}: it requires an admitted document, and the generated output was written with its diagnostics instead`;
    case "stage_not_applied":
      return `${stageName(shortfall.stage)} not applied: its output could not be admitted (${stageRefusalText(shortfall.refusal)}); the admitted document from before it was written instead`;
    case "stage_held_out": {
      const first = shortfall.first_held_out[0];
      const firstText = first !== undefined ? ` (first: utterance ${first})` : "";
      return `${stageName(shortfall.stage)} applied except to ${shortfall.held_out_utterances} utterance(s) that carry the findings, which keep their generated form${firstText}`;
    }
    case "timing_incomplete": {
      const first = shortfall.first_untimed[0];
      const firstText = first
        ? ` (first: utterance ${first.utterance}, ${first.untimed_words} of ${first.words} words, ${untimedCauseText(first.cause)})`
        : "";
      return `timing incomplete: ${shortfall.untimed_words} of ${shortfall.required_words} words in ${shortfall.untimed_utterances} utterance(s) have no timing and were written without it${firstText}`;
    }
  }
}

/** The server's `OptionalStage::name`. */
function stageName(stage: OptionalStage): string {
  switch (stage) {
    case "utterance_segmentation":
      return "utterance segmentation";
    case "morphosyntax":
      return "morphosyntax";
  }
}

/** The server's `UntimedCauseRecord` Display. */
function untimedCauseText(cause: UntimedCauseRecord): string {
  switch (cause.kind) {
    case "not_placed":
      return "no alignment request, the audio left for it could not contain its words";
    case "no_usable_timing":
      return "no usable timing";
    case "window_refused":
      return `no alignment request, its audio window was refused (${refusedWindowText(cause.window)})`;
    case "not_in_recording":
      return `not in the recording: [+ ${cause.postcode}]`;
  }
}

/** The cause phrase of the server's `UntimedCauseRecord` Display. */
function refusedWindowText(window: RefusedWindowTrace): string {
  switch (window.cause) {
    case "over_budget":
      return "longer than the alignment budget";
    case "empty":
      return "empty";
    case "inverted":
      return "inverted";
    case "past_recording":
      return "past the end of the recording";
    case "anchor_gap":
      return "over budget, with a gap between recovered anchors longer than the budget";
    case "anchors_unusable":
      return "over budget, with no usable recovered anchor to split at";
  }
}

/** The line saying how many findings are not shown and where they are. */
function moreFindingsLine(more: number, full: FullFindings): string {
  switch (full.kind) {
    case "inline":
      return `... and ${more} more finding(s)`;
    case "sidecar":
      return `... and ${more} more finding(s); the full list is in ${full.path}`;
    case "unwritten":
      return `... and ${more} more finding(s); the full list could not be written to ${full.path}: ${full.error}`;
  }
}

/**
 * A stage refusal as one phrase: the bar and how many findings with the
 * first one, or the producer's statement. The same text as the server's
 * `StageRefusalRecord` Display.
 */
function stageRefusalText(refusal: StageRefusalRecord): string {
  switch (refusal.kind) {
    case "judged": {
      const bar = judgementBarText(refusal.bar);
      const first = refusal.first_findings[0];
      const firstText = first
        ? `, first: ${first.code ? `${first.code} ${first.message}` : first.message}`
        : "";
      return `${bar}: ${refusal.finding_count} finding(s)${firstText}`;
    }
    case "unestablished":
      return refusal.reason;
  }
}

/** The server's `JudgementBar` Display. */
function judgementBarText(bar: JudgementBar): string {
  switch (bar) {
    case "construction":
      return "complete CHAT construction required";
    case "preservation":
      return "complete CHAT construction and preservation required";
  }
}
