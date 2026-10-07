/**
 * How the dashboard presents a failed file: by its error's own category.
 *
 * The server classifies every failure (`FailureCategory`, from the Rust
 * `scheduling::FailureCategory` enum) and the dashboard groups and labels
 * files by that classification, never by a guess. Before 2026-10-07 the
 * `validation` category, which is the input failing a command's own checks
 * (an `@Media` header align cannot use, a refused transcript), was labelled
 * "Pipeline Bug" with a banner telling the user it was not their input. That
 * was backwards, and three categories had no entry at all.
 *
 * Both maps are `Record`s over closed unions, so a category the server adds
 * fails type-checking here until it is given a group (the generated
 * `FailureCategory` union is the server's list), and a group without a
 * presentation cannot exist.
 */
import type { components } from "./generated/api";

/** The server's failure classification, as the wire carries it. */
export type FailureCategory = components["schemas"]["FailureCategory"];

/** The groups failed files are shown in. */
export type ErrorGroupKind =
  | "input_refused"
  | "parse"
  | "input_missing"
  | "evidence"
  | "analysis"
  | "model_access"
  | "engine"
  | "worker_setup"
  | "system"
  | "cancelled"
  | "unclassified";

/** How one group is shown. */
export type ErrorGroupPresentation = {
  /** The group's name, in the error panel and on a file's chip. */
  label: string;
  /** Position in the error panel: the input's own problems first. */
  order: number;
  /** Chip colours on a file row. */
  chip: string;
  /** Left border of the group in the error panel. */
  border: string;
  /** Text colour of the group's name. */
  text: string;
  /** One sentence under the group's name, where the category alone says
   * something the user needs; absent otherwise. */
  banner?: { text: string; className: string };
};

/** Every category's group. */
const CATEGORY_GROUP: Record<FailureCategory, ErrorGroupKind> = {
  // The input failed the command's own checks; the message names the change.
  validation: "input_refused",
  parse_error: "parse",
  input_missing: "input_missing",
  evidence_unavailable: "evidence",
  analysis_unavailable: "analysis",
  // A configuration/credential condition on the SERVER's machine (a gated
  // Hugging Face model, a missing token): neither bad input nor a defect.
  model_access_denied: "model_access",
  provider_transient: "engine",
  provider_terminal: "engine",
  // A deterministic worker start-up failure (model load, missing language
  // pack); the message is the worker's own and actionable.
  worker_bootstrap: "worker_setup",
  worker_crash: "system",
  worker_timeout: "system",
  worker_protocol: "system",
  memory_pressure: "system",
  system: "system",
  cancelled: "cancelled",
};

/** How each group is shown. */
export const ERROR_GROUPS: Record<ErrorGroupKind, ErrorGroupPresentation> = {
  input_refused: {
    label: "Input Refused",
    order: 0,
    chip: "bg-amber-50 text-amber-700",
    border: "border-amber-300",
    text: "text-amber-700",
    banner: {
      text: "The input was refused before any output was written. Each file's error below names what to change in it.",
      className: "text-amber-700",
    },
  },
  parse: {
    label: "CHAT Parse Error",
    order: 1,
    chip: "bg-amber-50 text-amber-600",
    border: "border-amber-200",
    text: "text-amber-700",
  },
  input_missing: {
    label: "Input Not Found",
    order: 2,
    chip: "bg-purple-50 text-purple-600",
    border: "border-purple-200",
    text: "text-purple-700",
  },
  evidence: {
    label: "Evidence Unavailable",
    order: 3,
    chip: "bg-indigo-50 text-indigo-600",
    border: "border-indigo-200",
    text: "text-indigo-700",
  },
  analysis: {
    label: "Analysis Unavailable",
    order: 4,
    chip: "bg-indigo-50 text-indigo-600",
    border: "border-indigo-200",
    text: "text-indigo-700",
  },
  model_access: {
    label: "Model Access Required",
    order: 5,
    chip: "bg-sky-50 text-sky-600",
    border: "border-sky-200",
    text: "text-sky-700",
    banner: {
      text: "A required model needs access approval or a Hugging Face token on this machine. This is not a problem with your input files.",
      className: "text-sky-600",
    },
  },
  engine: {
    label: "Engine Error",
    order: 6,
    chip: "bg-orange-50 text-orange-600",
    border: "border-orange-200",
    text: "text-orange-700",
  },
  worker_setup: {
    label: "Worker Setup Error",
    order: 7,
    chip: "bg-orange-50 text-orange-600",
    border: "border-orange-200",
    text: "text-orange-700",
  },
  system: {
    label: "System Error",
    order: 8,
    chip: "bg-red-50 text-red-600",
    border: "border-red-200",
    text: "text-red-700",
  },
  cancelled: {
    label: "Cancelled",
    order: 9,
    chip: "bg-zinc-100 text-zinc-600",
    border: "border-zinc-200",
    text: "text-zinc-600",
  },
  unclassified: {
    label: "Unclassified Error",
    order: 10,
    chip: "bg-zinc-100 text-zinc-500",
    border: "border-zinc-200",
    text: "text-zinc-600",
  },
};

/**
 * The group a failed file belongs to. A file whose failure carries no
 * category (a status restored from an older database row) is
 * `unclassified`, and so is a category a newer server sends that this build
 * does not know: shown as what it is, never folded into another group.
 */
export function errorGroupOf(category: FailureCategory | null | undefined): ErrorGroupKind {
  if (category == null) return "unclassified";
  // The wire is a boundary: the generated type is this build's list, and a
  // newer server may send a value outside it.
  return CATEGORY_GROUP[category] ?? "unclassified";
}
