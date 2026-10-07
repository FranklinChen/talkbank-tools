/** Structured error display with suggested recovery actions.
 *
 * Maps server error categories to user-friendly messages and actionable
 * suggestions. Shown in the processing progress view when files fail.
 * Replaces raw server error strings with context-appropriate guidance
 * for nontechnical users.
 */

import type { FileStatusEntry } from "../../types";
import type { components } from "../../generated/api";

type FailureCategory = components["schemas"]["FailureCategory"];

interface ErrorInfo {
  /** User-friendly title for the error category. */
  title: string;
  /** Plain-language explanation of what went wrong. */
  explanation: string;
  /** Suggested actions the user can take. */
  suggestions: string[];
}

/**
 * Every server failure category's guidance. A `Record` over the generated
 * `FailureCategory` union, so a category the server adds fails type-checking
 * here until it has guidance; it used to be a `switch` whose `default` showed
 * "Unexpected error" for four categories it did not name.
 */
const RECOVERY: Record<FailureCategory, ErrorInfo> = {
  validation: {
    title: "Input refused",
    explanation:
      "The file was refused before any output was written, because it fails a check this command makes. The error message names what to change.",
    suggestions: [
      "Read the error message next to the file: it names the change to make",
      "Make that change in the file (or the option it names) and run again",
    ],
  },
  parse_error: {
    title: "File could not be read",
    explanation: "The file format was not recognized or is corrupted.",
    suggestions: [
      "Make sure the file is a valid .cha file (not renamed from another format)",
      "Check that the file encoding is UTF-8",
      "Try opening the file in a text editor to verify its contents",
    ],
  },
  input_missing: {
    title: "File not found",
    explanation: "The input file could not be found at the expected location.",
    suggestions: [
      "Make sure the file hasn't been moved or deleted",
      "Check that the file path is correct",
      "Try selecting the folder again",
    ],
  },
  evidence_unavailable: {
    title: "Required evidence unavailable",
    explanation:
      "The file needs evidence this run could not produce, such as timing a linked transcript must carry. The error message names what would provide it.",
    suggestions: [
      "Read the error message next to the file: it names the remedies",
      "Try the option or header change it names, then run again",
    ],
  },
  analysis_unavailable: {
    title: "Analysis unavailable",
    explanation:
      "The file asks for an analysis the configured engine does not provide (for example a language it has no model for).",
    suggestions: [
      "Check the language and engine options for this file",
      "Read the error message next to the file for the analysis it could not run",
    ],
  },
  worker_crash: {
    title: "Processing engine crashed",
    explanation:
      "The ML model encountered an unexpected error while processing this file.",
    suggestions: [
      "Try processing the file again, transient crashes often resolve on retry",
      "If it keeps failing, the file may have unusual content that triggers a bug",
      "Check that your machine has enough free memory",
    ],
  },
  worker_timeout: {
    title: "Processing timed out",
    explanation: "The file took too long to process and was stopped.",
    suggestions: [
      "Large audio files may need more time, try again with fewer files",
      "Make sure your machine isn't running low on memory or CPU",
      "For very long recordings, consider splitting them into shorter segments",
    ],
  },
  worker_protocol: {
    title: "Internal communication error",
    explanation: "The processing engine sent an unexpected response.",
    suggestions: [
      "Try restarting the server and processing again",
      "This is likely a bug, please report it if it persists",
    ],
  },
  worker_bootstrap: {
    title: "Processing engine could not start",
    explanation:
      "A model or language pack the engine needs could not be loaded. Retrying will not help until that is fixed.",
    suggestions: [
      "Read the error message next to the file: it is the engine's own report",
      "Check the network connection and free disk space, then run again",
    ],
  },
  provider_transient: {
    title: "Temporary service error",
    explanation: "The cloud ASR service (Rev.AI) returned a temporary error.",
    suggestions: [
      "Wait a moment and try again, the service may be briefly overloaded",
      "Check your internet connection",
      "If using Rev.AI, verify your API key is still valid",
    ],
  },
  provider_terminal: {
    title: "Service rejected the request",
    explanation: "The cloud ASR service returned a permanent error for this file.",
    suggestions: [
      "Check that your Rev.AI API key is valid and has remaining credit",
      "The audio file may be in an unsupported format",
      "Try switching to Whisper (local) as the ASR engine",
    ],
  },
  memory_pressure: {
    title: "Not enough memory",
    explanation:
      "The server ran out of available memory and had to stop processing.",
    suggestions: [
      "Close other applications to free up memory",
      "Try processing fewer files at a time",
      "Restart the server and try again",
    ],
  },
  cancelled: {
    title: "Processing was cancelled",
    explanation: "This file was cancelled before it finished processing.",
    suggestions: ["You can restart the job to try again"],
  },
  system: {
    title: "System error",
    explanation: "An internal or server-side step failed.",
    suggestions: [
      "Try processing the file again",
      "If the error persists, restart the server",
      "Check the server logs for more details",
    ],
  },
  model_access_denied: {
    title: "Model access required",
    explanation:
      "A required model needs access approval or a Hugging Face token on this machine. This is not a problem with your input files.",
    suggestions: [
      "Read the error message next to the file: it names the model and the remedy",
    ],
  },
};

/** A failure with no recorded category, or one this build does not know. */
const UNCLASSIFIED: ErrorInfo = {
  title: "Unclassified error",
  explanation: "The server recorded no category this dashboard knows for this failure.",
  suggestions: [
    "Read the error message next to the file",
    "Check the server logs for more details",
  ],
};

/** The guidance for one category. The wire is a boundary: a newer server
 * may send a category outside this build's list. */
function categorize(category: FailureCategory | null | undefined): ErrorInfo {
  if (category == null) return UNCLASSIFIED;
  return RECOVERY[category] ?? UNCLASSIFIED;
}

interface ErrorRecoveryProps {
  /** Files that have errors. */
  errorFiles: FileStatusEntry[];
}

export function ErrorRecovery({ errorFiles }: ErrorRecoveryProps) {
  if (errorFiles.length === 0) return null;

  // Group errors by category for a cleaner display
  // Keyed by the category itself; an unrecorded one is its own group,
  // never folded into "system".
  const byCategory = new Map<FailureCategory | null, FileStatusEntry[]>();
  for (const f of errorFiles) {
    const key = f.error_category ?? null;
    const group = byCategory.get(key) ?? [];
    group.push(f);
    byCategory.set(key, group);
  }

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-red-700">
        {errorFiles.length} file{errorFiles.length !== 1 ? "s" : ""} had errors
      </h3>

      {[...byCategory.entries()].map(([category, files]) => {
        const info = categorize(category);
        return (
          <div
            key={category ?? "unclassified"}
            className="bg-red-50 border border-red-200 rounded-lg p-4"
          >
            <div className="text-sm font-medium text-red-800">
              {info.title}
            </div>
            <p className="text-xs text-red-600 mt-1">{info.explanation}</p>

            {/* Affected files */}
            <div className="mt-2 space-y-1">
              {files.map((f) => (
                <div key={f.filename} className="text-xs text-red-700">
                  <span className="font-mono">{f.filename}</span>
                  {f.error && (
                    <span className="text-red-500 ml-1">: {f.error}</span>
                  )}
                </div>
              ))}
            </div>

            {/* Suggestions */}
            <div className="mt-3 border-t border-red-200 pt-2">
              <div className="text-xs font-medium text-red-700 mb-1">
                What to try:
              </div>
              <ul className="text-xs text-red-600 space-y-0.5 list-disc list-inside">
                {info.suggestions.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          </div>
        );
      })}
    </div>
  );
}
