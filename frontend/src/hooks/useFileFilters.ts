import { useMemo, useState } from "react";
import type { FileStatusEntry } from "../types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type FilterTab = "all" | "error" | "diagnosed" | "processing" | "done" | "queued";

export type ErrorGroup = {
  category: string;        // "input" | "media" | "system" | "processing"
  categoryLabel: string;   // "CHAT Parse Error" etc.
  label: string;           // first line of the first file's error
  files: FileStatusEntry[];
};

export type FileCounts = {
  all: number;
  error: number;
  /** Output written with admission diagnostics: not a success, not a failure. */
  diagnosed: number;
  processing: number;
  done: number;
  queued: number;
};

const PAGE_SIZE = 50;

const STATUS_ORDER: Record<string, number> = {
  error: 0,
  diagnosed: 1,
  processing: 2,
  done: 3,
  queued: 4,
};

/**
 * Maps backend `FailureCategory` wire values to display-friendly group names.
 *
 * Backend categories (from Rust `FailureCategory` enum) are kebab-cased:
 *   validation, parse_error, input_missing, worker_crash, worker_timeout,
 *   worker_protocol, provider_transient, provider_terminal, memory_pressure,
 *   cancelled, system, model_access_denied.
 *
 * We collapse these into 6 user-facing groups:
 *   input, media, system, processing, validation, model_access.
 *
 * `worker_bootstrap` has no entry here (a pre-existing gap, not introduced by
 * `model_access_denied`): it falls through to the raw-slug rendering the
 * fallback below describes, same as any category added without an entry.
 */
const CATEGORY_NORMALIZE: Record<string, string> = {
  validation: "validation",
  parse_error: "input",
  input_missing: "media",
  worker_crash: "system",
  worker_timeout: "system",
  worker_protocol: "system",
  provider_transient: "processing",
  provider_terminal: "processing",
  memory_pressure: "system",
  cancelled: "system",
  system: "system",
  // A configuration/credential condition on the SERVER's machine (a gated
  // Hugging Face model, a missing token), never the caller's bad input, so
  // it gets its own bucket rather than folding into "validation" (which
  // renders the "pipeline bug, not your input" banner) or "system".
  model_access_denied: "model_access",
  // Legacy/fallback values from older display groups
  input: "input",
  media: "media",
  processing: "processing",
};

const CATEGORY_DISPLAY: Record<string, string> = {
  input: "CHAT Parse Error",
  media: "Media Not Found",
  system: "System Error",
  processing: "Processing Error",
  validation: "Pipeline Bug",
  model_access: "Model Access Required",
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useFileFilters(files: FileStatusEntry[]) {
  const [activeTab, setActiveTab] = useState<FilterTab>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);

  // Counts per status
  const counts: FileCounts = useMemo(() => {
    let error = 0, diagnosed = 0, processing = 0, done = 0, queued = 0;
    for (const f of files) {
      if (f.status === "error") error++;
      else if (f.status === "diagnosed") diagnosed++;
      else if (f.status === "processing") processing++;
      else if (f.status === "done") done++;
      else if (f.status === "queued") queued++;
    }
    return { all: files.length, error, diagnosed, processing, done, queued };
  }, [files]);

  // Error groups: category -> code -> files
  const errorGroups: ErrorGroup[] = useMemo(() => {
    const errorFiles = files.filter((f) => f.status === "error");
    if (errorFiles.length === 0) return [];

    // Group by normalized display category. Backend sends fine-grained
    // FailureCategory values (worker_crash, provider_transient, etc.);
    // we collapse them into user-friendly groups.
    const catMap = new Map<string, FileStatusEntry[]>();
    for (const f of errorFiles) {
      const rawCat = f.error_category ?? "processing";
      const cat = CATEGORY_NORMALIZE[rawCat] ?? "processing";
      const list = catMap.get(cat);
      if (list) list.push(f);
      else catMap.set(cat, [f]);
    }

    const groups: ErrorGroup[] = [];
    for (const [cat, catFiles] of catMap) {
      // The first line of the first file's error labels the group.
      const firstError = catFiles[0]?.error ?? "Unknown error";
      groups.push({
        category: cat,
        categoryLabel: CATEGORY_DISPLAY[cat] ?? cat,
        label: firstError.split("\n")[0],
        files: catFiles,
      });
    }

    // Sort categories: validation first (pipeline bugs), then input, media, processing, system
    const catOrder: Record<string, number> = { validation: 0, input: 1, media: 2, processing: 3, system: 4 };
    groups.sort((a, b) => (catOrder[a.category] ?? 99) - (catOrder[b.category] ?? 99));
    return groups;
  }, [files]);

  // Filtered + sorted files
  const filteredFiles = useMemo(() => {
    let result = files;

    // Tab filter
    if (activeTab !== "all") {
      result = result.filter((f) => f.status === activeTab);
    }

    // Search filter (by filename)
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((f) => f.filename.toLowerCase().includes(q));
    }

    // Sort: errors first, then processing, done, queued; alphabetical within
    return [...result].sort(
      (a, b) =>
        (STATUS_ORDER[a.status] ?? 99) - (STATUS_ORDER[b.status] ?? 99) ||
        a.filename.localeCompare(b.filename),
    );
  }, [files, activeTab, searchQuery]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredFiles.length / PAGE_SIZE));

  // Clamp page when filters change
  const clampedPage = Math.min(page, totalPages);
  if (clampedPage !== page) {
    // Schedule state update for next render
    queueMicrotask(() => setPage(clampedPage));
  }

  const pageFiles = filteredFiles.slice(
    (clampedPage - 1) * PAGE_SIZE,
    clampedPage * PAGE_SIZE,
  );

  return {
    // State
    activeTab,
    setActiveTab: (tab: FilterTab) => { setActiveTab(tab); setPage(1); },
    searchQuery,
    setSearchQuery: (q: string) => { setSearchQuery(q); setPage(1); },
    page: clampedPage,
    setPage,
    // Derived
    counts,
    errorGroups,
    filteredFiles,
    pageFiles,
    totalPages,
    pageSize: PAGE_SIZE,
  };
}
