import { useMemo, useState } from "react";
import type { FileStatusEntry } from "../types";
import { ERROR_GROUPS, errorGroupOf, type ErrorGroupKind } from "../errorCategories";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type FilterTab = "all" | "error" | "diagnosed" | "processing" | "done" | "queued";

export type ErrorGroup = {
  /** The group the files' error category puts them in. */
  category: ErrorGroupKind;
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

    // Group by the error's own category: the server's fine-grained
    // FailureCategory values (worker_crash, provider_transient, etc.) map to
    // user-facing groups in one place (`errorCategories.ts`).
    const catMap = new Map<ErrorGroupKind, FileStatusEntry[]>();
    for (const f of errorFiles) {
      const cat = errorGroupOf(f.error_category);
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
        categoryLabel: ERROR_GROUPS[cat].label,
        label: firstError.split("\n")[0],
        files: catFiles,
      });
    }

    // The input's own problems first, then the engine's, then the system's.
    groups.sort((a, b) => ERROR_GROUPS[a.category].order - ERROR_GROUPS[b.category].order);
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
