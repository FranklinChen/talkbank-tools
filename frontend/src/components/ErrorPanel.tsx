import { useState } from "react";
import type { ErrorGroup } from "../hooks/useFileFilters";
import { ErrorFileGroup } from "./ErrorFileGroup";

const CATEGORY_COLORS: Record<string, string> = {
  input: "border-amber-200",
  media: "border-purple-200",
  system: "border-red-200",
  processing: "border-orange-200",
  validation: "border-rose-300",
  model_access: "border-sky-200",
};

const CATEGORY_TEXT: Record<string, string> = {
  input: "text-amber-700",
  media: "text-purple-700",
  system: "text-red-700",
  processing: "text-orange-700",
  validation: "text-rose-700",
  model_access: "text-sky-700",
};

export function ErrorPanel({ errorGroups }: { errorGroups: ErrorGroup[] }) {
  const [collapsed, setCollapsed] = useState(false);

  if (errorGroups.length === 0) return null;

  const totalErrors = errorGroups.reduce((sum, g) => sum + g.files.length, 0);

  return (
    <div className="bg-red-50/50 border border-red-100 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        type="button"
        className="w-full flex items-center gap-2 px-4 py-2.5 text-left cursor-pointer hover:bg-red-50/80"
        onClick={() => setCollapsed(!collapsed)}
      >
        <span className="text-[10px] text-zinc-400">
          {collapsed ? "\u25B6" : "\u25BC"}
        </span>
        <span className="text-sm font-medium text-red-700">
          {totalErrors} {totalErrors === 1 ? "error" : "errors"}
        </span>
        <span className="text-xs text-red-500">
          {errorGroups.map((g) => `${g.categoryLabel} (${g.files.length})`).join(" \u00b7 ")}
        </span>
      </button>

      {/* Body */}
      {!collapsed && (
        <div className="px-4 pb-3">
          {errorGroups.map((group) => (
            <div
              key={group.category}
              className={`border-l-2 ${CATEGORY_COLORS[group.category] ?? "border-zinc-200"} pl-3 mb-3 last:mb-0`}
            >
              {/* Category header */}
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`text-xs font-semibold ${CATEGORY_TEXT[group.category] ?? "text-zinc-600"}`}
                >
                  {group.categoryLabel}
                </span>
                <span className="text-[10px] text-zinc-400">
                  ({group.files.length} {group.files.length === 1 ? "file" : "files"})
                </span>
              </div>

              {/* Validation bug banner */}
              {group.category === "validation" && (
                <p className="text-[11px] text-rose-600 mb-2 italic">
                  This is a pipeline bug, not your input. The file's error below
                  is the report; please pass it on.
                </p>
              )}

              {/* Model access banner: a configuration/credential condition on
                  the server's machine, not a batchalign defect and not the
                  caller's bad input. Each file's own message below already
                  names the repository and the remedy. */}
              {group.category === "model_access" && (
                <p className="text-[11px] text-sky-600 mb-2 italic">
                  A required model needs access approval or a Hugging Face
                  token on this machine. This is not a pipeline bug and not a
                  problem with your input files.
                </p>
              )}

              <ErrorFileGroup label={group.label} files={group.files} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
