import { useState } from "react";
import type { ErrorGroup } from "../hooks/useFileFilters";
import { ErrorFileGroup } from "./ErrorFileGroup";
import { ERROR_GROUPS } from "../errorCategories";

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
          {errorGroups.map((group) => {
            const shown = ERROR_GROUPS[group.category];
            return (
              <div
                key={group.category}
                className={`border-l-2 ${shown.border} pl-3 mb-3 last:mb-0`}
              >
                {/* Category header */}
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs font-semibold ${shown.text}`}>
                    {group.categoryLabel}
                  </span>
                  <span className="text-[10px] text-zinc-400">
                    ({group.files.length} {group.files.length === 1 ? "file" : "files"})
                  </span>
                </div>

                {/* What the category alone tells the user, where it does. */}
                {shown.banner && (
                  <p className={`text-[11px] ${shown.banner.className} mb-2 italic`}>
                    {shown.banner.text}
                  </p>
                )}

                <ErrorFileGroup label={group.label} files={group.files} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
