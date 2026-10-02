import { useState } from "react";
import type { FileStatusEntry } from "../types";

function splitBasename(filename: string): string {
  const idx = filename.lastIndexOf("/");
  return idx === -1 ? filename : filename.slice(idx + 1);
}

function ErrorFileEntry({ file }: { file: FileStatusEntry }) {
  const [expanded, setExpanded] = useState(false);
  const basename = splitBasename(file.filename);
  const hasDetail = file.error != null && file.error.length > 0;

  return (
    <div>
      <div className="flex items-center gap-2 py-0.5">
        <button
          type="button"
          className={`font-mono text-xs ${hasDetail ? "hover:underline cursor-pointer" : ""} text-zinc-700`}
          onClick={() => hasDetail && setExpanded(!expanded)}
        >
          {basename}
        </button>
        {hasDetail && (
          <span className="text-[10px] text-zinc-400">
            {expanded ? "\u25BC" : "\u25B6"}
          </span>
        )}
      </div>
      {expanded && hasDetail && (
        <pre className="mt-1 mb-2 ml-4 p-2 bg-red-50 rounded text-[11px] text-red-700 font-mono whitespace-pre-wrap overflow-x-auto max-h-40 overflow-y-auto">
          {file.error}
        </pre>
      )}
    </div>
  );
}

export function ErrorFileGroup({
  label,
  files,
}: {
  label: string;
  files: FileStatusEntry[];
}) {
  const [collapsed, setCollapsed] = useState(files.length > 10);

  return (
    <div className="ml-4 mb-2">
      <button
        type="button"
        className="flex items-center gap-2 text-xs cursor-pointer hover:bg-zinc-50 rounded px-1 py-0.5 -ml-1"
        onClick={() => setCollapsed(!collapsed)}
      >
        <span className="text-[10px] text-zinc-400">
          {collapsed ? "\u25B6" : "\u25BC"}
        </span>
        <span className="text-zinc-600 truncate max-w-sm">{label}</span>
        <span className="text-zinc-400">
          ({files.length} {files.length === 1 ? "file" : "files"})
        </span>
      </button>
      {!collapsed && (
        <div className="ml-5 mt-1">
          {files.map((f) => (
            <ErrorFileEntry key={f.filename} file={f} />
          ))}
        </div>
      )}
    </div>
  );
}
