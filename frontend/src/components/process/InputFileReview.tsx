/** Read-only review of the exact selected population. Native details/buttons
 * provide keyboard access; only one bounded page is mounted at a time. */
import { useState } from "react";
import { InputFilePage, type SelectedInputs } from "./inputSelection";

/** Inspection changes only the displayed page, never which files are submitted. */
export function InputFileReview({ input }: { input: SelectedInputs }) {
  const [page, setPage] = useState(() => InputFilePage.first(input));
  const current = page.forInput(input);
  const next = current.next;
  const previous = current.previous;
  return <details className="mt-3 rounded border border-gray-200 p-3 text-sm">
    <summary className="cursor-pointer font-medium">Review {current.total} input {current.total === 1 ? "file" : "files"}</summary>
    <section aria-label="Input files to process" className="mt-3 space-y-3">
      <p>Showing {current.firstNumber}–{current.lastNumber} of {current.total}. The job uses all {current.total} input files; reviewing pages does not change the selection.</p>
      <ul className="max-h-60 overflow-y-auto space-y-1 font-mono text-xs">
        {current.files.map(path => <li key={path} className="break-all">{path}</li>)}
      </ul>
      <nav aria-label="Input file pages" className="flex gap-3">
        <button type="button" disabled={!previous} onClick={previous ? () => setPage(previous) : undefined}
          className="rounded border px-3 py-1 disabled:opacity-50">Previous files</button>
        <button type="button" disabled={!next} onClick={next ? () => setPage(next) : undefined}
          className="rounded border px-3 py-1 disabled:opacity-50">Next files</button>
      </nav>
    </section>
  </details>;
}
