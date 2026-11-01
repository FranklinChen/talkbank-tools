/** Present the form-owned directory selection. A local busy state must never
 * leave the submission owner believing that an earlier folder is still ready. */
import { useDesktopEnvironment } from "../../desktop/DesktopContext";
import type { SelectedInputs } from "./inputSelection";
import type { INPUT_POLICIES } from "./CommandPicker";
import { InputFileReview } from "./InputFileReview";

export type FolderSelection = { kind: "empty" }
  | { kind: "no-matches"; folder: string }
  | { kind: "ready"; input: SelectedInputs }
  | { kind: "picking" }
  | { kind: "scanning"; folder: string };

interface FolderPickerProps {
  label: string; selection: FolderSelection; onChoose: () => void;
  policy: typeof INPUT_POLICIES[keyof typeof INPUT_POLICIES];
}

/** Render only the selection owned by the surrounding process state machine. */
export function FolderPicker({ label, selection, policy, onChoose }: FolderPickerProps) {
  const environment = useDesktopEnvironment();
  function selectedLabel() {
    switch (selection.kind) {
      case "picking": return "Choosing input folder...";
      case "scanning": return <>Scanning files...<span className="block text-xs">{selection.folder}</span></>;
      case "ready": return <><span>{selection.input.files.length} {selection.input.files.length === 1 ? "file" : "files"} found</span><span className="block text-xs">{selection.input.folder}</span></>;
      case "no-matches": return <>0 files found<span className="block text-xs">{selection.folder}</span></>;
      case "empty": return environment.isDesktop ? "Click to choose a folder" : "Folder selection requires the desktop app";
    }
    return selection satisfies never;
  }
  return <div>
    <label className="block text-sm font-medium mb-2">{label}</label>
    <button type="button" onClick={onChoose} disabled={!environment.isDesktop || selection.kind === "picking" || selection.kind === "scanning"}
      className="w-full border rounded p-5 text-left disabled:opacity-50">
      {selectedLabel()}
    </button>
    <p className="mt-2 text-xs text-gray-600">Looks for {policy.label}: {policy.extensions.map(ext => "." + ext).join(", ")}. Includes subfolders.</p>
    {selection.kind === "no-matches" && <p role="status" className="mt-2 text-sm text-amber-800">No {policy.label} found. Check the file types above or choose another folder.</p>}
    {(selection.kind === "ready" || selection.kind === "no-matches") && <p className="text-sm text-gray-600">Links inside the selected folder are skipped. Choose their original folder to include them.</p>}
    {selection.kind === "ready" && <InputFileReview input={selection.input} />}
  </div>;
}
