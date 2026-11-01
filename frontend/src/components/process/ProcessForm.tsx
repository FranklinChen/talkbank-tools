/** One-command launcher. Configuration and submitted jobs are distinct states;
 * the submitted state retains the admitted plan and returned job identity. */
import { useEffect, useRef, useState } from "react";
import { CommandPicker, INPUT_POLICIES, type CommandDef } from "./CommandPicker";
import { FolderPicker, type FolderSelection } from "./FolderPicker";
import { OutputModeSelector } from "./OutputModeSelector";
import { ProcessingProgress } from "./ProcessingProgress";
import { UnconfirmedSubmission } from "./UnconfirmedSubmission";
import { JobPlan, PreparedJob, type OutputChoice } from "./jobPlan";
import { SelectedInputs, type DiscoveredInput } from "./inputSelection";
import { useSubmitJob } from "../../hooks/useSubmitJob";
import { useDesktopFiles } from "../../desktop/DesktopContext";

type Configuration = {
  kind: "configure"; session: symbol; command: CommandDef; input: DiscoveredInput | null;
  output: OutputChoice; lang: string; speakers: string; error: string | null;
};
type InputAttempt = { kind: "input"; configuration: Configuration };
type SelectionAttempt = InputAttempt | { kind: "output"; configuration: Configuration };
type FormState = { kind: "choose" } | Configuration
  | { kind: "picking"; attempt: SelectionAttempt }
  | { kind: "scanning"; attempt: InputAttempt; folder: string }
  | { kind: "unconfirmed"; command: CommandDef; plan: JobPlan; message: string }
  | { kind: "processing"; plan: JobPlan; jobId: string };

/** Configure one checked plan and submit it once to the admitted local service. */
export function ProcessForm({ isServerReady }: { isServerReady: boolean }) {
  const files = useDesktopFiles();
  const [form, setForm] = useState<FormState>({ kind: "choose" });
  const submit = useSubmitJob();
  const selectionOwner = useRef<SelectionAttempt | null>(null);
  useEffect(() => () => { selectionOwner.current = null; }, []);
  function reset() { selectionOwner.current = null; setForm({ kind: "choose" }); submit.reset(); }
  if (form.kind === "processing") return <ProcessingProgress
    jobId={form.jobId} totalFiles={form.plan.request.sourcePaths.length}
    command={form.plan.request.command} outputFolder={form.plan.outputFolder} onReset={reset} />;
  if (form.kind === "unconfirmed") return <UnconfirmedSubmission command={form.command}
    plan={form.plan} message={form.message} onConfigure={reset} />;
  switch (submit.state.kind) {
    case "checking-paths": return <p role="status">Checking input and output files...</p>;
    case "reading-preferences": return <p role="status">Reading ASR preferences...</p>;
    case "sending": return <p role="status">Submitting job...</p>;
    case "available": case "finished": case "retired": break;
  }
  if (form.kind === "choose") return <CommandPicker onSelect={command => setForm({
    kind: "configure", session: Symbol(), command, input: null,
    output: { kind: "separate", folder: null }, lang: "eng", speakers: "1", error: null,
  })} />;
  const current = form.kind === "configure" ? form : form.attempt.configuration;
  const inputPolicy = INPUT_POLICIES[current.command.input];
  const inputView: FolderSelection = form.kind === "scanning" ? { kind: "scanning", folder: form.folder }
    : form.kind === "picking" && form.attempt.kind === "input" ? { kind: "picking" }
    : current.input?.kind === "selected" ? { kind: "ready", input: current.input.input }
    : current.input?.kind === "empty" ? { kind: "no-matches", folder: current.input.folder } : { kind: "empty" };
  function update(change: (previous: Configuration) => Configuration) {
    setForm(previous => previous.kind === "configure" && previous.session === current.session ? change(previous) : previous);
  }
  async function chooseFolder(kind: SelectionAttempt["kind"]) {
    if (form.kind !== "configure" || selectionOwner.current || submit.state.kind !== "available") return;
    const attempt: SelectionAttempt = kind === "input"
      ? { kind: "input", configuration: current } : { kind: "output", configuration: current };
    selectionOwner.current = attempt;
    setForm({ kind: "picking", attempt });
    // Only the owner may settle this interaction. Back or unmount withdraws
    // ownership before a delayed dialog or scan can affect another screen.
    function settle(configuration: Configuration) {
      if (selectionOwner.current === attempt) setForm(configuration);
    }
    try {
      const folder = await (attempt.kind === "input"
        ? files.pickFolder("Select " + current.command.label + " input folder") : files.pickOutputFolder());
      if (selectionOwner.current !== attempt) return;
      if (!folder) { settle(current); return; }
      switch (attempt.kind) {
        case "output": settle({ ...current, output: { kind: "separate", folder }, error: null }); break;
        case "input": {
          setForm({ kind: "scanning", attempt, folder });
          const discovered = await files.discoverFiles(folder, [...inputPolicy.extensions]);
          settle({ ...current, input: SelectedInputs.fromDiscovery(folder, discovered), error: null });
          break;
        }
      }
    } catch (error) {
      switch (attempt.kind) {
        case "input": settle({ ...current, input: null, error: "Input folder selection failed. Choose the input folder again. " + String(error) }); break;
        case "output": settle({ ...current, error: String(error) }); break;
      }
    } finally {
      if (selectionOwner.current === attempt) selectionOwner.current = null;
    }
  }
  async function start() {
    if (form.kind !== "configure" || current.input?.kind !== "selected" || selectionOwner.current || submit.state.kind !== "available") return;
    let attempt: PreparedJob;
    try {
      attempt = PreparedJob.prepare(current.command, current.input.input, current.output, current.lang, current.speakers);
    } catch (error) { update(previous => ({ ...previous, error: String(error) })); return; }
    // The available snapshot owns the claim. A stale handler cannot start it
    // twice, and unmount retires pre-POST work inside the launch controller.
    const outcome = await submit.state.start(attempt);
    switch (outcome.kind) {
      case "accepted": setForm({ kind: "processing", plan: outcome.plan, jobId: outcome.job.id }); return;
      case "refused": update(previous => ({ ...previous, error: outcome.message })); return;
      case "unconfirmed": setForm({ kind: "unconfirmed", command: current.command, plan: outcome.plan, message: outcome.message }); return;
      case "retired": return;
    }
    return outcome satisfies never;
  }
  const outputMissing = current.output.kind === "separate" && !current.output.folder;
  return <div className="max-w-xl mx-auto space-y-5">
    <button type="button" onClick={reset} disabled={submit.state.kind !== "available"} aria-label="Back">Back</button>
    {form.kind === "picking" && <p role="status">Choosing {form.attempt.kind} folder...</p>}
    {form.kind === "scanning" && <p role="status">Scanning input folder: {form.folder}</p>}
    <fieldset disabled={form.kind !== "configure"} className="space-y-5">
    <h2 className="text-lg font-semibold">{current.command.label}</h2>
    <FolderPicker label="Choose input files" selection={inputView} policy={inputPolicy} onChoose={() => { void chooseFolder("input"); }} />
    {current.command.id === "benchmark" && <p className="text-sm text-gray-600">
      Choose recordings with matching reference transcripts beside them (for example, clip.wav and clip.cha).
    </p>}
    <OutputModeSelector mode={current.output.kind} allowInPlace={current.command.id !== "benchmark"}
      onChange={mode => update(previous => ({ ...previous, output: mode === "in-place" ? { kind: "in-place" } : { kind: "separate", folder: null }, error: null }))} />
    {current.output.kind === "separate" && <div>
      <label className="block text-sm font-medium mb-2">Output folder</label>
      <button type="button" onClick={() => { void chooseFolder("output"); }} disabled={submit.state.kind !== "available"}
        className="w-full border border-dashed rounded p-4 text-left">
        {current.output.folder ?? "Click to choose output folder"}
      </button>
    </div>}
    {current.command.language === "per-file" && <p className="text-sm text-gray-600">Uses each transcript’s @Languages header.</p>}
    {current.command.language === "job" && <label className="block text-sm">Language (three-letter code)
      <input aria-label="Language" value={current.lang} disabled={submit.state.kind !== "available"}
        onChange={event => update(previous => ({ ...previous, lang: event.target.value, error: null }))}
        className="block border rounded p-2 mt-1" />
    </label>}
    {(current.command.id === "transcribe" || current.command.id === "benchmark") && <p className="text-sm text-gray-600">
      Uses the ASR engine selected in Settings. Rev.AI uploads recordings to a paid cloud service; Whisper runs locally.
    </p>}
    {(current.command.id === "transcribe" || current.command.id === "benchmark") && <label className="block text-sm">Number of speakers
      <input aria-label="Number of speakers" type="number" min="1" step="1" value={current.speakers}
        onChange={event => update(previous => ({ ...previous, speakers: event.target.value, error: null }))}
        className="block border rounded p-2 mt-1" />
    </label>}
    {current.error && <p role="alert" className="text-red-700">{current.error}</p>}
    <button type="button" onClick={start}
      disabled={!isServerReady || current.input?.kind !== "selected" || outputMissing || submit.state.kind !== "available"}
      className="w-full rounded p-3 bg-indigo-600 text-white disabled:opacity-50">
      Start Processing
    </button>
    {!isServerReady && <p role="status">Waiting for the processing service...</p>}
    </fieldset>
  </div>;
}
