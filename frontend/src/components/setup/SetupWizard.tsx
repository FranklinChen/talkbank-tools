/** First-run choice, credential entry, saving and acknowledged completion are
 * distinct states. No screen can declare setup done before the native receipt. */
import { useEffect, useRef, useState } from "react";
import { useDesktopConfig } from "../../desktop/DesktopContext";
import { DESKTOP_APP_NAME } from "../../desktop/protocol";
import { PreferenceDraft, type PreferenceEdit, type SavedPreferences } from "../../desktop/preferences";
import { EngineCard } from "./EngineCard";

type Credentials = { key: string; confirmation: string };
type EngineSelection = { kind: "whisper" } | { kind: "rev"; credentials: Credentials };
type EngineStep = { kind: "engine"; selection: EngineSelection; error: string | null };
type KeyStep = { kind: "api-key"; credentials: Credentials; error: string | null };
type EditableStep = { kind: "welcome" } | EngineStep | KeyStep;
type WizardState = EditableStep
  | { kind: "saving"; edit: PreferenceEdit; previous: EngineStep | KeyStep }
  | { kind: "done"; receipt: SavedPreferences };
interface SetupWizardProps { onComplete: () => void }

/** Configure one validated edit and keep failures on the screen that owns it. */
export function SetupWizard({ onComplete }: SetupWizardProps) {
  const config = useDesktopConfig();
  const [state, setState] = useState<WizardState>({ kind: "welcome" });
  const operation = useRef<PreferenceEdit | null>(null);
  useEffect(() => () => { operation.current = null; }, []);
  function navigate(next: EditableStep) { if (!operation.current) setState(next); }

  async function advance(previous: EngineStep | KeyStep) {
    if (operation.current) return;
    let edit: PreferenceEdit;
    try {
      switch (previous.kind) {
        case "engine":
          if (previous.selection.kind === "rev") {
            navigate({ kind: "api-key", credentials: previous.selection.credentials, error: null });
            return;
          }
          edit = PreferenceDraft.firstRun("whisper", "").prepare();
          break;
        case "api-key":
          if (previous.credentials.key !== previous.credentials.confirmation) throw new Error("Keys do not match. Please try again.");
          edit = PreferenceDraft.firstRun("rev", previous.credentials.key).prepare();
          break;
      }
    } catch (error) { setState({ ...previous, error: String(error) }); return; }
    operation.current = edit;
    setState({ kind: "saving", edit, previous });
    try {
      const receipt = await config.writeConfig(edit);
      if (operation.current === edit) setState({ kind: "done", receipt });
    } catch (error) {
      if (operation.current === edit) setState({ ...previous, error: "Could not confirm setup saved. " + String(error) });
    } finally { if (operation.current === edit) operation.current = null; }
  }

  const view = state.kind === "saving" ? state.previous : state;
  return <div className="min-h-screen flex items-center justify-center bg-[var(--bg-page)]">
    <div className="max-w-lg w-full mx-4 space-y-4">
      <fieldset disabled={state.kind === "saving"} className="space-y-5">
        {view.kind === "welcome" && <div className="text-center space-y-6">
          <h1 className="text-2xl font-bold">Welcome to {DESKTOP_APP_NAME}</h1>
          <p>Set up {DESKTOP_APP_NAME} to use your installed batchalign3 CLI.</p>
          <p>Rev.AI uploads recordings to a paid cloud service. Whisper runs locally and needs model downloads. You can change the preference later in Settings.</p>
          <button type="button" onClick={() => navigate({ kind: "engine", selection: { kind: "whisper" }, error: null })}
            className="rounded bg-indigo-600 text-white px-6 py-3">Let's Get Started</button>
        </div>}
        {view.kind === "engine" && <div className="space-y-5">
          <h2 className="text-lg font-semibold">Choose your ASR engine</h2>
          <p className="text-sm">This sets the default for speech-to-text tasks.</p>
          <EngineCard name="Rev.AI" description="Paid cloud service; uploads recordings."
            pros={["No local ASR model required"]} cons={["Requires an API key", "Uses a paid service"]}
            selected={view.selection.kind === "rev"} onSelect={() => navigate({ ...view, selection: view.selection.kind === "rev" ? view.selection : { kind: "rev", credentials: { key: "", confirmation: "" } }, error: null })} />
          <EngineCard name="Whisper" description="Local inference; requires model downloads."
            pros={["No ASR API key required"]} cons={["Uses local compute", "First-use downloads"]}
            selected={view.selection.kind === "whisper"} onSelect={() => navigate({ ...view, selection: { kind: "whisper" }, error: null })} />
          {view.error && <p role="alert">{view.error}</p>}
          <div className="flex justify-between">
            <button type="button" onClick={() => navigate({ kind: "welcome" })}>Back</button>
            <button type="button" onClick={() => { void advance(view); }} className="rounded border px-4 py-2">{view.selection.kind === "rev" ? "Next" : "Finish Setup"}</button>
          </div>
        </div>}
        {view.kind === "api-key" && <div className="space-y-5">
          <h2 className="text-lg font-semibold">Rev.AI API Key</h2>
          <p className="text-sm">Enter your Rev.AI key. It is saved locally in ~/.batchalign.ini.</p>
          <label className="block text-sm">API key
            <input aria-label="API key" type="password" autoComplete="off" value={view.credentials.key}
              onChange={event => navigate({ ...view, credentials: { ...view.credentials, key: event.target.value }, error: null })}
              className="block w-full rounded border p-2" />
          </label>
          <label className="block text-sm">Confirm API key
            <input aria-label="Confirm API key" type="password" autoComplete="off" value={view.credentials.confirmation}
              onChange={event => navigate({ ...view, credentials: { ...view.credentials, confirmation: event.target.value }, error: null })}
              className="block w-full rounded border p-2" />
          </label>
          {view.error && <p role="alert">{view.error}</p>}
          <div className="flex justify-between">
            <button type="button" onClick={() => navigate({ kind: "engine", selection: { kind: "rev", credentials: view.credentials }, error: null })}>Back</button>
            <button type="button" onClick={() => { void advance(view); }} className="rounded border px-4 py-2">Save &amp; Finish</button>
          </div>
        </div>}
        {view.kind === "done" && <div className="text-center space-y-6">
          <h2 className="text-xl font-bold">You're all set!</h2>
          <p>Default ASR engine: {view.receipt.engine === "rev" ? "Rev.AI" : "Whisper"}</p>
          <p className="text-sm">You can change this in Settings. Other provider credentials are preserved.</p>
          <button type="button" onClick={onComplete} className="rounded bg-emerald-600 text-white px-6 py-3">Get Started</button>
        </div>}
      </fieldset>
      {state.kind === "saving" && <p role="status">Saving setup...</p>}
    </div>
  </div>;
}
