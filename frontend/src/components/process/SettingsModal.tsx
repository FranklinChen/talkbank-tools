/** A modal session cannot edit unloaded settings or replay a read credential.
 * Closing retires pending reads; saving keeps this session open until settled. */
import { useEffect, useId, useRef, useState } from "react";
import { useDesktopConfig, useDesktopEnvironment } from "../../desktop/DesktopContext";
import { PreferenceDraft, type PreferenceEdit, type SavedPreferences } from "../../desktop/preferences";

interface SettingsModalProps { open: boolean; onClose: () => void }
type Notice = { kind: "none" } | { kind: "error"; message: string } | { kind: "saved"; receipt: SavedPreferences };
type SettingsState = { kind: "loading" } | { kind: "load-failed"; message: string }
  | { kind: "editing"; draft: PreferenceDraft; notice: Notice }
  | { kind: "saving"; draft: PreferenceDraft; edit: PreferenceEdit };
type Operation = { kind: "read" } | { kind: "save"; edit: PreferenceEdit };

/** Reopening gets a fresh read owner rather than the prior modal's stale fields. */
export function SettingsModal({ open, onClose }: SettingsModalProps) {
  const environment = useDesktopEnvironment();
  return open && environment.isDesktop ? <SettingsSession onClose={onClose} /> : null;
}

function SettingsSession({ onClose }: { onClose: () => void }) {
  const config = useDesktopConfig();
  const [state, setState] = useState<SettingsState>({ kind: "loading" });
  const operation = useRef<Operation | null>(null);
  const closeTarget = useRef(onClose);
  const modal = useRef<HTMLDivElement>(null);
  const keyId = useId();
  useEffect(() => { closeTarget.current = onClose; }, [onClose]);

  async function load() {
    if (operation.current) return;
    const owner: Operation = { kind: "read" };
    operation.current = owner;
    setState({ kind: "loading" });
    try {
      const draft = await config.readConfig();
      if (operation.current === owner) setState({ kind: "editing", draft, notice: { kind: "none" } });
    } catch (error) {
      if (operation.current === owner) setState({ kind: "load-failed", message: String(error) });
    } finally { if (operation.current === owner) operation.current = null; }
  }
  useEffect(() => { void load(); return () => { operation.current = null; }; }, [config]);

  function close() { if (operation.current?.kind !== "save") closeTarget.current(); }
  useEffect(() => {
    // Callback refreshes must not steal focus while the user edits a key.
    const opener = document.activeElement;
    modal.current?.querySelector<HTMLButtonElement>("button")?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") { event.preventDefault(); close(); }
      if (event.key !== "Tab") return;
      const controls = modal.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), a[href], input:not(:disabled), [tabindex]:not([tabindex="-1"])',
      );
      if (!controls?.length) { event.preventDefault(); modal.current?.focus(); return; }
      const first = controls[0]; const last = controls[controls.length - 1];
      if (![...controls].some(control => control === document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); }
      else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    document.addEventListener("keydown", keydown);
    return () => { document.removeEventListener("keydown", keydown); if (opener instanceof HTMLElement && opener.isConnected) opener.focus(); };
  }, []);

  function change(edit: (draft: PreferenceDraft) => PreferenceDraft) {
    setState(previous => previous.kind === "editing"
      ? { kind: "editing", draft: edit(previous.draft), notice: { kind: "none" } } : previous);
  }
  async function save() {
    if (state.kind !== "editing" || operation.current) return;
    const draft = state.draft;
    let edit: PreferenceEdit;
    try { edit = draft.prepare(); }
    catch (error) { setState({ kind: "editing", draft, notice: { kind: "error", message: String(error) } }); return; }
    const owner: Operation = { kind: "save", edit };
    operation.current = owner;
    setState({ kind: "saving", draft, edit });
    try {
      const receipt = await config.writeConfig(edit);
      if (operation.current === owner) setState({ kind: "editing", draft: draft.afterSave(receipt), notice: { kind: "saved", receipt } });
    } catch (error) {
      if (operation.current === owner) setState({ kind: "editing", draft, notice: { kind: "error", message: "Could not confirm settings saved. " + String(error) } });
    } finally { if (operation.current === owner) operation.current = null; }
  }

  function body() {
    if (state.kind === "loading") return <p role="status">Reading settings...</p>;
    if (state.kind === "load-failed") return <div>
      <p role="alert">Could not read settings. No edit was sent. {state.message}</p>
      <button type="button" onClick={() => { void load(); }} className="mt-3 rounded border px-3 py-2">Retry reading settings</button>
    </div>;
    const draft = state.draft;
    return <>
      {draft.choice.kind === "unsupported" && <p role="status">Your configured engine is {draft.choice.name}. Use the CLI for that engine, or explicitly choose Rev.AI or Whisper here.</p>}
      <fieldset disabled={state.kind === "saving"} className="space-y-4">
        <legend className="text-sm font-medium mb-2">Default ASR engine</legend>
        <div className="flex gap-3">
          <button type="button" aria-pressed={draft.choice.kind === "supported" && draft.choice.engine === "rev"}
            onClick={() => change(value => value.select("rev"))} className="group flex-1 rounded border p-3 aria-pressed:bg-indigo-50 aria-pressed:border-indigo-600 aria-pressed:font-semibold"><span aria-hidden="true" className="hidden group-aria-pressed:inline">✓ </span>Rev.AI<span className="block text-xs">Paid cloud service; uploads recordings</span></button>
          <button type="button" aria-pressed={draft.choice.kind === "supported" && draft.choice.engine === "whisper"}
            onClick={() => change(value => value.select("whisper"))} className="group flex-1 rounded border p-3 aria-pressed:bg-indigo-50 aria-pressed:border-indigo-600 aria-pressed:font-semibold"><span aria-hidden="true" className="hidden group-aria-pressed:inline">✓ </span>Whisper<span className="block text-xs">Local inference; model downloads required</span></button>
        </div>
        {draft.choice.kind === "supported" && draft.choice.engine === "rev" && <div>
          <label htmlFor={keyId} className="block text-sm font-medium">Rev.AI API key</label>
          <input id={keyId} type="password" autoComplete="off" value={draft.keyText}
            onChange={event => change(value => value.replaceKey(event.target.value))}
            className="w-full rounded border p-2" />
          <p className="text-xs mt-1">{draft.keyHint}</p>
        </div>}
        <div className="flex gap-3">
          <button type="button" onClick={() => { void save(); }} disabled={draft.choice.kind === "unsupported"}
            className="rounded border px-4 py-2 disabled:opacity-50">{state.kind === "saving" ? "Saving..." : "Save"}</button>
          <button type="button" onClick={() => { void load(); }} className="rounded border px-4 py-2">Reload settings</button>
        </div>
      </fieldset>
      {state.kind === "saving" && <p role="status">Saving settings. Wait for confirmation before closing.</p>}
      {state.kind === "editing" && state.notice.kind === "error" && <p role="alert">{state.notice.message}</p>}
      {state.kind === "editing" && state.notice.kind === "saved" && <p role="status">Settings saved.</p>}
      <p className="text-xs text-gray-500">Preferences use ~/.batchalign.ini. Other provider credentials are preserved.</p>
    </>;
  }
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4"
    onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <div ref={modal} role="dialog" aria-modal="true" aria-label="Settings" tabIndex={-1}
      className="bg-white rounded-xl shadow-xl max-w-md w-full p-6 space-y-5">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">Settings</h2>
        <button type="button" aria-label="Close settings" disabled={state.kind === "saving"} onClick={close}>Close</button>
      </div>
      {body()}
    </div>
  </div>;
}
