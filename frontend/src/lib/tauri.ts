/** Low-level native transport. Failures are reported to the owning UI instead
 * of being turned into default credentials or successful empty discovery. */
import { isDesktop } from "../runtime";
import { PreferenceDraft, SavedPreferences, type PreferenceEdit } from "../desktop/preferences";
import {
  DESKTOP_COMMANDS,
  type DesktopCommandKey, type DesktopCommandRequest, type DesktopCommandResponse,
  type DesktopServerStatus,
  type DesktopPathProposal, type DesktopPathReceipt,
} from "../desktop/protocol";

function desktopOnlyError(action: string): Error {
  return new Error(action + " requires the desktop app");
}

async function tauriInvoke<K extends DesktopCommandKey>(
  command: K,
  ...args: DesktopCommandRequest<K> extends void ? [] : [DesktopCommandRequest<K>]
): Promise<DesktopCommandResponse<K>> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DesktopCommandResponse<K>>(DESKTOP_COMMANDS[command].name, args[0] as Record<string, unknown> | undefined);
}

/** Open the native directory picker; null means the user cancelled. */
export async function pickFolder(title?: string): Promise<string | null> {
  if (!isDesktop()) throw desktopOnlyError("Folder selection");
  const { open } = await import("@tauri-apps/plugin-dialog");
  const result = await open({ directory: true, multiple: false, title: title ?? "Choose a folder" });
  return Array.isArray(result) ? result[0] ?? null : result;
}

/** Choose the output directory through the same native capability. */
export function pickOutputFolder(): Promise<string | null> {
  return pickFolder("Choose output folder");
}

/** Enumerate selected local inputs. */
export function discoverFiles(dir: string, extensions: string[]): Promise<string[]> {
  if (!isDesktop()) return Promise.reject(desktopOnlyError("File discovery"));
  return tauriInvoke("discoverFiles", { dir, extensions });
}

/** Resolve local file identities without writing or submitting any job. */
export function admitJobPaths(proposal: DesktopPathProposal): Promise<DesktopPathReceipt> {
  if (!isDesktop()) return Promise.reject(desktopOnlyError("Input/output admission"));
  return tauriInvoke("admitJobPaths", { proposal });
}

/** Reveal the actual output directory in the file manager. */
export async function openPath(path: string): Promise<void> {
  if (!isDesktop()) throw desktopOnlyError("Opening an output folder");
  await tauriInvoke("openOutputDirectory", { path });
}

/** Detect genuine first-run absence; read failures remain failures. */
export function isFirstLaunch(): Promise<boolean> {
  return isDesktop() ? tauriInvoke("isFirstLaunch") : Promise.resolve(false);
}

/** Read preferences without inventing values after a failed read. */
export async function readConfig(): Promise<PreferenceDraft> {
  if (!isDesktop()) return Promise.reject(desktopOnlyError("Reading settings"));
  return PreferenceDraft.loaded(await tauriInvoke("readConfig"));
}

/** Apply a narrow, native-validated settings edit. */
export async function writeConfig(edit: PreferenceEdit): Promise<SavedPreferences> {
  if (!isDesktop()) throw desktopOnlyError("Saving settings");
  const acknowledgement = await tauriInvoke("writeConfig", { config: edit.config });
  return SavedPreferences.admit(edit, acknowledgement);
}

/** Observe the CLI-managed service without owning it. */
export function serverStatus(): Promise<DesktopServerStatus> {
  return tauriInvoke("serverStatus");
}

/** Ask the CLI to start a missing managed service; refuse incompatible ones. */
export function startServer(): Promise<DesktopServerStatus> {
  return tauriInvoke("startServer");
}
