/** Focused desktop capabilities exposed to the shared React tree.
 *
 * The Tauri adapter still owns raw IPC and plugin transport in `lib/tauri.ts`.
 * This module defines the narrower capability interfaces that hooks and
 * components should depend on so the desktop seam stays explicit and small.
 */

import type { DesktopServerStatus, DesktopPathProposal, DesktopPathReceipt } from "./protocol";
import type { PreferenceDraft, PreferenceEdit, SavedPreferences } from "./preferences";

/** Runtime-only capability: answers whether the UI runs in the desktop shell. */
export interface DesktopEnvironmentCapability {
  /** True when the current UI is running inside the Tauri desktop shell. */
  isDesktop: boolean;
}

/** File-system and native-dialog helpers needed by the process flow. */
export interface DesktopFilesCapability {
  /** Inspect real filesystem identities before a local job is submitted. */
  admitJobPaths: (proposal: DesktopPathProposal) => Promise<DesktopPathReceipt>;
  /** Open the native folder picker. */
  pickFolder: (title?: string) => Promise<string | null>;
  /** Open the native output-folder picker. */
  pickOutputFolder: () => Promise<string | null>;
  /** Enumerate files inside one selected directory. */
  discoverFiles: (dir: string, extensions: string[]) => Promise<string[]>;
  /** Reveal one local path in the native shell. */
  openPath: (path: string) => Promise<void>;
}

/** Config persistence helpers needed by the setup flow. */
export interface DesktopConfigCapability {
  /** Detect whether the desktop shell should show the first-run wizard. */
  isFirstLaunch: () => Promise<boolean>;
  /** Read the desktop shell's persisted user configuration. */
  readConfig: () => Promise<PreferenceDraft>;
  /** Persist one user-configuration update through the desktop shell. */
  writeConfig: (edit: PreferenceEdit) => Promise<SavedPreferences>;
}

/** Server lifecycle helpers needed by the process flow. */
export interface DesktopServerCapability {
  /** Query the desktop shell's managed server status. */
  serverStatus: () => Promise<DesktopServerStatus>;
  /** Ask the desktop shell to start the local server process. */
  startServer: () => Promise<DesktopServerStatus>;
}
