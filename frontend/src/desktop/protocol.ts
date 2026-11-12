/** Typed IPC boundary for the small native launcher. The CLI owns the service;
 * observing it conveys an endpoint, never ownership of its PID. */

/** Native product identity, distinct from other Batchalign applications. */
export const DESKTOP_APP_NAME = "BA3 Runner";

/** User preferences editable without replacing the credentials document. */
export interface DesktopConfig {
  engine: string;
  rev_key: string | null;
}

/** Acknowledgement of a completed settings edit. */
export interface DesktopCommandAck { message: string }

/** Read-only filesystem admission derives paths from actual canonical inputs. */
export interface DesktopPathProposal {
  command: "transcribe" | "morphotag" | "align" | "translate" | "utseg" | "benchmark";
  source_root: string; inputs: readonly string[];
  destination: { mode: "separate"; folder: string } | { mode: "in_place" };
}

/** Native observation; only JobPlan admits it to the HTTP submission path. */
export interface DesktopPathReceipt {
  source_root: string; inputs: string[]; output_root: string; outputs: string[];
}

/** Exhaustive native discovery result; only ready carries an endpoint. */
export type DesktopServerStatus =
  | { state: "missing_binary" }
  | { state: "stopped" }
  | { state: "ready"; url: string; build_hash: string }
  | { state: "refused"; message: string };

/** Paired request/response types for native commands. */
export interface DesktopCommandMap {
  admitJobPaths: { name: "admit_job_paths"; request: { proposal: DesktopPathProposal }; response: DesktopPathReceipt };
  openOutputDirectory: { name: "open_output_directory"; request: { path: string }; response: void };
  discoverFiles: { name: "discover_files"; request: { dir: string; extensions: string[] }; response: string[] };
  isFirstLaunch: { name: "is_first_launch"; request: void; response: boolean };
  readConfig: { name: "read_config"; request: void; response: DesktopConfig };
  writeConfig: { name: "write_config"; request: { config: DesktopConfig }; response: DesktopCommandAck };
  serverStatus: { name: "server_status"; request: void; response: DesktopServerStatus };
  startServer: { name: "start_server"; request: void; response: DesktopServerStatus };
  signSameUser: { name: "sign_same_user"; request: { method: string; path: string; body: string }; response: string };
}

/** Runtime names owned by the native protocol adapter. */
export const DESKTOP_COMMANDS = {
  admitJobPaths: { name: "admit_job_paths" },
  openOutputDirectory: { name: "open_output_directory" },
  discoverFiles: { name: "discover_files" },
  isFirstLaunch: { name: "is_first_launch" },
  readConfig: { name: "read_config" },
  writeConfig: { name: "write_config" },
  serverStatus: { name: "server_status" },
  startServer: { name: "start_server" },
  signSameUser: { name: "sign_same_user" },
} satisfies { [K in keyof DesktopCommandMap]: { name: DesktopCommandMap[K]["name"] } };

export type DesktopCommandKey = keyof DesktopCommandMap;
export type DesktopCommandRequest<K extends DesktopCommandKey> = DesktopCommandMap[K]["request"];
export type DesktopCommandResponse<K extends DesktopCommandKey> = DesktopCommandMap[K]["response"];
