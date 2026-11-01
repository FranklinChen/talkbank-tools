/** Output authority crosses two edges: native discovery binds the server, and
 * a matching persisted job response supplies its folders. Existence remains
 * the native opener's action-time check. No input path can fill missing data.
 */
import { fetchJobOutputFolders } from "../api";
import type { DesktopEndpoint } from "../runtime";
import type { DesktopFilesCapability } from "./capabilities";
import type { components } from "../generated/api";

type WireAccess = components["schemas"]["LocalOutputAccess"];
type AdmittedWireAccess = Exclude<WireAccess, { state: "recorded" }>
  | { state: "recorded"; directories: readonly [string, ...string[]] };

function record(raw: unknown): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) throw new Error("The service returned an unsupported output-location response.");
  return raw as Record<string, unknown>;
}

function directoryText(raw: unknown): string {
  if (typeof raw !== "string" || !raw.trim()) throw new Error("The service returned an invalid output-folder path.");
  return raw;
}

function wireAccess(raw: unknown): AdmittedWireAccess {
  const value = record(raw);
  switch (value.state) {
    case "content_mode": return { state: "content_mode" };
    case "not_recorded": return { state: "not_recorded" };
    case "refused":
      if (typeof value.message !== "string" || !value.message.trim()) break;
      return { state: "refused", message: value.message };
    case "recorded": {
      if (!Array.isArray(value.directories)) break;
      const values: readonly unknown[] = value.directories;
      const [first, ...remaining] = values;
      return { state: "recorded", directories: [directoryText(first), ...remaining.map(directoryText)] };
    }
  }
  throw new Error("The service returned unsupported or incomplete output-location evidence.");
}

/** A folder admitted from one identity-matched job response. */
export class RecordedOutputFolder {
  private constructor(readonly path: string) { Object.freeze(this); }

  /** Admit all wire facts before publishing any native-open capability. */
  static fromResponse(request: LocalJobOutputRequest, raw: unknown): JobOutputAccess {
    const response = record(raw);
    if (response.job_id !== request.jobId) throw new Error("The output-location response belongs to a different job.");
    const access = wireAccess(response.access);
    switch (access.state) {
      case "content_mode": return { kind: "unavailable", message: "This job used uploaded content; it has no recorded local output folder." };
      case "not_recorded": return { kind: "unavailable", message: "No output folder was recorded for this job." };
      case "refused": throw new Error(access.message);
      case "recorded": {
        const [first, ...remaining] = access.directories;
        const unique = [...new Set(remaining.filter(path => path !== first))];
        const folders: readonly [RecordedOutputFolder, ...RecordedOutputFolder[]] = Object.freeze([
          new RecordedOutputFolder(first), ...unique.map(path => new RecordedOutputFolder(path)),
        ]);
        return { kind: "recorded", folders };
      }
    }
  }

  /** Native admission reports moved/missing folders and OS refusals visibly. */
  open(files: Pick<DesktopFilesCapability, "openPath">): Promise<void> {
    return files.openPath(this.path);
  }
}

/** Empty evidence cannot carry an actionable folder list. */
export type JobOutputAccess = Readonly<
  | { kind: "unavailable"; message: string }
  | { kind: "recorded"; folders: readonly [RecordedOutputFolder, ...RecordedOutputFolder[]] }>;

/** The query scope owns the local service and record identity together. */
export class LocalJobOutputRequest {
  private constructor(private readonly endpoint: DesktopEndpoint, readonly jobId: string) {}

  /** Route admission prevents URL path normalization from changing the job. */
  static forJob(endpoint: DesktopEndpoint, jobId: string): LocalJobOutputRequest {
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(jobId)) throw new Error("This job has an unsupported identifier.");
    return new LocalJobOutputRequest(endpoint, jobId);
  }

  /** Fetch only the native-admitted origin, never the page origin. */
  async load(): Promise<JobOutputAccess> {
    return RecordedOutputFolder.fromResponse(this, await fetchJobOutputFolders(this.jobId, this.endpoint.url().origin));
  }
}

/** One action owner prevents parallel opens without mirrored React/ref flags. */
export type OutputFolderActionState = Readonly<{ kind: "idle" } | { kind: "opening" } | { kind: "failed"; message: string }>;

/** A route-local state machine; subscribers only observe its immutable state. */
export class OutputFolderAction {
  private state: OutputFolderActionState = Object.freeze({ kind: "idle" });
  private readonly listeners = new Set<() => void>();

  constructor(private readonly files: Pick<DesktopFilesCapability, "openPath">) {}

  /** Subscribe only for this action owner's lifetime. */
  readonly subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  /** React receives the same snapshot until a real transition occurs. */
  readonly snapshot = (): OutputFolderActionState => this.state;

  private transition(state: OutputFolderActionState) {
    this.state = Object.freeze(state);
    this.listeners.forEach(listener => listener());
  }

  /** Claim synchronously; completion of an old owner cannot settle a new route. */
  async open(folder: RecordedOutputFolder): Promise<void> {
    if (this.state.kind === "opening") return;
    this.transition({ kind: "opening" });
    try { await folder.open(this.files); this.transition({ kind: "idle" }); }
    catch (error) { this.transition({ kind: "failed", message: String(error) }); }
  }
}
