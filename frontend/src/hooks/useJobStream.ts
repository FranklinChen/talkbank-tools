/** Authoritative progress observations. Polling survives a lost SSE connection;
 * a missing/malformed terminal status can never become successful completion. */
import { useQuery } from "@tanstack/react-query";
import { fetchJob } from "../api";
import { controlPlaneOrigin } from "../runtime";
import type { FileStatusEntry, JobInfo, JobStatusValue } from "../types";

/** All terminal outcomes are explicit, including interrupted/writeback failure. */
export const TERMINAL_JOB: Record<JobStatusValue, boolean> = {
  queued: false, running: false, completed: true, failed: true,
  cancelled: true, interrupted: true, writeback_failed: true,
};

/** One admitted server observation, retaining the actual file-level findings. */
export class ObservedProgress {
  private constructor(
    readonly files: ReadonlyMap<string, FileStatusEntry>,
    readonly completedFiles: number,
    private readonly observedStatus: JobStatusValue,
  ) {}
  get jobStatus(): JobStatusValue { return this.observedStatus; }
  /** Admit the fields needed by the progress view at the HTTP boundary. */
  static admit(job: JobInfo, expectedId: string): ObservedProgress {
    if (job == null || job.job_id !== expectedId || !Object.prototype.hasOwnProperty.call(TERMINAL_JOB, job.status)
      || !Number.isInteger(job.completed_files) || job.completed_files < 0
      || !Array.isArray(job.file_statuses)) throw new Error("The server returned incomplete job progress.");
    const files = new Map<string, FileStatusEntry>();
    for (const file of job.file_statuses) {
      if (typeof file?.filename !== "string" || typeof file.status !== "string"
        || !["queued", "processing", "done", "diagnosed", "error", "interrupted"].includes(file.status)
        || files.has(file.filename)) throw new Error("The server returned invalid file progress.");
      files.set(file.filename, file);
    }
    return new ObservedProgress(files, job.completed_files, job.status);
  }
}

type JobProgressState =
  | { kind: "connecting" }
  | { kind: "unavailable"; message: string; retry: () => void }
  | { kind: "observed"; progress: ObservedProgress };

/** Observe one identified job; no terminal result is inferred from transport EOF. */
export function useJobStream(jobId: string): JobProgressState {
  const base = controlPlaneOrigin();
  const query = useQuery({
    queryKey: ["desktop-job-progress", base, jobId],
    queryFn: async () => ObservedProgress.admit(await fetchJob(jobId, base), jobId),
    refetchInterval: query => query.state.data && TERMINAL_JOB[query.state.data.jobStatus] ? false : 1500,
    retry: 1,
  });
  if (query.isError) return { kind: "unavailable", message: query.error.message, retry: () => { void query.refetch(); } };
  if (!query.data) return { kind: "connecting" };
  return { kind: "observed", progress: query.data };
}
