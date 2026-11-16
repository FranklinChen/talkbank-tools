/** Admit raw job bytes and retain the file index required by progress consumers. */
import { decodeJob } from "../wire";
import type { FileStatusEntry } from "../types";
import { classifyJobStatus, type ObservedStatus } from "./job-status";

/** Only complete, identified job observations can publish progress or completion. */
export class ObservedProgress {
  private constructor(
    private readonly fileIndex: ReadonlyMap<string, FileStatusEntry>,
    readonly completedFiles: number,
    readonly state: ObservedStatus,
  ) {}
  get files(): ReadonlyMap<string, FileStatusEntry> {
    return this.fileIndex;
  }
  get jobStatus() {
    return this.state.status;
  }

  /** Schema admission precedes identity, count policy and unique file indexing. */
  static decode(value: unknown, expectedId: string): ObservedProgress {
    const job = decodeJob(value);
    if (job.job_id !== expectedId || job.completed_files < 0) {
      throw new Error("The server returned incomplete job progress.");
    }
    const files = new Map<string, FileStatusEntry>();
    for (const file of job.file_statuses) {
      if (files.has(file.filename))
        throw new Error("The server returned invalid file progress.");
      files.set(file.filename, file);
    }
    return new ObservedProgress(
      files,
      job.completed_files,
      classifyJobStatus(job.status),
    );
  }
}
