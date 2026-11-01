/** Submission acknowledgement and full job observation are different facts.
 * A lost response cannot certify rejection or authorize an automatic retry. */
import type { ApiError } from "./api";

/** The only submission result that can open progress for a named job. */
export class SubmittedJob {
  private constructor(private readonly identity: string) {}
  get id(): string { return this.identity; }
  static admit(value: unknown): SubmittedJob {
    if (value === null || typeof value !== "object" || !("job_id" in value)
      || typeof value.job_id !== "string" || !value.job_id
      || /[\s/\\?#%]/u.test(value.job_id) || value.job_id === "." || value.job_id === "..") {
      throw new Error("The service returned no usable job identifier.");
    }
    return new SubmittedJob(value.job_id);
  }
}

export type SubmissionOutcome = Readonly<{ kind: "accepted"; job: SubmittedJob }
  | { kind: "refused"; error: ApiError }
  | { kind: "unconfirmed"; message: string }>;
