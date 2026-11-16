/** One exhaustive classification shared by production observations and fixtures. */
import type { JobStatus } from "../generated/decoders";

type ActiveStatus = "queued" | "running";
export type ObservedStatus =
  | { readonly kind: "active"; readonly status: ActiveStatus }
  | {
      readonly kind: "terminal";
      readonly status: Exclude<JobStatus, ActiveStatus>;
    };

/** Payload types prevent assigning a terminal wire value to an active state. */
export function classifyJobStatus(status: JobStatus): ObservedStatus {
  switch (status) {
    case "queued":
    case "running":
      return { kind: "active", status };
    case "completed":
    case "failed":
    case "cancelled":
    case "interrupted":
    case "writeback_failed":
      return { kind: "terminal", status };
  }
}
