/** A cancellation response is not a complete job observation. Keep its owner
 * above reconnecting progress views so a transport outage cannot re-enable it. */
import { useState, useSyncExternalStore } from "react";
import { cancelJob } from "../api";
import { controlPlaneOrigin } from "../runtime";

type CancelOutcome = "cancelled" | "completed" | "failed";

/** Only the route's closed response vocabulary can acknowledge this action. */
class CancellationReceipt {
  private constructor(readonly outcome: CancelOutcome, private readonly explanation: string) {}
  get message(): string { return this.explanation; }
  static admit(value: unknown): CancellationReceipt {
    if (value === null || typeof value !== "object" || !("status" in value) || !("message" in value)
      || typeof value.message !== "string" || !value.message.trim()) {
      throw new Error("The service returned an incomplete cancellation response.");
    }
    switch (value.status) {
      case "cancelled": case "completed": case "failed": return new CancellationReceipt(value.status, value.message);
      default: throw new Error("The service returned an unknown cancellation outcome.");
    }
  }
}

export type JobCancellation = Readonly<{ kind: "available"; request: () => void }
  | { kind: "requesting" }
  | { kind: "acknowledged"; receipt: CancellationReceipt }
  | { kind: "failed"; message: string; retry: () => void }>;

/** One job-bound owner publishes transitions synchronously before awaiting HTTP.
 * Only available/failed states expose an action; acknowledged actions stay closed. */
class CancellationAction {
  private state: JobCancellation;
  private readonly listeners = new Set<() => void>();
  constructor(private readonly jobId: string, private readonly origin: string) {
    this.state = { kind: "available", request: () => { void this.request(); } };
  }
  /** A different job or service gets a fresh owner, never the old action state. */
  forJob(jobId: string, origin: string): CancellationAction {
    return this.jobId === jobId && this.origin === origin ? this : new CancellationAction(jobId, origin);
  }
  read = (): JobCancellation => this.state;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  private publish(state: JobCancellation) {
    this.state = state;
    this.listeners.forEach(listener => listener());
  }
  private async request() {
    switch (this.state.kind) {
      case "requesting": case "acknowledged": return;
      case "available": case "failed": break;
    }
    this.publish({ kind: "requesting" });
    try {
      const receipt = CancellationReceipt.admit(await cancelJob(this.jobId, this.origin));
      this.publish({ kind: "acknowledged", receipt });
    } catch (error) {
      this.publish({ kind: "failed", message: String(error), retry: () => { void this.request(); } });
    }
  }
}

/** Retain cancellation intent for this job across progress-view disconnections. */
export function useJobCancellation(jobId: string): JobCancellation {
  const origin = controlPlaneOrigin();
  const [owner, setOwner] = useState(() => new CancellationAction(jobId, origin));
  const current = owner.forJob(jobId, origin);
  // Ownership is React state, not a disposable memo cache. Retargeting retires
  // the old owner; any late HTTP response can publish only to that old owner.
  if (current !== owner) setOwner(current);
  return useSyncExternalStore(current.subscribe, current.read);
}
