// Owns admission of wire statuses and the closed outcomes of dashboard polling.
type ActiveStatus = "queued" | "running";
type TerminalStatus = "completed" | "failed" | "cancelled";
type JobStatus =
  | { kind: "active"; status: ActiveStatus }
  | { kind: "terminal"; status: TerminalStatus };
type WaitTarget = "running" | "completed" | "failed" | "terminal";
type Observation =
  | { kind: "waiting"; status: ActiveStatus }
  | { kind: "reached"; status: ActiveStatus | TerminalStatus }
  | { kind: "unexpected-terminal"; status: TerminalStatus };

function admitStatus(value: unknown): JobStatus {
  if (typeof value !== "object" || value === null || !("status" in value)) {
    throw new Error("job observation lacks status");
  }
  switch (value.status) {
    case "queued":
    case "running":
      return { kind: "active", status: value.status };
    case "completed":
    case "failed":
    case "cancelled":
      return { kind: "terminal", status: value.status };
    default:
      throw new Error(`invalid job status: ${String(value.status)}`);
  }
}

/** A closed wait target cannot accidentally keep polling a finished job. */
export class JobWait {
  private constructor(private readonly target: WaitTarget) {}

  static running(): JobWait { return new JobWait("running"); }
  static completed(): JobWait { return new JobWait("completed"); }
  static failed(): JobWait { return new JobWait("failed"); }
  static terminal(): JobWait { return new JobWait("terminal"); }

  observe(value: unknown): Observation {
    const state = admitStatus(value);
    if (state.status === this.target ||
        (this.target === "terminal" && state.kind === "terminal")) {
      return { kind: "reached", status: state.status };
    }
    switch (state.kind) {
      case "active": return { kind: "waiting", status: state.status };
      case "terminal": return { kind: "unexpected-terminal", status: state.status };
    }
  }
}
