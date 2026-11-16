// Owns admission of wire statuses and the closed outcomes of dashboard polling.
import * as Schema from "effect/Schema";
import { JobStatus as StatusSchema } from "../src/generated/decoders";

import {
  classifyJobStatus,
  type ObservedStatus,
} from "../src/observations/job-status";
type ActiveStatus = Extract<ObservedStatus, { kind: "active" }>["status"];
type TerminalStatus = Extract<ObservedStatus, { kind: "terminal" }>["status"];
const decodeObservation = Schema.decodeUnknownSync(
  Schema.Struct({ status: StatusSchema }),
);
type WaitTarget = "running" | "completed" | "failed" | "terminal";
type Observation =
  | { kind: "waiting"; status: ActiveStatus }
  | { kind: "reached"; status: ActiveStatus | TerminalStatus }
  | { kind: "unexpected-terminal"; status: TerminalStatus };

/** A closed wait target cannot accidentally keep polling a finished job. */
export class JobWait {
  private constructor(private readonly target: WaitTarget) {}

  static running(): JobWait {
    return new JobWait("running");
  }
  static completed(): JobWait {
    return new JobWait("completed");
  }
  static failed(): JobWait {
    return new JobWait("failed");
  }
  static terminal(): JobWait {
    return new JobWait("terminal");
  }

  observe(value: unknown): Observation {
    const state = classifyJobStatus(decodeObservation(value).status);
    if (
      state.status === this.target ||
      (this.target === "terminal" && state.kind === "terminal")
    ) {
      return { kind: "reached", status: state.status };
    }
    switch (state.kind) {
      case "active":
        return { kind: "waiting", status: state.status };
      case "terminal":
        return { kind: "unexpected-terminal", status: state.status };
    }
  }
}
