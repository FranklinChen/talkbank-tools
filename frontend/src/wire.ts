/** Admit server payloads once, before they enter cache or presentation state. */
import * as Schema from "effect/Schema";
import * as Effect from "effect/Effect";
import * as Wire from "./generated/decoders";

export const DashboardMessage = Schema.Union([
  Schema.Struct({
    type: Schema.Literal("snapshot"),
    jobs: Schema.Array(Wire.JobListItem),
    health: Wire.HealthResponse,
  }),
  Schema.Struct({ type: Schema.Literal("job_update"), job: Wire.JobListItem }),
  Schema.Struct({
    type: Schema.Literal("file_update"),
    job_id: Wire.JobId,
    file: Wire.FileStatusEntry,
    completed_files: Schema.Int,
  }),
  Schema.Struct({ type: Schema.Literal("job_deleted"), job_id: Wire.JobId }),
]);
export type DashboardMessage = typeof DashboardMessage.Type;

/** Parse and decode complete wire frames, preserving the declared payloads. */
const decodeFrame = Schema.decodeUnknownEffect(
  Schema.fromJsonString(DashboardMessage),
);

/** A real server address admitted before any socket is created. */
export class DashboardServer {
  private constructor(
    private readonly endpointUrl: string,
    readonly label: string,
  ) {}
  get url(): string {
    return this.endpointUrl;
  }
  static prepare(hostPort: string, scheme: "ws:" | "wss:"): DashboardServer {
    const url = new URL(`${scheme}//${hostPort}/ws`);
    if (
      url.pathname !== "/ws" ||
      url.search ||
      url.hash ||
      url.username ||
      url.password
    ) {
      throw new Error("Invalid dashboard server address.");
    }
    return new DashboardServer(url.href, url.hostname.split(".")[0]);
  }
}

/** Carries both completed wire admission and the connection that produced it. */
export class DashboardUpdate {
  private constructor(
    readonly server: DashboardServer,
    private readonly decoded: DashboardMessage,
  ) {}
  get message(): DashboardMessage {
    return this.decoded;
  }
  static decode(server: DashboardServer, text: string) {
    return decodeFrame(text).pipe(
      Effect.map((message) => new DashboardUpdate(server, message)),
    );
  }
}

// Compile each decoder once and carry the prepared parser across observations.
const decodeJobInfo = Schema.decodeUnknownSync(Wire.JobInfo);
const decodeJobList = Schema.decodeUnknownSync(Schema.Array(Wire.JobListItem));

/** File observations are mandatory for the detailed UI's admitted job model. */
export function decodeJob(value: unknown) {
  let job: Wire.JobInfo;
  try {
    job = decodeJobInfo(value);
  } catch {
    throw new Error("The server returned incomplete job progress.");
  }
  if (job.file_statuses === undefined) {
    throw new Error("The server returned incomplete job progress.");
  }
  return { ...job, file_statuses: job.file_statuses };
}

/** An immutable wire list becomes the cache's independently owned outer array. */
export const decodeJobs = (value: unknown) => [...decodeJobList(value)];
/** Admit the shared health contract before publication. */
export const decodeHealth = Schema.decodeUnknownSync(Wire.HealthResponse);
/** Decode action acknowledgements before their operation-specific policy. */
export const decodeStatusMessage = Schema.decodeUnknownSync(
  Wire.StatusMessageResponse,
);
