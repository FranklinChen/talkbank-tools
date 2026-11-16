/** REST fetch helpers for the Batchalign server API. */

import type { JobInfo, JobListItem } from "./types";
import { ObservedProgress } from "./observations/progress";
import {
  decodeJob,
  decodeJobs,
  decodeHealth,
  decodeStatusMessage,
} from "./wire";
import { SubmittedJob, type SubmissionOutcome } from "./submission";
import { signSameUser } from "./lib/tauri";
import {
  controlPlaneHostPort,
  controlPlaneHttpScheme,
  controlPlaneOrigin,
  desktopEndpointForServer,
  isSameControlPlaneAsPage,
} from "./runtime";

/** The header that proves a request as this account, only for a request to
 * the desktop's own admitted endpoint named in full (an empty base goes to
 * the page's own origin, which is not that endpoint). Signed natively over
 * exactly `method`, `path` and `body`. With none, the server judges the job's
 * paths by its configured folders and says so if it refuses. */
async function sameUserHeadersFor(
  base: string,
  method: string,
  path: string,
  body: string,
): Promise<Record<string, string>> {
  if (!base || !desktopEndpointForServer(base)) return {};
  try {
    return { [SAME_USER_HEADER]: await signSameUser(method, path, body) };
  } catch {
    return {};
  }
}

const SAME_USER_HEADER = "x-batchalign-same-user";

/** Preserve an actual server explanation, raw refusal, or explicit absence. */
type ApiRefusal =
  | { kind: "explanation" | "response-body"; message: string }
  | { kind: "unavailable" };

async function readRefusal(res: Response): Promise<ApiRefusal> {
  let body: string;
  try {
    body = await res.text();
  } catch {
    return { kind: "unavailable" };
  }
  if (!body.trim()) return { kind: "unavailable" };
  try {
    const raw: unknown = JSON.parse(body);
    if (typeof raw === "object" && raw !== null && "detail" in raw) {
      const detail: unknown = raw.detail;
      if (typeof detail === "string" && detail.trim())
        return { kind: "explanation", message: detail };
      if (
        typeof detail === "object" &&
        detail !== null &&
        "message" in detail &&
        typeof detail.message === "string" &&
        detail.message.trim()
      ) {
        return { kind: "explanation", message: detail.message };
      }
    }
  } catch {
    /* A non-JSON refusal is still evidence; preserve its body below. */
  }
  return { kind: "response-body", message: body };
}

export class ApiError extends Error {
  readonly status: number;

  constructor(
    context: string,
    status: number,
    readonly refusal: ApiRefusal,
  ) {
    super(
      `${context} failed (${status}). ${refusal.kind === "unavailable" ? "The server returned no readable explanation." : refusal.message}`,
    );
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseJson<A>(
  res: Response,
  context: string,
  decode: (value: unknown) => A,
): Promise<A> {
  if (!res.ok) {
    throw new ApiError(context, res.status, await readRefusal(res));
  }
  return decode(await res.json());
}

/** Build the HTTP base URL for a server host:port string. */
export function serverBaseUrl(hostPort: string): string {
  if (isSameControlPlaneAsPage(hostPort)) return "";
  if (hostPort === controlPlaneHostPort()) return controlPlaneOrigin();
  return `${controlPlaneHttpScheme()}//${hostPort}`;
}

export async function fetchJobs(base = ""): Promise<JobListItem[]> {
  const res = await fetch(`${base}/jobs`);
  return parseJson(res, "GET /jobs", decodeJobs);
}

async function readJob<A>(
  jobId: string,
  base: string,
  decode: (value: unknown) => A,
): Promise<A> {
  const res = await fetch(`${base}/jobs/${jobId}`);
  return parseJson(res, `GET /jobs/${jobId}`, decode);
}

/** Decode the detailed record once for cache and detail consumers. */
export function fetchJob(jobId: string, base = ""): Promise<JobInfo> {
  return readJob(jobId, base, decodeJob);
}

/** Carry admitted progress and its completed file index directly to the view. */
export function fetchJobProgress(
  jobId: string,
  base = "",
): Promise<ObservedProgress> {
  return readJob(jobId, base, (value) => ObservedProgress.decode(value, jobId));
}

/** Read recorded output evidence; its consumer admits identity and wire states. */
export async function fetchJobOutputFolders(
  jobId: string,
  base: string,
): Promise<unknown> {
  const res = await fetch(
    `${base}/jobs/${encodeURIComponent(jobId)}/output-folders`,
    { redirect: "error" },
  );
  return parseJson(res, "Reading recorded output folders", (value) => value);
}

export async function cancelJob(
  jobId: string,
  base = "",
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${base}/jobs/${jobId}/cancel`, { method: "POST" });
  return parseJson(res, "POST cancel", decodeStatusMessage);
}

export async function restartJob(jobId: string, base = ""): Promise<JobInfo> {
  // A restart admits the job's paths again: as this account, when this is
  // the desktop's own admitted server.
  const path = `/jobs/${jobId}/restart`;
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: await sameUserHeadersFor(base, "POST", path, ""),
  });
  return parseJson(res, "POST restart", decodeJob);
}

export async function deleteJob(
  jobId: string,
  base = "",
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${base}/jobs/${jobId}`, { method: "DELETE" });
  return parseJson(res, "DELETE job", decodeStatusMessage);
}

/** Submit once. Transport/server ambiguity must not become a retryable refusal. */
export async function submitJob(
  body: {
    command: string;
    lang?: string;
    num_speakers?: number;
    paths_mode: boolean;
    source_paths: readonly string[];
    output_paths: readonly string[];
    source_dir?: string;
    options: Record<string, unknown>;
  },
  base = "",
): Promise<SubmissionOutcome> {
  try {
    // Signed over exactly these bytes, then sent as they are.
    const encoded = JSON.stringify(body);
    const res = await fetch(`${base}/jobs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(await sameUserHeadersFor(base, "POST", "/jobs", encoded)),
      },
      body: encoded,
    });
    if (!res.ok) {
      const error = new ApiError(
        "POST /jobs",
        res.status,
        await readRefusal(res),
      );
      // Request/authentication/admission refusals happen before acceptance.
      // Timeouts and server failures do not establish whether a job was stored.
      switch (res.status) {
        case 400:
        case 401:
        case 403:
        case 404:
        case 409:
        case 413:
        case 415:
        case 422:
        case 429:
          return { kind: "refused", error };
        default:
          return { kind: "unconfirmed", message: error.message };
      }
    }
    return { kind: "accepted", job: SubmittedJob.admit(await res.json()) };
  } catch (error) {
    return { kind: "unconfirmed", message: String(error) };
  }
}

/** Check server health. */
export async function fetchHealth(
  base = "",
): Promise<import("./types").HealthResponse> {
  const res = await fetch(`${base}/health`);
  return parseJson(res, "GET /health", decodeHealth);
}
