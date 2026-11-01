/** REST fetch helpers for the Batchalign server API. */

import type { JobInfo, JobListItem } from "./types";
import { SubmittedJob, type SubmissionOutcome } from "./submission";
import {
  controlPlaneHostPort,
  controlPlaneHttpScheme,
  controlPlaneOrigin,
  isSameControlPlaneAsPage,
} from "./runtime";

/** Preserve an actual server explanation, raw refusal, or explicit absence. */
type ApiRefusal = { kind: "explanation" | "response-body"; message: string }
  | { kind: "unavailable" };

async function readRefusal(res: Response): Promise<ApiRefusal> {
  let body: string;
  try { body = await res.text(); } catch { return { kind: "unavailable" }; }
  if (!body.trim()) return { kind: "unavailable" };
  try {
    const raw: unknown = JSON.parse(body);
    if (typeof raw === "object" && raw !== null && "detail" in raw) {
      const detail: unknown = raw.detail;
      if (typeof detail === "string" && detail.trim()) return { kind: "explanation", message: detail };
      if (typeof detail === "object" && detail !== null && "message" in detail && typeof detail.message === "string" && detail.message.trim()) {
        return { kind: "explanation", message: detail.message };
      }
    }
  } catch { /* A non-JSON refusal is still evidence; preserve its body below. */ }
  return { kind: "response-body", message: body };
}

export class ApiError extends Error {
  readonly status: number;

  constructor(context: string, status: number, readonly refusal: ApiRefusal) {
    super(`${context} failed (${status}). ${refusal.kind === "unavailable" ? "The server returned no readable explanation." : refusal.message}`);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseJson<T>(res: Response, context: string): Promise<T> {
  if (!res.ok) {
    throw new ApiError(context, res.status, await readRefusal(res));
  }
  return (await res.json()) as T;
}

/** Build the HTTP base URL for a server host:port string. */
export function serverBaseUrl(hostPort: string): string {
  if (isSameControlPlaneAsPage(hostPort)) return "";
  if (hostPort === controlPlaneHostPort()) return controlPlaneOrigin();
  return `${controlPlaneHttpScheme()}//${hostPort}`;
}

export async function fetchJobs(base = ""): Promise<JobListItem[]> {
  const res = await fetch(`${base}/jobs`);
  return parseJson<JobListItem[]>(res, "GET /jobs");
}

export async function fetchJob(jobId: string, base = ""): Promise<JobInfo> {
  const res = await fetch(`${base}/jobs/${jobId}`);
  return parseJson<JobInfo>(res, `GET /jobs/${jobId}`);
}

/** Read recorded output evidence; its consumer admits identity and wire states. */
export async function fetchJobOutputFolders(jobId: string, base: string): Promise<unknown> {
  const res = await fetch(`${base}/jobs/${encodeURIComponent(jobId)}/output-folders`, { redirect: "error" });
  return parseJson<unknown>(res, "Reading recorded output folders");
}

export async function cancelJob(
  jobId: string,
  base = ""
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${base}/jobs/${jobId}/cancel`, { method: "POST" });
  return parseJson<{ status: string; message: string }>(res, "POST cancel");
}

export async function restartJob(jobId: string, base = ""): Promise<JobInfo> {
  const res = await fetch(`${base}/jobs/${jobId}/restart`, { method: "POST" });
  return parseJson<JobInfo>(res, "POST restart");
}

export async function deleteJob(
  jobId: string,
  base = ""
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${base}/jobs/${jobId}`, { method: "DELETE" });
  return parseJson<{ status: string; message: string }>(res, "DELETE job");
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
    const res = await fetch(`${base}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const error = new ApiError("POST /jobs", res.status, await readRefusal(res));
      // Request/authentication/admission refusals happen before acceptance.
      // Timeouts and server failures do not establish whether a job was stored.
      switch (res.status) {
        case 400: case 401: case 403: case 404: case 409:
        case 413: case 415: case 422: case 429:
          return { kind: "refused", error };
        default: return { kind: "unconfirmed", message: error.message };
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
  return parseJson<import("./types").HealthResponse>(res, "GET /health");
}
