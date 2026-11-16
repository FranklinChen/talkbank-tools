/** Re-export server types from auto-generated OpenAPI definitions. */

import type * as Wire from "./generated/decoders";

// Preserve the declared contract's named fields while withholding uninspected
// extension properties from UI operations. Decoding retains those JSON bytes.
type DeclaredFields<T> = {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

// Server types (generated from the repo-root openapi.json via OpenAPI)
export type JobStatusValue = Wire.JobStatus;
export type FileStatusEntry = Wire.FileStatusEntry;
export type HealthResponse = Wire.HealthResponse;
export type JobControlPlaneInfo = Wire.JobControlPlaneInfo;

// JobInfo: override file_statuses to required (server always sends it),
// add client-side `server` enrichment.
export type JobInfo = Omit<DeclaredFields<Wire.JobInfo>, "file_statuses"> & {
  file_statuses: readonly FileStatusEntry[];
  server?: string;
};

// Enriched with client-side `server` field (not in OpenAPI)
export type JobListItem = Wire.JobListItem & {
  server?: string;
};
