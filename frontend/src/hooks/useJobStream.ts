/** Authoritative progress observations. Polling survives a lost SSE connection;
 * a missing/malformed terminal status can never become successful completion. */
import { useQuery } from "@tanstack/react-query";
import { fetchJobProgress } from "../api";
import { controlPlaneOrigin } from "../runtime";
import type { ObservedProgress } from "../observations/progress";
export type { ObservedProgress } from "../observations/progress";

type JobProgressState =
  | { kind: "connecting" }
  | { kind: "unavailable"; message: string; retry: () => void }
  | { kind: "observed"; progress: ObservedProgress };

/** Observe one identified job; no terminal result is inferred from transport EOF. */
export function useJobStream(jobId: string): JobProgressState {
  const base = controlPlaneOrigin();
  const query = useQuery({
    queryKey: ["desktop-job-progress", base, jobId],
    queryFn: () => fetchJobProgress(jobId, base),
    refetchInterval: (query) =>
      query.state.data && query.state.data.state.kind === "terminal"
        ? false
        : 1500,
    retry: 1,
  });
  if (query.isError)
    return {
      kind: "unavailable",
      message: query.error.message,
      retry: () => {
        void query.refetch();
      },
    };
  if (!query.data) return { kind: "connecting" };
  return { kind: "observed", progress: query.data };
}
