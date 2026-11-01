/** Observe the service admitted at startup. A desktop window does not own or
 * stop the shared service; reconnecting repeats native startup admission. */
import { useDesktopEnvironment } from "../desktop/DesktopContext";
import { useServerHealth } from "./useServerHealth";

export type ServerLifecycleStatus = "starting" | "running" | "stopped" | "external";
export interface ServerLifecycleState {
  status: ServerLifecycleStatus;
  isReady: boolean;
  start: () => Promise<void>;
  error: string | null;
  health: ReturnType<typeof useServerHealth>;
}

/** Read readiness through HTTP; never infer it from a child process handle. */
export function useServerLifecycle(): ServerLifecycleState {
  const environment = useDesktopEnvironment();
  const health = useServerHealth();
  const status: ServerLifecycleStatus = !environment.isDesktop ? "external"
    : health.isLoading ? "starting" : health.isReachable ? "running" : "stopped";
  return {
    status, health, isReady: health.isReachable,
    error: health.statusLabel === "unreachable" ? "Cannot reach the processing service. Existing jobs may still be running." : null,
    start: async () => { window.location.reload(); },
  };
}
