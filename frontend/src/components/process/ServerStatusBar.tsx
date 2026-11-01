/** Read-only readiness for the shared processing service. Reconnecting repeats
 * admission; this window never exposes a stop action for other users' jobs. */
import type { ServerLifecycleState } from "../../hooks/useServerLifecycle";

/** Display observed readiness and a reconnect action after transport failure. */
export function ServerStatusBar({ lifecycle }: { lifecycle: ServerLifecycleState }) {
  return <div className="border rounded p-3 space-y-2">
    <p role="status">{lifecycle.isReady ? "Processing service ready" : "Processing service unavailable"}</p>
    {lifecycle.error && <p role="alert">{lifecycle.error}</p>}
    {!lifecycle.isReady && <button type="button" onClick={lifecycle.start}>Reconnect</button>}
    <p className="text-xs text-gray-500">Closing the desktop window leaves processing jobs running.</p>
  </div>;
}
