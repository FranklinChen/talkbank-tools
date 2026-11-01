/** Admit the native service endpoint before mounting queries or job controls.
 * This prevents a window from using a guessed port or a remote path-mode host. */
import { useEffect, useState, type ReactNode } from "react";
import { isDesktop, DesktopEndpoint, bindDesktopEndpoint } from "../runtime";
import { serverStatus, startServer } from "../lib/tauri";
import { DESKTOP_APP_NAME } from "./protocol";

type Startup = { kind: "connecting" } | { kind: "ready" } | { kind: "failed"; message: string };

/** Connect the existing CLI-managed local service before the application starts. */
export function DesktopBootstrap({ children }: { children: ReactNode }) {
  const [startup, setStartup] = useState<Startup>(isDesktop() ? { kind: "connecting" } : { kind: "ready" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!isDesktop()) return;
    document.title = DESKTOP_APP_NAME;
    let active = true;
    async function connect() {
      try {
        let observed = await serverStatus();
        if (observed.state === "stopped") observed = await startServer();
        if (!active) return;
        switch (observed.state) {
          case "ready": {
            bindDesktopEndpoint(DesktopEndpoint.fromReady(observed));
            setStartup({ kind: "ready" });
            break;
          }
          case "missing_binary":
            setStartup({ kind: "failed", message: "Install the current Batchalign CLI before opening the desktop app, then retry." });
            break;
          case "stopped":
            setStartup({ kind: "failed", message: "The managed server did not become ready. Check the CLI server diagnostics, then retry." });
            break;
          case "refused":
            setStartup({ kind: "failed", message: observed.message });
            break;
        }
      } catch (error) {
        if (active) setStartup({ kind: "failed", message: String(error) });
      }
    }
    void connect();
    return () => { active = false; };
  }, [attempt]);
  if (startup.kind === "ready") return children;
  return <main className="max-w-xl mx-auto p-8 space-y-4">
    <h1 className="text-xl font-semibold">{DESKTOP_APP_NAME}</h1>
    {startup.kind === "connecting"
      ? <p role="status">Connecting to the local processing service...</p>
      : <><p role="alert">{startup.message}</p>
        <button type="button" onClick={() => { setStartup({ kind: "connecting" }); setAttempt(value => value + 1); }}>Retry connection</button></>}
    <p className="text-sm text-gray-500">Processing uses the installed CLI and its managed service. Closing this window leaves jobs running.</p>
  </main>;
}
