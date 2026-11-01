/** Dashboard application composition root.
 *
 * This file intentionally stays tiny. Bootstrapping fleet discovery,
 * query synchronization, and WebSocket live updates happens inside
 * `useFleetDashboardSync()`, while route composition lives in `AppRoutes`.
 *
 * In desktop mode, checks for first-launch (no `~/.batchalign.ini`) and
 * shows the setup wizard before the main route tree. This matches the
 * batchalign2 behavior where `config_read(interactive=True)` triggers
 * `interactive_setup()` on first CLI invocation.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { AppRoutes } from "./AppRoutes";
import { useFleetDashboardSync } from "./hooks/useFleetDashboardSync";
import { SetupWizard } from "./components/setup/SetupWizard";
import {
  useDesktopConfig,
  useDesktopEnvironment,
} from "./desktop/DesktopContext";

/** Mount the dashboard route tree after starting the fleet-sync side effects. */
export function App() {
  useFleetDashboardSync();
  const environment = useDesktopEnvironment();
  const desktopConfig = useDesktopConfig();

  const [setup, setSetup] = useState<{ kind: "checking" } | { kind: "needed" } | { kind: "ready" } | { kind: "failed"; message: string }>({ kind: "checking" });
  const checkOwner = useRef<object | null>(null);

  const handleSetupComplete = useCallback(() => {
    setSetup({ kind: "ready" });
    if (environment.isDesktop && window.location.pathname === "/") {
      window.history.replaceState(
        null,
        "",
        `/process${window.location.search}`,
      );
      window.dispatchEvent(new PopStateEvent("popstate"));
    }
  }, [environment]);

  async function loadFirstLaunchState() {
    if (checkOwner.current) return;
    const owner = {};
    checkOwner.current = owner;
    setSetup({ kind: "checking" });
    try {
      const next = environment.isDesktop ? await desktopConfig.isFirstLaunch() : false;
      if (typeof next !== "boolean") throw new Error("The desktop returned an incomplete first-launch check.");
      if (checkOwner.current === owner) setSetup(next ? { kind: "needed" } : { kind: "ready" });
    } catch (error) {
      if (checkOwner.current === owner) setSetup({ kind: "failed", message: String(error) });
    } finally { if (checkOwner.current === owner) checkOwner.current = null; }
  }
  useEffect(() => { void loadFirstLaunchState(); return () => { checkOwner.current = null; }; }, [desktopConfig, environment]);

  // Still checking: render nothing briefly
  if (setup.kind === "checking" && environment.isDesktop) return <p role="status">Reading configuration...</p>;
  if (setup.kind === "failed") return <div>
    <p role="alert">Could not read configuration: {setup.message}. No settings were changed.</p>
    <button type="button" onClick={() => { void loadFirstLaunchState(); }}>Retry configuration check</button>
  </div>;

  if (setup.kind === "needed") {
    return <SetupWizard onComplete={handleSetupComplete} />;
  }

  return <AppRoutes />;
}
