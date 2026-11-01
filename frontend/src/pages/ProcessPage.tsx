/** Route shell for `/process`, the end-user processing flow.
 *
 * This is the default landing page in the desktop app. Researchers see the
 * command picker, configure a job, and watch progress, all without opening
 * a terminal. The existing `/dashboard` routes are preserved for power users.
 *
 * Native startup has already admitted the CLI-managed service. This page
 * observes its availability without owning or stopping its process.
 */

import { ProcessForm } from "../components/process/ProcessForm";
import { RecentJobs } from "../components/process/RecentJobs";
import { ServerStatusBar } from "../components/process/ServerStatusBar";
import { OnboardingOverlay } from "../components/process/OnboardingOverlay";
import { useServerLifecycle } from "../hooks/useServerLifecycle";
import { useSortedJobs } from "../state";
import { useDesktopEnvironment } from "../desktop/DesktopContext";

export function ProcessPage() {
  const environment = useDesktopEnvironment();
  const lifecycle = useServerLifecycle();
  const jobs = useSortedJobs();

  return (
    <div className="space-y-6">
      {/* Server status bar (desktop only, or when server is down in web mode) */}
      {(environment.isDesktop || !lifecycle.isReady) && (
        <ServerStatusBar lifecycle={lifecycle} />
      )}

      {/* First-time onboarding hint */}
      <OnboardingOverlay hasJobs={jobs.length > 0} />

      <ProcessForm isServerReady={lifecycle.isReady} />
      <RecentJobs />
    </div>
  );
}
