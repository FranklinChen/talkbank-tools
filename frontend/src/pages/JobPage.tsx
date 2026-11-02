/** Route shell for `/dashboard/jobs/:id`.
 *
 * The page is intentionally thin: it parses routing state, delegates all data
 * loading to `useJobPageController`, and hands the resolved view model to the
 * presentational detail component.
 */
import type { ReactElement } from "react";
import { useLocation, useRoute } from "wouter";
import { Layout } from "../components/Layout";
import { JobDetailPageView } from "../components/JobDetailPageView";
import { useJobPageController } from "../hooks/useJobPageController";

/** Render the job-detail route or the small route-level loading/error states. */
function SelectedJobPage({ jobId }: { jobId: string }): ReactElement {
  const [, navigate] = useLocation();
  const state = useJobPageController(jobId);
  switch (state.kind) {
    case "loading":
      return <Layout><p className="text-zinc-400 text-sm">Loading...</p></Layout>;
    case "unavailable":
      return (
        <Layout>
          <p role="alert" className="text-zinc-500 text-sm">Job details could not be loaded. The job may still exist on its processing service.</p>
          <button type="button" onClick={state.retry} className="text-sm underline">Retry job lookup</button>
        </Layout>
      );
    case "ready":
      return (
        <JobDetailPageView
          detail={state.detail}
          wsJob={state.wsJob}
          multiServer={state.multiServer}
          effectiveServer={state.effectiveServer}
          serverBase={state.serverBase}
          onDeleted={() => navigate("/dashboard")}
        />
      );
  }
}

/** A missing route has no query scope; selected jobs own keyed controllers. */
export function JobPage() {
  const [, params] = useRoute("/dashboard/jobs/:id");
  return params === null
    ? <Layout><p className="text-zinc-500 text-sm">Job not found.</p></Layout>
    : <SelectedJobPage key={params.id} jobId={params.id} />;
}
