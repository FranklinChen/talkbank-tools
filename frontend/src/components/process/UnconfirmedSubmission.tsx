/** Keep a possibly accepted submission distinct from a form the user may retry.
 * Inspection and a deliberate new configuration are the available actions. */
import { Link } from "wouter";
import type { CommandDef } from "./CommandPicker";
import type { JobPlan } from "./jobPlan";

/** Preserve the submitted context while explaining uncertainty without a fake ID. */
export function UnconfirmedSubmission({ command, plan, message, onConfigure }: {
  command: CommandDef; plan: JobPlan; message: string; onConfigure: () => void;
}) {
  return <section className="max-w-xl mx-auto space-y-4" aria-label="Unconfirmed submission">
    <div role="alert">
      <h2 className="font-semibold">Submission could not be confirmed</h2>
      <p>The service may already have started this job. The request has not been sent again.</p>
      <p className="mt-2 text-sm">{message}</p>
    </div>
    <dl className="text-sm space-y-1">
      <dt className="font-medium">Task</dt><dd>{command.label} — {plan.request.sourcePaths.length} input {plan.request.sourcePaths.length === 1 ? "file" : "files"}</dd>
      <dt className="font-medium">Input folder</dt><dd className="break-all">{plan.request.sourceDir}</dd>
      <dt className="font-medium">Output folder</dt><dd className="break-all">{plan.outputFolder}</dd>
    </dl>
    <p>Check Recent Tasks below or open the task list. Look for this command and these files before starting them again. An empty list does not confirm that the request was rejected.</p>
    <Link href="/dashboard" className="block underline">Open task list</Link>
    <button type="button" onClick={onConfigure} className="rounded border px-3 py-2">I checked the tasks; configure another job</button>
  </section>;
}
