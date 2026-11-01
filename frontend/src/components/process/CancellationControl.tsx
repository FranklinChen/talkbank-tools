/** Render action capabilities from cancellation typestate. Acknowledgement
 * leaves job completion to the authoritative progress observer. */
import type { JobCancellation } from "../../hooks/useJobCancellation";

/** Never turn an HTTP acknowledgement into a fabricated cancelled job result. */
export function CancellationControl({ cancellation }: { cancellation: JobCancellation }) {
  switch (cancellation.kind) {
    case "available": return <button type="button" onClick={cancellation.request} className="text-sm text-red-700">Cancel</button>;
    case "requesting": return <p role="status">Requesting cancellation...</p>;
    case "acknowledged": return <div role="status">
      <p>{cancellation.receipt.outcome === "cancelled" ? "Cancellation acknowledged." : `The service reports that the job already finished (${cancellation.receipt.outcome}).`}</p>
      <p>Waiting for the final job record. Completed output may remain.</p>
      <p className="text-xs">{cancellation.receipt.message}</p>
    </div>;
    case "failed": return <div>
      <p role="alert">Could not confirm cancellation. The job may still be running. {cancellation.message}</p>
      <button type="button" onClick={cancellation.retry} className="text-sm text-red-700">Retry cancellation</button>
    </div>;
  }
  return cancellation satisfies never;
}
