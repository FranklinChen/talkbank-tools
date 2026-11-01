/** Native-only output actions owned by one job/server scope. React Query owns
 * lookup lifetime; the keyed action session retires when route identity changes.
 */
import { useState, useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";
import { desktopEndpointForServer, type DesktopEndpoint } from "../runtime";
import { useDesktopEnvironment, useDesktopFiles } from "../desktop/DesktopContext";
import { LocalJobOutputRequest, OutputFolderAction } from "../desktop/jobOutputs";

function LocalFolders({ endpoint, jobId }: { endpoint: DesktopEndpoint; jobId: string }) {
  const files = useDesktopFiles();
  const query = useQuery({
    queryKey: ["local-job-output-folders", endpoint.url().origin, jobId],
    queryFn: () => LocalJobOutputRequest.forJob(endpoint, jobId).load(),
    retry: false,
    staleTime: Infinity,
  });
  const [owner] = useState(() => new OutputFolderAction(files));
  const action = useSyncExternalStore(owner.subscribe, owner.snapshot);
  switch (query.status) {
    case "pending": return <p role="status" className="text-xs text-zinc-500">Loading recorded output folders…</p>;
    case "error": return <div><p role="alert" className="text-xs text-red-700">{String(query.error)}</p><button type="button" onClick={() => void query.refetch()} className="text-xs underline">Retry output lookup</button></div>;
    case "success": {
      const access = query.data;
      switch (access.kind) {
        case "unavailable": return <p className="text-xs text-zinc-500">{access.message}</p>;
        case "recorded": return <div className="space-y-2">
          <p className="text-xs text-zinc-500">Recorded output {access.folders.length === 1 ? "folder" : "folders"}</p>
          {access.folders.map(folder => <div key={folder.path} className="flex flex-wrap items-center gap-2">
            <button type="button" disabled={action.kind === "opening"} onClick={() => void owner.open(folder)}
              aria-label={access.folders.length === 1 ? "Open Output Folder" : `Open Output Folder ${folder.path}`}
              className="px-3 py-1.5 rounded bg-emerald-700 text-white text-xs disabled:opacity-50">Open Output Folder</button>
            <span className="text-xs font-mono text-zinc-500 break-all">{folder.path}</span>
          </div>)}
          {action.kind === "opening" && <p role="status" className="text-xs text-zinc-500">Opening output folder…</p>}
          {action.kind === "failed" && <p role="alert" className="text-xs text-red-700">{action.message}</p>}
        </div>;
      }
    }
  }
}

/** Reveal only persisted destinations from this desktop's managed service. */
export function JobOutputFolders({ jobId, serverBase }: { jobId: string; serverBase: string }) {
  const { isDesktop } = useDesktopEnvironment();
  if (!isDesktop) return null;
  const endpoint = desktopEndpointForServer(serverBase);
  if (endpoint === null) return <p className="text-xs text-zinc-500">Output folders can be opened on the processing computer.</p>;
  return <LocalFolders key={JSON.stringify([endpoint.url().origin, jobId])} endpoint={endpoint} jobId={jobId} />;
}
