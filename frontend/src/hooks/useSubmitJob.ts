/** Bind the launch state machine to a mounted screen and admitted service.
 * This hook owns HTTP/cache effects; the controller owns all launch phases. */
import { useLayoutEffect, useState, useSyncExternalStore } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { submitJob } from "../api";
import { controlPlaneOrigin } from "../runtime";
import { useDesktopConfig, useDesktopFiles } from "../desktop/DesktopContext";
import { JobLaunch } from "../desktop/jobLaunch";

/** Submit one checked plan and retain the server's actual job identity. */
export function useSubmitJob() {
  const base = controlPlaneOrigin();
  const queryClient = useQueryClient();
  const files = useDesktopFiles();
  const preferences = useDesktopConfig();
  const create = () => new JobLaunch({
    inspect: files.admitJobPaths,
    preferences: preferences.readConfig,
    send: request => {
      const { plan } = request;
      return submitJob({
        command: plan.request.command, lang: plan.request.lang, num_speakers: plan.request.numSpeakers,
        paths_mode: true, source_paths: plan.request.sourcePaths, output_paths: plan.request.outputPaths,
        source_dir: plan.request.sourceDir,
        options: { command: plan.request.command, ...(request.kind === "with-asr" ? { asr_engine: request.engine } : {}) },
      }, base);
    },
    // An unconfirmed response may still have created a job. Refresh the list
    // for inspection without guessing a match or submitting anything again.
    observed: () => { void queryClient.invalidateQueries({ queryKey: ["jobs"] }); },
  });
  const [owner, setOwner] = useState(create);
  // Retire at the unmount commit, before a queued native reply can continue;
  // passive effect cleanup would leave an unnecessary post-commit window.
  useLayoutEffect(() => owner.attach(), [owner]);
  const state = useSyncExternalStore(owner.subscribe, owner.read);
  return { state, reset: () => { setOwner(create()); } };
}
