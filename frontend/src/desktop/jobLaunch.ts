/** The launcher owns unsent work only while its screen is attached. Each async
 * reply must advance its own phase; a retired phase cannot reach the POST.
 * Sent work belongs to the service and is observed, never cancelled by unmount. */
import { JobPlan, type PreparedJob } from "../components/process/jobPlan";
import type { DesktopConfigCapability, DesktopFilesCapability } from "./capabilities";
import type { PreferenceEngine } from "./preferences";
import type { SubmissionOutcome, SubmittedJob } from "../submission";

/** ASR selection is present only for commands that require it. */
export type LaunchRequest = Readonly<
  | { kind: "with-asr"; plan: JobPlan; engine: PreferenceEngine }
  | { kind: "without-asr"; plan: JobPlan }
>;

/** Only an available snapshot offers the action that can claim this launcher. */
export type JobLaunchState = Readonly<
  | { kind: "available"; start: (proposal: PreparedJob) => Promise<LaunchResult> }
  | { kind: "checking-paths" }
  | { kind: "reading-preferences"; plan: JobPlan }
  | { kind: "sending"; plan: JobPlan }
  | { kind: "finished" }
  | { kind: "retired" }
>;

type AvailableLaunch = Extract<JobLaunchState, { kind: "available" }>;
type PreparingLaunch = Extract<JobLaunchState, { kind: "checking-paths" | "reading-preferences" }>;

/** Retirement is absence of launch authority, not an HTTP refusal or failure. */
export type LaunchResult = Readonly<
  | { kind: "accepted"; plan: JobPlan; job: SubmittedJob }
  | { kind: "unconfirmed"; plan: JobPlan; message: string }
  | { kind: "refused"; message: string }
  | { kind: "retired" }
>;

interface LaunchPorts {
  inspect: DesktopFilesCapability["admitJobPaths"];
  preferences: DesktopConfigCapability["readConfig"];
  send: (request: LaunchRequest) => Promise<SubmissionOutcome>;
  observed: () => void;
}

/** One screen-bound state machine, including the single-flight claim. Snapshot
 * identity binds delayed replies to a phase, even across detach/reattach. */
export class JobLaunch {
  private state: JobLaunchState = { kind: "retired" };
  private readonly listeners = new Set<() => void>();
  constructor(private readonly ports: LaunchPorts) {}

  read = (): JobLaunchState => this.state;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  private publish(state: JobLaunchState) {
    this.state = state;
    this.listeners.forEach(listener => listener());
  }
  private available(): AvailableLaunch {
    const state: AvailableLaunch = { kind: "available", start: proposal => this.prepare(state, proposal) };
    return state;
  }
  /** An effect attachment owns its phases. Reattachment creates a fresh claim,
   * so an old reply cannot acquire authority from the new available state. */
  attach(): () => void {
    this.publish(this.available());
    return () => { this.publish({ kind: "retired" }); };
  }
  private refuse(phase: PreparingLaunch, error: unknown): LaunchResult {
    if (this.read() !== phase) return { kind: "retired" };
    this.publish(this.available());
    return { kind: "refused", message: String(error) };
  }
  private async prepare(available: AvailableLaunch, proposal: PreparedJob): Promise<LaunchResult> {
    if (this.read() !== available) return { kind: "retired" };
    const checking: PreparingLaunch = { kind: "checking-paths" };
    this.publish(checking);
    let plan: JobPlan;
    try { plan = await JobPlan.admit(proposal, this.ports.inspect); }
    catch (error) { return this.refuse(checking, error); }
    if (this.read() !== checking) return { kind: "retired" };

    switch (plan.request.command) {
      case "transcribe": case "benchmark": {
        const reading: PreparingLaunch = { kind: "reading-preferences", plan };
        this.publish(reading);
        let engine: string;
        try { engine = (await this.ports.preferences()).engine; }
        catch (error) { return this.refuse(reading, error); }
        if (this.read() !== reading) return { kind: "retired" };
        if (engine !== "rev" && engine !== "whisper") return this.refuse(reading,
          "This simple launcher supports Rev.AI and Whisper. Use the CLI for your configured engine, or explicitly change the ASR preference in Settings.");
        return this.send(reading, { kind: "with-asr", plan, engine });
      }
      case "morphotag": case "translate": case "align": case "utseg":
        return this.send(checking, { kind: "without-asr", plan });
    }
    return plan.request.command satisfies never;
  }
  private async send(previous: PreparingLaunch, request: LaunchRequest): Promise<LaunchResult> {
    if (this.read() !== previous) return { kind: "retired" };
    const sending: JobLaunchState = { kind: "sending", plan: request.plan };
    // Claim and initiate HTTP before notifying observers. No callback or await
    // can retire this phase between claiming the POST and actually sending it.
    this.state = sending;
    let pending: Promise<SubmissionOutcome>;
    try { pending = this.ports.send(request); }
    catch (error) { pending = Promise.resolve({ kind: "unconfirmed", message: String(error) }); }
    this.listeners.forEach(listener => listener());
    let outcome: SubmissionOutcome;
    try { outcome = await pending; }
    catch (error) { outcome = { kind: "unconfirmed", message: String(error) }; }
    // Even a retired view may have sent a real job. Refresh discovery without
    // publishing its reply into another launcher's configuration.
    this.ports.observed();
    if (this.read() !== sending) return { kind: "retired" };
    switch (outcome.kind) {
      case "refused":
        this.publish(this.available());
        return { kind: "refused", message: outcome.error.message };
      case "accepted":
        this.publish({ kind: "finished" });
        return { kind: "accepted", plan: request.plan, job: outcome.job };
      case "unconfirmed":
        this.publish({ kind: "finished" });
        return { kind: "unconfirmed", plan: request.plan, message: outcome.message };
    }
    return outcome satisfies never;
  }
}
