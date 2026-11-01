/** A proposal contains choices; only native filesystem admission can turn it
 * into the JobPlan accepted by the HTTP submit owner. No path equality guess
 * in the webview can certify that two local names refer to distinct files. */
import type { CommandDef } from "./CommandPicker";
import type { SelectedInputs } from "./inputSelection";
import type { DesktopPathProposal, DesktopPathReceipt } from "../../desktop/protocol";

/** Separate output is incomplete until the user actually chooses a folder. */
export type OutputChoice = { kind: "separate"; folder: string | null } | { kind: "in-place" };

/** Synchronous admission captures all user choices before an async boundary. */
export class PreparedJob {
  private constructor(readonly paths: DesktopPathProposal, readonly lang: string, readonly numSpeakers: number) {}

  /** Check scalar choices and own an immutable snapshot of discovered inputs. */
  static prepare(command: CommandDef, input: SelectedInputs, destination: OutputChoice, lang: string, speakers: string): PreparedJob {
    if (command.language === "job" && !/^[a-z]{3}$/.test(lang)) throw new Error("Enter a three-letter language code, such as eng.");
    const audio = command.id === "transcribe" || command.id === "benchmark";
    const numSpeakers = audio ? Number(speakers) : 1;
    if (audio && (!/^\d+$/.test(speakers) || !Number.isInteger(numSpeakers) || numSpeakers < 1 || numSpeakers > 4294967295)) {
      throw new Error("Enter a positive whole number of speakers.");
    }
    let output: DesktopPathProposal["destination"];
    switch (destination.kind) {
      case "in-place": output = { mode: "in_place" }; break;
      case "separate":
        if (!destination.folder) throw new Error("Choose an output folder before processing.");
        output = { mode: "separate", folder: destination.folder };
        break;
    }
    return new PreparedJob(Object.freeze({ command: command.id, source_root: input.folder,
      inputs: Object.freeze([...input.files]), destination: Object.freeze(output),
    }), command.language === "job" ? lang : "per-file", numSpeakers);
  }
}

type PreparedRequest = Readonly<{
  command: CommandDef["id"]; lang: string; sourceDir: string; numSpeakers: number;
  sourcePaths: readonly string[]; outputPaths: readonly string[];
}>;

/** The only HTTP-submit input, minted from its own native admission response. */
export class JobPlan {
  private constructor(private readonly admitted: PreparedRequest, readonly outputFolder: string) {}
  get request(): PreparedRequest { return this.admitted; }

  /** Obtain a filesystem observation; incomplete transport never becomes a job. */
  static async admit(proposal: PreparedJob, inspect: (paths: DesktopPathProposal) => Promise<DesktopPathReceipt>): Promise<JobPlan> {
    const paths = await inspect(proposal.paths);
    if (!paths || typeof paths.source_root !== "string" || !paths.source_root ||
        typeof paths.output_root !== "string" || !paths.output_root ||
        !Array.isArray(paths.inputs) || !Array.isArray(paths.outputs) ||
        paths.inputs.length !== proposal.paths.inputs.length || paths.outputs.length !== paths.inputs.length ||
        ![...paths.inputs, ...paths.outputs].every(path => typeof path === "string" && path.length > 0)) {
      throw new Error("The desktop returned incomplete input/output admission. No job was submitted.");
    }
    return new JobPlan(Object.freeze({ command: proposal.paths.command, lang: proposal.lang, numSpeakers: proposal.numSpeakers,
      sourceDir: paths.source_root, sourcePaths: Object.freeze([...paths.inputs]), outputPaths: Object.freeze([...paths.outputs]),
    }), paths.output_root);
  }
}
