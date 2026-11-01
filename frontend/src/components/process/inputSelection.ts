/** Discovery owns a folder-bound immutable file population. Empty discovery is
 * useful feedback, but cannot manufacture the SelectedInputs job preparation
 * requires. Neither this type nor a filename display certifies filesystem identity. */

export type DiscoveredInput = { kind: "empty"; folder: string }
  | { kind: "selected"; input: SelectedInputs };

/** A complete, nonempty native discovery result, before filesystem admission. */
export class SelectedInputs {
  private constructor(readonly folder: string, private readonly entries: readonly [string, ...string[]]) {}
  get files(): readonly [string, ...string[]] { return this.entries; }

  /** Reject incomplete transport; retain genuine empty discovery explicitly. */
  static fromDiscovery(folder: string, files: readonly string[]): DiscoveredInput {
    if (typeof folder !== "string" || !folder || !Array.isArray(files)
      || !files.every(path => typeof path === "string" && path.length > 0)) {
      throw new Error("The desktop returned an incomplete file list. Choose the input folder again.");
    }
    const [first, ...rest] = files;
    if (first === undefined) return { kind: "empty", folder };
    return { kind: "selected", input: new SelectedInputs(folder, Object.freeze([first, ...rest] as const)) };
  }
}

const PAGE_SIZE = 25;

/** Pages are bound to a selected population; consumers cannot supply an
 * out-of-range offset or carry a page number over to another folder's files. */
export class InputFilePage {
  private constructor(private readonly input: SelectedInputs, private readonly offset: number) {}
  static first(input: SelectedInputs): InputFilePage { return new InputFilePage(input, 0); }
  get firstNumber(): number { return this.offset + 1; }
  get lastNumber(): number { return Math.min(this.offset + PAGE_SIZE, this.input.files.length); }
  get total(): number { return this.input.files.length; }
  get files(): readonly string[] { return this.input.files.slice(this.offset, this.lastNumber); }
  /** A newly discovered population retires the earlier population's page. */
  forInput(input: SelectedInputs): InputFilePage { return this.input === input ? this : InputFilePage.first(input); }
  get next(): InputFilePage | null {
    return this.lastNumber === this.total ? null : new InputFilePage(this.input, this.lastNumber);
  }
  get previous(): InputFilePage | null {
    return this.offset === 0 ? null : new InputFilePage(this.input, Math.max(0, this.offset - PAGE_SIZE));
  }
}
