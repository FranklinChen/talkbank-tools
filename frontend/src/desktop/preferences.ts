/** Preference intent is different from an observed configuration. An untouched
 * credential is retained at native write time, never replayed from an old read. */

export type PreferenceEngine = "rev" | "whisper";
type EngineChoice = Readonly<{ kind: "supported"; engine: PreferenceEngine }
  | { kind: "unsupported"; name: string }>;
type KeyEdit = Readonly<{ kind: "retain" } | { kind: "replace"; text: string }>;
type StoredKey = "usable" | "unavailable";
type PreferenceRequest = Readonly<{ engine: PreferenceEngine; rev_key: string | null }>;

function usableKey(text: string): boolean { return !!text.trim() && !/\p{Cc}/u.test(text); }

/** A loaded or explicitly chosen draft is not yet permission to write. */
export class PreferenceDraft {
  private constructor(readonly choice: EngineChoice, private readonly storedKey: StoredKey, private readonly keyEdit: KeyEdit) {}
  static loaded(value: unknown): PreferenceDraft {
    if (!value || typeof value !== "object" || !("engine" in value) || !("rev_key" in value)
      || typeof value.engine !== "string" || !value.engine.trim()
      || !(value.rev_key === null || typeof value.rev_key === "string")) {
      throw new Error("The desktop returned incomplete settings. Reload settings before editing.");
    }
    const choice: EngineChoice = value.engine === "rev" || value.engine === "whisper"
      ? { kind: "supported", engine: value.engine } : { kind: "unsupported", name: value.engine };
    return new PreferenceDraft(choice, typeof value.rev_key === "string" && usableKey(value.rev_key) ? "usable" : "unavailable", { kind: "retain" });
  }
  /** First-run choices are explicit UI policy, not a substitute for failed reads. */
  static firstRun(engine: PreferenceEngine, key: string): PreferenceDraft {
    return new PreferenceDraft({ kind: "supported", engine }, "unavailable", key === "" ? { kind: "retain" } : { kind: "replace", text: key });
  }
  select(engine: PreferenceEngine): PreferenceDraft { return new PreferenceDraft({ kind: "supported", engine }, this.storedKey, this.keyEdit); }
  get engine(): string { return this.choice.kind === "supported" ? this.choice.engine : this.choice.name; }
  replaceKey(text: string): PreferenceDraft {
    return new PreferenceDraft(this.choice, this.storedKey, text === "" ? { kind: "retain" } : { kind: "replace", text });
  }
  get keyText(): string { return this.keyEdit.kind === "replace" ? this.keyEdit.text : ""; }
  get keyHint(): string {
    return this.storedKey === "usable" ? "A stored key is available. Leave this blank to keep it." : "Enter a Rev.AI key before selecting Rev.AI for use.";
  }
  /** Only this edge validates an edit before the native capability receives it. */
  admitRequest(): PreferenceRequest {
    if (this.choice.kind === "unsupported") throw new Error("Choose Rev.AI or Whisper explicitly before saving.");
    const engine = this.choice.engine;
    if (engine === "whisper") return Object.freeze({ engine, rev_key: null });
    switch (this.keyEdit.kind) {
      case "retain":
        if (this.storedKey !== "usable") throw new Error("Enter a Rev.AI API key before saving.");
        return Object.freeze({ engine, rev_key: null });
      case "replace":
        if (!usableKey(this.keyEdit.text)) throw new Error("The API key must be a nonempty single line.");
        return Object.freeze({ engine, rev_key: this.keyEdit.text.trim() });
    }
    return this.keyEdit satisfies never;
  }
  prepare(): PreferenceEdit { return PreferenceEdit.prepare(this); }
  /** Acknowledgement retires replacement intent before any later edit. */
  afterSave(receipt: SavedPreferences): PreferenceDraft {
    return new PreferenceDraft({ kind: "supported", engine: receipt.engine },
      receipt.keyChange === "replaced" ? "usable" : this.storedKey, { kind: "retain" });
  }
}

/** The write capability accepts only a native-compatible validated edit. */
export class PreferenceEdit {
  private constructor(private readonly request: PreferenceRequest) {}
  static prepare(draft: PreferenceDraft): PreferenceEdit { return new PreferenceEdit(draft.admitRequest()); }
  get config(): PreferenceRequest { return this.request; }
}

/** Saving is complete only after a valid native acknowledgement. */
export class SavedPreferences {
  private constructor(private readonly selectedEngine: PreferenceEngine, readonly keyChange: "retained" | "replaced", readonly message: string) {}
  get engine(): PreferenceEngine { return this.selectedEngine; }
  static admit(edit: PreferenceEdit, value: unknown): SavedPreferences {
    if (!value || typeof value !== "object" || !("message" in value) || typeof value.message !== "string" || !value.message.trim()) {
      throw new Error("The desktop did not confirm that settings were saved. Reload settings to check them.");
    }
    return new SavedPreferences(edit.config.engine, edit.config.rev_key === null ? "retained" : "replaced", value.message);
  }
}
