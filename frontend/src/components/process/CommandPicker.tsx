/** 2x3 grid of command cards for the home screen.
 *
 * Each card represents a batchalign command with a user-friendly label and
 * description. Clicking a card navigates to the new-task form for that command.
 */

import { useRef, type KeyboardEvent } from "react";
import { commandStyle } from "../../utils";

/** Command metadata for the picker grid. */
interface CommandLabels {
  /** User-facing label. */
  label: string;
  /** One-line description of what the command does. */
  description: string;
}

/** The command selects its language policy; a text-NLP command cannot carry a
 * job-level language selector. These are the launcher's supported commands. */
export type CommandDef = CommandLabels & (
  | { id: "morphotag" | "translate" | "align"; language: "per-file"; input: "transcripts" }
  | { id: "utseg"; language: "job"; input: "transcripts" }
  | { id: "transcribe" | "benchmark"; language: "job"; input: "recordings" }
);

/** One input policy supplies discovery and the matching user-facing guidance. */
export const INPUT_POLICIES = {
  recordings: { label: "audio or video recordings", extensions: ["wav", "mp3", "mp4", "m4a", "flac", "ogg", "webm"] },
  transcripts: { label: "CHAT transcripts", extensions: ["cha"] },
} as const;

export const COMMANDS: CommandDef[] = [
  {
    id: "transcribe",
    label: "Transcribe Audio",
    description: "Turn audio or video recordings into written transcripts in CHAT format",
    language: "job",
    input: "recordings",
  },
  {
    id: "morphotag",
    label: "Add Grammar",
    description: "Add part-of-speech tags and grammatical structure, needed for CLAN commands like MLU and DSS",
    language: "per-file",
    input: "transcripts",
  },
  {
    id: "align",
    label: "Align to Audio",
    description: "Estimate word timing in a transcript using its audio; review the inferred timing before relying on it",
    language: "per-file",
    input: "transcripts",
  },
  {
    id: "translate",
    label: "Translate",
    description: "Add an English translation line under each utterance in a non-English transcript",
    language: "per-file",
    input: "transcripts",
  },
  {
    id: "utseg",
    label: "Segment Utterances",
    description: "Identify utterance boundaries in an existing transcript",
    language: "job",
    input: "transcripts",
  },
  {
    id: "benchmark",
    label: "Score Accuracy",
    description: "Measure how closely a machine transcript matches a human-verified one",
    language: "job",
    input: "recordings",
  },
];

interface CommandPickerProps {
  onSelect: (command: CommandDef) => void;
}

export function CommandPicker({ onSelect }: CommandPickerProps) {
  const gridRef = useRef<HTMLDivElement>(null);

  /** Arrow-key navigation within the command grid. */
  function handleKeyDown(e: KeyboardEvent, index: number) {
    const cols = window.innerWidth >= 640 ? 3 : 2; // sm:grid-cols-3 vs grid-cols-2
    let next = index;
    switch (e.key) {
      case "ArrowRight":
        next = Math.min(index + 1, COMMANDS.length - 1);
        break;
      case "ArrowLeft":
        next = Math.max(index - 1, 0);
        break;
      case "ArrowDown":
        next = Math.min(index + cols, COMMANDS.length - 1);
        break;
      case "ArrowUp":
        next = Math.max(index - cols, 0);
        break;
      default:
        return;
    }
    e.preventDefault();
    const buttons = gridRef.current?.querySelectorAll<HTMLButtonElement>("button");
    buttons?.[next]?.focus();
  }

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-800 mb-4">
        What would you like to do?
      </h2>
      <div ref={gridRef} className="grid grid-cols-2 sm:grid-cols-3 gap-3" role="group" aria-label="Choose a command">
        {COMMANDS.map((cmd, i) => {
          const [bg, text] = commandStyle(cmd.id);
          return (
            <button
              key={cmd.id}
              onClick={() => onSelect(cmd)}
              onKeyDown={(e) => handleKeyDown(e, i)}
              className={`${bg} rounded-lg p-4 text-left transition-all hover:scale-[1.02] hover:shadow-md active:scale-[0.98] cursor-pointer`}
            >
              <div className={`text-sm font-semibold ${text}`}>
                {cmd.label}
              </div>
              <div className="text-xs text-gray-600 mt-1">
                {cmd.description}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
