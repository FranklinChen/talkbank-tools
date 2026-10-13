// Pure presentation-boundary tests: no browser, server, model or clock needed.
import { test, expect } from "@playwright/test";
import { diagnosticLines, diagnosedSummary, exclusionSummary, exclusionText } from "../../src/utils.ts";

const span = { utterance: 3, first_word: 2, words: 2, language: "spa" };

test("unavailable UTR is not a CHAT validity failure or deliberate omission", () => {
  const diagnostics = { findings: null, shortfalls: [{
    kind: "utr_unavailable", engine: "rev_utr", pass: "initial",
    category: "provider_transient", detail: "provider unavailable",
  }] };
  expect(diagnosedSummary(diagnostics)).toBe("written, 0 diagnostics, UTR initial pass unavailable");
  expect(diagnosticLines(diagnostics)).toEqual([
    "UTR initial pass unavailable (rev_utr, provider_transient): provider unavailable; CHAT and available timing retained",
  ]);
});

test("provenance shortfalls do not claim CHAT or morphology loss", () => {
  const diagnostics = { findings: null, shortfalls: [{
    kind: "morphology_provenance_incomplete", detail: "prior repair count unreadable",
  }] };
  expect(diagnosedSummary(diagnostics)).toBe("written, 0 diagnostics, morphology provenance incomplete");
  expect(diagnosticLines(diagnostics)).toEqual([
    "morphology provenance incomplete: prior repair count unreadable; morphology and original records retained",
  ]);
});

test("secondary shortfalls remain visible without CHAT validation findings", () => {
  const diagnostics = {
    findings: null,
    shortfalls: [{
      kind: "secondary_morphology_incomplete", spans: 1, words: 2,
      first_spans: [{ span, cause: "merge", detail: "analysis could not be merged" }],
    }],
  };
  expect(diagnosedSummary(diagnostics)).toBe(
    "written, 0 diagnostics, secondary morphology incomplete for 2 words in 1 span(s)",
  );
  expect(diagnosticLines(diagnostics)).toEqual([
    "secondary morphology incomplete: 1 span(s), 2 words retain L2 placeholders (first: utterance 3, spa: analysis could not be merged)",
  ]);
});

test("unsupported secondary languages are not mislabeled as alignment exclusions", () => {
  const exclusion = {
    kind: "unsupported_secondary_morphology", spans: 1, words: 2, first_spans: [span],
  };
  expect(exclusionSummary([exclusion])).toBe("1 span(s) excluded from secondary morphology: unsupported languages");
  expect(exclusionText(exclusion)).toBe("secondary morphology excluded: 1 span(s), 2 words in unsupported languages; L2 placeholders retained");
  expect(exclusionSummary([])).toBe("");
  expect(exclusionSummary([exclusion, {
    kind: "not_in_recording", excluded_utterances: 1, excluded_words: 3, first_excluded: [],
  }])).toContain("1 utterance not in the recording");
});

test("unplaced secondary words report an utterance without guessing a word address", () => {
  const diagnostics = { findings: null, shortfalls: [{
    kind: "secondary_morphology_unplaced", utterances: 1, words: 2,
    first_utterances: [{ utterance: 3, words: 2, cause: "alignment", detail: "primary analysis does not align" }],
  }] };
  expect(diagnosedSummary(diagnostics)).toContain("secondary morphology not dispatched for 2 words in 1 utterance(s)");
  expect(diagnosticLines(diagnostics)).toEqual([
    "secondary morphology not dispatched: 2 words in 1 utterance(s) could not be placed (first: utterance 3: primary analysis does not align)",
  ]);
});
