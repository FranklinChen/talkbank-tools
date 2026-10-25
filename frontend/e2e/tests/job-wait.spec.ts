import { expect, test } from "@playwright/test";
import { JobWait } from "../job-wait";

test("a failure wait rejects successful completion immediately", () => {
  expect(JobWait.failed().observe({ status: "completed" })).toEqual({
    kind: "unexpected-terminal", status: "completed",
  });
});

test("only active jobs can keep a wait pending", () => {
  for (const status of ["queued", "running"]) {
    expect(JobWait.failed().observe({ status }).kind).toBe("waiting");
  }
  for (const status of ["failed", "cancelled"]) {
    expect(JobWait.completed().observe({ status }).kind).toBe("unexpected-terminal");
  }
  for (const status of ["completed", "failed", "cancelled"]) {
    expect(JobWait.terminal().observe({ status }).kind).toBe("reached");
  }
  expect(JobWait.failed().observe({ status: "failed" }).kind).toBe("reached");
  expect(JobWait.running().observe({ status: "running" }).kind).toBe("reached");
});

test("malformed observations never become pending jobs", () => {
  for (const value of [null, {}, { status: "unknown" }, { status: 1 }]) {
    expect(() => JobWait.completed().observe(value)).toThrow();
  }
});
