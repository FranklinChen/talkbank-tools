/** Polling policy and byte admission use the same compiled waiter as journeys. */
import { expect, it } from "vitest";
import { JobWait } from "../e2e/job-wait";

/**
 * Guards: waiting for failure reports an observed successful completion immediately.
 * Kind: policy
 * Not a type: the requested target and an independently delivered terminal observation can legitimately disagree.
 * State: immutable wire observations exercise target interpretation without a browser, request or clock.
 * Concurrency: each waiter owns its target and reads only its local literal observation.
 */
it("a failure wait reports successful completion as a different outcome", () => {
  expect(JobWait.failed().observe({ status: "completed" })).toEqual({
    kind: "unexpected-terminal",
    status: "completed",
  });
});

/**
 * Guards: unknown external statuses fail admission while all currently emitted terminal failures finish polling.
 * Kind: wire
 * Not a type: an external response can omit status, change its primitive type or add an unknown wire spelling.
 * State: literal decoded JSON values cross the shared schema admission edge without HTTP or browser resources.
 * Concurrency: observations and wait targets are created within each invocation and share no mutable state.
 */
it("wire status admission includes interrupted and writeback failures", () => {
  for (const value of [null, {}, { status: "unknown" }, { status: 1 }]) {
    expect(() => JobWait.completed().observe(value)).toThrow();
  }
  for (const status of ["interrupted", "writeback_failed"]) {
    expect(JobWait.terminal().observe({ status })).toEqual({
      kind: "reached",
      status,
    });
  }
});
