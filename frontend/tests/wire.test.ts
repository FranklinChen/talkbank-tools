/** Literal wire data crosses the same admission boundary used by live sockets. */
import { expect, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import { DashboardServer, DashboardUpdate } from "../src/wire";

/**
 * Guards: malformed frames cannot become admitted updates, and valid updates retain their producing server.
 * Kind: wire
 * Not a type: remote JSON bytes can violate generated payload schemas regardless of the sender's declared types.
 * State: literal frames and a prepared server value exercise decoding without opening any transport.
 * Concurrency: each decode is local and immutable, with no shared service or ambient browser state.
 */
it.effect(
  "wire admission retains provenance and rejects malformed payloads",
  () =>
    Effect.gen(function* () {
      const server = DashboardServer.prepare("server.test:8001", "ws:");
      const admitted = yield* DashboardUpdate.decode(
        server,
        '{"type":"job_deleted","job_id":"job-1"}',
      );
      expect(admitted.server).toBe(server);
      expect(admitted.message).toEqual({
        type: "job_deleted",
        job_id: "job-1",
      });
      for (const text of [
        "{",
        '{"type":"job_deleted","job_id":42}',
        '{"type":"job_update","job":{}}',
      ]) {
        const exit = yield* Effect.exit(DashboardUpdate.decode(server, text));
        expect(exit._tag).toBe("Failure");
      }
    }),
);
