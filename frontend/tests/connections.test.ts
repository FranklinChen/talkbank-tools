/** External socket observations share one admitted service double. */
import { expect, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as Fiber from "effect/Fiber";
import * as Queue from "effect/Queue";
import type { DashboardUpdate } from "../src/wire";
import * as TestClock from "effect/testing/TestClock";
import { Socket } from "effect/socket";
import { ConnectionPlan, runConnections } from "../src/liveSync/connections";
import { ScriptedSockets } from "./fixtures/sockets";

/**
 * Guards: reconnect closes the old transport, resets heartbeat ownership, and stops on cancellation.
 * Kind: contract
 * Not a type: timer delivery, transport close events and finalizer ordering occur in the Effect runtime after scoped acquisition.
 * State: one scripted constructor service records socket effects, with an Effect test clock for heartbeat and retry deadlines.
 * Concurrency: each test owns its service, clock and fiber; peer readiness comes from the fixture queue.
 */
it.effect("scoped connections own heartbeat, reconnect and shutdown", () =>
  Effect.gen(function* () {
    const sockets = yield* ScriptedSockets.make;
    const statuses: boolean[] = [];
    const updates = yield* Queue.unbounded<DashboardUpdate>();
    const fiber = yield* runConnections(
      ConnectionPlan.prepare(["server.test:8001", "SERVER.TEST:8001"], "ws:"),
      (update) => {
        Queue.offerUnsafe(updates, update);
      },
      (_server, connected) => statuses.push(connected),
    ).pipe(
      Effect.provideService(Socket.WebSocketConstructor, sockets.connect),
      Effect.forkChild,
    );
    const first = (yield* sockets.next).open();
    yield* TestClock.adjust("30 seconds");
    expect(first.sent).toEqual(["ping"]);
    first.receive('{"type":"job_deleted","job_id":"job-1"}');
    const update = yield* Queue.take(updates);
    expect(update.server.url).toBe("ws://server.test:8001/ws");
    first.disconnect();
    yield* TestClock.adjust("1 second");
    const second = (yield* sockets.next).open();
    yield* TestClock.adjust("30 seconds");
    expect(second.sent).toEqual(["ping"]);
    yield* Fiber.interrupt(fiber);
    second
      .observeClosed()
      .deliverQueuedFrame('{"type":"job_deleted","job_id":"late"}');
    yield* TestClock.adjust("1 minute");
    expect(yield* Queue.size(updates)).toBe(0);
    expect(sockets.attempts).toBe(2);
    expect(statuses).toEqual([true, false, true, false]);
    // Effect retains a harmless late-error sink when closing an aborted socket.
    expect([first.events, second.events]).toEqual([["error"], ["error"]]);
    expect(second.state).toBe(3);
  }),
);

/**
 * Guards: unmount during acquisition releases the connecting socket without publishing a ready session.
 * Kind: contract
 * Not a type: a browser can deliver its open event after cancellation; removing actual listener registrations is a runtime effect.
 * State: the shared constructor double retains the connecting peer so late delivery can be observed after scope cleanup.
 * Concurrency: a private fiber and queue establish the pending acquisition before interruption, with no real sockets or sleeps.
 */
it.effect("cancellation owns a pending handshake", () =>
  Effect.gen(function* () {
    const sockets = yield* ScriptedSockets.make;
    const statuses: boolean[] = [];
    const fiber = yield* runConnections(
      ConnectionPlan.prepare(["server.test:8001"], "ws:"),
      () => {},
      (_server, connected) => statuses.push(connected),
    ).pipe(
      Effect.provideService(Socket.WebSocketConstructor, sockets.connect),
      Effect.forkChild,
    );
    const pending = yield* sockets.next;
    yield* Fiber.interrupt(fiber);
    expect(pending.state).toBe(3);
    expect(pending.events).toEqual(["error"]);
    pending.observeClosed().deliverQueuedOpen();
    yield* TestClock.adjust("1 minute");
    expect(statuses).toEqual([]);
    expect(sockets.attempts).toBe(1);
  }),
);
