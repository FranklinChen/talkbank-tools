/** Scoped dashboard transports. Only an acquired reader permits live operations. */
import * as Effect from "effect/Effect";
import type * as Scope from "effect/Scope";
import { Socket } from "effect/socket";
import { DashboardServer, DashboardUpdate } from "../wire";

export type MessageHandler = (update: DashboardUpdate) => undefined;
export type StatusHandler = (server: string, connected: boolean) => void;

type ConnectionEnd =
  | { readonly phase: "opening"; readonly cause: Socket.SocketError }
  | { readonly phase: "opened"; readonly cause: Socket.SocketError };

/** A prepared fleet has one connection owner per exact server address. */
export class ConnectionPlan {
  private constructor(private readonly prepared: readonly DashboardServer[]) {}
  get targets(): readonly DashboardServer[] {
    return this.prepared;
  }

  static prepare(
    servers: readonly string[],
    scheme: "ws:" | "wss:",
  ): ConnectionPlan {
    const addresses = [...new Set(servers)];
    const prepared = addresses.map((hostPort) =>
      DashboardServer.prepare(hostPort, scheme),
    );
    return new ConnectionPlan([
      ...new Map(prepared.map((server) => [server.url, server])).values(),
    ]);
  }
}

/** Carries the reader/writer obtained after the transport's real open event. */
class OpenedConnection {
  private constructor(
    private readonly reader: Socket.Reader,
    private readonly writer: Socket.Writer,
  ) {}

  static acquire(
    socket: Socket.Socket,
  ): Effect.Effect<OpenedConnection, ConnectionEnd, Scope.Scope> {
    return Effect.gen(function* () {
      const reader = yield* socket.reader.pipe(
        Effect.mapError((cause) => ({ phase: "opening" as const, cause })),
      );
      const writer = yield* socket.writer;
      return new OpenedConnection(reader, writer);
    });
  }

  listen(
    server: DashboardServer,
    onMessage: MessageHandler,
  ): Effect.Effect<never, Socket.SocketError> {
    const receive = this.reader.pull.pipe(
      Effect.flatMap((frames) =>
        Effect.forEach(
          frames,
          (frame) => {
            const text =
              typeof frame === "string"
                ? frame
                : new TextDecoder().decode(frame);
            if (text === "pong") return Effect.void;
            return DashboardUpdate.decode(server, text).pipe(
              Effect.match({
                onFailure: () =>
                  console.warn(
                    "Rejected malformed dashboard WebSocket message.",
                  ),
                onSuccess: (update) => onMessage(update),
              }),
            );
          },
          { discard: true },
        ),
      ),
    );
    const heartbeat = Effect.sleep("30 seconds").pipe(
      Effect.andThen(this.writer.write("ping")),
    );
    return Effect.raceFirst(Effect.forever(receive), Effect.forever(heartbeat));
  }
}

function connect(
  target: ConnectionPlan["targets"][number],
  onMessage: MessageHandler,
  onStatus: StatusHandler,
): Effect.Effect<never, ConnectionEnd, Socket.WebSocketConstructor> {
  return Effect.scoped(
    Effect.gen(function* () {
      const socket = yield* Socket.makeWebSocket(target.url);
      const opened = yield* OpenedConnection.acquire(socket);
      yield* Effect.acquireRelease(
        Effect.sync(() => onStatus(target.label, true)),
        () => Effect.sync(() => onStatus(target.label, false)),
      );
      return yield* opened
        .listen(target, onMessage)
        .pipe(
          Effect.mapError((cause) => ({ phase: "opened" as const, cause })),
        );
    }),
  );
}

/** Reconnection is sequenced after scope cleanup; cancellation interrupts sleep. */
function reconnect(
  target: ConnectionPlan["targets"][number],
  onMessage: MessageHandler,
  onStatus: StatusHandler,
  delayMs = 1000,
): Effect.Effect<never, never, Socket.WebSocketConstructor> {
  return connect(target, onMessage, onStatus).pipe(
    Effect.catch((ending) => {
      // An established connection resets the existing 1s..30s retry policy.
      const waitMs = ending.phase === "opened" ? 1000 : delayMs;
      return Effect.sleep(waitMs).pipe(
        Effect.andThen(
          Effect.suspend(() =>
            reconnect(
              target,
              onMessage,
              onStatus,
              Math.min(waitMs * 2, 30_000),
            ),
          ),
        ),
      );
    }),
  );
}

/** The constructor service is the same effect boundary for browsers and doubles. */
export function runConnections(
  plan: ConnectionPlan,
  onMessage: MessageHandler,
  onStatus: StatusHandler,
) {
  return Effect.all(
    plan.targets.map((target) => reconnect(target, onMessage, onStatus)),
    {
      concurrency: "unbounded",
      discard: true,
    },
  );
}
