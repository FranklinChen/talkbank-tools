/** Fleet discovery and the single React-owned lifetime for scoped connections. */

import {
  ConnectionPlan,
  runConnections,
  type MessageHandler,
  type StatusHandler,
} from "./liveSync/connections";
import * as Effect from "effect/Effect";
import * as Fiber from "effect/Fiber";
import * as Schema from "effect/Schema";
import { Socket } from "effect/socket";
import { DashboardServer } from "./wire";
import {
  controlPlaneHostPort,
  controlPlaneOrigin,
  controlPlaneWsScheme,
} from "./runtime";

const decodeFleet = Schema.decodeUnknownSync(
  Schema.Array(Schema.Struct({ name: Schema.String, url: Schema.String })),
);

function normalizeHostPort(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  try {
    if (trimmed.includes("://")) return new URL(trimmed).host;
    return trimmed;
  } catch {
    return null;
  }
}

/** Parse `?servers=` from the URL, always including the active control-plane host. */
export function getServerList(): string[] {
  const params = new URLSearchParams(window.location.search);
  const raw = params.get("servers") ?? "";
  const extra = raw
    .split(",")
    .map((s) => normalizeHostPort(s))
    .filter((s): s is string => Boolean(s));
  const primary = controlPlaneHostPort();
  const all = [...new Set([primary, ...extra])];
  return all;
}

/**
 * Discover fleet servers from `GET /fleet` on the connected server.
 * Falls back to `getServerList()` if the endpoint is unavailable.
 */
async function discoverFleet(signal: AbortSignal): Promise<string[]> {
  const params = new URLSearchParams(window.location.search);
  if (params.get("servers")) {
    // Explicit servers in URL, don't override with fleet discovery
    return getServerList();
  }

  try {
    const resp = await fetch(`${controlPlaneOrigin()}/fleet`, { signal });
    if (!resp.ok) return getServerList();
    const fleet = decodeFleet(await resp.json());
    if (!fleet.length) return getServerList();

    // Extract host:port from each fleet URL, deduplicate with current control-plane host.
    const primary = controlPlaneHostPort();
    const fleetHosts = fleet
      .map((s) => normalizeHostPort(s.url))
      .filter((s): s is string => Boolean(s));
    const unique = [...new Set([primary, ...fleetHosts])];
    return unique;
  } catch {
    return getServerList();
  }
}

/** Short hostname label (e.g. "<host>" from "<host>:8000"). */
export function serverLabel(hostPort: string): string {
  return DashboardServer.prepare(hostPort, "ws:").label;
}

/** Discovery, ready publication and connection cleanup share one interruptible fiber. */
export function startFleetSync(
  onServers: (servers: string[]) => void,
  onMessage: MessageHandler,
  onStatus: StatusHandler,
): { close: () => void } {
  const program = Effect.tryPromise((signal) => discoverFleet(signal)).pipe(
    Effect.flatMap((servers) =>
      Effect.gen(function* () {
        const plan = ConnectionPlan.prepare(servers, controlPlaneWsScheme());
        yield* Effect.sync(() => onServers(servers));
        yield* runConnections(plan, onMessage, onStatus);
      }),
    ),
    Effect.provide(Socket.layerWebSocketConstructorGlobal),
  );
  const fiber = Effect.runFork(program);
  return {
    close: () => {
      Effect.runFork(Fiber.interrupt(fiber));
    },
  };
}
