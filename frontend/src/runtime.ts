/** Control-plane ownership. Native discovery admits an endpoint before queries
 * mount; route changes cannot replace it with an origin or guessed port. */
import type { DesktopServerStatus } from "./desktop/protocol";

/** A complete local endpoint admitted at the native transport boundary. */
export class DesktopEndpoint {
  private constructor(private readonly origin: string) {}

  /** Admit the ready wire state; other service states cannot construct it. */
  static fromReady(observation: Extract<DesktopServerStatus, { state: "ready" }>): DesktopEndpoint {
    const url = new URL(observation.url);
    if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || !url.port
      || url.username || url.password || url.pathname !== "/" || url.search || url.hash
      || !observation.build_hash) throw new Error("The native service did not identify a local endpoint.");
    return new DesktopEndpoint(url.origin);
  }

  /** Return a copy, so consumers cannot mutate the admitted endpoint. */
  url(): URL { return new URL(this.origin); }
}

type DesktopControlPlane = { kind: "awaiting-discovery" } | { kind: "admitted"; endpoint: DesktopEndpoint };
let desktopControlPlane: DesktopControlPlane = { kind: "awaiting-discovery" };

/** Bind the discovery proof independently of the current document URL. */
export function bindDesktopEndpoint(endpoint: DesktopEndpoint): void {
  desktopControlPlane = { kind: "admitted", endpoint };
}

/** True when running inside a Tauri webview (desktop app). */
export function isDesktop(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

/** Return the native discovery capability only for this exact job server.
 * A remote dashboard record can never borrow the local computer's opener.
 */
export function desktopEndpointForServer(base: string): DesktopEndpoint | null {
  if (!isDesktop() || desktopControlPlane.kind === "awaiting-discovery") return null;
  const endpoint = desktopControlPlane.endpoint;
  if (!base) return endpoint;
  try {
    const url = new URL(base);
    return url.origin === endpoint.url().origin && url.pathname === "/"
      && !url.username && !url.password && !url.search && !url.hash ? endpoint : null;
  } catch { return null; }
}

function normalizeServerUrl(raw: string): URL | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  try {
    if (trimmed.includes("://")) return new URL(trimmed);
    return new URL(`http://${trimmed}`);
  } catch {
    return null;
  }
}

function serverOverrideFromQuery(): URL | null {
  const params = new URLSearchParams(window.location.search);
  return (
    normalizeServerUrl(params.get("server") ?? "") ??
    normalizeServerUrl(params.get("api") ?? "")
  );
}

export function controlPlaneUrl(): URL {
  if (isDesktop()) {
    switch (desktopControlPlane.kind) {
      case "awaiting-discovery": throw new Error("Native service discovery has not completed.");
      case "admitted": return desktopControlPlane.endpoint.url();
    }
  }
  const override = serverOverrideFromQuery();
  if (override) return override;

  if (location.protocol === "http:" || location.protocol === "https:") {
    return new URL(`${location.protocol}//${location.host}`);
  }

  throw new Error("The browser page has no HTTP control-plane endpoint.");
}

export function controlPlaneOrigin(): string {
  return controlPlaneUrl().origin;
}

export function controlPlaneHostPort(): string {
  return controlPlaneUrl().host;
}

export function controlPlaneHttpScheme(): "http:" | "https:" {
  return controlPlaneUrl().protocol === "https:" ? "https:" : "http:";
}

export function controlPlaneWsScheme(): "ws:" | "wss:" {
  return controlPlaneUrl().protocol === "https:" ? "wss:" : "ws:";
}

export function isSameControlPlaneAsPage(hostPort: string): boolean {
  const control = controlPlaneUrl();
  if (hostPort !== control.host) return false;

  if (location.protocol !== "http:" && location.protocol !== "https:") {
    return false;
  }

  return control.protocol === location.protocol && control.host === location.host;
}
