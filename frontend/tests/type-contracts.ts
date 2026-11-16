/** Compiler-only admission contracts. This module is never a runtime test. */
import { ConnectionPlan, type MessageHandler } from "../src/liveSync/connections";
import { handleDashboardMessage } from "../src/liveSync/handleDashboardMessage";
import {
  DashboardServer,
  DashboardUpdate,
  type DashboardMessage,
} from "../src/wire";

import type { ObservedStatus } from "../src/observations/job-status";

export function admissionContracts(raw: DashboardMessage) {
  // @ts-expect-error Raw payloads lack the private update admission receipt.
  handleDashboardMessage(raw);
  // @ts-expect-error A caller cannot manufacture a decoded update.
  new DashboardUpdate(raw);
  // @ts-expect-error A terminal status cannot inhabit the active variant.
  const active: ObservedStatus = { kind: "active", status: "completed" };
  void active;
  // @ts-expect-error Matching public getters cannot manufacture the private decoded payload.
  const fabricated: DashboardUpdate = {
    server: DashboardServer.prepare("server.test", "ws:"),
    message: raw,
  };
  void fabricated;
  // @ts-expect-error An asynchronous callback would escape the owning connection fiber.
  const escaped: MessageHandler = async () => {};
  void escaped;
  // @ts-expect-error Fleet effects require a prepared connection plan.
  new ConnectionPlan([]);
}
