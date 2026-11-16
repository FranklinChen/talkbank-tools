/** A shared event-driven WebSocket double at Effect's real constructor seam. */
import * as Effect from "effect/Effect";
import * as Queue from "effect/Queue";
import { Socket } from "effect/socket";

type EventName = "open" | "message" | "error" | "close";
type Listener = (event: Socket.WebSocketEvent) => void;
type Phase = "connecting" | "opened" | "closed";

class WireSocket implements Socket.WebSocketLike {
  private phase: Phase = "connecting";
  private readonly listeners = new Map<
    EventName,
    Map<Listener, { readonly once: boolean }>
  >();
  readonly sent: string[] = [];
  get readyState() {
    return { connecting: 0, opened: 1, closed: 3 }[this.phase];
  }
  get events() {
    return [...this.listeners]
      .filter(([, listeners]) => listeners.size > 0)
      .map(([event]) => event);
  }
  addEventListener(
    type: EventName,
    listener: Listener,
    options?: { readonly once?: boolean },
  ) {
    const listeners = this.listeners.get(type) ?? new Map();
    listeners.set(listener, { once: options?.once === true });
    this.listeners.set(type, listeners);
  }
  removeEventListener(type: EventName, listener: Listener) {
    this.listeners.get(type)?.delete(listener);
  }
  emit(type: EventName, event: Socket.WebSocketEvent) {
    for (const [listener, options] of this.listeners.get(type) ?? []) {
      if (options.once) this.removeEventListener(type, listener);
      listener(event);
    }
  }
  open() {
    this.phase = "opened";
    this.emit("open", {});
  }
  close(code = 1000, reason = "scope closed") {
    this.phase = "closed";
    this.emit("close", { code, reason });
  }
  send(data: string | Uint8Array<ArrayBuffer>) {
    this.sent.push(
      typeof data === "string" ? data : new TextDecoder().decode(data),
    );
  }
}

/** Only the open transition provides a peer that can deliver message bytes. */
class ConnectingPeer {
  private constructor(private readonly socket: WireSocket) {}
  static make() {
    const socket = new WireSocket();
    const transport: Socket.WebSocketLike = socket;
    return { transport, peer: new ConnectingPeer(socket) };
  }
  open(): OpenPeer {
    this.socket.open();
    return OpenPeer.fromOpened(this.socket);
  }
  observeClosed(): ClosedPeer {
    return ClosedPeer.admit(this.socket);
  }
  get state() {
    return this.socket.readyState;
  }
  get events() {
    return this.socket.events;
  }
}

/** Payload delivery and network closure are capabilities of an opened peer. */
class OpenPeer {
  private constructor(private readonly socket: WireSocket) {}
  static fromOpened(socket: WireSocket) {
    return new OpenPeer(socket);
  }
  receive(text: string) {
    this.socket.emit("message", { data: text });
  }
  disconnect() {
    this.socket.close(1006, "connection lost");
  }
  get sent(): readonly string[] {
    return this.socket.sent;
  }
  get state() {
    return this.socket.readyState;
  }
  get events() {
    return this.socket.events;
  }
  observeClosed(): ClosedPeer {
    return ClosedPeer.admit(this.socket);
  }
}

/** A queued browser event can arrive after the transport's close observation. */
class ClosedPeer {
  private constructor(private readonly socket: WireSocket) {}
  static admit(socket: WireSocket): ClosedPeer {
    if (socket.readyState !== 3)
      throw new Error("The transport has not closed.");
    return new ClosedPeer(socket);
  }
  deliverQueuedOpen() {
    this.socket.emit("open", {});
  }
  deliverQueuedFrame(text: string) {
    this.socket.emit("message", { data: text });
  }
}

/** Every acquired transport has an explicit peer readiness observation. */
export class ScriptedSockets {
  private constructor(private readonly pending: Queue.Queue<ConnectingPeer>) {}
  private acquired = 0;
  get attempts(): number { return this.acquired; }
  static make = Effect.map(
    Queue.unbounded<ConnectingPeer>(),
    (pending) => new ScriptedSockets(pending),
  );
  readonly connect = () => {
    const { transport, peer } = ConnectingPeer.make();
    this.acquired += 1;
    Queue.offerUnsafe(this.pending, peer);
    return transport;
  };
  get next() {
    return Queue.take(this.pending);
  }
}
