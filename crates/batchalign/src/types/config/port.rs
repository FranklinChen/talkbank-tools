//! The port a server is ASKED to listen on.
//!
//! This exists to keep a request apart from an observation. `PortRequest` is
//! what an operator wrote in `server.yaml`; the port the server ends up
//! listening on is [`BoundPort`](crate::server_handshake::BoundPort), and the
//! only way to obtain one is from a listener that has already bound.
//!
//! Before this type, both were `u16`, so any `format!("http://127.0.0.1:{}",
//! cfg.port)` silently claimed the request had been granted. That assumption
//! holds for a fixed port and fails for an ephemeral one, and nothing in the
//! type said which case a given `u16` was.
//!
//! It also reclaims `0`. TCP spends `0` on "let the OS choose a free port",
//! which is the standing way to ask for an ephemeral bind; `ServerConfig::validate`
//! used to rewrite it to 8000 with a warning, so the one value that could
//! express the request was the one value that could not survive being written
//! down.

use std::num::NonZeroU16;

use serde::{Deserialize, Serialize};

/// Default HTTP port when `server.yaml` does not name one.
const DEFAULT_PORT: u16 = 8000;

/// The port a server is asked to listen on.
///
/// Serializes as a bare integer so `server.yaml` keeps its existing shape:
/// `0` is [`Ephemeral`](Self::Ephemeral), anything else is
/// [`Fixed`](Self::Fixed).
/// The wire form is the bare integer `server.yaml` has always carried, via
/// `from`/`into` rather than hand-written impls: `from_u16` already IS the
/// conversion, and it is total, so there is nothing a manual `Deserialize`
/// could reject that this does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "u16", into = "u16")]
pub enum PortRequest {
    /// Bind exactly this port. Failing to get it is a startup error, because
    /// the operator named a port other things are expecting to reach.
    Fixed(NonZeroU16),
    /// Let the OS choose any free port.
    ///
    /// There is no number to hand around in advance, which is the point: a
    /// caller learns the port from the server's published handshake, after the
    /// bind, rather than predicting it beforehand.
    Ephemeral,
}

impl PortRequest {
    /// Interpret an integer as a port request: `0` asks the OS to choose.
    ///
    /// Total by construction: every `u16` names a request, so there is no
    /// rejected value to report and no default to silently substitute. This is
    /// deliberately public and easy to reach; the type that must not be
    /// fabricated from a loose integer is
    /// [`BoundPort`](crate::server_handshake::BoundPort), not this one.
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        match NonZeroU16::new(value) {
            Some(port) => Self::Fixed(port),
            None => Self::Ephemeral,
        }
    }

    /// The value to hand to `TcpListener::bind`, and the wire form.
    ///
    /// `0` for [`Ephemeral`](Self::Ephemeral) is not a sentinel here: it is
    /// what the sockets API itself defines as "choose one for me".
    #[must_use]
    pub const fn bind_value(self) -> u16 {
        match self {
            Self::Fixed(port) => port.get(),
            Self::Ephemeral => 0,
        }
    }

    /// The requested port when one was named.
    ///
    /// Returns `None` for [`Ephemeral`](Self::Ephemeral), where no port is
    /// knowable until the server has bound. Callers that want to build a URL
    /// want the published [`BoundPort`](crate::server_handshake::BoundPort)
    /// instead; this is for reporting what was ASKED for.
    #[must_use]
    pub const fn fixed(self) -> Option<NonZeroU16> {
        match self {
            Self::Fixed(port) => Some(port),
            Self::Ephemeral => None,
        }
    }

    /// Operator-facing rendering: the port number, or the word `ephemeral`.
    ///
    /// There is deliberately no `Display`: an operator reading a startup line
    /// wants the word, a child's `--port` argument wants the number, and one
    /// `Display` would make `.to_string()` the natural call for both while only
    /// one of them is right. Use [`bind_value`](Self::bind_value) for anything
    /// a machine reads.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Fixed(port) => port.to_string(),
            Self::Ephemeral => "ephemeral".to_string(),
        }
    }
}

impl Default for PortRequest {
    /// The documented default port, 8000.
    fn default() -> Self {
        Self::from_u16(DEFAULT_PORT)
    }
}

impl From<PortRequest> for u16 {
    fn from(request: PortRequest) -> Self {
        request.bind_value()
    }
}

impl From<u16> for PortRequest {
    /// Infallible on purpose: every `u16` names a request.
    fn from(value: u16) -> Self {
        Self::from_u16(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0` is the ephemeral request, not an error and not 8000.
    ///
    /// The behaviour this replaces warned and substituted 8000, so a test
    /// server asking for "any free port" silently contended for the one port
    /// most likely to be in use on a developer machine.
    #[test]
    fn zero_is_the_ephemeral_request() {
        assert_eq!(PortRequest::from_u16(0), PortRequest::Ephemeral);
        assert_eq!(PortRequest::Ephemeral.bind_value(), 0);
        assert_eq!(PortRequest::Ephemeral.fixed(), None);
    }

    /// A named port survives the round trip through the wire form unchanged.
    #[test]
    fn a_named_port_round_trips() {
        let request = PortRequest::from_u16(9123);
        assert_eq!(request.fixed().map(NonZeroU16::get), Some(9123));
        assert_eq!(request.bind_value(), 9123);
    }

    /// The absent-key default stays the documented 8000.
    #[test]
    fn the_default_is_the_documented_port() {
        assert_eq!(
            PortRequest::default().fixed().map(NonZeroU16::get),
            Some(8000)
        );
    }
}
