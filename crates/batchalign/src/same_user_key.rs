//! The same-user key: how a server tells its own account's requests from
//! anyone else's, without the key ever leaving that account's processes.
//!
//! A paths-mode job names files on the server's own disk, and the server has
//! no authentication, so by default it admits paths only inside the folders
//! its configuration lists (`paths_mode_roots`). The CLI and the desktop app
//! of the account that runs the server are that account already: they may
//! name any path that account can, exactly as a direct run does.
//!
//! Each time a server starts it makes a random key ([`ServerKey`]) and
//! publishes it inside its handshake record, which is owner-only and is
//! written and retired as one file under the slot lock: a key never outlives,
//! precedes or belongs to a different server than the handshake beside it.
//!
//! The key is never sent. A client of the same account reads it from the
//! handshake ([`OwnServer::read`]), asks the server for a single-use nonce,
//! and sends a proof: a keyed hash, under the key, of the nonce, the method,
//! the path and the body ([`OwnServer::sign`]). The server checks the proof
//! in constant time, from a loopback peer only, consuming the nonce. Anything
//! listening in the middle (another account on the same port of another
//! loopback address, say) learns nothing it can reuse: a proof is good for
//! that one request, once, to the server that issued its nonce.
//!
//! The desktop app never holds the key either: its native side asks the CLI
//! to sign each request (`serve sign-request`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::server_handshake::{HandshakeError, HandshakeSlot, ServerHandshake};

/// The request header that carries a proof: `<nonce>.<proof>`, both
/// hexadecimal.
pub const HEADER: &str = "x-batchalign-same-user";

/// How long an issued nonce stays usable.
const NONCE_LIFETIME: Duration = Duration::from_secs(60);

/// At most this many nonces are outstanding; issuing more forgets the expired
/// ones first, then refuses. Bounds what an anonymous client can make the
/// server remember.
const MAX_OUTSTANDING_NONCES: usize = 4096;

/// A server's own key: 32 random bytes. Only a starting server makes one;
/// only the server checks proofs with it. Neither `Clone`, `PartialEq` nor
/// `Debug` shows it.
pub struct ServerKey([u8; 32]);

impl std::fmt::Debug for ServerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ServerKey(..)")
    }
}

impl ServerKey {
    /// A fresh key for a server that is starting.
    pub(crate) fn generate() -> Self {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        bytes[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        Self(bytes)
    }

    /// The key as the handshake record stores it.
    pub(crate) fn record_value(&self) -> String {
        hex(&self.0)
    }

    /// Who sent a request, from its [`HEADER`] value (if any), the peer it
    /// came from, and what it asks for. A proof is checked against this key
    /// and a nonce this server issued and has not seen used.
    pub(crate) fn requester(
        &self,
        presented: Option<&str>,
        peer: PeerLocality,
        request: &RequestBinding<'_>,
        nonces: &NonceStore,
    ) -> Result<Requester, KeyRefusal> {
        let Some(value) = presented else {
            return Ok(Requester::Unidentified);
        };
        let (nonce, proof) = value.split_once('.').ok_or(KeyRefusal::Malformed)?;
        let proof = blake3::Hash::from_hex(proof).map_err(|_| KeyRefusal::Malformed)?;
        if !nonces.take(nonce) {
            return Err(KeyRefusal::StaleNonce);
        }
        // `blake3::Hash` compares in constant time.
        if proof != request_proof(&self.0, nonce, request) {
            return Err(KeyRefusal::Foreign);
        }
        match peer {
            PeerLocality::Loopback => Ok(Requester::SameUser),
            PeerLocality::Remote => Err(KeyRefusal::NotLocal),
        }
    }
}

/// Where a request came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerLocality {
    /// This machine, over a loopback address.
    Loopback,
    /// Anywhere else.
    Remote,
}

impl PeerLocality {
    /// The locality of a connection's peer.
    pub(crate) fn of(peer: std::net::SocketAddr) -> Self {
        match peer.ip().to_canonical().is_loopback() {
            true => Self::Loopback,
            false => Self::Remote,
        }
    }
}

/// What a signed request asks for: the proof covers all of it, so a proof
/// cannot be moved to another method, path or body.
#[derive(Clone, Copy)]
pub struct RequestBinding<'a> {
    /// The HTTP method, upper case.
    pub method: &'a str,
    /// The request path, without query.
    pub path: &'a str,
    /// The exact body bytes.
    pub body: &'a [u8],
}

/// The proof for one request: a keyed hash of the nonce, method, path and
/// body hash, each separated by a zero byte.
fn request_proof(key: &[u8; 32], nonce: &str, request: &RequestBinding<'_>) -> blake3::Hash {
    let body = blake3::hash(request.body);
    let mut hasher = blake3::Hasher::new_keyed(key);
    for part in [
        nonce.as_bytes(),
        request.method.as_bytes(),
        request.path.as_bytes(),
    ] {
        hasher.update(part);
        hasher.update(&[0]);
    }
    hasher.update(body.as_bytes());
    hasher.finalize()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}

/// Nonces this server issued and has not seen used: each is good for one
/// proof, for [`NONCE_LIFETIME`].
#[derive(Default)]
pub(crate) struct NonceStore(Mutex<HashMap<String, Instant>>);

/// No nonce can be issued now: too many are outstanding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("too many same-user nonces are outstanding; retry shortly")]
pub struct NoncesExhausted;

impl NonceStore {
    /// Issue a fresh nonce.
    pub(crate) fn issue(&self) -> Result<String, NoncesExhausted> {
        let now = Instant::now();
        let mut issued = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if issued.len() >= MAX_OUTSTANDING_NONCES {
            issued.retain(|_, expires| *expires > now);
        }
        if issued.len() >= MAX_OUTSTANDING_NONCES {
            return Err(NoncesExhausted);
        }
        let nonce = uuid::Uuid::new_v4().simple().to_string();
        issued.insert(nonce.clone(), now + NONCE_LIFETIME);
        Ok(nonce)
    }

    /// Use up `nonce`: true only for one this store issued, unexpired, the
    /// first time it is presented.
    fn take(&self, nonce: &str) -> bool {
        let mut issued = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        issued
            .remove(nonce)
            .is_some_and(|expires| expires > Instant::now())
    }
}

/// This account's own server: its address and key, from its owner-only
/// handshake record in this account's state folder. Made only by
/// [`OwnServer::read`]; signs requests to that server and no other.
pub struct OwnServer {
    address: std::net::SocketAddr,
    key: [u8; 32],
}

impl std::fmt::Debug for OwnServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnServer")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

/// Why this account's own server could not be read.
#[derive(Debug, thiserror::Error)]
pub enum OwnServerError {
    /// The handshake could not be read, or is not this account's alone.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// The handshake's key is not a key.
    #[error("the same-user key in {0} is not a key; it is not used")]
    MalformedKey(std::path::PathBuf),
}

impl OwnServer {
    /// The server of `slot` this account published in `state_dir`; `None`
    /// when there is none, or its record is from a build without keys.
    pub fn read(state_dir: &Path, slot: HandshakeSlot) -> Result<Option<Self>, OwnServerError> {
        let Some(record) = ServerHandshake::own_server(state_dir, slot)? else {
            return Ok(None);
        };
        let key = parse_key(&record.key).ok_or_else(|| {
            OwnServerError::MalformedKey(ServerHandshake::path_in(state_dir, slot))
        })?;
        Ok(Some(Self {
            address: record.address,
            key,
        }))
    }

    /// The base URL of this server, from its own record.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Whether `url` is a loopback URL for this server's port: a URL the
    /// user typed that names this account's own server. Requests are then
    /// sent to [`Self::url`], never to the typed host.
    pub fn is_named_by(&self, url: &str) -> bool {
        let Ok(parsed) = reqwest::Url::parse(url) else {
            return false;
        };
        let loopback = parsed.host_str().is_some_and(|host| {
            let host = host.trim_start_matches('[').trim_end_matches(']');
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.to_canonical().is_loopback())
        });
        loopback && parsed.port() == Some(self.address.port())
    }

    /// The [`HEADER`] value for one request, with a nonce this server
    /// issued.
    pub fn sign(&self, nonce: &str, request: &RequestBinding<'_>) -> String {
        format!(
            "{nonce}.{}",
            request_proof(&self.key, nonce, request).to_hex()
        )
    }
}

/// A 64-character lowercase hexadecimal key.
fn parse_key(text: &str) -> Option<[u8; 32]> {
    let text = text.trim();
    let lower_hex = text
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if text.len() != 64 || !lower_hex {
        return None;
    }
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(index * 2..index * 2 + 2)?, 16).ok()?;
    }
    Some(key)
}

/// What a request's proof says about who sent it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Requester {
    /// A valid proof, from this machine: this server's own account.
    SameUser,
    /// No proof: anyone who can reach the server.
    Unidentified,
}

/// A presented proof that grants nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KeyRefusal {
    /// Not a proof under this server's key.
    #[error(
        "this server was started by another account, or from another state folder, on this \
         machine: its same-user key does not match yours. Use your own server (`batchalign3 \
         serve start`), or ask the owner of this one to list your folders under paths_mode_roots"
    )]
    Foreign,
    /// A valid proof from another machine.
    #[error("same-user requests are accepted only from this machine")]
    NotLocal,
    /// The nonce was not issued here, has been used, or has expired.
    #[error("the same-user nonce is unknown, used or expired; request a new one and retry")]
    StaleNonce,
    /// The header is not `<nonce>.<proof>`.
    #[error("the same-user header is not a nonce and a proof")]
    Malformed,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SURVIVES: boundary. The proof is the trust boundary between accounts
    /// on one machine: its key is published only in this account's own
    /// handshake, and a proof is good for one request once, from this
    /// machine only.
    #[test]
    fn a_proof_identifies_one_request_of_this_account_once()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let key = ServerKey::generate();
        assert!(OwnServer::read(dir.path(), HandshakeSlot::Main)?.is_none());
        let _published = ServerHandshake::publish_listening(
            dir.path(),
            HandshakeSlot::Main,
            std::net::SocketAddr::from((std::net::Ipv4Addr::UNSPECIFIED, 41871)),
            &key,
        )?;
        let own = OwnServer::read(dir.path(), HandshakeSlot::Main)?.ok_or("own")?;
        // Bound everywhere: reached at loopback, on its own port only.
        assert_eq!(own.url(), "http://127.0.0.1:41871");
        assert!(own.is_named_by("http://localhost:41871/"));
        assert!(!own.is_named_by("http://127.0.0.1:41872/"));
        assert!(!own.is_named_by("http://example.org:41871/"));

        let nonces = NonceStore::default();
        let request = RequestBinding {
            method: "POST",
            path: "/jobs",
            body: b"{}",
        };
        let local = PeerLocality::Loopback;
        let signed = own.sign(&nonces.issue()?, &request);
        assert_eq!(
            key.requester(Some(&signed), local, &request, &nonces),
            Ok(Requester::SameUser)
        );
        // Once only.
        assert_eq!(
            key.requester(Some(&signed), local, &request, &nonces),
            Err(KeyRefusal::StaleNonce)
        );
        // Bound to its request.
        let other_body = RequestBinding {
            body: b"{\"x\":1}",
            ..request
        };
        let resigned = own.sign(&nonces.issue()?, &request);
        assert_eq!(
            key.requester(Some(&resigned), local, &other_body, &nonces),
            Err(KeyRefusal::Foreign)
        );
        // Only from this machine, only under this server's key.
        let remote = own.sign(&nonces.issue()?, &request);
        assert_eq!(
            key.requester(Some(&remote), PeerLocality::Remote, &request, &nonces),
            Err(KeyRefusal::NotLocal)
        );
        let foreign = own.sign(&nonces.issue()?, &request);
        assert_eq!(
            ServerKey::generate().requester(Some(&foreign), local, &request, &nonces),
            Err(KeyRefusal::Foreign)
        );
        assert_eq!(
            key.requester(None, local, &request, &nonces),
            Ok(Requester::Unidentified)
        );
        assert_eq!(
            key.requester(Some("nonsense"), local, &request, &nonces),
            Err(KeyRefusal::Malformed)
        );
        Ok(())
    }
}
