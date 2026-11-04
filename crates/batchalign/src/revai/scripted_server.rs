//! A loopback stand-in for Rev.AI that answers a scripted sequence of replies.
//!
//! Test only. The Rev.AI client is a real HTTP client, so the right place to
//! test its wait is at the socket: a real request goes out, a real reply (or a
//! real hang-up) comes back. This server binds the loopback address, which the
//! macOS application firewall lets a test binary receive on (a LAN address it
//! does not: see `.trail/learnings/2026-10-30-a-test-binary-cannot-receive-lan-franklin.md`),
//! and answers each connection with the next reply of its script, repeating the
//! last reply once the script is spent so a "forever in progress" job needs no
//! infinite script.
//!
//! The protocol handling is the minimum a blocking reqwest client needs: read
//! the request head, read `Content-Length` bytes of body, answer with
//! `Connection: close` so the client opens a new connection per request and
//! every request reaches the script in order.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// One scripted answer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScriptedReply {
    /// HTTP 200 with this JSON body.
    Json(&'static str),
    /// This HTTP status with a short plain body.
    Status(u16),
    /// Accept the connection, read the request, then drop the socket without
    /// answering: the transport fault the fleet met on 2026-09-03.
    HangUp,
}

/// A request the server saw: its method and path, as the request line spelled
/// them, so a test can assert what the client did (and did not) ask for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeenRequest {
    pub(crate) method: String,
    pub(crate) path: String,
}

/// The running server: where to point the client, and what it has been asked.
pub(crate) struct ScriptedRevServer {
    /// The speech-to-text root to build a test client on.
    pub(crate) speech_to_text_base: String,
    seen: Arc<Mutex<Vec<SeenRequest>>>,
}

impl ScriptedRevServer {
    /// Start a server that answers `script` in order, then repeats its last
    /// reply. The script must not be empty: a server with nothing to say is a
    /// test with no subject.
    pub(crate) fn start(script: Vec<ScriptedReply>) -> Self {
        assert!(!script.is_empty(), "test: a scripted server needs a script");
        let listener = TcpListener::bind("127.0.0.1:0").expect("test: bind loopback");
        let addr = listener.local_addr().expect("test: local addr");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        // Detached: it ends with the process. The listener lives in the thread,
        // and the loop ends only when accepting fails.
        std::thread::spawn(move || {
            let mut script = script.into_iter();
            let mut last = None;
            for stream in listener.incoming() {
                let Ok(stream) = stream else { return };
                let reply = match script.next() {
                    Some(reply) => {
                        last = Some(reply);
                        reply
                    }
                    None => last.expect("test: the script is non-empty"),
                };
                answer(stream, reply, &recorded);
            }
        });
        Self {
            speech_to_text_base: format!("http://{addr}/speechtotext/v1"),
            seen,
        }
    }

    /// Every request seen so far, in order.
    pub(crate) fn seen(&self) -> Vec<SeenRequest> {
        self.seen.lock().expect("test: seen requests").clone()
    }
}

/// Serve one connection: read the whole request, record it, reply per script.
fn answer(mut stream: TcpStream, reply: ScriptedReply, seen: &Mutex<Vec<SeenRequest>>) {
    let Some(request) = read_request(&mut stream) else {
        return;
    };
    seen.lock().expect("test: seen requests").push(request);
    let response = match reply {
        ScriptedReply::Json(body) => Some(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )),
        ScriptedReply::Status(status) => {
            let body = format!("scripted status {status}");
            Some(format!(
                "HTTP/1.1 {status} Scripted\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            ))
        }
        ScriptedReply::HangUp => None,
    };
    if let Some(response) = response {
        // A client that has already gone away is not this test's concern.
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    }
}

/// Read one HTTP/1.1 request: the head up to the blank line, then
/// `Content-Length` bytes of body. `None` if the client sent nothing usable.
fn read_request(stream: &mut TcpStream) -> Option<SeenRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split(' ');
    let method = parts.next()?.to_owned();
    let path = parts.next()?.to_owned();
    let content_length: usize = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .unwrap_or(0);
    let mut body_read = buffer.len() - head_end;
    while body_read < content_length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        body_read += read;
    }
    Some(SeenRequest { method, path })
}
