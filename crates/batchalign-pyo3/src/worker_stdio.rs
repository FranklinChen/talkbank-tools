//! Interruptible delivery from the worker process's stdin.
//!
//! The OS read cannot be cancelled portably. A single native thread owns it,
//! while Python waits on a bounded mailbox that terminal failures can close.
//! The reader holds no Python objects or interpreter locks, and is deliberately
//! not joined on shutdown: an open parent pipe must not prevent process exit.

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// Stdin is process-global. This lease prevents competing native readers, even
// when a stopped reader is still blocked in the OS until process termination.
static STDIN_CLAIMED: AtomicBool = AtomicBool::new(false);

struct StdinClaim;

impl StdinClaim {
    fn acquire() -> PyResult<Self> {
        STDIN_CLAIMED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Self)
            .map_err(|_| PyRuntimeError::new_err("protocol stdin already has a reader"))
    }
}

impl Drop for StdinClaim {
    fn drop(&mut self) {
        STDIN_CLAIMED.store(false, Ordering::Release);
    }
}

enum InputState {
    Waiting,
    Line(String),
    Eof,
    ReadError(io::Error),
    Stopped,
}

struct Mailbox {
    state: Mutex<InputState>,
    changed: Condvar,
}

impl Mailbox {
    fn lock(&self) -> MutexGuard<'_, InputState> {
        // No user code executes under this lock. On poisoning, preserve the
        // state rather than introducing a second panic during worker teardown.
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn wait<'a>(&self, state: MutexGuard<'a, InputState>) -> MutexGuard<'a, InputState> {
        self.changed
            .wait(state)
            .unwrap_or_else(|error| error.into_inner())
    }

    fn publish(&self, next: InputState) -> bool {
        let mut state = self.lock();
        while matches!(*state, InputState::Line(_)) {
            state = self.wait(state);
        }
        if !matches!(*state, InputState::Waiting) {
            return false;
        }
        *state = next;
        self.changed.notify_all();
        true
    }

    fn read_line(&self) -> io::Result<Option<String>> {
        let mut state = self.lock();
        loop {
            match std::mem::replace(&mut *state, InputState::Waiting) {
                InputState::Waiting => state = self.wait(state),
                InputState::Eof => {
                    *state = InputState::Eof;
                    return Ok(None);
                }
                InputState::Stopped => {
                    *state = InputState::Stopped;
                    return Ok(None);
                }
                InputState::Line(line) => {
                    self.changed.notify_all();
                    return Ok(Some(line));
                }
                InputState::ReadError(error) => {
                    *state = InputState::Eof;
                    return Err(error);
                }
            }
        }
    }

    fn stop(&self) {
        *self.lock() = InputState::Stopped;
        self.changed.notify_all();
    }

    fn receive_stdin(&self, _claim: StdinClaim) {
        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        loop {
            let mut line = String::new();
            let next = match stdin.read_line(&mut line) {
                Ok(0) => InputState::Eof,
                Ok(_) => InputState::Line(line),
                Err(error) => InputState::ReadError(error),
            };
            let terminal = !matches!(next, InputState::Line(_));
            if !self.publish(next) || terminal {
                return;
            }
        }
    }
}

/// Process-owned stdin delivery. Only `open_protocol_stdin` can construct it.
#[pyclass(frozen)]
pub(crate) struct ProtocolStdin {
    mailbox: Arc<Mailbox>,
}

#[pymethods]
impl ProtocolStdin {
    /// Wait without the GIL. Explicit stop and input EOF both end iteration.
    fn read_line(&self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| self.mailbox.read_line()).map_err(PyErr::from)
    }

    /// Stop delivery and wake the Python reader, including during an OS read.
    fn stop(&self) {
        self.mailbox.stop();
    }

    /// EOF is not cancellation: already queued requests may still complete.
    #[getter]
    fn stopped(&self) -> bool {
        matches!(*self.mailbox.lock(), InputState::Stopped)
    }
}

impl Drop for ProtocolStdin {
    fn drop(&mut self) {
        self.mailbox.stop();
    }
}

/// Start the sole native reader for this worker process.
#[pyfunction]
pub(crate) fn open_protocol_stdin() -> PyResult<ProtocolStdin> {
    let claim = StdinClaim::acquire()?;
    let mailbox = Arc::new(Mailbox {
        state: Mutex::new(InputState::Waiting),
        changed: Condvar::new(),
    });
    let reader = Arc::clone(&mailbox);
    std::thread::Builder::new()
        .name("worker-stdin".into())
        .spawn(move || reader.receive_stdin(claim))
        .map_err(PyErr::from)?;
    Ok(ProtocolStdin { mailbox })
}
