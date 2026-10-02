//! The server's one source of the current time.
//!
//! The job store owns a [`Clock`], and every time the store and the runner
//! record (a file starting or finishing, a lease taken or renewed, a retry
//! deadline, a job's completion, the retention cutoff) is read from it. In
//! production it is [`SystemClock`]. Tests give the store a clock they
//! control, so lease expiry, retry deadlines and retention can be checked at
//! exact instants instead of against wall time.

use std::fmt::Debug;

use crate::api::MachineTime;

/// A source of the current time.
pub trait Clock: Send + Sync + Debug {
    /// Now, to the millisecond.
    fn now(&self) -> MachineTime;
}

/// The real clock.
#[derive(Debug, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> MachineTime {
        MachineTime::now()
    }
}

/// A clock a test sets and advances by hand.
#[derive(Debug)]
pub struct ManualClock {
    now: std::sync::Mutex<MachineTime>,
}

impl ManualClock {
    /// A clock reading `now` until it is moved.
    pub fn at(now: MachineTime) -> Self {
        Self {
            now: std::sync::Mutex::new(now),
        }
    }

    /// Move the clock forward by `by`.
    pub fn advance(&self, by: std::time::Duration) {
        let mut now = self.reading();
        *now = now.plus(by);
    }

    /// The reading, also after a panic elsewhere poisoned the lock: a time
    /// cannot be left half-written, so the value is still whole.
    fn reading(&self) -> std::sync::MutexGuard<'_, MachineTime> {
        match self.now.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl Clock for ManualClock {
    fn now(&self) -> MachineTime {
        *self.reading()
    }
}
