//! Millisecond quantities, and the difference between a LENGTH and a POSITION.
//!
//! # Why these live here and not under `chat_ops::fa`
//!
//! They were defined in `chat_ops::fa::coordinates`, which is where they are
//! used most, and that made them invisible to `media`. `media::MediaWindow`
//! therefore stored its two bounds as `DurationMs`: POSITIONS typed as
//! DURATIONS, which is the same conflation `coordinates` exists to end, one
//! layer out. `media` deliberately does not depend on `chat_ops::fa` (a media
//! primitive has no business knowing about CHAT analysis) and inverting that
//! would be worse, so the types moved UP to a module both can see rather than
//! either module reaching sideways.
//!
//! `chat_ops::fa::coordinates` re-exports all three, so FA code keeps its
//! existing spelling and there is still exactly one definition of each.
//!
//! **Why here and not `batchalign-types`, which `crates/batchalign/AGENTS.md`
//! names as the home for domain newtypes.** The argument above does not settle
//! it: `batchalign-types` would satisfy "somewhere both can see" just as well.
//! The reason that does settle it is PRIVACY.
//! [`WindowMs::offset_from_window_start`] is `pub(crate)` and is read by
//! exactly one function, `FaWindow::to_file`. Moving these into another crate
//! forces it `pub`, and a public bare-`u64` reader on `WindowMs` destroys the
//! containment property the type exists for. That argument covers `WindowMs`
//! and only `WindowMs`; if [`Ms`] and [`FileMs`] ever move down to
//! `batchalign-types`, this one stays.
//!
//! # The distinction the types carry
//!
//! * [`Ms`] is a LENGTH. It has no origin, so it cannot be confused with a
//!   position in either space, and a zero is meaningful ("no time passed").
//! * [`FileMs`] is a POSITION measured from the start of the recording. This is
//!   the space CHAT bullets are written in.
//! * [`WindowMs`] is a POSITION measured from the start of an FA window, which
//!   is what an alignment engine reports and is meaningless without the window.
//!
//! The conversion between the two position spaces is
//! `chat_ops::fa::coordinates::FaWindow::to_file`, and it is the only one, so
//! the containment question gets asked exactly where it cannot be forgotten.
//!
//! # Not yet merged with `DurationMs`, and why
//!
//! `batchalign_types::DurationMs` is a fourth millisecond newtype whose own
//! docstring used to read "Duration OR audio position", which is the conflation
//! stated outright. It is now duration-only by documentation, but [`Ms`] and it
//! are still two spellings of one concept, and merging them is blocked on four
//! things, none of which is effort:
//!
//! * `numeric_id!`, which declares `DurationMs`, is NOT EXPORTED. It is
//!   `#[macro_use] mod macros` inside `batchalign-types` with no
//!   `#[macro_export]`, so this crate cannot invoke it at all. That alone
//!   settles why these three are hand-written.
//! * It generates `pub` inner fields plus `Deref`, `From` both ways and
//!   `PartialEq<u64>`. Each is an unconditional route back to a bare integer,
//!   and a public field would make `WindowMs(4_000)` constructible by anyone,
//!   which is exactly what [`WindowMs::offset_from_window_start`] being
//!   `pub(crate)` exists to prevent.
//! * It derives `PartialOrd` but never `Ord`, even under `[Eq]`. `Ord` is
//!   required by `coordinates::FaWindow::within` (`start.cmp(&end)`) and by
//!   `timing.rs`'s `worst_overshoot.max(exceeds_by)`.
//! * `Display` prints a bare number where [`Ms`] prints `500ms` and
//!   [`WindowMs`] prints `+500ms`, so a merge either strips the unit from every
//!   FA diagnostic or changes Display across the whole worker-IPC surface, and
//!   `DurationMs` is carried by `worker_v2` types mirroring the Python side.
//!
//! An earlier version of this paragraph said "no `Ord` use on `Ms`". That was
//! FALSE (`timing.rs` uses `Ms::max`) and false in the dangerous direction: it
//! made the merge read as cheaper than it is.
//!
//! **So if the merge ever happens, it goes `DurationMs` becomes [`Ms`], never
//! the reverse.** Merging the other way would hand every FA position a
//! `Default`, a `Deref` escape to `u64`, an unnamed `From<u64>` constructor and
//! `pos == 5000`, which are four of the affordances these types were written to
//! deny. Widening `numeric_id!` with opt-outs is the other route.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A length of time. Deliberately not a point: a duration has no origin, so it
/// cannot be confused with a position in either coordinate space.
///
/// `Default` is `Ms(0)`, which is safe here in a way it is not for the position
/// types: a zero DURATION is a real, meaningful quantity ("no time passed"),
/// whereas a zero POSITION is a real instant masquerading as "unset". That
/// asymmetry is why `FileMs` and `WindowMs` deliberately have no `Default`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Ms(pub u64);

impl fmt::Display for Ms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms", self.0)
    }
}

/// A point in time measured from the start of the RECORDING.
///
/// This is the coordinate space CHAT bullets are written in. Producing one from
/// an engine's report is a transition that can fail; see
/// [`crate::chat_ops::fa::coordinates::FaWindow::to_file`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileMs(u64);

impl FileMs {
    /// A point in file coordinates.
    ///
    /// Public because timings READ from an existing transcript are already in
    /// this space and need no transition. Points DERIVED from an engine report
    /// must come through
    /// [`crate::chat_ops::fa::coordinates::FaWindow::to_file`] instead.
    pub const fn new(ms: u64) -> Self {
        Self(ms)
    }

    /// The underlying millisecond count, for serialization into a bullet.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// How far this point lies after `earlier`, saturating at zero.
    pub const fn since(self, earlier: Self) -> Ms {
        Ms(self.0.saturating_sub(earlier.0))
    }
}

impl fmt::Display for FileMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms", self.0)
    }
}

/// A point in time measured from the start of an FA WINDOW.
///
/// This is what an alignment engine reports, and it is meaningless without the
/// window it was measured against. It has no accessor returning a bare `u64`
/// for writing: the only thing a caller can do with one is hand it to the
/// window that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WindowMs(u64);

impl WindowMs {
    /// A time as reported by an alignment engine, relative to the window it was
    /// given.
    pub const fn reported(ms: u64) -> Self {
        Self(ms)
    }

    /// The raw offset from the window's start.
    ///
    /// `pub(crate)` and named, rather than a public field or a `get()`. The
    /// only legitimate reader is the single conversion into file coordinates,
    /// `chat_ops::fa::coordinates::FaWindow::to_file`, which needs the number
    /// to add the window's offset to it. Anything else holding a `WindowMs`
    /// still cannot turn it into a bare `u64`, which is what stops an
    /// engine-relative time being written into a transcript as though it were
    /// measured from the recording.
    pub(crate) const fn offset_from_window_start(self) -> u64 {
        self.0
    }
}

impl fmt::Display for WindowMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+{}ms", self.0)
    }
}
