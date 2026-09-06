//! Comparison preparation belongs to the alignment input, not each DP cell.

pub(super) trait Alignable {
    type Policy: Copy;
    fn matches(&self, other: &Self, policy: Self::Policy) -> bool;
    fn to_key(&self) -> String;
}

/// Unprepared strings and characters support only allocation-free comparison.
#[derive(Clone, Copy)]
pub(super) enum LiteralMatchMode {
    Exact,
    AsciiInsensitive,
}

impl Alignable for String {
    type Policy = LiteralMatchMode;

    fn matches(&self, other: &Self, policy: Self::Policy) -> bool {
        match policy {
            LiteralMatchMode::Exact => self == other,
            LiteralMatchMode::AsciiInsensitive => self.eq_ignore_ascii_case(other),
        }
    }

    fn to_key(&self) -> String {
        self.clone()
    }
}

impl Alignable for char {
    type Policy = LiteralMatchMode;

    fn matches(&self, other: &Self, policy: Self::Policy) -> bool {
        match policy {
            LiteralMatchMode::Exact => self == other,
            LiteralMatchMode::AsciiInsensitive => self.eq_ignore_ascii_case(other),
        }
    }

    fn to_key(&self) -> String {
        self.to_string()
    }
}

/// A word whose Unicode lowercase form was computed once on admission.
/// Original spelling remains the owner of result keys and the ASCII fast path.
pub(super) struct PreparedFuzzyWord<'source> {
    original: &'source str,
    lowercase: String,
}

impl<'source> PreparedFuzzyWord<'source> {
    pub(super) fn new(original: &'source str) -> Self {
        Self {
            original,
            lowercase: original.to_lowercase(),
        }
    }
}

/// Preserve the public MatchMode threshold's existing comparison semantics.
/// Range validation remains the caller's policy (for example UTR's typed
/// threshold); this preparation does not reinterpret legacy float inputs.
#[derive(Clone, Copy)]
pub(super) struct FuzzyComparison {
    threshold: f64,
}

impl FuzzyComparison {
    pub(super) fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Alignable for PreparedFuzzyWord<'_> {
    type Policy = FuzzyComparison;

    fn matches(&self, other: &Self, policy: Self::Policy) -> bool {
        self.original.eq_ignore_ascii_case(other.original)
            || strsim::jaro_winkler(&self.lowercase, &other.lowercase) >= policy.threshold
    }

    fn to_key(&self) -> String {
        self.original.to_owned()
    }
}
