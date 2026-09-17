//! Processing provenance: injects `@Comment` headers recording what
//! batchalign3 did to a CHAT file, when, and with what engines.
//!
//! Format written: `[fc-ba3 <command> | key=val ; key=val | ISO-8601]`
//!
//! Each command run adds one comment. Re-running the same command
//! replaces the previous comment for that command, preserving comments
//! from other commands (e.g., morphotag comment survives align re-run).
//!
//! # Two names, one grammar
//!
//! Builds before the rename wrote the same grammar under the name `ba3`
//! (`[ba3 <command> | ...]`). A file keeps whichever name wrote it, so every
//! reader here (replacement on re-run, no-op write detection, extraction for
//! the job-detail API) recognizes both through [`StampName`], while the writer
//! can only produce [`StampName::WRITTEN`]. The unchecked-ASR warning follows
//! the same rule through [`OUR_WARNING_SHAPES`]. Neither recognizer matches the
//! lines another Batchalign distribution writes (a bare `Unchecked output of
//! ASR model`, or `batchalign3 <sha> | <stage>: ...`), so re-running over such
//! a file never deletes them. The literal pieces of both grammars are written
//! once, as macros below, and shared by the writers and the recognizers.
//!
//! # One codec
//!
//! [`StampCodec`] writes the stamp for [`ProvenanceComment::format`] and reads
//! it back for [`extract_provenance`] and the no-op write gate, from the same
//! separator constants. A field value is [`StampSafeText`]
//! ([`StampFieldValue`]), so it cannot change the grammar's structure, and
//! every conversion into one is total: the text was admitted where it was born
//! (a worker's reported name, a language code, a compile-time literal, a
//! checkpoint validated when the transcribe plan was admitted). Field keys are
//! a closed set ([`StampField`]), several engines are joined by one owner
//! ([`EngineNames`]), and a comment that opens as one of our stamps but does
//! not parse is a typed [`UnparseableStamp`], never silently skipped.
//!
//! # One recognizer, in two steps
//!
//! "Did we write this comment?" has one owner, [`recognize::RecognizedComment`],
//! and it answers in two steps because its readers need different amounts of
//! it. Step one, [`recognize::RecognizedComment::classify`], reads the opening
//! and the command: it says whether a comment is ours and, for a stamp, WHICH
//! command wrote it, without committing to the body. Step two,
//! [`recognize::NamedStamp::parse`], reads the fields and the timestamp.
//!
//! The split is what keeps the no-op write gate safe. The gate must know whose
//! stamp it is looking at BEFORE a damaged one means anything, because a
//! damaged stamp belonging to a command the job does not run is not the gate's
//! business and must not provoke a rewrite. A recognizer that only ever
//! answered "parsed" or "did not parse" would collapse "damaged" together with
//! "not ours", which is exactly the confusion that would make the gate rewrite
//! files it should leave alone. So the answer has four cases, not two:
//! a stamp whose command is named, a stamp too damaged to name one, our
//! warning, and a comment written by something else.
//!
//! # What counts as a meaningful difference
//!
//! Re-running a command writes fresh stamps, and the runner skips the write
//! when nothing else changed ([`is_provenance_only_difference`]). The stamps
//! set aside are those of every command the job's recipe composes, read from
//! the recipe catalog (`crate::command_model::commands_stamped_by`): a
//! `transcribe` run also writes `utseg` and `morphotag` stamps, and all three
//! are compared, not only its own. Two stamps for one command count as the
//! same only when they differ in the stamp NAME (`fc-ba3` or the legacy `ba3`)
//! and the TIMESTAMP. Any field difference (an engine, a language, a flag) is
//! meaningful, so the file is written and the stamp names what actually
//! produced it now. Our unchecked-ASR warning is compared the same way: only
//! its form (current or legacy) and the build identity it names may differ; a
//! different ASR engine is meaningful.
//!
//! Full spec: `book/src/batchalign/architecture/provenance.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use crate::api::{
    InvalidStampSafeText, LanguageCode3, ReleasedCommand, ReportedEngineName, StampJoiner,
    StampSafeText,
};
use crate::chat_ops::fa::utr::UtrResult;
use crate::chat_ops::{ChatFile, Header, Line};
use crate::engine_reports::FaCacheNamespace;
use crate::morphosyntax::identity::AppliedAnalyses;
use crate::transcribe::types::AsrIdentity;
use crate::types::engines::UtrEngine;
use crate::utseg::AdmittedUtsegPrediction;
use crate::utseg_evidence::UtsegEngineIdentities;
use batchalign_transform::parse::parse_strict;
use batchalign_transform::serialize::to_chat_string;

// ---------------------------------------------------------------------------
// Shared literals
// ---------------------------------------------------------------------------
//
// Macros rather than `const`s so each piece composes with `concat!` and
// format strings at compile time: the writers and the recognizers are built
// from the same tokens, so they cannot drift apart.

/// The name this build writes, in stamps and in the warning.
macro_rules! written_name {
    () => {
        "fc-ba3"
    };
}

/// The name builds before the rename wrote stamps under; read only.
macro_rules! legacy_name {
    () => {
        "ba3"
    };
}

/// The sentence both forms of our warning carry. Shared with other
/// distributions' warnings, so it is never matched on its own.
macro_rules! unchecked_sentence {
    () => {
        "Unchecked output of ASR model"
    };
}

/// What follows the build identity in the current warning.
macro_rules! current_warning_engine_marker {
    () => {
        ", ASR engine "
    };
}

/// How the warning builds before the rename wrote begins.
macro_rules! legacy_warning_product {
    () => {
        "Batchalign "
    };
}

/// What follows the version in the legacy warning.
macro_rules! legacy_warning_engine_marker {
    () => {
        ", ASR Engine "
    };
}

/// The explicit safety ending the current warning always has.
macro_rules! do_not_use_ending {
    () => {
        ", DO NOT USE."
    };
}

/// What ends a stamp's command: the section separator without its trailing
/// space. Step one of recognition finds this to name the command, and the full
/// separator is built from it below, so the two cannot drift apart.
macro_rules! command_end {
    () => {
        " |"
    };
}

// ---------------------------------------------------------------------------
// Field keys and values
// ---------------------------------------------------------------------------

/// A key a stamp field is written under.
///
/// Closed, so no caller can invent a key (or one containing `=` or a
/// separator). Ordered by the written key ([`Ord`] below compares
/// [`Self::as_str`]), so a map of these writes its fields in the alphabetical
/// order they had as strings whatever order the variants are declared in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StampField {
    Asr,
    AsrModel,
    Diarize,
    Engine,
    Fa,
    Incremental,
    Lang,
    Retokenize,
    UdRepairs,
    Utr,
    Wor,
}

impl StampField {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::AsrModel => "asr_model",
            Self::Diarize => "diarize",
            Self::Engine => "engine",
            Self::Fa => "fa",
            Self::Incremental => "incremental",
            Self::Lang => "lang",
            Self::Retokenize => "retokenize",
            Self::UdRepairs => "ud_repairs",
            Self::Utr => "utr",
            Self::Wor => "wor",
        }
    }
}

impl Ord for StampField {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for StampField {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A stamp field value: [`StampSafeText`], so it cannot change the stamp's
/// structure.
///
/// There is no fallible constructor here. Every conversion in is total,
/// because every source is already stamp-safe text: a reported engine name, a
/// language code, a compile-time literal, or a join of those ([`EngineNames`]).
/// Text that could be unsafe (a checkpoint a user selected, a boundary model
/// id a worker named) is admitted as [`StampSafeText`] where it enters, not
/// here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StampFieldValue(StampSafeText);

impl StampFieldValue {
    /// The value a flag field is written with.
    const TRUE: Self = Self(StampSafeText::from_static("true"));

    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<StampSafeText> for StampFieldValue {
    fn from(text: StampSafeText) -> Self {
        Self(text)
    }
}

impl From<&LanguageCode3> for StampFieldValue {
    fn from(code: &LanguageCode3) -> Self {
        Self(StampSafeText::from(code))
    }
}

/// A transcript's language: its code, or a pair's two codes joined by a comma
/// (`eng,spa`), in `@Languages` order. One `lang=` field either way; a second
/// `lang=` field would be a damaged stamp.
impl From<&crate::api::TranscriptLanguage> for StampFieldValue {
    fn from(language: &crate::api::TranscriptLanguage) -> Self {
        match language {
            crate::api::TranscriptLanguage::One(code) => Self::from(code),
            crate::api::TranscriptLanguage::Pair(pair) => Self(StampSafeText::from(pair)),
        }
    }
}

impl From<&ReportedEngineName> for StampFieldValue {
    fn from(name: &ReportedEngineName) -> Self {
        Self(name.as_stamp_text().clone())
    }
}

impl From<NonZeroUsize> for StampFieldValue {
    fn from(count: NonZeroUsize) -> Self {
        Self(StampSafeText::from_count(count))
    }
}

impl From<InvalidStampSafeText> for crate::error::ServerError {
    fn from(error: InvalidStampSafeText) -> Self {
        Self::Validation(error.to_string())
    }
}

/// Distinct engine names written as one `engine=` value, joined with `+` in
/// text order.
///
/// The one joiner for every command that names several engines (morphotag,
/// utseg, translate, coref), so the separator and the order are decided once.
/// Every name is already [`StampSafeText`], and so is their join, so writing
/// the value needs no check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EngineNames(BTreeSet<StampSafeText>);

impl EngineNames {
    /// The separator between names, shared by the join and by
    /// [`incremental_morphotag_provenance`], which reads a written list back.
    const JOINER: StampJoiner = StampJoiner::Plus;

    /// Whether no engine is named.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The joined value, or `None` when no engine is named.
    pub(crate) fn joined(&self) -> Option<StampSafeText> {
        let mut names = self.0.iter();
        let first = names.next()?;
        Some(StampSafeText::join(first, names, Self::JOINER))
    }
}

impl FromIterator<StampSafeText> for EngineNames {
    fn from_iter<I: IntoIterator<Item = StampSafeText>>(names: I) -> Self {
        Self(names.into_iter().collect())
    }
}

impl Extend<StampSafeText> for EngineNames {
    fn extend<I: IntoIterator<Item = StampSafeText>>(&mut self, names: I) {
        self.0.extend(names);
    }
}

impl IntoIterator for EngineNames {
    type Item = StampSafeText;
    type IntoIter = std::collections::btree_set::IntoIter<StampSafeText>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// ---------------------------------------------------------------------------
// The comment and its codec
// ---------------------------------------------------------------------------

/// Processing provenance metadata for one batchalign3 command invocation.
#[derive(Debug, Clone)]
pub struct ProvenanceComment {
    /// The command that ran.
    command: ReleasedCommand,
    /// Semantic key-value pairs (engines, options that affect output), in
    /// written order.
    fields: BTreeMap<StampField, StampFieldValue>,
}

impl ProvenanceComment {
    /// A comment for `command` with no fields yet.
    fn new(command: ReleasedCommand) -> Self {
        Self {
            command,
            fields: BTreeMap::new(),
        }
    }

    /// Add a field.
    fn field(mut self, key: StampField, value: StampFieldValue) -> Self {
        self.fields.insert(key, value);
        self
    }

    /// Add a flag field only when it is set.
    fn field_if(self, key: StampField, value: bool) -> Self {
        if value {
            self.field(key, StampFieldValue::TRUE)
        } else {
            self
        }
    }

    /// Add a field only when there is a value for it.
    ///
    /// The key's ABSENCE is then the statement that there was none, which is
    /// how a count field says "nothing happened" without giving that fact a
    /// second spelling (`ud_repairs=0`) that a reader would have to be told
    /// means the same thing. The caller's `Option` is what makes the two
    /// cases different values rather than a number to compare against zero.
    fn field_if_some(self, key: StampField, value: Option<StampFieldValue>) -> Self {
        match value {
            Some(value) => self.field(key, value),
            None => self,
        }
    }

    /// Add `engine=` naming `names`, or nothing when no engine is named (never
    /// a placeholder).
    fn engines(self, names: &EngineNames) -> Self {
        match names.joined() {
            Some(joined) => self.field(StampField::Engine, StampFieldValue::from(joined)),
            None => self,
        }
    }

    /// Format as the `[fc-ba3 ...]` comment string (without `@Comment:\t`
    /// prefix), under [`StampName::WRITTEN`], the only name this build writes.
    pub fn format(&self) -> String {
        let timestamp = chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%:z")
            .to_string();
        StampCodec::write(
            StampName::WRITTEN.opening(),
            self.command.as_str(),
            self.fields
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
            &timestamp,
        )
    }
}

/// Between the command, the fields, and the timestamp.
const SECTION_SEPARATOR: &str = concat!(command_end!(), " ");
/// What ends a stamp's command, and so begins the rest of its body.
const COMMAND_END: &str = command_end!();
/// Between two fields.
const FIELD_SEPARATOR: &str = " ; ";
/// Between a field's key and its value.
const KEY_VALUE_SEPARATOR: char = '=';
/// Closes a stamp.
const STAMP_CLOSE: char = ']';

/// The stamp body grammar, written and read from the constants above:
/// `<opening><command> | k=v ; k=v | <timestamp>]`, or
/// `<opening><command> | <timestamp>]` with no fields.
struct StampCodec;

/// What follows a stamp's command, read back and borrowing the comment text.
///
/// The command is NOT here: [`recognize::NamedStamp`] read it in step one and
/// owns it. One stamp therefore has one command held in one place, and no
/// reader can pair a command with another stamp's fields.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StampBody<'a> {
    fields: BTreeMap<&'a str, &'a str>,
    timestamp: &'a str,
}

/// What is wrong with a stamp that opens with our name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StampDefect {
    /// No closing `]`.
    #[error("has no closing `]`")]
    Unterminated,
    /// The command or timestamp section is missing or empty.
    #[error("does not have a command and a timestamp separated by ` | `")]
    MissingSections,
    /// A field is not `key=value` with a non-empty key.
    #[error("has a field {0:?} that is not key=value")]
    FieldWithoutValue(String),
    /// A key appears twice.
    #[error("repeats the field {0:?}")]
    DuplicateField(String),
}

/// A comment that opens as one of our stamps but does not parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("provenance stamp {stamp:?} {defect}")]
pub struct UnparseableStamp {
    /// The comment text, as found.
    pub stamp: String,
    /// What is wrong with it.
    pub defect: StampDefect,
}

impl StampCodec {
    /// Write one stamp.
    fn write<'a>(
        opening: &str,
        command: &str,
        fields: impl Iterator<Item = (&'a str, &'a str)>,
        timestamp: &str,
    ) -> String {
        let mut stamp = String::new();
        stamp.push_str(opening);
        stamp.push_str(command);
        stamp.push_str(SECTION_SEPARATOR);
        let mut wrote_field = false;
        for (key, value) in fields {
            if wrote_field {
                stamp.push_str(FIELD_SEPARATOR);
            }
            stamp.push_str(key);
            stamp.push(KEY_VALUE_SEPARATOR);
            stamp.push_str(value);
            wrote_field = true;
        }
        if wrote_field {
            stamp.push_str(SECTION_SEPARATOR);
        }
        stamp.push_str(timestamp);
        stamp.push(STAMP_CLOSE);
        stamp
    }

    /// The text inside a stamp: everything before the closing `]`.
    ///
    /// The one place that knows the terminator, so both steps of recognition
    /// report a missing `]` as the same defect.
    fn inner(after_opening: &str) -> Result<&str, StampDefect> {
        after_opening
            .strip_suffix(STAMP_CLOSE)
            .ok_or(StampDefect::Unterminated)
    }

    /// Read the fields and the timestamp of a stamp whose command step one has
    /// already read. `after_command` begins at the [`COMMAND_END`] that ends
    /// the command, so this sees the grammar from the first separator on.
    fn read_body(after_command: &str) -> Result<StampBody<'_>, StampDefect> {
        let rest = Self::inner(after_command)?
            .strip_prefix(SECTION_SEPARATOR)
            .ok_or(StampDefect::MissingSections)?;
        let (fields, timestamp) = match rest.split_once(SECTION_SEPARATOR) {
            Some((fields, timestamp)) => (Some(fields), timestamp),
            None => (None, rest),
        };
        if timestamp.is_empty() {
            return Err(StampDefect::MissingSections);
        }
        let mut parsed = BTreeMap::new();
        for field in fields
            .into_iter()
            .flat_map(|fields| fields.split(FIELD_SEPARATOR))
        {
            let (key, value) = field
                .split_once(KEY_VALUE_SEPARATOR)
                .filter(|(key, _)| !key.is_empty())
                .ok_or_else(|| StampDefect::FieldWithoutValue(field.to_owned()))?;
            if parsed.insert(key, value).is_some() {
                return Err(StampDefect::DuplicateField(key.to_owned()));
            }
        }
        Ok(StampBody {
            fields: parsed,
            timestamp,
        })
    }
}

// ---------------------------------------------------------------------------
// Stamp names: the one owner of "is this comment one of our stamps?"
// ---------------------------------------------------------------------------

/// A name our per-command stamps have been written under.
///
/// Closed, and private to this module: writing goes through
/// [`StampName::WRITTEN`], so no caller can produce the legacy name, while
/// every reader matches both through [`recognize::RecognizedComment::classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StampName {
    /// `[fc-ba3 ...]`, written by this build.
    FcBa3,
    /// `[ba3 ...]`, written by builds before the rename; read only.
    LegacyBa3,
}

impl StampName {
    /// The name this build writes.
    const WRITTEN: Self = Self::FcBa3;

    /// Every name a reader must recognize, in match order.
    const RECOGNIZED: [Self; 2] = [Self::FcBa3, Self::LegacyBa3];

    /// The bracket and name that open a stamp, including the separating space.
    const fn opening(self) -> &'static str {
        match self {
            Self::FcBa3 => concat!("[", written_name!(), " "),
            Self::LegacyBa3 => concat!("[", legacy_name!(), " "),
        }
    }
}

/// The one owner of "did this build write this comment, and what is it?".
///
/// [`NamedStamp`] and [`UnnamedStamp`] are minted in this module and nowhere
/// else: their fields are private to it, so the only route to one is
/// [`RecognizedComment::classify`] over real comment text. No caller can pair
/// one stamp's command with another stamp's body, because it cannot build the
/// pair at all. The two payload-free variants (our warning and `Foreign`) carry
/// no such pairing, and are not protected this way.
mod recognize {
    use super::{
        COMMAND_END, ReleasedCommand, StampBody, StampCodec, StampDefect, StampName,
        UnparseableStamp,
    };

    /// What one `@Comment` body is, decided from its opening and its command,
    /// before any stamp body is parsed.
    ///
    /// Four answers, and every reader needs them apart. "Damaged", "not ours"
    /// and "ours, but another command's" are three different facts: the no-op
    /// write gate rewrites a file over the first only when the stamp is one the
    /// job itself writes, and must leave the other two exactly where they are.
    #[derive(Debug)]
    pub(super) enum RecognizedComment<'a> {
        /// One of our stamps, recording the command it names. The fields and
        /// the timestamp are NOT read yet; [`NamedStamp::parse`] reads those.
        Stamp(NamedStamp<'a>),
        /// One of our stamps, too damaged to say which command wrote it:
        /// nothing before the first [`COMMAND_END`] names one.
        Unnamed(UnnamedStamp<'a>),
        /// Our unchecked-ASR warning, naming the ASR engine it records.
        UncheckedAsrWarning {
            /// The engine text the warning names.
            engine: &'a str,
        },
        /// Written by something else, another Batchalign distribution
        /// included: never ours to read, replace or count.
        Foreign,
    }

    impl<'a> RecognizedComment<'a> {
        /// Step one: recognize `body`, naming a stamp's command without
        /// committing to the rest of its body.
        pub(super) fn classify(body: &'a str) -> Self {
            let stamp = body.trim();
            let Some(after_opening) = StampName::RECOGNIZED
                .into_iter()
                .find_map(|name| stamp.strip_prefix(name.opening()))
            else {
                return match super::our_warning_engine(stamp) {
                    Some(engine) => Self::UncheckedAsrWarning { engine },
                    None => Self::Foreign,
                };
            };
            match after_opening.find(COMMAND_END) {
                // The command is what precedes the first COMMAND_END, and the
                // rest of the body starts AT it, so step two reads exactly the
                // grammar the writer wrote. Searching for COMMAND_END rather
                // than the whole separator is deliberate: it keeps a body like
                // `[fc-ba3 align |x]` attributed to align, so the run that
                // writes align still replaces it instead of leaving it forever.
                Some(end) if end > 0 => {
                    let (command, after_command) = after_opening.split_at(end);
                    Self::Stamp(NamedStamp {
                        stamp,
                        command,
                        after_command,
                    })
                }
                // Ours, with no command to attribute it to: either no command
                // terminator at all, or nothing before it.
                Some(_) | None => Self::Unnamed(UnnamedStamp {
                    stamp,
                    defect: match StampCodec::inner(after_opening) {
                        // Terminated, so what is missing is the command.
                        Ok(_) => StampDefect::MissingSections,
                        Err(defect) => defect,
                    },
                }),
            }
        }
    }

    /// One of our stamps whose command has been read, before its body is.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct NamedStamp<'a> {
        /// The whole comment body, so an error can quote what was read.
        stamp: &'a str,
        /// The command this stamp records, as the file spells it.
        command: &'a str,
        /// The body from the separator that ends the command onwards.
        after_command: &'a str,
    }

    impl<'a> NamedStamp<'a> {
        /// The command this stamp records, as written.
        pub(super) fn command(&self) -> &'a str {
            self.command
        }

        /// Whether this stamp records `command`. The whole command matches or
        /// nothing does, so a longer name sharing a prefix is a different
        /// stamp (`transcribe_s` is not `transcribe`).
        pub(super) fn records(&self, command: ReleasedCommand) -> bool {
            self.command == command.as_str()
        }

        /// The one of `commands` this stamp records, if any.
        pub(super) fn recorded_among(
            &self,
            commands: &[ReleasedCommand],
        ) -> Option<ReleasedCommand> {
            commands
                .iter()
                .copied()
                .find(|&command| self.records(command))
        }

        /// Step two: read the fields and the timestamp.
        pub(super) fn parse(&self) -> Result<StampBody<'a>, UnparseableStamp> {
            StampCodec::read_body(self.after_command)
                .map_err(|defect| unparseable(self.stamp, defect))
        }
    }

    /// One of our stamps that names no command.
    #[derive(Debug, Clone)]
    pub(super) struct UnnamedStamp<'a> {
        stamp: &'a str,
        defect: StampDefect,
    }

    impl UnnamedStamp<'_> {
        /// The typed error, for a reader that cannot pass it by.
        pub(super) fn into_error(self) -> UnparseableStamp {
            unparseable(self.stamp, self.defect)
        }
    }

    /// The one PRODUCTION construction of [`UnparseableStamp`], so every reader
    /// reports a damaged stamp with the same text and the same defect. The type
    /// is public with public fields, so this is the module's discipline rather
    /// than something the compiler enforces.
    fn unparseable(stamp: &str, defect: StampDefect) -> UnparseableStamp {
        UnparseableStamp {
            stamp: stamp.to_owned(),
            defect,
        }
    }
}

use recognize::RecognizedComment;

/// Whether a comment's text is one of our stamps for `command`.
fn is_stamp_for(text: &str, command: ReleasedCommand) -> bool {
    matches!(
        RecognizedComment::classify(text),
        RecognizedComment::Stamp(stamp) if stamp.records(command)
    )
}

/// The text of a `@Comment:` line after the header, trimmed; `None` for any
/// other line.
fn comment_body(line: &str) -> Option<&str> {
    line.trim_start().strip_prefix("@Comment:").map(str::trim)
}

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

/// Find the index at which a new changeable header (such as a provenance
/// `@Comment`) should be inserted: immediately after the last header that must
/// precede changeable headers, i.e. the last `@ID` or constant participant
/// header (`@Birth of` / `@Birthplace of` / `@L1 of`).
///
/// Per the CHAT format the constant participant headers must immediately follow
/// the `@ID` block. Inserting after the last `@ID` *only* (ignoring the
/// constant headers) displaces `@Birth of` and produces CLAN CHECK error 127,
/// so we scan for the last of all four header kinds.
fn insert_pos_after_constant_headers(file: &ChatFile) -> usize {
    file.lines
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, line)| {
            if let Line::Header { header, .. } = line
                && matches!(
                    header.as_ref(),
                    Header::ID(_)
                        | Header::Birth { .. }
                        | Header::Birthplace { .. }
                        | Header::L1Of { .. }
                )
            {
                return Some(i + 1);
            }
            None
        })
        .unwrap_or(0)
}

/// Inject a provenance comment into a CHAT file's AST.
///
/// Replaces any existing stamp for the same command, under either name
/// (`[fc-ba3 <command> |` or the legacy `[ba3 <command> |`). New comments are
/// placed immediately after the constant participant headers (the last `@ID` /
/// `@Birth of` / `@Birthplace of` / `@L1 of`).
pub fn inject_provenance(file: &mut ChatFile, comment: &ProvenanceComment) {
    let new_content = comment.format();

    // Remove existing provenance comment for this command.
    file.lines.retain(|line| {
        if let Line::Header { header, .. } = line
            && let Header::Comment { content } = header.as_ref()
        {
            return !is_stamp_for(&content.to_chat_string(), comment.command);
        }
        true
    });

    // Insert after the constant participant headers (not merely after the last
    // @ID), so @Birth of / @Birthplace of / @L1 of stay adjacent to @ID.
    let insert_pos = insert_pos_after_constant_headers(file);

    let bullet_content = crate::chat_ops::BulletContent::from_text(new_content);

    file.lines.insert(
        insert_pos,
        Line::header(Header::Comment {
            content: bullet_content,
        }),
    );
}

/// Inject a provenance comment into serialized CHAT text.
///
/// Parses the text, injects the comment into the AST, and re-serializes.
/// This is a convenience wrapper for pipelines that work with CHAT strings
/// rather than AST objects. Parse failure returns the original diagnostics;
/// a recovered document must not be serialized as successful output merely
/// because the caller requested a provenance comment.
pub fn inject_provenance_into_text(
    chat_text: &str,
    comment: &ProvenanceComment,
) -> Result<String, talkbank_model::ParseErrors> {
    let parser = crate::chat_parser();
    let mut file = parse_strict(&parser, chat_text)?;
    inject_provenance(&mut file, comment);
    Ok(to_chat_string(&file))
}

// ---------------------------------------------------------------------------
// Per-command builders
// ---------------------------------------------------------------------------

/// Build a provenance comment for a full morphotag run, naming the Stanza
/// models behind the responses it applied and counting the relations repaired
/// inside them (see [`AppliedAnalyses`]).
///
/// `engine=` is absent when no model analyzed anything, never a placeholder,
/// and `ud_repairs=` is absent when nothing was repaired.
///
/// A full run clears every `%mor` and `%gra` tier before it analyzes, so the
/// repairs behind the tiers in the written document are exactly this run's.
/// The incremental builder below is the one that must account for two runs.
pub(crate) fn morphotag_provenance(
    lang: &LanguageCode3,
    applied: &AppliedAnalyses,
    retokenize: bool,
) -> ProvenanceComment {
    morphotag_comment(
        lang,
        &applied.engine_names(),
        retokenize,
        applied.repair_count(),
    )
}

/// The morphotag stamp's fields, shared by the full and incremental runs.
fn morphotag_comment(
    lang: &LanguageCode3,
    names: &EngineNames,
    retokenize: bool,
    repairs: Option<NonZeroUsize>,
) -> ProvenanceComment {
    ProvenanceComment::new(ReleasedCommand::Morphotag)
        .field(StampField::Lang, StampFieldValue::from(lang))
        .engines(names)
        .field_if(StampField::Retokenize, retokenize)
        .field_if_some(StampField::UdRepairs, repairs.map(StampFieldValue::from))
}

/// Why an incremental morphotag run cannot name the models behind the tiers it
/// kept.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IncrementalMorphotagStampError {
    /// The `--before` document's morphotag stamp does not parse.
    #[error(transparent)]
    UnparseablePriorStamp(#[from] UnparseableStamp),
    /// A name in the `--before` document's `engine=` list is not stamp-safe
    /// text.
    #[error("the prior morphotag stamp names an engine that cannot be written back: {0}")]
    UnwritablePriorEngine(#[from] InvalidStampSafeText),
    /// A model that ran now has a name containing the list separator, so a
    /// later incremental run could not read this stamp's list back into the
    /// same names.
    #[error(
        "morphotag model {0} contains the engine-list separator `{sep}`, so an incremental \
         stamp naming it could not be read back",
        sep = EngineNames::JOINER.as_str()
    )]
    SeparatorInModelName(StampSafeText),
    /// The `--before` document's morphotag stamp states a repair count that is
    /// not a count, so the repairs behind the tiers it keeps cannot be carried
    /// forward. Refused rather than treated as none, which would silently
    /// undercount the written document.
    #[error("the prior morphotag stamp states an unreadable repair count {0:?}")]
    UnreadablePriorRepairCount(String),
}

impl From<IncrementalMorphotagStampError> for crate::error::ServerError {
    fn from(error: IncrementalMorphotagStampError) -> Self {
        Self::Validation(error.to_string())
    }
}

/// Build the stamp for an incremental morphotag run, which reanalyzes only the
/// utterances an edit changed and copies every other `%mor`/`%gra` tier from
/// `prior`, the `--before` document.
///
/// `None` when no model ran: nothing was reanalyzed, so the document keeps the
/// stamp it already carries, which names the models behind the tiers it kept.
/// Otherwise the output holds tiers from both runs, so `engine=` names the
/// models `prior`'s morphotag stamps name (read back through [`StampCodec`])
/// together with the models that ran now, and `incremental=true` is set.
///
/// A written list is read back by splitting on the separator [`EngineNames`]
/// joins with. That is exact only when no name contains the separator, so a
/// model that ran now with such a name is refused rather than written into a
/// list the next incremental run would misread.
pub(crate) fn incremental_morphotag_provenance(
    prior: &ChatFile,
    lang: &LanguageCode3,
    applied: &AppliedAnalyses,
    retokenize: bool,
) -> Result<Option<ProvenanceComment>, IncrementalMorphotagStampError> {
    if applied.ran_no_model() {
        return Ok(None);
    }
    let separator = EngineNames::JOINER.as_str();
    let prior_record = stamped_morphotag_record(prior, ReleasedCommand::Morphotag)?;
    let mut names = prior_record.engines;
    for name in applied.engine_names() {
        if name.as_str().contains(separator) {
            return Err(IncrementalMorphotagStampError::SeparatorInModelName(name));
        }
        names.extend([name]);
    }
    // The output holds tiers from both runs, so the count covers both, exactly
    // as `engine=` names the models of both. Saturating because a count that
    // large is not a fact about any transcript, and wrapping it would be the
    // one way to report FEWER repairs than were made.
    let repairs = NonZeroUsize::new(
        prior_record
            .repairs
            .saturating_add(applied.repair_count().map_or(0, NonZeroUsize::get)),
    );
    Ok(Some(
        morphotag_comment(lang, &names, retokenize, repairs)
            .field_if(StampField::Incremental, true),
    ))
}

/// What the stamps a `--before` document already carries record about the
/// morphotag tiers it keeps.
///
/// One walk produces both, because both are read from the same stamps for the
/// same reason: the incremental output holds those tiers beside the new ones,
/// so its stamp must account for them together.
#[derive(Debug, Default)]
struct StampedMorphotagRecord {
    /// Every engine name the stamps record. A stamp with no `engine=` field
    /// names nothing.
    engines: EngineNames,
    /// How many relation repairs the stamps record. A stamp with no
    /// `ud_repairs=` field records none, which is what the grammar's absence
    /// means. A stamp written by a build older than the field records none
    /// too, and is indistinguishable from one that repaired nothing: the
    /// grammar cannot say "unknown", and inventing a spelling for it would
    /// change what every existing file means.
    repairs: usize,
}

/// Read what the stamps for `command` in `file` record, through the codec.
fn stamped_morphotag_record(
    file: &ChatFile,
    command: ReleasedCommand,
) -> Result<StampedMorphotagRecord, IncrementalMorphotagStampError> {
    let separator = EngineNames::JOINER.as_str();
    let mut record = StampedMorphotagRecord::default();
    for line in &file.lines {
        let Line::Header { header, .. } = line else {
            continue;
        };
        let Header::Comment { content } = header.as_ref() else {
            continue;
        };
        let text = content.to_chat_string();
        let stamp = match RecognizedComment::classify(&text) {
            RecognizedComment::Stamp(stamp) if stamp.records(command) => stamp,
            // Another command's stamp, our warning, someone else's comment, or
            // a stamp too damaged to attribute to any command: none of them
            // names an engine behind THIS command's tiers.
            RecognizedComment::Stamp(_)
            | RecognizedComment::Unnamed(_)
            | RecognizedComment::UncheckedAsrWarning { .. }
            | RecognizedComment::Foreign => continue,
        };
        let parsed = stamp.parse()?;
        if let Some(list) = parsed.fields.get(StampField::Engine.as_str()) {
            for name in list.split(separator) {
                record.engines.extend([StampSafeText::try_from(name)?]);
            }
        }
        if let Some(count) = parsed.fields.get(StampField::UdRepairs.as_str()) {
            let count: usize = count.parse().map_err(|_| {
                IncrementalMorphotagStampError::UnreadablePriorRepairCount((*count).to_owned())
            })?;
            record.repairs = record.repairs.saturating_add(count);
        }
    }
    Ok(record)
}

/// Whether utterance timing recovery contributed to an aligned file, and with
/// which engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UtrContribution {
    /// No recovery pass ran for this file.
    NotRun,
    /// A recovery pass ran with this engine.
    Ran(UtrEngine),
}

impl UtrContribution {
    /// Fold one completed recovery pass in. A pass that ran records its
    /// engine; one that found nothing untimed leaves the earlier record.
    pub(crate) fn record_pass(&mut self, engine: &UtrEngine, result: &UtrResult) {
        if result.ran() {
            *self = Self::Ran(engine.clone());
        }
    }
}

/// Build a provenance comment for align: `fa=` is the FA engine the selected
/// worker reported (also the FA cache namespace), and `utr=` names the timing
/// recovery engine when a recovery pass ran.
pub(crate) fn align_provenance(
    lang: &LanguageCode3,
    fa: &FaCacheNamespace,
    utr: &UtrContribution,
    wor: bool,
    incremental: bool,
) -> ProvenanceComment {
    let comment = ProvenanceComment::new(ReleasedCommand::Align)
        .field(StampField::Fa, StampFieldValue::from(fa.name()))
        .field(StampField::Lang, StampFieldValue::from(lang));
    let comment = match utr {
        UtrContribution::NotRun => comment,
        UtrContribution::Ran(engine) => {
            comment.field(StampField::Utr, StampFieldValue::from(engine.stamp_name()))
        }
    };
    comment
        .field_if(StampField::Wor, wor)
        .field_if(StampField::Incremental, incremental)
}

/// Build a provenance comment for transcribe: `asr=` names the engine and
/// `asr_model=` the selected checkpoint, when there is one. Infallible: the
/// checkpoint was admitted as stamp-safe text with the transcribe plan.
pub(crate) fn transcribe_provenance(
    lang: &crate::api::TranscriptLanguage,
    asr: &AsrIdentity,
    diarize: bool,
    wor: bool,
) -> ProvenanceComment {
    let comment = ProvenanceComment::new(ReleasedCommand::Transcribe)
        .field(StampField::Asr, StampFieldValue::from(asr.engine().clone()))
        .field(StampField::Lang, StampFieldValue::from(lang))
        .field_if(StampField::Diarize, diarize)
        .field_if(StampField::Wor, wor);
    match asr.asr_model() {
        Some(model) => comment.field(StampField::AsrModel, StampFieldValue::from(model)),
        None => comment,
    }
}

/// Why a text command's output carries no provenance stamp.
///
/// A stamp names the engines behind what was applied, so a run that applied
/// nothing has nothing to name. This is the reason, carried as a value: the
/// pipelines used to receive a bare `None` here and could only guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum NoStampReason {
    /// Every item of this file was blank, unresolved, or otherwise produced
    /// nothing that was applied, so no engine ran.
    #[error("no engine produced anything that was applied to this file")]
    NothingApplied,
    /// Work was applied, but the worker did not say what produced it: utseg
    /// results that carry no model evidence. A stamp would have to invent a
    /// name for the source, and an invented name is what this program removed
    /// everywhere else.
    #[error("the worker applied results without naming what produced them")]
    SourceNotNamed,
}

/// The stamp a text command's output carries, or the reason it carries none.
///
/// Returned by every [`crate::pipeline::text_infer::TextProvenance`] source and
/// matched exhaustively by the pipelines, so "no stamp" is a state with a
/// reason rather than an absent value.
#[derive(Debug, Clone)]
pub(crate) enum TextStamp {
    /// Stamp this comment on the output.
    Stamped(ProvenanceComment),
    /// Write no stamp, for this reason.
    NotStamped(NoStampReason),
}

/// Build a provenance comment for utseg naming the inference sources behind
/// the predictions it applied (see [`UtsegEngineIdentities`]);
/// [`NoStampReason::NothingApplied`] when there were none.
///
/// Fallible, unlike the other builders: a boundary model's id and revision
/// come from the worker's prediction evidence, which is not admitted as
/// stamp-safe text where it is born.
pub(crate) fn utseg_provenance(
    lang: &LanguageCode3,
    predictions: &[AdmittedUtsegPrediction],
) -> Result<TextStamp, InvalidStampSafeText> {
    let Some(engines) = UtsegEngineIdentities::from_predictions(predictions) else {
        return Ok(TextStamp::NotStamped(NoStampReason::NothingApplied));
    };
    let names = engines.engine_names()?;
    if names.is_empty() {
        // Boundaries were applied, but every source declined to name itself.
        // A stamp with no `engine=` would say a run happened and nothing about
        // what produced it, and a placeholder name would be worse.
        return Ok(TextStamp::NotStamped(NoStampReason::SourceNotNamed));
    }
    Ok(TextStamp::Stamped(
        ProvenanceComment::new(ReleasedCommand::Utseg)
            .engines(&names)
            .field(StampField::Lang, StampFieldValue::from(lang)),
    ))
}

/// A command whose stamp names the engines its applied results named, one
/// reported name per result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResultNamedCommand {
    /// Each translation names the engine that produced it.
    Translate,
    /// Each resolved document names the engine that resolved it.
    Coref,
}

impl ResultNamedCommand {
    const fn command(self) -> ReleasedCommand {
        match self {
            Self::Translate => ReleasedCommand::Translate,
            Self::Coref => ReleasedCommand::Coref,
        }
    }
}

/// Build the stamp for translate or coref, naming the engines of the results
/// it applied; [`NoStampReason::NothingApplied`] when no result named one,
/// because nothing was applied.
pub(crate) fn result_named_provenance<'a>(
    command: ResultNamedCommand,
    lang: &LanguageCode3,
    engines: impl IntoIterator<Item = &'a ReportedEngineName>,
) -> TextStamp {
    let names: EngineNames = engines
        .into_iter()
        .map(|name| name.as_stamp_text().clone())
        .collect();
    if names.is_empty() {
        return TextStamp::NotStamped(NoStampReason::NothingApplied);
    }
    TextStamp::Stamped(
        ProvenanceComment::new(command.command())
            .engines(&names)
            .field(StampField::Lang, StampFieldValue::from(lang)),
    )
}

// ---------------------------------------------------------------------------
// The unchecked-ASR warning
// ---------------------------------------------------------------------------

/// The human-readable warning transcribe writes over unchecked ASR output:
/// `fc-ba3 <build identity>, ASR engine <engine> (<checkpoint>). Unchecked
/// output of ASR model, DO NOT USE.` The parenthetical is present only when the
/// [`AsrIdentity`] carries a checkpoint (its `Display` owns that choice). The
/// build identity is [`crate::build_hash`], never the semver, because
/// staleness is judged by build identity.
struct UncheckedAsrWarning<'a> {
    asr: &'a AsrIdentity,
}

impl std::fmt::Display for UncheckedAsrWarning<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            concat!(
                written_name!(),
                " {}",
                current_warning_engine_marker!(),
                "{}. ",
                unchecked_sentence!(),
                do_not_use_ending!()
            ),
            crate::build_hash(),
            self.asr
        )
    }
}

/// One shape our warning has been written in:
/// `<product><identity><engine marker><engine>. <sentence><ending>`, where the
/// identity is one whitespace-free token and the engine is non-empty.
struct WarningShape {
    product: &'static str,
    engine_marker: &'static str,
    endings: &'static [&'static str],
}

/// Every shape of OUR warning: the one this build writes, and the one builds
/// before the rename wrote (`Batchalign <version>, ASR Engine <engine>.
/// Unchecked output of ASR model.`, optionally ending `, DO NOT USE.`).
///
/// Recognition is deliberately narrow: each shape opens with our own product
/// token and names an engine, so a comment that is exactly `Unchecked output of
/// ASR model` (written by another Batchalign distribution) is not ours and
/// survives a re-transcription.
const OUR_WARNING_SHAPES: [WarningShape; 2] = [
    WarningShape {
        product: concat!(written_name!(), " "),
        engine_marker: current_warning_engine_marker!(),
        endings: &[do_not_use_ending!()],
    },
    WarningShape {
        product: legacy_warning_product!(),
        engine_marker: legacy_warning_engine_marker!(),
        endings: &[".", do_not_use_ending!()],
    },
];

impl WarningShape {
    /// The engine text, if `text` has this shape.
    fn engine_in<'a>(&self, text: &'a str) -> Option<&'a str> {
        let (identity, rest) = text
            .strip_prefix(self.product)?
            .split_once(self.engine_marker)?;
        if identity.is_empty() || identity.contains(char::is_whitespace) {
            return None;
        }
        self.endings.iter().find_map(|ending| {
            rest.strip_suffix(ending)
                .and_then(|rest| rest.strip_suffix(unchecked_sentence!()))
                .and_then(|engine| engine.strip_suffix(". "))
                .filter(|engine| !engine.is_empty())
        })
    }
}

/// The ASR engine text of OUR unchecked-ASR warning, in either shape, or
/// `None` when `text` is not one of ours. The engine is what the no-op write
/// gate compares; the shape and the build identity are not meaningful.
fn our_warning_engine(text: &str) -> Option<&str> {
    let text = text.trim();
    OUR_WARNING_SHAPES
        .iter()
        .find_map(|shape| shape.engine_in(text))
}

/// Inject a human-readable "unchecked ASR" warning comment into a CHAT file.
///
/// This is separate from the machine-readable provenance comment. Readers rely
/// on this text to know a transcript has not been human-reviewed.
///
/// A previous warning of ours, in either shape, is replaced rather than
/// duplicated when re-transcribing; see [`OUR_WARNING_SHAPES`] for what counts
/// as ours.
pub(crate) fn inject_unchecked_warning(file: &mut ChatFile, asr: &AsrIdentity) {
    let warning_text = UncheckedAsrWarning { asr }.to_string();

    // Remove existing unchecked warning (if re-transcribing).
    file.lines.retain(|line| {
        if let Line::Header { header, .. } = line
            && let Header::Comment { content } = header.as_ref()
        {
            return our_warning_engine(&content.to_chat_string()).is_none();
        }
        true
    });

    // Insert after the constant participant headers (same position as
    // provenance), so @Birth of / @Birthplace of / @L1 of stay adjacent to @ID.
    let insert_pos = insert_pos_after_constant_headers(file);

    let bullet_content = crate::chat_ops::BulletContent::from_text(warning_text);

    file.lines.insert(
        insert_pos,
        Line::header(Header::Comment {
            content: bullet_content,
        }),
    );
}

// ---------------------------------------------------------------------------
// No-op write detection: recognize when a candidate output text would only
// differ from the on-disk text in what re-running a command always changes
// (the stamp name and timestamp, and the warning's build identity), so the
// runner can skip pointless disk writes and the git-status churn they cause.
// ---------------------------------------------------------------------------

/// What one side's provenance lines said, modulo what never counts.
#[derive(Debug, Default, PartialEq, Eq)]
struct ProvenanceSeen<'a> {
    /// The fields of each stamp set aside, by the command it records, each
    /// command's stamps in file order. The key is the command's own name
    /// ([`ReleasedCommand::as_str`]), never text borrowed from a file, so the
    /// two sides cannot key one command under two spellings.
    stamps: BTreeMap<&'static str, Vec<BTreeMap<&'a str, &'a str>>>,
    /// The ASR engine text of each of our warnings, in file order.
    warning_engines: Vec<&'a str>,
    /// Whether a stamp set aside did not parse.
    unparseable_stamp: bool,
}

/// Returns `true` when `old_text` and `new_text` differ ONLY in what
/// re-running a `command` job always changes:
///
/// - the stamp of every command the job's recipe composes
///   (`crate::command_model::commands_stamped_by`), in its NAME (`fc-ba3` or
///   the legacy `ba3`) and its TIMESTAMP. A `transcribe` job writes
///   `transcribe`, `utseg` and `morphotag` stamps, so all three are set aside
///   and compared; setting aside only the command's own would make every
///   re-run on a new build a write; and
/// - our unchecked-ASR warning, in its SHAPE (current or legacy) and the
///   BUILD IDENTITY it names.
///
/// Any other difference is meaningful: a different field on a set-aside stamp
/// (an engine, a language, a flag), a set-aside stamp present on one side
/// only, a different ASR engine in the warning, the stamp of a command the job
/// does not compose, `%mor` / `%gra` / `%wor` content, or anything else. A
/// set-aside stamp that does not parse is also treated as meaningful, so the
/// file is written and the stamp replaced.
///
/// Returns `false` for byte-equal inputs: there is no diff to suppress.
///
/// Comparison is line-based and streamed: both inputs are walked in lockstep,
/// stamp and warning lines are set aside on either side (so a stamp that moved
/// position is not a difference by itself), and the set-aside provenance is
/// compared once both walks finish. No intermediate copy of either text is
/// made.
pub(crate) fn is_provenance_only_difference(
    old_text: &str,
    new_text: &str,
    command: ReleasedCommand,
) -> bool {
    // A non-difference is not a provenance-only difference. The caller's
    // contract is "should I suppress this write?", and writing identical
    // bytes is already a no-op the OS will short-circuit; we don't need
    // to claim ownership of that case.
    if old_text == new_text {
        return false;
    }
    let stamped = crate::command_model::commands_stamped_by(command);
    let mut old_seen = ProvenanceSeen::default();
    let mut new_seen = ProvenanceSeen::default();
    let mut old_lines = old_text.split_inclusive('\n');
    let mut new_lines = new_text.split_inclusive('\n');
    loop {
        let old = next_content_line(&mut old_lines, &stamped, &mut old_seen);
        let new = next_content_line(&mut new_lines, &stamped, &mut new_seen);
        if old != new {
            return false;
        }
        if old.is_none() {
            // Both streams ended on identical content; the provenance set
            // aside must agree too.
            return !old_seen.unparseable_stamp
                && !new_seen.unparseable_stamp
                && old_seen == new_seen;
        }
    }
}

/// Advance `lines` past any stamp for a command in `stamped` and any warning
/// of ours, recording what they said into `seen`, and return the next other
/// chunk. Returns `None` once the iterator is exhausted. Each chunk includes
/// its own line terminator (from `split_inclusive`), so terminator differences
/// propagate naturally: we are not normalizing newlines.
fn next_content_line<'a>(
    lines: &mut std::str::SplitInclusive<'a, char>,
    stamped: &[ReleasedCommand],
    seen: &mut ProvenanceSeen<'a>,
) -> Option<&'a str> {
    for chunk in lines.by_ref() {
        let Some(body) = comment_body(chunk) else {
            return Some(chunk);
        };
        match RecognizedComment::classify(body) {
            RecognizedComment::Stamp(stamp) => match stamp.recorded_among(stamped) {
                Some(command) => match stamp.parse() {
                    Ok(parsed) => seen
                        .stamps
                        .entry(command.as_str())
                        .or_default()
                        .push(parsed.fields),
                    // A stamp this job writes that does not parse: the write
                    // goes through and replaces it with one that does.
                    Err(_) => seen.unparseable_stamp = true,
                },
                // Ours, but for a command this job does not run. Whether it is
                // well formed or damaged is not this job's business, so it is
                // compared as content: identical on both sides, no rewrite.
                None => return Some(chunk),
            },
            RecognizedComment::UncheckedAsrWarning { engine } => seen.warning_engines.push(engine),
            // Too damaged to say whose stamp it is. Writing would not repair it
            // either, because replacement matches a stamp by its command, so it
            // is compared as content like any other line.
            RecognizedComment::Unnamed(_) => return Some(chunk),
            RecognizedComment::Foreign => return Some(chunk),
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Extraction: parse provenance comments from CHAT text
// ---------------------------------------------------------------------------

/// One extracted provenance entry from a CHAT file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct ProvenanceEntry {
    /// Command name (e.g., "morphotag", "align").
    pub command: String,
    /// Key-value fields (engine, lang, etc.).
    pub fields: BTreeMap<String, String>,
    /// ISO 8601 timestamp string.
    pub timestamp: String,
}

/// A `@Comment` this build generates, recognized by the codec that writes it.
///
/// The two kinds a run adds to a file: its per-command stamp and its
/// unchecked-ASR warning. Anything else a file carries, including the bare
/// warning another Batchalign distribution writes, is not ours and has no
/// variant here.
///
/// Comparable, but not `Eq`: a stamp carries [`ProvenanceEntry`], which is a
/// read-back of what a file says rather than a value with a total equality.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GeneratedComment {
    /// One of our stamps, under either name, read with [`StampCodec`].
    Stamp(ProvenanceEntry),
    /// Our unchecked-ASR warning, naming the ASR engine it records.
    UncheckedAsrWarning {
        /// The engine text the warning names.
        engine: String,
    },
}

/// Recognize one `@Comment` body as a comment this build generates.
///
/// `None` for every comment that is not ours. Shared by [`extract_provenance`]
/// and by the offline replays, which must leave what a run generates out of a
/// comparison: a stamp's timestamp and a warning's build identity change on
/// every run, so a generated comment can never match by equality.
///
/// Every reader of a `@Comment` goes through [`recognize::RecognizedComment`],
/// and they take different amounts of it. This one, and the offline replays
/// through it, need both steps, because they report a whole entry.
/// [`is_stamp_for`], [`stamped_engine_names`] and the no-op write gate need
/// step one and their own comparison. The gate above all: it must know WHOSE
/// stamp it is looking at before a damaged one means anything, because a
/// damaged stamp belonging to a command the job does not run must not provoke
/// a rewrite.
///
/// A comment that opens as one of our stamps but does not parse is an
/// [`UnparseableStamp`] error rather than an unrecognized comment: silently
/// treating it as someone else's would misreport what produced the file.
pub(crate) fn recognize_generated_comment(
    body: &str,
) -> Result<Option<GeneratedComment>, UnparseableStamp> {
    match RecognizedComment::classify(body) {
        RecognizedComment::Stamp(stamp) => {
            let parsed = stamp.parse()?;
            Ok(Some(GeneratedComment::Stamp(ProvenanceEntry {
                command: stamp.command().to_owned(),
                fields: parsed
                    .fields
                    .into_iter()
                    .map(|(key, value)| (key.to_owned(), value.to_owned()))
                    .collect(),
                timestamp: parsed.timestamp.to_owned(),
            })))
        }
        // A stamp of ours too damaged to name its command is still ours:
        // reporting it as someone else's comment would misreport what produced
        // the file, and leaving it out would report a shorter history.
        RecognizedComment::Unnamed(stamp) => Err(stamp.into_error()),
        RecognizedComment::UncheckedAsrWarning { engine } => {
            Ok(Some(GeneratedComment::UncheckedAsrWarning {
                engine: engine.to_owned(),
            }))
        }
        RecognizedComment::Foreign => Ok(None),
    }
}

/// Extract all batchalign3 provenance entries from CHAT text.
///
/// Scans for `@Comment:` lines holding one of our stamps, under either name
/// (`[fc-ba3 ...]` or the legacy `[ba3 ...]`), and reads them through
/// [`recognize_generated_comment`], which uses the same codec the writer uses.
/// The entry shape does not record which name wrote the stamp. A comment that
/// opens as one of our stamps but does not parse is an [`UnparseableStamp`]
/// error: silently leaving it out would report a file's history as shorter than
/// it is.
pub fn extract_provenance(chat_text: &str) -> Result<Vec<ProvenanceEntry>, UnparseableStamp> {
    let mut entries = Vec::new();
    for line in chat_text.lines() {
        let Some(body) = comment_body(line) else {
            continue;
        };
        if let Some(GeneratedComment::Stamp(entry)) = recognize_generated_comment(body)? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::parse::parse_lenient;

    fn value(text: &'static str) -> StampFieldValue {
        StampFieldValue::from(StampSafeText::try_from(text).expect("valid test field value"))
    }

    fn empty_ud() -> crate::chat_ops::nlp::UdResponse {
        crate::chat_ops::nlp::UdResponse {
            sentences: Vec::new(),
        }
    }

    /// The ASR identity a transcribe plan admits for `engine` with `extras`.
    fn asr_identity(
        engine: crate::types::engines::AsrEngineName,
        extras: &BTreeMap<String, String>,
    ) -> AsrIdentity {
        let backend = crate::transcribe::types::AsrBackend::try_from_engine(&engine)
            .expect("engine is implemented");
        crate::transcribe::types::TranscribeAsrPlan::from_request(
            backend,
            false,
            1,
            extras,
            &crate::api::LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
        )
        .expect("plan is admitted")
        .identity()
    }

    /// The stamp records the models a run actually LOADED, auxiliaries
    /// included, not merely a checkpoint the request happened to name.
    ///
    /// Before this, `asr_model=` could only carry an override a backend read
    /// from its own key, so the engines with no such key recorded nothing at
    /// all, and even where one existed it named a request rather than a result.
    /// Qwen is the case worth pinning because its composition has a second
    /// model: a transcript that names the recognizer but not the aligner that
    /// produced its word timings is only half attributed.
    #[test]
    fn asr_model_records_every_loaded_model_with_its_revision() {
        use crate::types::worker_v2::{
            AsrModelIdentityV2, HubCommitV2, LoadedModelV2, ModelIdV2, ObservedRevisionV2,
            RequestedRevisionV2,
        };

        const ASR: &str = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3";
        const ALIGNER: &str = "c07281df297b9905d24a508279258cccf987a064";

        fn loaded(id: &str, commit: &str) -> LoadedModelV2 {
            let commit = HubCommitV2::try_from(commit).expect("a 40 hex character commit");
            LoadedModelV2 {
                id: ModelIdV2::try_from(id).expect("a hub model id is stamp safe"),
                requested: RequestedRevisionV2::Commit {
                    commit: commit.clone(),
                },
                observed: ObservedRevisionV2::Commit { commit },
            }
        }

        let qwen = asr_identity(
            crate::types::engines::AsrEngineName::HkQwen,
            &BTreeMap::new(),
        )
        .with_loaded_models(AsrModelIdentityV2::Qwen {
            asr: loaded("Qwen/Qwen3-ASR-1.7B-hf", ASR),
            aligner: loaded("Qwen/Qwen3-ForcedAligner-0.6B-hf", ALIGNER),
        });

        let comment = transcribe_provenance(
            &crate::api::TranscriptLanguage::One(LanguageCode3::yue()),
            &qwen,
            false,
            false,
        )
        .format();
        assert!(
            comment.contains(&format!(
                "asr_model=Qwen/Qwen3-ASR-1.7B-hf@{ASR}\
                 +aligner:Qwen/Qwen3-ForcedAligner-0.6B-hf@{ALIGNER}"
            )),
            "{comment}"
        );
    }

    #[test]
    fn format_morphotag_provenance() {
        let comment = ProvenanceComment::new(ReleasedCommand::Morphotag)
            .field(StampField::Engine, value("stanza-1.11.1"))
            .field(StampField::Lang, value("eng"));
        let formatted = comment.format();
        assert!(formatted.starts_with("[fc-ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | "));
        assert!(formatted.ends_with(']'));
    }

    #[test]
    fn format_empty_fields() {
        let formatted = ProvenanceComment::new(ReleasedCommand::Utseg).format();
        assert!(formatted.starts_with("[fc-ba3 utseg | "));
        assert!(formatted.ends_with(']'));
    }

    /// The one stamp `text` opens as, which must name a command.
    fn named(text: &str) -> recognize::NamedStamp<'_> {
        match RecognizedComment::classify(text) {
            RecognizedComment::Stamp(stamp) => stamp,
            other => panic!("expected a stamp naming a command, got {other:?}"),
        }
    }

    /// The writer and the reader are one codec: what is written reads back,
    /// the command through step one and the rest through step two.
    #[test]
    fn the_stamp_codec_reads_what_it_writes() {
        let written = StampCodec::write(
            StampName::WRITTEN.opening(),
            "align",
            [("fa", "wave2vec-fa-v1"), ("lang", "eng"), ("wor", "true")].into_iter(),
            "2026-09-15T19:15:00-04:00",
        );
        let stamp = named(&written);
        assert_eq!(stamp.command(), "align");
        let parsed = stamp.parse().expect("reads back");
        assert_eq!(
            parsed.fields,
            BTreeMap::from([("fa", "wave2vec-fa-v1"), ("lang", "eng"), ("wor", "true")])
        );
        assert_eq!(parsed.timestamp, "2026-09-15T19:15:00-04:00");

        let bare = StampCodec::write(
            StampName::WRITTEN.opening(),
            "utseg",
            std::iter::empty(),
            "2026-09-15T19:20:00-04:00",
        );
        let stamp = named(&bare);
        assert_eq!(stamp.command(), "utseg");
        assert!(stamp.parse().expect("reads back").fields.is_empty());
    }

    /// Recognition answers in two steps, and step one names the command even
    /// when the body is damaged.
    ///
    /// Four answers, not two. A damaged stamp, a comment that is not ours, and
    /// a stamp for another command are different facts, and the no-op write
    /// gate acts differently on each; a recognizer that could only say "parsed"
    /// or "did not parse" would force two of them together.
    #[test]
    fn classification_names_a_command_before_the_body_is_parsed() {
        // Ours, named, and the body parses.
        let whole = named("[fc-ba3 align | fa=wave2vec-fa-v1 | 2026-09-15T19:15:00-04:00]");
        assert_eq!(whole.command(), "align");
        assert!(whole.records(ReleasedCommand::Align));
        assert!(!whole.records(ReleasedCommand::Morphotag));
        assert!(whole.parse().is_ok());

        // Ours, named, and the body does NOT parse. Step one still answers,
        // which is what lets the gate ask whose stamp this is.
        let damaged = named("[ba3 align | fa ; lang=eng | 2026-09-15T19:15:00-04:00]");
        assert_eq!(damaged.command(), "align");
        assert_eq!(
            damaged.parse().expect_err("the field has no value").defect,
            StampDefect::FieldWithoutValue("fa".to_owned())
        );
        assert_eq!(
            damaged.recorded_among(&[ReleasedCommand::Morphotag, ReleasedCommand::Align]),
            Some(ReleasedCommand::Align)
        );
        assert_eq!(damaged.recorded_among(&[ReleasedCommand::Morphotag]), None);

        // Ours, but nothing before a separator to name a command with.
        for unnamed in ["[fc-ba3 align]", "[fc-ba3  | 2026-09-15T19:15:00-04:00]"] {
            assert!(
                matches!(
                    RecognizedComment::classify(unnamed),
                    RecognizedComment::Unnamed(_)
                ),
                "{unnamed}"
            );
        }

        // Not ours, including the bare warning another distribution writes.
        for foreign in [
            "Unchecked output of ASR model",
            "batchalign3 abc1234 | asr: rev | 2026-06-02T17:04:12Z",
            "[ba3transcribe | x]",
            "This is a regular user comment",
        ] {
            assert!(
                matches!(
                    RecognizedComment::classify(foreign),
                    RecognizedComment::Foreign
                ),
                "{foreign}"
            );
        }

        // Ours, and not a stamp at all.
        assert!(matches!(
            RecognizedComment::classify(
                "fc-ba3 build-1, ASR engine rev. Unchecked output of ASR model, DO NOT USE."
            ),
            RecognizedComment::UncheckedAsrWarning { engine: "rev" }
        ));
    }

    /// Recognition takes both names and nothing else, and a command must match
    /// whole: a longer command sharing the prefix is a different stamp.
    #[test]
    fn stamps_are_recognized_under_both_names_only() {
        for text in [
            "[fc-ba3 transcribe | asr=rev | 2026-09-15T10:00:00-04:00]",
            "[ba3 transcribe | asr=rev | 2026-03-28T10:00:00-04:00]",
            "  [fc-ba3 transcribe | 2026-09-15T10:00:00-04:00]",
        ] {
            assert!(is_stamp_for(text, ReleasedCommand::Transcribe), "{text}");
        }
        for text in [
            "[fc-ba3 transcribe_s | asr=rev | 2026-09-15T10:00:00-04:00]",
            "[ba3transcribe | x]",
            "batchalign3 abc1234 | transcribe: rev | 2026-06-02T17:04:12Z",
            "Unchecked output of ASR model",
        ] {
            assert!(!is_stamp_for(text, ReleasedCommand::Transcribe), "{text}");
        }
    }

    #[test]
    fn provenance_refuses_recovered_output() {
        let comment = ProvenanceComment::new(ReleasedCommand::Align);
        for body in ["*PAR:\thello [ .", "*PAR:\thello .\n%mor:\t|"] {
            let chat = format!(
                "@UTF8\n@Begin\n@Languages:\teng\n\
                 @Participants:\tPAR Participant\n\
                 @ID:\teng|test|PAR|||||Participant|||\n{body}\n@End\n"
            );
            let errors = inject_provenance_into_text(&chat, &comment)
                .expect_err("provenance must not hide main-tier or generated-tier parse errors");
            assert!(!errors.errors.is_empty());
        }
    }

    #[test]
    fn inject_replaces_existing_comment_for_same_command() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.10.0 ; lang=eng | 2026-03-28T10:00:00-04:00]
*PAR:\thello .
@End
";
        let new_comment = ProvenanceComment::new(ReleasedCommand::Morphotag)
            .field(StampField::Engine, value("stanza-1.11.1"))
            .field(StampField::Lang, value("eng"));

        let result = inject_provenance_into_text(chat, &new_comment).expect("valid CHAT");

        // The legacy-named comment is ours, so it is replaced, not kept.
        assert!(!result.contains("stanza-1.10.0"));
        assert!(!result.contains("[ba3 morphotag"), "{result}");
        // New comment should be present, under the written name only.
        assert!(result.contains("stanza-1.11.1"));
        assert_eq!(
            result.matches("[fc-ba3 morphotag").count(),
            1,
            "should have exactly one morphotag provenance comment"
        );

        // A second run replaces the stamp the first one wrote.
        let rerun = inject_provenance_into_text(
            &result,
            &ProvenanceComment::new(ReleasedCommand::Morphotag)
                .field(StampField::Engine, value("stanza-1.12.0")),
        )
        .expect("valid CHAT");
        assert!(!rerun.contains("stanza-1.11.1"), "{rerun}");
        assert_eq!(rerun.matches("[fc-ba3 morphotag").count(), 1, "{rerun}");
    }

    #[test]
    fn inject_preserves_comments_from_other_commands() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 transcribe | asr=whisper ; lang=eng | 2026-03-28T10:00:00-04:00]
*PAR:\thello .
@End
";
        let morphotag_comment = ProvenanceComment::new(ReleasedCommand::Morphotag)
            .field(StampField::Engine, value("stanza-1.11.1"))
            .field(StampField::Lang, value("eng"));

        let result = inject_provenance_into_text(chat, &morphotag_comment).expect("valid CHAT");

        // Another command's stamp survives, whichever name wrote it.
        assert!(result.contains("[ba3 transcribe"));
        // Morphotag comment should be added
        assert!(result.contains("[fc-ba3 morphotag"));
    }

    /// The provenance `@Comment` must land AFTER the constant participant
    /// headers (`@Birth of` / `@Birthplace of` / `@L1 of`), not between the
    /// `@ID` block and `@Birth of`. Inserting it right after the last `@ID`
    /// displaces `@Birth of`, which CLAN CHECK flags as error 127.
    #[test]
    fn inject_provenance_lands_after_constant_headers() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Birth of PAR:\t15-DEC-1970
*PAR:\thello .
@End
";
        let comment = ProvenanceComment::new(ReleasedCommand::Morphotag)
            .field(StampField::Engine, value("stanza-1.11.1"))
            .field(StampField::Lang, value("eng"));

        let result = inject_provenance_into_text(chat, &comment).expect("valid CHAT");

        let birth_pos = result
            .find("@Birth of")
            .expect("@Birth of header must be present");
        let stamp_pos = result
            .find("[fc-ba3 morphotag")
            .expect("provenance comment must be present");
        assert!(
            birth_pos < stamp_pos,
            "provenance @Comment must follow @Birth of (constant headers must \
             immediately follow @ID); got:\n{result}"
        );
    }

    /// The machine-readable stamp and the human warning both name the models
    /// that RAN, and they name the SAME thing.
    ///
    /// Agreement is the property under test. Two comments describing one run
    /// that could drift apart is the defect this workstream removes, which is
    /// why both render from one accessor rather than from two formatters.
    #[test]
    fn asr_model_is_recorded_in_provenance_and_warning_text() {
        use crate::types::engines::AsrEngineName;
        use crate::types::worker_v2::{
            AsrModelIdentityV2, HubCommitV2, LoadedModelV2, ModelIdV2, ObservedRevisionV2,
            RequestedRevisionV2,
        };

        const SENSEVOICE: &str = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7";
        const FSMN_VAD: &str = "df20e6b30c653645fa4ff125cacfcabd1020a669";

        fn loaded(id: &str, commit: &str) -> LoadedModelV2 {
            let commit = HubCommitV2::try_from(commit).expect("a 40 hex character commit");
            LoadedModelV2 {
                id: ModelIdV2::try_from(id).expect("a hub model id is stamp safe"),
                requested: RequestedRevisionV2::Commit {
                    commit: commit.clone(),
                },
                observed: ObservedRevisionV2::Commit { commit },
            }
        }

        let expected =
            format!("FunAudioLLM/SenseVoiceSmall@{SENSEVOICE}+vad:funasr/fsmn-vad@{FSMN_VAD}");
        let funaudio = asr_identity(AsrEngineName::HkFunaudio, &BTreeMap::new())
            .with_loaded_models(AsrModelIdentityV2::SenseVoice {
                asr: loaded("FunAudioLLM/SenseVoiceSmall", SENSEVOICE),
                vad: loaded("funasr/fsmn-vad", FSMN_VAD),
            });

        let comment = transcribe_provenance(
            &crate::api::TranscriptLanguage::One(LanguageCode3::zho()),
            &funaudio,
            true,
            false,
        )
        .format();
        assert!(
            comment.contains(&format!("asr_model={expected}")),
            "{comment}"
        );
        assert_eq!(funaudio.to_string(), format!("funaudio ({expected})"));
        assert_eq!(
            UncheckedAsrWarning { asr: &funaudio }.to_string(),
            format!(
                "fc-ba3 {}, ASR engine funaudio ({expected}). Unchecked output of ASR model, DO NOT USE.",
                crate::build_hash()
            )
        );
    }

    /// A run that reported no models names none, in EITHER line, and neither
    /// one falls back to what the request selected.
    ///
    /// The request here does select a checkpoint, and it must still appear
    /// nowhere: printing it would record what was asked for as though it had
    /// been observed, the same substitution the ASR bridge refuses. Replaying
    /// legacy evidence is the live case; the warning simply says less.
    #[test]
    fn an_unreported_identity_names_no_model_in_either_line() {
        use crate::types::engines::{AsrEngineName, FUNAUDIO_MODEL_OVERRIDE_KEY};

        let extras = BTreeMap::from([(
            FUNAUDIO_MODEL_OVERRIDE_KEY.to_owned(),
            "paraformer-zh".to_owned(),
        )]);
        let unreported = asr_identity(AsrEngineName::HkFunaudio, &extras);

        let comment = transcribe_provenance(
            &crate::api::TranscriptLanguage::One(LanguageCode3::zho()),
            &unreported,
            true,
            false,
        )
        .format();
        assert!(!comment.contains("asr_model="), "{comment}");
        assert!(!comment.contains("paraformer-zh"), "{comment}");
        assert_eq!(unreported.to_string(), "funaudio");
        assert_eq!(
            UncheckedAsrWarning { asr: &unreported }.to_string(),
            format!(
                "fc-ba3 {}, ASR engine funaudio. Unchecked output of ASR model, DO NOT USE.",
                crate::build_hash()
            )
        );
    }

    /// A checkpoint that would break the stamp is refused when the transcribe
    /// plan is admitted, so no stamp can ever be built from it.
    #[test]
    fn a_checkpoint_with_stamp_structure_is_refused_at_plan_admission() {
        use crate::transcribe::types::{AsrBackend, TranscribeAsrPlan, TranscribeAsrPlanError};
        use crate::types::engines::{AsrEngineName, FUNAUDIO_MODEL_OVERRIDE_KEY};

        let extras = BTreeMap::from([(
            FUNAUDIO_MODEL_OVERRIDE_KEY.to_owned(),
            "paraformer | zh".to_owned(),
        )]);
        let backend = AsrBackend::try_from_engine(&AsrEngineName::HkFunaudio)
            .expect("funaudio is implemented");
        assert!(matches!(
            TranscribeAsrPlan::from_request(
                backend,
                false,
                1,
                &extras,
                &crate::api::LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
            ),
            Err(TranscribeAsrPlanError::InvalidCheckpoint {
                key: FUNAUDIO_MODEL_OVERRIDE_KEY,
                reason: InvalidStampSafeText::StampStructure(_),
            })
        ));
    }

    /// Our warning is recognized in its current and legacy shapes, with the
    /// engine it names, and never in a line that merely shares the sentence.
    #[test]
    fn unchecked_warning_shapes_are_ours_only() {
        let whisper = asr_identity(
            crate::types::engines::AsrEngineName::Whisper,
            &BTreeMap::new(),
        );
        let current = UncheckedAsrWarning { asr: &whisper }.to_string();
        assert_eq!(our_warning_engine(&current), Some("whisper"));
        for (legacy, engine) in [
            (
                "Batchalign 0.3.0, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.",
                "rev",
            ),
            (
                "Batchalign 0.8.2-post.9, ASR Engine funaudio. Unchecked output of ASR model.",
                "funaudio",
            ),
            (
                "Batchalign 0.9.0, ASR Engine funaudio (paraformer-zh). Unchecked output of ASR model, DO NOT USE.",
                "funaudio (paraformer-zh)",
            ),
        ] {
            assert_eq!(our_warning_engine(legacy), Some(engine), "{legacy}");
        }
        for foreign in [
            "Unchecked output of ASR model",
            "Unchecked output of ASR model.",
            "batchalign3 abc1234 | asr: rev | 2026-06-02T17:04:12Z",
            "Batchalign, ASR Engine rev. Unchecked output of ASR model.",
            "fc-ba3 0.9.0, ASR engine . Unchecked output of ASR model, DO NOT USE.",
            "Please note: Unchecked output of ASR model, DO NOT USE.",
        ] {
            assert_eq!(our_warning_engine(foreign), None, "{foreign}");
        }
    }

    /// Re-transcription replaces every warning of ours, in either shape, with
    /// one current warning, and leaves another distribution's lines alone.
    #[test]
    fn re_transcription_replaces_our_warnings_and_keeps_foreign_lines() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\tBatchalign 0.8.5, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.
@Comment:\tfc-ba3 0.9.0-old-build, ASR engine rev. Unchecked output of ASR model, DO NOT USE.
@Comment:\tUnchecked output of ASR model
@Comment:\tbatchalign3 abc1234 | asr: rev | 2026-06-02T17:04:12Z
*PAR:\thello .
@End
";
        let parser = crate::chat_parser();
        let (mut file, _) = parse_lenient(&parser, chat);
        let rev = asr_identity(
            crate::types::engines::AsrEngineName::RevAi,
            &BTreeMap::new(),
        );
        inject_unchecked_warning(&mut file, &rev);
        let result = to_chat_string(&file);

        assert!(!result.contains("Batchalign 0.8.5"), "{result}");
        assert!(!result.contains("0.9.0-old-build"), "{result}");
        assert_eq!(result.matches("fc-ba3 ").count(), 1, "{result}");
        assert!(result.contains(&format!("fc-ba3 {}, ASR engine rev.", crate::build_hash())));
        assert!(
            result.contains("@Comment:\tUnchecked output of ASR model\n"),
            "{result}"
        );
        assert!(
            result.contains("batchalign3 abc1234 | asr: rev"),
            "{result}"
        );
    }

    /// The unchecked-ASR warning shares the same insertion logic and must also
    /// land after the constant participant headers.
    #[test]
    fn inject_unchecked_warning_lands_after_constant_headers() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Birth of PAR:\t15-DEC-1970
*PAR:\thello .
@End
";
        let parser = crate::chat_parser();
        let (mut file, _) = parse_lenient(&parser, chat);
        let whisper = asr_identity(
            crate::types::engines::AsrEngineName::Whisper,
            &BTreeMap::new(),
        );
        inject_unchecked_warning(&mut file, &whisper);
        let result = to_chat_string(&file);

        let birth_pos = result
            .find("@Birth of")
            .expect("@Birth of header must be present");
        let warning_pos = result
            .find("Unchecked output of ASR model")
            .expect("unchecked warning must be present");
        assert!(
            birth_pos < warning_pos,
            "unchecked warning must follow @Birth of; got:\n{result}"
        );
        assert!(
            result.contains(&format!(
                "fc-ba3 {}, ASR engine whisper. Unchecked output of ASR model, DO NOT USE.",
                crate::build_hash()
            )),
            "warning must identify the compiling build; got:\n{result}"
        );
    }

    /// Extraction reads both names into the same entry shape: a legacy stamp
    /// from an older build and a current one from this build.
    #[test]
    fn extract_provenance_from_chat_under_both_names() {
        let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:\t[fc-ba3 align | fa=whisper-fa-large-v2 ; lang=eng | 2026-09-15T19:15:00-04:00]
@Comment:\t[fc-ba3 utseg | 2026-09-15T19:20:00-04:00]
@Comment:\tThis is a regular user comment
*PAR:\thello .
@End
";
        let entries = extract_provenance(chat).expect("every stamp parses");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].command, "morphotag");
        assert_eq!(entries[0].fields["engine"], "stanza-1.11.1");
        assert_eq!(entries[0].timestamp, "2026-03-29T18:30:00-04:00");
        assert_eq!(entries[1].command, "align");
        assert_eq!(entries[1].fields["fa"], "whisper-fa-large-v2");
        assert_eq!(entries[1].timestamp, "2026-09-15T19:15:00-04:00");
        assert_eq!(entries[2].command, "utseg");
        assert!(entries[2].fields.is_empty());
    }

    #[test]
    fn extract_provenance_ignores_comments_that_are_not_our_stamps() {
        let chat = "\
@Comment:\tBatchalign 0.1.0, ASR Engine rev.
@Comment:\tUnchecked output of ASR model
@Comment:\tbatchalign3 abc1234 | utr: rev:rev_lang_en | 2026-06-02T17:04:12Z
";
        let entries = extract_provenance(chat).expect("no stamps of ours");
        assert!(entries.is_empty(), "{entries:?}");
    }

    /// A comment that opens as our stamp but does not parse is an error that
    /// names the stamp and the defect, never a silently shorter history.
    ///
    /// The last case is the one the two-step recognizer changed. A command must
    /// be followed by the exact separator, so stray text after it is a defect.
    /// Reading the command with the codec's own section split used to swallow
    /// that text INTO the command, and this function then reported a stamp
    /// whose command was `align |x junk`: a name no command has, keyed as if it
    /// were one. Naming the command in step one is what makes it a defect.
    #[test]
    fn an_unparseable_stamp_of_ours_is_a_typed_error() {
        for (stamp, defect) in [
            (
                "[fc-ba3 unterminated | 2026-09-15T19:20:00-04:00",
                StampDefect::Unterminated,
            ),
            ("[fc-ba3 align]", StampDefect::MissingSections),
            (
                "[ba3 align | fa ; lang=eng | 2026-09-15T19:20:00-04:00]",
                StampDefect::FieldWithoutValue("fa".to_owned()),
            ),
            (
                "[fc-ba3 align | lang=eng ; lang=spa | 2026-09-15T19:20:00-04:00]",
                StampDefect::DuplicateField("lang".to_owned()),
            ),
            (
                "[fc-ba3 align |x junk | 2026-09-15T19:20:00-04:00]",
                StampDefect::MissingSections,
            ),
        ] {
            let chat = format!("@Comment:\t{stamp}\n");
            assert_eq!(
                extract_provenance(&chat),
                Err(UnparseableStamp {
                    stamp: stamp.to_owned(),
                    defect
                }),
                "{stamp}"
            );
        }
    }

    #[test]
    fn field_if_omits_false_values() {
        let comment = ProvenanceComment::new(ReleasedCommand::Morphotag)
            .field(StampField::Engine, value("stanza-1.11.1"))
            .field_if(StampField::Retokenize, false)
            .field_if(StampField::Incremental, true);
        let formatted = comment.format();
        assert!(!formatted.contains("retokenize"));
        assert!(formatted.contains("incremental=true"));
    }

    /// Coref and translate name the engines their applied results named, and
    /// stamp nothing when no result named one.
    #[test]
    fn coref_and_translate_provenance_name_the_engines_of_their_results() {
        let stanza = ReportedEngineName::try_from("stanza-1.11.1").expect("valid");
        let google = ReportedEngineName::try_from("googletrans-v1").expect("valid");
        let TextStamp::Stamped(coref) = result_named_provenance(
            ResultNamedCommand::Coref,
            &LanguageCode3::eng(),
            [&stanza, &stanza],
        ) else {
            panic!("a resolved document names its engine");
        };
        let coref = coref.format();
        assert!(
            coref.starts_with("[fc-ba3 coref | engine=stanza-1.11.1 ; lang=eng | "),
            "{coref}"
        );
        let TextStamp::Stamped(translate) = result_named_provenance(
            ResultNamedCommand::Translate,
            &LanguageCode3::eng(),
            [&google, &stanza],
        ) else {
            panic!("translations name their engines");
        };
        let translate = translate.format();
        assert!(
            translate.starts_with(
                "[fc-ba3 translate | engine=googletrans-v1+stanza-1.11.1 ; lang=eng | "
            ),
            "{translate}"
        );
        // Nothing applied is a state with a reason, not an absent stamp.
        assert!(matches!(
            result_named_provenance(
                ResultNamedCommand::Translate,
                &LanguageCode3::eng(),
                std::iter::empty(),
            ),
            TextStamp::NotStamped(NoStampReason::NothingApplied)
        ));
    }

    /// Morphotag names the distinct models behind the responses it applied,
    /// and no engine at all when none analyzed anything.
    #[test]
    fn morphotag_provenance_names_the_models_it_applied() {
        use crate::morphosyntax::identity::AdmittedMorphosyntaxResponse;
        let (_, applied) = AppliedAnalyses::take_applied(vec![
            AdmittedMorphosyntaxResponse::for_test(empty_ud(), "1.11.1", "eng"),
            AdmittedMorphosyntaxResponse::for_test(empty_ud(), "1.11.1", "eng"),
        ]);
        let comment = morphotag_provenance(&LanguageCode3::eng(), &applied, true).format();
        assert!(
            comment.starts_with(
                "[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng ; retokenize=true | "
            ),
            "{comment}"
        );
        let nothing =
            morphotag_provenance(&LanguageCode3::eng(), &AppliedAnalyses::none(), false).format();
        assert!(
            nothing.starts_with("[fc-ba3 morphotag | lang=eng | "),
            "{nothing}"
        );
    }

    /// One relation repair, as a worker reports it.
    fn a_repair(
        from_relation: &str,
        to_relation: &str,
    ) -> crate::types::worker_v2::UdRelationRepairV2 {
        crate::types::worker_v2::UdRelationRepairV2 {
            kind: crate::types::worker_v2::UdRelationRepairKindV2::RelationAlias,
            word: "ne".to_owned(),
            from_relation: from_relation.to_owned(),
            to_relation: to_relation.to_owned(),
        }
    }

    /// A run that repaired relations counts them; one that repaired none
    /// writes no count at all, because the field's ABSENCE is how the grammar
    /// says "none" and `ud_repairs=0` would be a second spelling of it.
    #[test]
    fn morphotag_provenance_counts_the_relations_it_repaired() {
        use crate::morphosyntax::identity::AdmittedMorphosyntaxResponse;
        let (_, applied) = AppliedAnalyses::take_applied(vec![
            AdmittedMorphosyntaxResponse::for_test_with_repairs(
                empty_ud(),
                "1.11.1",
                "eng",
                vec![a_repair("iob", "iobj"), a_repair("IOB", "iobj")],
            ),
            AdmittedMorphosyntaxResponse::for_test(empty_ud(), "1.11.1", "eng"),
        ]);
        let comment = morphotag_provenance(&LanguageCode3::eng(), &applied, false).format();
        assert!(
            comment.starts_with(
                "[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng ; ud_repairs=2 | "
            ),
            "{comment}"
        );

        let (_, repaired_nothing) =
            AppliedAnalyses::take_applied(vec![AdmittedMorphosyntaxResponse::for_test(
                empty_ud(),
                "1.11.1",
                "eng",
            )]);
        let comment =
            morphotag_provenance(&LanguageCode3::eng(), &repaired_nothing, false).format();
        assert!(!comment.contains("ud_repairs"), "{comment}");
    }

    /// The `--before` document of an incremental morphotag test, carrying a
    /// morphotag stamp (under the legacy name) whose first field is
    /// `engine_field`.
    fn prior_document(engine_field: &str) -> ChatFile {
        let chat = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
             @ID:\teng|test|PAR|||||Participant|||\n\
             @Comment:\t[ba3 morphotag | {engine_field} ; lang=eng | 2026-03-28T10:00:00-04:00]\n\
             *PAR:\thello .\n@End\n"
        );
        let (file, _) = parse_lenient(&crate::chat_parser(), &chat);
        file
    }

    /// What one analyzed utterance reported, as the incremental path collects
    /// it.
    fn applied_that_ran(stanza_version: &str) -> AppliedAnalyses {
        use crate::morphosyntax::identity::AdmittedMorphosyntaxResponse;
        AppliedAnalyses::take_applied(vec![AdmittedMorphosyntaxResponse::for_test(
            empty_ud(),
            stanza_version,
            "eng",
        )])
        .1
    }

    /// The same, having repaired `repairs` relations.
    fn applied_that_repaired(repairs: usize) -> AppliedAnalyses {
        use crate::morphosyntax::identity::AdmittedMorphosyntaxResponse;
        AppliedAnalyses::take_applied(vec![AdmittedMorphosyntaxResponse::for_test_with_repairs(
            empty_ud(),
            "1.11.1",
            "eng",
            vec![a_repair("iob", "iobj"); repairs],
        )])
        .1
    }

    /// An incremental run's output holds tiers from both runs, so its stamp
    /// accounts for both runs' repairs, exactly as `engine=` names both runs'
    /// models. The prior count is read back through the codec and added.
    #[test]
    fn incremental_morphotag_sums_prior_and_new_repair_counts() {
        let stamp = incremental_morphotag_provenance(
            &prior_document("engine=stanza-1.10.0 ; ud_repairs=3"),
            &LanguageCode3::eng(),
            &applied_that_repaired(1),
            false,
        )
        .expect("the prior stamp reads back")
        .expect("a model ran")
        .format();
        assert!(stamp.contains(" ud_repairs=4 "), "{stamp}");

        // A prior stamp with no count states none, so only this run's repairs
        // are counted.
        let stamp = incremental_morphotag_provenance(
            &prior_document("engine=stanza-1.10.0"),
            &LanguageCode3::eng(),
            &applied_that_repaired(2),
            false,
        )
        .expect("the prior stamp reads back")
        .expect("a model ran")
        .format();
        assert!(stamp.contains(" ud_repairs=2 "), "{stamp}");
    }

    /// A prior count that is not a count is refused, rather than read as none:
    /// treating it as zero would silently undercount the written document.
    #[test]
    fn incremental_morphotag_refuses_an_unreadable_prior_repair_count() {
        assert_eq!(
            incremental_morphotag_provenance(
                &prior_document("ud_repairs=many"),
                &LanguageCode3::eng(),
                &applied_that_ran("1.11.1"),
                false,
            )
            .map(|stamp| stamp.map(|stamp| stamp.format())),
            Err(IncrementalMorphotagStampError::UnreadablePriorRepairCount(
                "many".to_owned()
            ))
        );
    }

    /// An incremental run that ran no model writes no stamp, so the document
    /// keeps the one naming the models behind the tiers it kept.
    #[test]
    fn incremental_morphotag_that_ran_no_model_leaves_the_existing_stamp() {
        let prior = prior_document("engine=stanza-1.10.0");
        assert_eq!(
            incremental_morphotag_provenance(
                &prior,
                &LanguageCode3::eng(),
                &AppliedAnalyses::none(),
                false,
            )
            .map(|stamp| stamp.map(|stamp| stamp.format())),
            Ok(None)
        );
    }

    /// An incremental run that reanalyzed utterances names the models the
    /// prior stamp named, read back through the codec, alongside the ones that
    /// ran now. Reading its own stamp back on the next run changes nothing.
    #[test]
    fn incremental_morphotag_names_prior_and_new_models() {
        let expected = "[fc-ba3 morphotag | engine=stanza-1.10.0+stanza-1.11.1:eng:standard ; \
                        incremental=true ; lang=eng | ";
        for prior_engines in [
            "engine=stanza-1.10.0",
            "engine=stanza-1.10.0+stanza-1.11.1:eng:standard",
        ] {
            let stamp = incremental_morphotag_provenance(
                &prior_document(prior_engines),
                &LanguageCode3::eng(),
                &applied_that_ran("1.11.1"),
                false,
            )
            .expect("the prior stamp reads back")
            .expect("a model ran")
            .format();
            assert!(stamp.starts_with(expected), "{prior_engines}: {stamp}");
        }
    }

    /// A prior stamp that does not parse, or a model name its list could not be
    /// read back into, fails the run rather than writing a stamp that
    /// understates or garbles what produced the tiers.
    #[test]
    fn incremental_morphotag_refuses_what_it_cannot_read_back() {
        assert!(matches!(
            incremental_morphotag_provenance(
                &prior_document("engine=stanza-1.10.0 ; lang"),
                &LanguageCode3::eng(),
                &applied_that_ran("1.11.1"),
                false,
            ),
            Err(IncrementalMorphotagStampError::UnparseablePriorStamp(_))
        ));
        assert!(matches!(
            incremental_morphotag_provenance(
                &prior_document("engine=stanza-1.10.0"),
                &LanguageCode3::eng(),
                &applied_that_ran("1.11.1+cpu"),
                false,
            ),
            Err(IncrementalMorphotagStampError::SeparatorInModelName(_))
        ));
    }

    /// Align records the reported FA engine byte for byte, and `utr=` only
    /// when a recovery pass actually ran.
    #[test]
    fn align_provenance_records_utr_only_when_recovery_ran() {
        let fa = FaCacheNamespace::for_test("wave2vec-fa-v1");
        let ran = align_provenance(
            &LanguageCode3::eng(),
            &fa,
            &UtrContribution::Ran(UtrEngine::Whisper),
            false,
            false,
        )
        .format();
        assert!(
            ran.starts_with("[fc-ba3 align | fa=wave2vec-fa-v1 ; lang=eng ; utr=whisper | "),
            "{ran}"
        );

        let mut contribution = UtrContribution::NotRun;
        contribution.record_pass(&UtrEngine::RevAi, &UtrResult::not_run_no_untimed(3));
        assert_eq!(contribution, UtrContribution::NotRun);
        let not_run =
            align_provenance(&LanguageCode3::eng(), &fa, &contribution, false, false).format();
        assert!(
            not_run.starts_with("[fc-ba3 align | fa=wave2vec-fa-v1 ; lang=eng | "),
            "{not_run}"
        );
    }

    // ---- is_provenance_only_difference tests ----
    //
    // The decision predicate that lets the runner skip a disk write when
    // re-running a provenance-injecting command would change only the stamp's
    // name and timestamp (and the warning's build identity). Each test pins
    // one shape of difference we care about.

    /// Re-running morphotag against an unchanged corpus produces a candidate
    /// that differs from the on-disk text only in the stamp's timestamp: do
    /// not write.
    #[test]
    fn provenance_only_diff_detects_timestamp_only_change() {
        let old_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
*PAR:\thello .
%mor:\tco|hello .
@End
";
        let new_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tco|hello .
@End
";
        assert!(is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// A stamp whose ENGINE changed is a meaningful difference: the file is
    /// written, so the stamp names the model that produced it now. Before,
    /// the whole stamp line was ignored and a new engine never reached disk.
    #[test]
    fn provenance_only_diff_treats_a_changed_engine_as_meaningful() {
        let old_text = "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]\n*PAR:\thello .\n";
        let new_text = "@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-09-15T02:52:17-04:00]\n*PAR:\thello .\n";
        assert!(!is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
        // So is an added or removed flag.
        let flagged = "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng ; retokenize=true | 2026-09-15T02:52:17-04:00]\n*PAR:\thello .\n";
        assert!(!is_provenance_only_difference(
            old_text,
            flagged,
            ReleasedCommand::Morphotag,
        ));
    }

    /// A file stamped by a build before the rename, re-run by this build with
    /// the same fields: the only difference is the stamp's name and
    /// timestamp, so nothing is written. The rename alone never churns a
    /// corpus.
    #[test]
    fn provenance_only_diff_accepts_legacy_stamp_against_current_stamp() {
        let old_text = "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]\n*PAR:\thello .\n";
        let new_text = "@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-09-15T02:52:17-04:00]\n*PAR:\thello .\n";
        assert!(is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
        // Under either name, another command's stamp is still a real change.
        let other_command =
            "@Comment:\t[fc-ba3 align | fa=wave2vec | 2026-09-15T02:52:17-04:00]\n*PAR:\thello .\n";
        assert!(!is_provenance_only_difference(
            old_text,
            other_command,
            ReleasedCommand::Morphotag,
        ));
    }

    /// Our unchecked-ASR warning is compared like a stamp: a new build
    /// identity or the legacy-to-current shape change alone is not worth a
    /// write, but a different ASR engine is.
    #[test]
    fn provenance_only_diff_compares_our_warning_by_engine() {
        let old_text = "\
@Comment:\t[ba3 transcribe | asr=rev ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:\tBatchalign 0.8.5, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
";
        let rebuilt = "\
@Comment:\t[fc-ba3 transcribe | asr=rev ; lang=eng | 2026-09-15T18:30:00-04:00]
@Comment:\tfc-ba3 new-build-1, ASR engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
";
        assert!(is_provenance_only_difference(
            old_text,
            rebuilt,
            ReleasedCommand::Transcribe,
        ));
        let other_engine = "\
@Comment:\t[fc-ba3 transcribe | asr=rev ; lang=eng | 2026-09-15T18:30:00-04:00]
@Comment:\tfc-ba3 new-build-1, ASR engine whisper. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
";
        assert!(!is_provenance_only_difference(
            old_text,
            other_engine,
            ReleasedCommand::Transcribe,
        ));
    }

    /// A stamp for the command that does not parse is not suppressed: the
    /// write goes through and replaces it with a stamp that does.
    #[test]
    fn provenance_only_diff_writes_over_an_unparseable_stamp() {
        let old_text = "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang | 2026-03-29T18:30:00-04:00]\n*PAR:\thello .\n";
        let new_text = "@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-09-15T02:52:17-04:00]\n*PAR:\thello .\n";
        assert!(!is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// A damaged stamp belonging to a command this job does not run must NOT
    /// provoke a rewrite.
    ///
    /// This is the property the two-step recognizer exists to keep. Sending the
    /// gate through a recognizer that only ever parses a stamp whole would make
    /// EVERY damaged stamp its business, and it would rewrite every file in a
    /// corpus carrying one, repairing nothing (replacement matches a stamp by
    /// its command) and churning history for no change. Here only morphotag's
    /// timestamp moved: the damaged align stamp is identical on both sides, and
    /// so is a stamp too damaged to name any command at all.
    #[test]
    fn provenance_only_diff_leaves_a_damaged_stamp_of_another_command_alone() {
        let untouched = "@Comment:\t[ba3 align | fa ; lang=eng | 2026-03-29T19:15:00-04:00]\n\
                         @Comment:\t[fc-ba3 align]\n";
        let old_text = format!(
            "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]\n\
             {untouched}*PAR:\thello .\n"
        );
        let new_text = format!(
            "@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-09-15T02:52:17-04:00]\n\
             {untouched}*PAR:\thello .\n"
        );
        assert!(is_provenance_only_difference(
            &old_text,
            &new_text,
            ReleasedCommand::Morphotag,
        ));

        // The same damage in the stamp the job itself writes IS its business.
        let damaged_own = old_text.replace("engine=stanza-1.11.1 ; lang=eng", "engine ; lang=eng");
        assert!(!is_provenance_only_difference(
            &damaged_own,
            &new_text,
            ReleasedCommand::Morphotag,
        ));

        // A damaged stamp that DIFFERS between the sides is an ordinary content
        // difference, so the write goes through.
        let edited = new_text.replace("fa ; lang=eng", "fa ; lang=spa");
        assert!(!is_provenance_only_difference(
            &old_text,
            &edited,
            ReleasedCommand::Morphotag,
        ));
    }

    /// Real %mor content change must NOT be classified as
    /// provenance-only. The predicate's whole purpose is to preserve
    /// real updates while suppressing pointless ones.
    #[test]
    fn provenance_only_diff_returns_false_for_mor_change() {
        let old_text = "\
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
*PAR:\thello .
%mor:\tco|hello .
";
        let new_text = "\
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tco|hi .
";
        assert!(!is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// Identical input should never be flagged; there is literally no
    /// difference to suppress. The predicate is a "diff-only-in-X"
    /// detector, not a "skip the write because everything matches"
    /// shortcut (the caller can byte-compare for that).
    #[test]
    fn provenance_only_diff_returns_false_for_identical_text() {
        let text = "@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]\n*PAR:\thello .\n";
        assert!(!is_provenance_only_difference(
            text,
            text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// When checking whether morphotag would write pointlessly, only the
    /// morphotag stamp is allowed to differ. A diff in the align stamp is a
    /// real change from morphotag's perspective and must not be hidden,
    /// otherwise re-running morphotag could clobber an unrelated align update
    /// on disk.
    #[test]
    fn provenance_only_diff_returns_false_when_other_commands_provenance_differs() {
        let old_text = "\
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:\t[ba3 align | fa=whisper-fa-large-v2 ; lang=eng | 2026-03-29T19:15:00-04:00]
*PAR:\thello .
";
        let new_text = "\
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-05-08T02:52:17-04:00]
@Comment:\t[ba3 align | fa=whisper-fa-large-v2 ; lang=eng | 2026-05-08T03:00:00-04:00]
*PAR:\thello .
";
        assert!(!is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// First-run case: file had no morphotag provenance before, and
    /// running morphotag now adds both the stamp AND fresh %mor tiers. The
    /// %mor addition is a real content change; the predicate must return
    /// false even though one side has no morphotag stamp to set aside.
    #[test]
    fn provenance_only_diff_returns_false_on_first_run_with_real_content_added() {
        let old_text = "\
@UTF8
*PAR:\thello .
@End
";
        let new_text = "\
@UTF8
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-05-08T02:52:17-04:00]
*PAR:\thello .
%mor:\tco|hello .
@End
";
        assert!(!is_provenance_only_difference(
            old_text,
            new_text,
            ReleasedCommand::Morphotag,
        ));
    }

    /// A transcribe job writes the `transcribe`, `utseg` and `morphotag`
    /// stamps (its recipe composes all three stages) and our warning. Re-run
    /// on a new build over identical content, each changes only in name,
    /// timestamp or build identity, so nothing is written. Setting aside only
    /// the `transcribe` stamp made every such re-run a write.
    #[test]
    fn provenance_only_diff_sets_aside_every_stamp_a_transcribe_job_writes() {
        let old_text = "\
@UTF8
@Begin
@Comment:\t[ba3 transcribe | asr=rev ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:\t[ba3 utseg | engine=talkbank/utterance-boundary@revision-1 ; lang=eng | 2026-03-29T18:31:00-04:00]
@Comment:\t[ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-03-29T18:32:00-04:00]
@Comment:\tBatchalign 0.8.5, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
%mor:\tintj|hello .
@End
";
        let rebuilt = "\
@UTF8
@Begin
@Comment:\t[fc-ba3 transcribe | asr=rev ; lang=eng | 2026-09-15T18:30:00-04:00]
@Comment:\t[fc-ba3 utseg | engine=talkbank/utterance-boundary@revision-1 ; lang=eng | 2026-09-15T18:31:00-04:00]
@Comment:\t[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-09-15T18:32:00-04:00]
@Comment:\tfc-ba3 new-build-2, ASR engine rev. Unchecked output of ASR model, DO NOT USE.
*PAR:\thello .
%mor:\tintj|hello .
@End
";
        assert!(is_provenance_only_difference(
            old_text,
            rebuilt,
            ReleasedCommand::Transcribe,
        ));

        // A composed stage's changed field is still meaningful.
        let new_model = rebuilt.replace("stanza-1.11.1:eng:standard", "stanza-1.12.0:eng:standard");
        assert!(!is_provenance_only_difference(
            old_text,
            &new_model,
            ReleasedCommand::Transcribe,
        ));

        // A stamp for a command the job does not compose is not set aside.
        let with_align = rebuilt.replace(
            "*PAR:",
            "@Comment:\t[fc-ba3 align | fa=wave2vec-fa-v1 ; lang=eng | 2026-09-15T18:33:00-04:00]\n*PAR:",
        );
        assert!(!is_provenance_only_difference(
            old_text,
            &with_align,
            ReleasedCommand::Transcribe,
        ));

        // A morphotag job composes no utseg stage, so a changed utseg stamp is
        // a real change from its point of view.
        let new_utseg = rebuilt.replace("revision-1", "revision-2");
        assert!(!is_provenance_only_difference(
            rebuilt,
            &new_utseg,
            ReleasedCommand::Morphotag,
        ));
    }

    /// A stamp far longer than a CHAT line stays one `@Comment` line and reads
    /// back unchanged. The serializer breaks bullet content only at explicit
    /// continuation segments, a stamp is injected as one text segment, and
    /// stamp-safe text cannot contain a line break, so nothing can wrap it.
    #[test]
    fn a_long_stamp_is_written_on_one_line_and_reads_back() {
        let name = format!("long model {}", "segment ".repeat(80).trim_end());
        let engine = ReportedEngineName::try_from(name.as_str()).expect("stamp-safe name");
        let TextStamp::Stamped(comment) = result_named_provenance(
            ResultNamedCommand::Translate,
            &LanguageCode3::eng(),
            [&engine],
        ) else {
            panic!("names an engine");
        };
        let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
                    @ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";
        let written = inject_provenance_into_text(chat, &comment).expect("valid CHAT");
        // Re-parsed and written again, the stamp is still one line.
        let rewritten = inject_provenance_into_text(&written, &comment).expect("valid CHAT");
        for text in [&written, &rewritten] {
            let stamp_lines: Vec<&str> = text
                .lines()
                .filter(|line| line.contains("fc-ba3 translate"))
                .collect();
            assert_eq!(stamp_lines.len(), 1, "{text}");
            assert!(
                stamp_lines[0].starts_with("@Comment:\t[fc-ba3 translate | engine=long model ")
                    && stamp_lines[0].ends_with(']'),
                "{text}"
            );
            let entries = extract_provenance(text).expect("the stamp parses");
            assert_eq!(entries.len(), 1, "{text}");
            assert_eq!(entries[0].fields["engine"], name);
        }
    }
}
