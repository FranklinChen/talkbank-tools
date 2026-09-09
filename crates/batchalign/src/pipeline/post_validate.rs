//! The post-validation gate: the only route from a finished CHAT model to
//! bytes a command is allowed to write.
//!
//! # Why this exists
//!
//! Every text seam used to run `validate_output(&chat, command)` and, on
//! failure, emit `warn!(... "post-validation warnings (non-fatal)")` and write
//! the file anyway. The rule that a command must not corrupt its own input was
//! therefore enforced by nothing: a `warn!` is where lost information goes to
//! look like it was handled. A file whose `%mor` had drifted out of alignment,
//! or whose terminator a transform had eaten, still landed on disk and still
//! reported success to the operator.
//!
//! [`PostValidated`] replaces that convention with a type. It owns the
//! document whose bytes a command may write, and the only way to obtain one is
//! to have passed the gate (or to be an explicit, named pass-through). The
//! bytes it hands out are serialized from the judged model and from nothing
//! else, so validating one document and writing another has no route. The CHAT
//! writer seams
//! (`recipe_runner::runtime::write_chat_output_artifact_with_provenance_gate`
//! and `runner::dispatch::audio_output::write_primary_chat_output_artifact`)
//! take a `PostValidated` rather than a bare `String`/`ChatFile`, so "write
//! output that failed validation" has no signature to travel through.
//!
//! That sentence was FALSE for the seam that touches disk until 2026-09-07:
//! `write_chat_output_artifact_with_provenance_gate` took `content: &str`, so
//! the proof stopped one call short of the write and every caller could have
//! handed it any string at all. It takes the proof now.
//!
//! `compare` and `benchmark` write their CHAT-shaped primary output through
//! `write_text_output_artifact` rather than through the CHAT writer seams, so
//! they are not covered by the "no signature to travel through" sentence above
//! and must reach the gate themselves. They do: `compare::materialize_released`
//! and `compare::materialize_main_annotated` hold the `ChatFile` they built and
//! hand back a [`PostValidated`], and their two writers
//! (`execution::kernel`'s `MaterializeOutputs` stage and
//! `runner::dispatch::benchmark_pipeline`) write `as_str()` off that proof.
//!
//! This paragraph said the opposite until 2026-09-07 ("that command builds no
//! `ChatFile` to gate"), which was false where it mattered: both materializers
//! had one in hand and called `to_chat_string` on it, and both writers then ran
//! `merge_abbreviations_in_chat_text` over the finished TEXT and wrote THAT, so
//! the bytes on disk were two transforms removed from anything judged.
//!
//! # The proof names its command, once
//!
//! A `PostValidated` carries the [`ReleasedCommand`] whose output it is, and
//! the writer reads it off the proof. It used to hold a `&'static str` while
//! the writer took a separate `ReleasedCommand` argument, so one command was
//! named twice in two types and the two could disagree: a document gated as
//! `"transcribe"` was written under whatever the job dispatched, and for a
//! `transcribe_s` job the provenance suppression therefore looked for a
//! `[ba3 transcribe_s |` line that the transcribe pipeline never writes.
//!
//! # The rule the gate enforces, and it depends on what was ADMITTED
//!
//! A command that admitted its input at a [`ValidityLevel`] promised not to
//! hand back something worse, so its output must satisfy BOTH of:
//!
//! 1. [`validate_to_level`] at that same level.
//! 2. [`validate_output`] for that command, which adds the command-specific
//!    checks (`%mor` item count for morphotag, backwards bullets for align)
//!    plus a terminator check over every utterance.
//!
//! A command that admitted its input at NO level can promise neither. Check 2
//! is the trap, because it looks like a preservation check and is not: its
//! terminator loop runs for EVERY command, before and independently of the
//! match on the command name, so what it actually asks is "does this document
//! have terminators", which is a property of the INPUT. Run against a document
//! nothing vouched for, it refuses files the command never damaged. That is
//! what compare and benchmark did between the introduction of their gate and
//! 2026-09-07: a gold companion carrying one terminator-less utterance had
//! always been written, and became a hard refusal with no output at all. The
//! level those two chose could not save them either, since L0 with no parse
//! errors to report is vacuous.
//!
//! So there are two judgements, and [`Judgement`] is which one a proof made:
//! `Admitted` for the pair above, and `Preserved` for the only claim available
//! about an output whose input nothing vouched for, which is that it did not
//! LOSE what that input had. Each transition below re-runs the judgement its
//! proof RECORDS rather than restating one, so the abbreviation merge and the
//! provenance stamp are judged exactly the way the document that reached them
//! was.
//!
//! No `parse_errors` are supplied to the level check: the output was BUILT
//! from an admitted model, never re-parsed, so there are no parse errors to
//! report and re-parsing our own serialized output is banned. L0 is therefore
//! vacuously satisfied here, which is correct: the model exists.
//!
//! That sentence was FALSE for the two transitions until 2026-09-07. Each of
//! them re-parsed the proof's own text with `parse_lenient`, discarded the
//! resulting `parse_errors` into a `_`, applied its transform to the recovered
//! model, and re-gated with `&[]`. So the banned re-parse happened, the L0
//! rung it fed was vacuous for a reason the module denied, and any
//! disagreement between the model the gate judged and what its bytes parse
//! back to was silently adopted. A `Gated` proof now IS the model, so each
//! transition edits it, re-judges it, and serializes nothing; there is no
//! parser in either signature for a re-parse to travel through, and no
//! serializer either. The bytes are made once, where they are wanted:
//! [`OutputProof::Gated::text`].
//!
//! # Every route to a `PostValidated`, and why each is allowed
//!
//! This is the weakest-constructor enumeration required of any proof type.
//! There are exactly eight, and no other constructor exists:
//!
//! - [`PostValidated::gate_owned`] is THE evidence path. It runs both checks
//!   and keeps the model they judged. Everything that produces command output
//!   arrives here.
//! - [`PostValidated::gate`] is the same function for a caller that only
//!   BORROWS its model: it clones and delegates, and it is a separate name so
//!   that the clone is visible at the call site rather than charged to every
//!   caller. It grants no privilege `gate_owned` does not, so it does not
//!   widen the proof.
//! - [`PostValidated::preserving`] is the evidence path for a command whose
//!   INPUT was never admitted at any level. It judges the output against a
//!   census of that input rather than against a bar, so it CAN refuse a
//!   command that destroyed something and CANNOT refuse a document that
//!   arrived that way. Its input half is a [`UtteranceCensus`], whose only
//!   constructor reads a real `ChatFile`, so a caller cannot state what an
//!   input had without having held it.
//! - [`PostValidated::pass_through`] is for a document the command did NOT
//!   modify: a `@Options: dummy` file, a CA file the command declines to
//!   touch, a file with no collectible payloads. Nothing was applied, so
//!   nothing can have been corrupted, and gating it would instead re-judge the
//!   INPUT against a bar the input never had to meet (a dummy file has no
//!   `@Participants`, so it fails L1 while being exactly the bytes the user
//!   handed us). The proof it carries is "unchanged", not "valid", and it
//!   takes the ORIGINAL INPUT TEXT rather than the parsed model, so
//!   "unchanged" is literally true: it used to re-serialize the model, which
//!   ships a document the docs call unchanged through a serializer no gate
//!   judged, so any parse-then-serialize difference reached disk unexamined.
//! - [`PostValidated::declined_stripping_decision_tiers`] is the neighbouring
//!   fact for a document the command declined to process but whose input bytes
//!   are not the answer: the model has been EDITED by the command's own
//!   declared no-op, and there is no text that says what it now holds. That
//!   no-op is the decision-tier strip morphotag owes a file an earlier align
//!   run wrote `%xalign` / `%xrev` onto, and the constructor PERFORMS it rather
//!   than trusting three call sites to have done it first. It is still not
//!   gated, for the same reason a pass-through is not. Align used this too
//!   until 2026-09-07, for a `@Options: dummy` / `NoAlign` document it changes
//!   NOTHING about, and re-serializing was therefore the wrong answer there;
//!   its entry point now carries the original text and takes `pass_through`.
//! - [`PostValidated::with_provenance_injected`] consumes a proof, stamps the
//!   run's `[ba3 ...]` comment onto the MODEL IT CARRIES, and re-runs the
//!   gate. Same
//!   shape as the merge below and for the same reason: the stamp used to run
//!   on the finished TEXT at the dispatch seam, so the bytes written carried a
//!   line no gate had judged. A pass-through is returned untouched, which is
//!   what makes `NoAlign`'s documented "zero modifications" true.
//! - [`PostValidated::with_abbreviations_merged`] consumes a proof and returns
//!   another, having applied the abbreviation merge to the MODEL IT CARRIES
//!   and re-run the SAME judgement over the merged model, for the same
//!   command. It exists because the merge used to run on the proven TEXT at
//!   the writer, and the writer then wrote the merge's output: the bytes on
//!   disk were not the bytes any gate had judged. The merge is now a step
//!   INSIDE the gate, so those two are the same bytes again. A refusal returns
//!   the UNMERGED proof alongside the failure, because the merge is cosmetic
//!   and losing an admissible output to it is not a policy anyone chose.
//! - [`PostValidated::for_test`] is `#[cfg(test)]` only. It exists for the
//!   fake `WorkerGateway` doubles, which stand in for a gateway whose real
//!   implementation has already run the gate; they synthesize output text
//!   without ever building a `ChatFile`. It is the ONE place here that parses,
//!   and it parses because a `Gated` proof carries a model: the double has to
//!   stand up the same pair the real gateway hands back, and a test-only parse
//!   of a string the test itself wrote is not the banned re-parse of a
//!   command's own output.

use std::sync::OnceLock;

use crate::api::ReleasedCommand;
use batchalign_transform::merge_abbreviations;
use batchalign_transform::serialize::to_chat_string;
use batchalign_transform::validate::{
    ValidationError, ValidityLevel, validate_output, validate_to_level,
};
use talkbank_model::ChatFile;
use talkbank_model::model::Line;

/// A command's output that has passed the post-validation gate, together with
/// the document that passed it.
///
/// Possession of one of these IS the proof. The bytes it hands out are
/// serialized FROM the judged model and from nothing else, so a caller cannot
/// validate one model and write another.
///
/// `Clone` and `PartialEq` are written out rather than derived because
/// [`OutputProof::Gated`] carries a materialization cache, and a cache is not
/// part of the document's identity. See the impls below.
#[derive(Debug)]
pub(crate) struct PostValidated {
    /// The command whose output this document is.
    ///
    /// Held here rather than inside [`OutputProof::Gated`] because it is true
    /// of every route: an unchanged document is still SOME command's output,
    /// and the writer needs the name whichever route produced the bytes.
    command: ReleasedCommand,
    /// What makes this document admissible, and at what bar.
    proof: OutputProof,
}

/// Why a [`PostValidated`]'s bytes may be written.
///
/// A sum rather than an `Option<Judgement>` beside a comment: "judged" and
/// "the command applied nothing" are different facts, and a later step that
/// wants to re-judge (the abbreviation merge, the provenance stamp) must ask
/// the proof what it claimed rather than restate one. Restating it is exactly
/// how the FA fast paths came to be judged at L1 against an input admitted at
/// L2.
#[derive(Debug)]
enum OutputProof {
    /// The model passed [`PostValidated::judge`] under this judgement, for
    /// this command, and the bytes are what it serializes to.
    Gated {
        /// What was CLAIMED about this model, and therefore what a transition
        /// on it must re-establish.
        judgement: Judgement,
        /// The model the gate judged, and the model the bytes are serialized
        /// from.
        ///
        /// Carried HERE rather than beside the proof because only a gated
        /// document has one: a pass-through's bytes are the input's own and
        /// there is no output model to transition. An `Option<ChatFile>` next
        /// to the sum would have made "gated with no model" a state the
        /// transitions had to handle and could get wrong.
        ///
        /// It is what lets the two transitions below run without a parser.
        /// Boxed because a `ChatFile` is far larger than the other fields and
        /// every `Ungated` proof would otherwise pay for it.
        file: Box<ChatFile>,
        /// The serialization of `file`, made the first time anyone asks for
        /// bytes and never again.
        ///
        /// # Why it is a cache and not a field
        ///
        /// The gate used to serialize eagerly, and a gated document is gated
        /// three times on the align path (`FaAdmission::finish`, then the
        /// provenance stamp, then the abbreviation merge), so an aligned file
        /// paid three whole-document serializations of which the first two
        /// were dropped unread. Validation is what every transition owes; a
        /// serialization is what the WRITER needs, once, at the end. Splitting
        /// them is what this cell is for.
        ///
        /// It is a `OnceLock` rather than a plain field so that
        /// [`PostValidated::as_str`] can keep handing out a borrow: the bytes
        /// are still derived from exactly the model the gate judged, and they
        /// are still derived at most once per proof.
        text: OnceLock<String>,
    },
    /// Final bytes that no gate judged and that carry no model to transition.
    ///
    /// The variant was called `Unchanged` until 2026-09-07 and that was a
    /// false statement for one of its two constructors:
    /// [`PostValidated::declined_stripping_decision_tiers`] EDITS the model
    /// before serializing it, so its bytes are not the input's and the
    /// document is not unchanged. What both constructors share, and all this
    /// variant claims, is that the bytes are final, there is no output model,
    /// and nothing judged them.
    ///
    /// It carried an `origin` field naming which of the two constructors made
    /// it, added and removed on 2026-09-07. Nothing ever read it but
    /// `PartialEq` and the one test written to observe that: no behaviour
    /// anywhere differs between the two origins, both being returned untouched
    /// by every transition, so the field made no wrong value unrepresentable
    /// and was a second thing to keep true. The false claim was in the
    /// variant's NAME, and renaming it is the whole of the fix; what each
    /// constructor did is stated on the constructor, where a reader is.
    Ungated {
        /// The final bytes.
        text: String,
    },
}

/// What a proof CLAIMS about the document it carries.
///
/// Which one a command can make is decided by what it ADMITTED, and the module
/// docs above give the full argument. The short form: a bar is a claim about
/// the OUTPUT alone and is only fair when the input had to clear the same bar;
/// preservation is a claim about the COMMAND, and is the only one available
/// when nothing vouched for the input.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Judgement {
    /// The input was admitted at this level, so the output must still satisfy
    /// it and must pass the command's own output checks.
    Admitted(ValidityLevel),
    /// Nothing admitted the input at any level, so the output is judged
    /// against WHAT THAT INPUT HAD, and this is the census of it.
    Preserved(UtteranceCensus),
}

impl Judgement {
    /// The bar a refusal reports.
    ///
    /// A projection rather than a second copy: a failure message needs to say
    /// which claim failed and has no business carrying the census it was
    /// measured against, and deriving it here means a new judgement must state
    /// how it renders.
    fn bar(&self) -> JudgementBar {
        match self {
            Self::Admitted(level) => JudgementBar::Level(*level),
            Self::Preserved(_) => JudgementBar::Preservation,
        }
    }
}

/// The bar an output was judged against, as a refusal reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgementBar {
    /// The level its input was admitted at, which the output must still meet.
    Level(ValidityLevel),
    /// What its input HAD, because nothing admitted that input.
    Preservation,
}

impl std::fmt::Display for JudgementBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Level(level) => write!(f, "output must still satisfy L{}", *level as u8),
            Self::Preservation => write!(f, "output must not lose what the input had"),
        }
    }
}

/// What a document's utterances WERE, before a command touched them.
///
/// # Why it exists
///
/// It is the input half of a preservation judgement. "Every utterance in this
/// document has a terminator" is a fact about the INPUT, and refusing an
/// output for failing it punishes a command for what it was handed. "This
/// output still has every terminator its input had" is a fact about the
/// COMMAND, and it is the one compare and benchmark are answerable for.
///
/// # Every route to one
///
/// [`UtteranceCensus::of`] and nothing else: the fields are private and no
/// other constructor exists, so a caller cannot state what an input had
/// without holding that input. That matters in both directions, because a
/// census decides acquittals as well as convictions: a forged empty one would
/// admit anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtteranceCensus {
    /// One entry per main tier, in document order.
    utterances: Vec<CensusEntry>,
}

/// One utterance, in as much detail as preservation is about.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CensusEntry {
    /// The speaker code. Carried so the positional correspondence between the
    /// two documents is CHECKED rather than assumed; see
    /// [`UtteranceCensus::losses_in`].
    speaker: String,
    /// Whether this utterance carried a terminator.
    has_terminator: bool,
}

impl UtteranceCensus {
    /// Take the census of a document.
    pub(crate) fn of(file: &ChatFile) -> Self {
        Self {
            utterances: file
                .lines
                .iter()
                .filter_map(|line| match line {
                    Line::Utterance(utt) => Some(CensusEntry {
                        speaker: utt.main.speaker.as_str().to_owned(),
                        has_terminator: utt.main.content.terminator.is_some(),
                    }),
                    // Headers and free-standing tiers are not what preservation
                    // is about here, and a command that adds one (compare's
                    // `@Media` line) must not read as a changed document.
                    _ => None,
                })
                .collect(),
        }
    }

    /// Everything `output` LOST relative to this census.
    ///
    /// The comparison is positional, and the speaker check is what makes that
    /// legitimate rather than assumed: the commands judged this way edit
    /// headers and dependent tiers, so utterance N of the output is utterance
    /// N of the input, and on the day that stops being true the mismatch is
    /// REPORTED instead of two unrelated utterances being silently compared.
    /// An output with MORE utterances than its input has lost nothing by that
    /// alone, so only a shortfall is a finding.
    fn losses_in(&self, output: &ChatFile, command: ReleasedCommand) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        // Walked lazily rather than censused: a second census would allocate a
        // speaker `String` per utterance of a document this only ever reads.
        let mut after = output.lines.iter().filter_map(|line| match line {
            Line::Utterance(utt) => Some(utt),
            _ => None,
        });
        let mut kept = 0usize;

        for before in &self.utterances {
            let Some(now) = after.next() else {
                // The output ran out, so `kept` is its whole length and every
                // remaining input utterance is missing from it.
                break;
            };
            kept += 1;

            let speaker = now.main.speaker.as_str();
            if before.speaker != speaker {
                errors.push(ValidationError {
                    message: format!(
                        "After {command}: an utterance by *{} is now by *{speaker}, so the \
                         output no longer lines up with the input",
                        before.speaker
                    ),
                    level: ValidityLevel::StructurallyComplete,
                });
                // These are not the same utterance, so anything further this
                // pair could be asked is meaningless.
                continue;
            }
            if before.has_terminator && now.main.content.terminator.is_none() {
                // Word for word what `validate_output` says for the same
                // event, because it IS the same event to an operator; what
                // changed is only when it counts as one.
                errors.push(ValidationError {
                    message: format!(
                        "After {command}: utterance by *{} lost its terminator",
                        before.speaker
                    ),
                    level: ValidityLevel::StructurallyComplete,
                });
            }
        }

        if kept < self.utterances.len() {
            errors.push(ValidationError {
                message: format!(
                    "After {command}: the output kept {kept} of the input's {} utterances",
                    self.utterances.len()
                ),
                level: ValidityLevel::StructurallyComplete,
            });
        }

        errors
    }
}

impl Clone for PostValidated {
    fn clone(&self) -> Self {
        Self {
            command: self.command,
            proof: self.proof.clone(),
        }
    }
}

impl Clone for OutputProof {
    fn clone(&self) -> Self {
        match self {
            Self::Gated {
                judgement,
                file,
                text,
            } => Self::Gated {
                judgement: judgement.clone(),
                file: file.clone(),
                // Carried across so a clone of an already-materialized proof
                // does not re-serialize, and so that a clone of a `for_test`
                // proof keeps the bytes the double supplied rather than
                // deriving fresh ones from its parse.
                text: match text.get() {
                    Some(materialized) => {
                        let cell = OnceLock::new();
                        // Infallible on a cell made one line above; the
                        // `Result` is dropped rather than unwrapped because
                        // this module does not panic.
                        let _ = cell.set(materialized.clone());
                        cell
                    }
                    None => OnceLock::new(),
                },
            },
            Self::Ungated { text } => Self::Ungated { text: text.clone() },
        }
    }
}

impl PartialEq for PostValidated {
    /// Two proofs are equal when they are the same command's output over the
    /// same document.
    ///
    /// The materialization cache is deliberately not compared: it is DERIVED
    /// from the model, so whether one side happens to have been serialized
    /// already is a fact about who has read it, not about the documents.
    fn eq(&self, other: &Self) -> bool {
        self.command == other.command && self.proof == other.proof
    }
}

impl PartialEq for OutputProof {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Gated {
                    judgement,
                    file: mine,
                    ..
                },
                Self::Gated {
                    judgement: other_judgement,
                    file: theirs,
                    ..
                },
                // The judgement is compared as well as the document: what a
                // proof CLAIMS is part of what it is, so the same bytes
                // admitted on different grounds are not the same proof.
            ) => judgement == other_judgement && mine == theirs,
            (Self::Ungated { text: mine }, Self::Ungated { text: theirs }) => mine == theirs,
            // Written out rather than a catch-all so a third route to a proof
            // has to say what it compares equal to.
            (Self::Gated { .. }, Self::Ungated { .. })
            | (Self::Ungated { .. }, Self::Gated { .. }) => false,
        }
    }
}

/// Why a command's output was refused.
///
/// Carries the bar the output was judged against, the command whose output
/// failed, and every failure the judgement found, so the operator sees what
/// broke rather than a count. Rendered into the per-file error a job reports.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{command} post-validation failed ({bar}): {}", render_errors(.errors))]
pub(crate) struct PostValidationFailure {
    /// The command whose own output failed its gate.
    pub(crate) command: ReleasedCommand,
    /// What the output was judged against. A level when the input was admitted
    /// at one, preservation when nothing admitted it; the message says which,
    /// so a reader is never told a document failed a bar it was never held to.
    pub(crate) bar: JudgementBar,
    /// Every failure the judgement found, in the order they were found.
    pub(crate) errors: Vec<ValidationError>,
}

/// The abbreviation merge produced a document that fails the judgement its
/// unmerged predecessor passed.
///
/// # Why the unmerged proof is in here
///
/// The transition CONSUMES the proof it merges, so without this the caller
/// could not fall back to the admissible output it was already holding, and a
/// cosmetic flag could turn a file with a valid judged output into a total
/// per-file failure with nothing written at all. That is reachable rather than
/// theoretical: `merge_abbreviations` edits the main tier and does not touch
/// `%mor`, so collapsing `F B I` into `FBI` in a morphotag output leaves three
/// `%mor` items against one word, which is exactly what `validate_output`
/// refuses. Pinned by
/// `a_merge_that_breaks_the_gate_hands_back_the_unmerged_proof`.
///
/// Deciding what to DO about it is the caller's, not this type's: writing the
/// unmerged output and failing the file are both defensible, and they differ
/// per seam. All five call sites currently fail the file, and each says so.
#[derive(Debug, thiserror::Error)]
#[error(
    "the abbreviation merge produced a document that fails its own gate \
         (the unmerged output is still admissible): {failure}"
)]
pub(crate) struct AbbreviationMergeRefused {
    /// The proof the merge was applied TO. Still admissible, still writable.
    pub(crate) unmerged: PostValidated,
    /// Why the MERGED document was refused.
    pub(crate) failure: PostValidationFailure,
}

/// Join gate failures into one line, matching the `; `-separated shape the
/// pre-validation gates already report.
fn render_errors(errors: &[ValidationError]) -> String {
    errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

impl PostValidated {
    /// Run the post-validation gate over a command's finished output.
    ///
    /// `level` must be the level the INPUT was admitted at; passing a lower
    /// one would let a command silently degrade the file it was given.
    ///
    /// This is the BORROWING form, and it is a one-clone wrapper over
    /// [`Self::gate_owned`]: the proof has to keep the model it judged, so a
    /// caller that only lends one forces a copy. Every caller that owns its
    /// model and drops it straight afterwards should call `gate_owned`
    /// instead; the two transitions below do, and so used to pay a
    /// whole-document clone to hand back a model they had just been given.
    pub(crate) fn gate(
        file: &ChatFile,
        level: ValidityLevel,
        command: ReleasedCommand,
    ) -> Result<Self, PostValidationFailure> {
        Self::gate_owned(file.clone(), level, command)
    }

    /// [`Self::gate`] for a caller that OWNS the model it is having judged.
    ///
    /// The evidence path proper. Nothing is serialized here: validation is
    /// what a transition owes, and the bytes are derived from this model when
    /// somebody actually asks for them. See [`OutputProof::Gated::text`].
    pub(crate) fn gate_owned(
        file: ChatFile,
        level: ValidityLevel,
        command: ReleasedCommand,
    ) -> Result<Self, PostValidationFailure> {
        Self::judge(file, Judgement::Admitted(level), command)
    }

    /// Admit the output of a command whose INPUT nothing admitted, on the only
    /// ground such a command has: PRESERVATION.
    ///
    /// `input` is the census of the document the output DESCENDS from, taken
    /// before the command edited it. The output is refused only for what the
    /// command destroyed, so a document that arrived already missing a
    /// terminator is still written, and a transform that drops one is not.
    ///
    /// # Why this route runs no `validate_output`
    ///
    /// Because that function's terminator loop is precisely the input-property
    /// check this route exists to replace, and its command-specific arms
    /// (`morphotag`, `align`) are unreachable for the commands that come here:
    /// compare and benchmark fall through its `match` to `_ => {}`. A command
    /// that arrives here later and DOES owe an output-specific check must add
    /// it to the census judgement rather than reach for the level one, which
    /// would refuse its input's own faults all over again.
    pub(crate) fn preserving(
        input: UtteranceCensus,
        output: ChatFile,
        command: ReleasedCommand,
    ) -> Result<Self, PostValidationFailure> {
        Self::judge(output, Judgement::Preserved(input), command)
    }

    /// Run one [`Judgement`] over a model and keep the model it judged.
    ///
    /// The single place a `Gated` proof is built, so the judgement a proof
    /// records is always the judgement that was actually run over it, and the
    /// two transitions below re-run it by handing back the same value.
    fn judge(
        file: ChatFile,
        judgement: Judgement,
        command: ReleasedCommand,
    ) -> Result<Self, PostValidationFailure> {
        let errors: Vec<ValidationError> = match &judgement {
            Judgement::Admitted(level) => {
                let mut errors = Vec::new();
                // (a) No degradation below the admission bar. The output was
                // built from a model, not parsed from text, so there are no
                // parse errors.
                if let Err(level_errors) = validate_to_level(&file, &[], *level) {
                    errors.extend(level_errors);
                }
                // (b) The command-specific output checks.
                if let Err(output_errors) = validate_output(&file, command.as_str()) {
                    errors.extend(output_errors);
                }
                errors
            }
            // (c) Nothing admitted the input, so the only question is what
            // this command destroyed.
            Judgement::Preserved(input) => input.losses_in(&file, command),
        };

        if errors.is_empty() {
            Ok(Self {
                command,
                proof: OutputProof::Gated {
                    judgement,
                    file: Box::new(file),
                    text: OnceLock::new(),
                },
            })
        } else {
            Err(PostValidationFailure {
                command,
                bar: judgement.bar(),
                errors,
            })
        }
    }

    /// Admit a document the command did not modify, carrying the ORIGINAL
    /// input bytes.
    ///
    /// See the module docs for why an unchanged document is not gated: the
    /// proof here is "this command applied nothing", which is a different
    /// (and stronger) fact than "this output is valid".
    ///
    /// It takes the input TEXT, not the parsed model, because "unchanged" is
    /// a claim about bytes. Re-serializing the model would make the claim
    /// false wherever a parse-then-serialize round trip is not the identity,
    /// and no gate would be looking: this is the one route that judges
    /// nothing, so it is the one route where a silent rewrite could not be
    /// caught.
    pub(crate) fn pass_through(original_text: &str, command: ReleasedCommand) -> Self {
        Self {
            command,
            proof: OutputProof::Ungated {
                text: original_text.to_owned(),
            },
        }
    }

    /// Admit a document the command DECLINED to process, applying the command's
    /// declared no-op and serializing the result.
    ///
    /// Distinct from [`Self::pass_through`], whose bytes are the input's own.
    /// Here the input text is not the answer: morphotag EDITS the model,
    /// stripping the `%xalign` / `%xrev` decision tiers an earlier align run
    /// wrote, so the input bytes would describe a document it deliberately did
    /// not return. That is also why the proof does not claim "unchanged", and
    /// why this constructor is NAMED for the no-op it applies: the fact that
    /// something was edited lives in the name a reader reads, since no value
    /// downstream ever acted on it.
    ///
    /// # The strip happens HERE, and that is the point
    ///
    /// The strip and the declaration were paired by convention across three
    /// call sites until 2026-09-07: each ran
    /// `batchalign_transform::decisions::strip_decision_tiers` and then called
    /// this constructor, and nothing made the second follow the first. A fourth
    /// declining site would have had to know. It is one operation now, so a
    /// declined proof whose bytes still carry decision tiers cannot be built.
    ///
    /// A caller that has already stripped (morphotag's incremental path strips
    /// once, up front, because its ANALYZE path needs it too) pays one
    /// idempotent traversal. That is deliberate: the guarantee is worth more
    /// than the traversal, and it must not depend on the caller.
    ///
    /// # Align is no longer a caller
    ///
    /// Align's AST entry point used this for `@Options: dummy` and `NoAlign`,
    /// on the grounds that it never saw the text its `ChatFile` came from. That
    /// was true and it was the wrong cure: align changes NOTHING about those
    /// documents, so re-serializing the model shipped a parse-and-serialize
    /// round trip as "unchanged", and this route judges nothing so no gate
    /// could catch it. The text now travels to that entry point and it takes
    /// [`Self::pass_through`] instead.
    ///
    /// It is still not gated, for the reason the module docs give: a CA file
    /// morphotag declines to analyze is exactly the kind of document that never
    /// had to meet the bar the command demands of material it does analyze, so
    /// gating would refuse a file the command was told to leave alone. The
    /// proof is "only the command's declared no-op was applied", which is why
    /// the no-op is named in the function's own name rather than left to a
    /// caller to supply.
    pub(crate) fn declined_stripping_decision_tiers(
        mut file: ChatFile,
        command: ReleasedCommand,
    ) -> Self {
        batchalign_transform::decisions::strip_decision_tiers(&mut file);
        Self {
            command,
            // Serialized here rather than lazily, because this route judges
            // nothing and so has no model to transition afterwards: the bytes
            // are final the moment they exist, and this is the one and only
            // serialization the document gets.
            proof: OutputProof::Ungated {
                text: to_chat_string(&file),
            },
        }
    }

    /// Merge single-letter abbreviations and re-run the gate over the merged
    /// MODEL, so the bytes returned are bytes a gate has judged.
    ///
    /// The writers used to call `merge_abbreviations_in_chat_text` on
    /// `as_str()` and write ITS result, which put a transform after the proof
    /// and made the proof describe a document that never reached disk. The
    /// merge is a step inside the gate now: it runs on the MODEL the proof
    /// carries and is re-judged under the SAME [`Judgement`] and for the SAME
    /// command that proof records, so the merge cannot lower the bar it is
    /// judged against, and a compare output is re-checked for preservation
    /// against the input census rather than against a level nothing admitted.
    ///
    /// # A refusal hands the unmerged proof BACK
    ///
    /// See [`AbbreviationMergeRefused`]: the merge is cosmetic and can break a
    /// document that was fine, so losing the proof it was applied to would
    /// turn a valid output into nothing written. The whole-document clone
    /// below is what that fallback costs, once per merged file, and it buys
    /// the only alternative to discarding an admissible output.
    ///
    /// Nothing is parsed. Until 2026-09-07 this re-parsed `self.text` and threw
    /// the parse errors away, so the re-gate's L0 rung was handed an empty
    /// error list and could not have refused anything, and the model merged was
    /// the parser's recovery of our own bytes rather than the model the gate
    /// had judged.
    ///
    /// # A pass-through is returned untouched, and that is the POLICY
    ///
    /// The merge is a policy applied to what the command WROTE. A document
    /// the command did not modify has no output for the merge to be a part
    /// of, so it is left exactly as it arrived and the tree it is written back
    /// into stays byte-identical. That is what pass-through means, and it is
    /// what `@Options: dummy`, `NoAlign`, a CA document a command declines to
    /// touch, and a document with no collectible payloads all get.
    ///
    /// This is a deliberate narrowing of the previous behaviour, where the
    /// merge ran on every success including those, because it ran on the
    /// finished TEXT at the writer rather than on a proof. Pinned by
    /// `merging_abbreviations_leaves_a_pass_through_byte_identical` below, and
    /// documented for operators in
    /// `book/src/batchalign/reference/command-io.md`.
    pub(crate) fn with_abbreviations_merged(self) -> Result<Self, AbbreviationMergeRefused> {
        let command = self.command;
        let OutputProof::Gated {
            judgement,
            file,
            text,
        } = self.proof
        else {
            return Ok(self);
        };
        let mut merged = file.clone();
        merge_abbreviations(&mut merged);
        Self::judge(*merged, judgement.clone(), command).map_err(|failure| {
            AbbreviationMergeRefused {
                // Rebuilt from the parts the merge did not touch, cache
                // included, so the caller gets back exactly the proof it
                // handed in rather than an equivalent one.
                unmerged: Self {
                    command,
                    proof: OutputProof::Gated {
                        judgement,
                        file,
                        text,
                    },
                },
                failure,
            }
        })
    }

    /// Stamp the run's provenance comment onto the MODEL and re-run the gate,
    /// so the bytes returned are bytes a gate has judged.
    ///
    /// `align` used to build its proof, throw it away, serialize the model
    /// again, inject the comment into that TEXT, and hand the writer a bare
    /// `String` which the writer then re-gated at a level nobody had admitted
    /// anything at. Two of those steps were a defect each: the written bytes
    /// were not the proven bytes, and the second gate refused documents the
    /// first had deliberately let through. The stamp is a transition on the
    /// proof now, applied to the MODEL that proof carries and re-judged under
    /// the SAME [`Judgement`] and for the SAME command it records, so it
    /// cannot lower the bar it is judged against. Nothing is parsed, for the
    /// reason [`Self::with_abbreviations_merged`] gives.
    ///
    /// Unlike the merge, a refusal here does not hand the pre-stamp proof
    /// back, and that is deliberate rather than an oversight: a document
    /// written without its `[ba3 ...]` line is not a lesser version of the
    /// same output, it is one the provenance gate can no longer recognise on
    /// the next run, so there is no fallback for a caller to choose.
    ///
    /// A pass-through is returned untouched, for the same reason
    /// [`Self::with_abbreviations_merged`] returns one untouched: the command
    /// applied nothing, so there is no run for a provenance line to describe.
    /// This is what makes `NoAlign`'s documented "strict pass-through, zero
    /// modifications" literally true; it used to acquire a `@Comment` line.
    pub(crate) fn with_provenance_injected(
        self,
        comment: &crate::provenance::ProvenanceComment,
    ) -> Result<Self, PostValidationFailure> {
        let command = self.command;
        let OutputProof::Gated {
            judgement, file, ..
        } = self.proof
        else {
            return Ok(self);
        };
        let mut file = *file;
        crate::provenance::inject_provenance(&mut file, comment);
        Self::judge(file, judgement, command)
    }

    /// The command whose output these bytes are.
    ///
    /// The writer reads the command from HERE rather than being told it
    /// separately, so there is no second place for the name to be written down
    /// and no way for the two to disagree.
    pub(crate) fn command(&self) -> ReleasedCommand {
        self.command
    }

    /// Borrow the validated CHAT text, serializing the judged model if this is
    /// the first time anyone has asked.
    ///
    /// THE materialization point for a proof that is being read rather than
    /// consumed, and the reason the transitions above are free: an aligned
    /// file is judged three times and serialized once, at the writer, instead
    /// of three times with the first two dropped unread.
    ///
    /// Call it only where the bytes are genuinely wanted. A caller that will
    /// discard them (a debug dump that is switched off, say) should ask
    /// whether it wants them BEFORE asking for them.
    pub(crate) fn as_str(&self) -> &str {
        match &self.proof {
            OutputProof::Gated { file, text, .. } => text.get_or_init(|| to_chat_string(file)),
            OutputProof::Ungated { text, .. } => text,
        }
    }

    /// Take ownership of the validated CHAT text.
    ///
    /// Used by consumers that are not the writer (comparison artifacts, the
    /// transcribe pipeline's intermediate stage), which legitimately need the
    /// string. This does not weaken the gate: the writer seams take the proof
    /// itself, so a bare `String` still cannot reach disk.
    pub(crate) fn into_text(self) -> String {
        match self.proof {
            OutputProof::Ungated { text, .. } => text,
            OutputProof::Gated { file, text, .. } => match text.into_inner() {
                // Already materialized by an earlier `as_str`; reuse rather
                // than serialize a second time.
                Some(materialized) => materialized,
                // Nobody has asked for bytes yet, so this is the proof's one
                // and only serialization.
                None => to_chat_string(&file),
            },
        }
    }

    /// Test-only route, for the fake `WorkerGateway` doubles.
    ///
    /// See the module docs: these doubles replace a gateway whose production
    /// implementation runs the gate, so the double is upstream of it and never
    /// builds a `ChatFile` to gate.
    ///
    /// It carries a GATED proof, not an unchanged one, because that is what
    /// the real gateway returns. This matters: `with_abbreviations_merged`
    /// returns a pass-through untouched, so a double claiming "unchanged"
    /// would make the write path skip the merge in tests while running it in
    /// production, which is the double modelling a different seam from the one
    /// it stands in for.
    #[cfg(test)]
    pub(crate) fn for_test(text: impl Into<String>, command: ReleasedCommand) -> Self {
        let text = text.into();
        // The double has no model and a `Gated` proof has one, so this is where
        // the pair is stood up. Parsing here is not the re-parse the module
        // bans: that ban is about re-deriving a command's own output from its
        // own bytes, and this string is one a test wrote.
        let parser = crate::chat_parser();
        let (file, _parse_errors) = batchalign_transform::parse::parse_lenient(&parser, &text);
        // The double's OWN bytes are pre-loaded into the materialization
        // cache, so READING this proof hands back exactly what the test wrote
        // rather than a round trip through the parser and the serializer. The
        // real gateway's bytes and its model always agree; a double's need
        // not, and the write-gate tests in `recipe_runner::runtime` assert on
        // bytes they supplied, several of which are CHAT fragments with no
        // `@UTF8` / `@Begin` that would not survive a round trip.
        //
        // That holds for READS ONLY, and the difference matters. Every
        // transition above consumes the proof and rebuilds it around the model
        // it just judged, with a FRESH cache, so the bytes after a transition
        // are the serialization of this parse and not what was passed in. A
        // double whose text does not round-trip must therefore be read, never
        // transitioned; put a document through `with_abbreviations_merged` or
        // `with_provenance_injected` in a test and it will come back
        // re-serialized. This comment said "reading this proof hands back
        // exactly what the test wrote" with no such qualification until
        // 2026-09-07, which reads as a promise about the value rather than
        // about one method of it.
        let materialized = OnceLock::new();
        // Infallible on a cell made one line above.
        let _ = materialized.set(text);
        Self {
            command,
            proof: OutputProof::Gated {
                judgement: Judgement::Admitted(ValidityLevel::StructurallyComplete),
                file: Box::new(file),
                text: materialized,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    /// A minimal file that satisfies L1 and carries a terminator.
    const VALID: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world .\n@End\n";

    /// [`VALID`] but for the missing terminator, which is a fault a document
    /// nothing admitted is allowed to ARRIVE with.
    const NO_TERMINATOR: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world\n@End\n";

    fn parse(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().expect("grammar loads");
        let (file, _errors) = parse_lenient(&parser, text);
        file
    }

    /// The gate admits an output that still satisfies its admission level and
    /// the command's own output checks, and the proof carries those bytes.
    #[test]
    fn gate_admits_output_that_still_validates() {
        let file = parse(VALID);
        let proof = PostValidated::gate(
            &file,
            ValidityLevel::StructurallyComplete,
            ReleasedCommand::Utseg,
        )
        .expect("a clean file must pass its own output gate");
        assert!(proof.as_str().contains("*CHI:"));
    }

    /// RED FIRST: an output that lost a terminator must be REFUSED, not
    /// warned about. This is the whole point of the type; before this change
    /// the same input produced a `warn!` and a written file.
    #[test]
    fn gate_refuses_output_that_dropped_a_terminator() {
        let mut file = parse(VALID);
        for line in &mut file.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }
        let failure = PostValidated::gate(
            &file,
            ValidityLevel::StructurallyComplete,
            ReleasedCommand::Utseg,
        )
        .expect_err("an output that lost its terminator must be refused");
        assert_eq!(failure.command, ReleasedCommand::Utseg);
        let rendered = failure.to_string();
        assert!(
            rendered.contains("lost its terminator"),
            "the refusal must name what broke, got: {rendered}"
        );
    }

    /// A pass-through is admitted without being judged against a bar it never
    /// had to meet: a dummy-shaped file with no `@Participants` would fail L1,
    /// yet the command applied nothing to it.
    #[test]
    fn pass_through_admits_an_unmodified_document_that_would_fail_the_gate() {
        let text = "@UTF8\n@Begin\n*PAR:\thello .\n@End\n";
        let file = parse(text);
        assert!(
            PostValidated::gate(
                &file,
                ValidityLevel::StructurallyComplete,
                ReleasedCommand::Utseg
            )
            .is_err(),
            "precondition: this document does not satisfy L1"
        );
        let proof = PostValidated::pass_through(text, ReleasedCommand::Utseg);
        assert!(proof.as_str().contains("*PAR:"));
    }

    /// RED FIRST: a pass-through must carry the input's OWN bytes. It used to
    /// re-serialize the parsed model, so a document the docs call unchanged
    /// shipped through a serializer no gate had judged. The trailing spaces
    /// here are the cheapest observable difference a round trip removes.
    #[test]
    fn pass_through_carries_the_original_bytes_not_a_reserialization() {
        let text = "@UTF8\n@Begin\n@Comment:\tkept   \n*PAR:\thello .\n@End\n";
        let proof = PostValidated::pass_through(text, ReleasedCommand::Utseg);
        assert_eq!(
            proof.as_str(),
            text,
            "a pass-through must be byte-identical to its input"
        );
    }

    /// RED FIRST (2026-09-07): a declining command's proof must not carry the
    /// decision tiers an earlier align run wrote, and the constructor is what
    /// guarantees it.
    ///
    /// The strip and the declaration were paired by convention across three
    /// call sites, so a fourth would have shipped `%xalign` back out. The
    /// `pass_through` half of the assertion is the control: it is the same
    /// document down the other route, and it keeps the tiers, which is exactly
    /// why the two routes are different constructors.
    #[test]
    fn declining_a_document_strips_the_decision_tiers_a_previous_run_wrote() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world .\n\
%xalign:\tmonotonicity:end_clamped\n@End\n";
        let declined = PostValidated::declined_stripping_decision_tiers(
            parse(text),
            ReleasedCommand::Morphotag,
        );
        assert!(
            !declined.as_str().contains("%xalign:"),
            "the constructor must apply the declared no-op, got: {}",
            declined.as_str()
        );
        assert!(
            declined.as_str().contains("*CHI:"),
            "only the decision tiers may go, got: {}",
            declined.as_str()
        );
        assert!(
            PostValidated::pass_through(text, ReleasedCommand::Morphotag)
                .as_str()
                .contains("%xalign:"),
            "control: a pass-through applies nothing, so it keeps them"
        );
    }

    /// RED FIRST (2026-09-07 review, item 1): preservation admits a document
    /// that ARRIVED without a terminator, which a level gate refuses.
    ///
    /// This is the failure the compare and benchmark gate shipped: a gold
    /// companion with one terminator-less non-CA utterance had always been
    /// written and became a hard refusal with no output at all, for a fault the
    /// command could not have caused.
    #[test]
    fn preserving_admits_an_output_whose_input_never_had_a_terminator() {
        let input = parse(NO_TERMINATOR);
        assert!(
            PostValidated::gate(
                &input,
                ValidityLevel::StructurallyComplete,
                ReleasedCommand::Compare
            )
            .is_err(),
            "precondition: a level gate refuses this document"
        );
        let proof = PostValidated::preserving(
            UtteranceCensus::of(&input),
            input.clone(),
            ReleasedCommand::Compare,
        )
        .expect("a command cannot destroy a terminator its input never had");
        assert!(proof.as_str().contains("*CHI:"));
    }

    /// RED FIRST (2026-09-07 review, item 1): the same output IS refused when
    /// its input had the terminator, so the check fires on a real loss.
    ///
    /// Read with the test above, these two differ only in the input, which is
    /// what makes either of them evidence: a check that ignored the input would
    /// fail one of them whichever way it leaned.
    #[test]
    fn preserving_refuses_an_output_that_lost_a_terminator_its_input_had() {
        let failure = PostValidated::preserving(
            UtteranceCensus::of(&parse(VALID)),
            parse(NO_TERMINATOR),
            ReleasedCommand::Compare,
        )
        .expect_err("an output that lost a terminator its input had must be refused");
        assert_eq!(failure.bar, JudgementBar::Preservation);
        let rendered = failure.to_string();
        assert!(
            rendered.contains("lost its terminator"),
            "the refusal must name what broke, got: {rendered}"
        );
        assert!(
            !rendered.contains("must still satisfy"),
            "a preserved refusal must not report a bar the input was never held to: {rendered}"
        );
    }

    /// An utterance the command DROPPED is a loss too, and the count is what
    /// notices it.
    #[test]
    fn preserving_refuses_an_output_that_dropped_an_utterance() {
        let two = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world .\n*CHI:\tbye now .\n@End\n";
        let failure = PostValidated::preserving(
            UtteranceCensus::of(&parse(two)),
            parse(VALID),
            ReleasedCommand::Compare,
        )
        .expect_err("an output with fewer utterances than its input must be refused");
        assert!(
            failure
                .to_string()
                .contains("kept 1 of the input's 2 utterances"),
            "the refusal must say what went missing, got: {failure}"
        );
    }

    /// The correspondence between the two documents is positional, so it is
    /// CHECKED rather than assumed: an output whose utterances no longer line
    /// up with the input is refused instead of having two unrelated utterances
    /// silently compared.
    #[test]
    fn preserving_refuses_an_output_whose_utterances_no_longer_line_up() {
        let other_speaker = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n\
@ID:\teng|test|MOT|||||Mother|||\n*MOT:\thello world .\n@End\n";
        let failure = PostValidated::preserving(
            UtteranceCensus::of(&parse(VALID)),
            parse(other_speaker),
            ReleasedCommand::Compare,
        )
        .expect_err("an output that no longer lines up with its input must be refused");
        assert!(
            failure.to_string().contains("no longer lines up"),
            "the refusal must say the correspondence broke, got: {failure}"
        );
    }

    /// RED FIRST (2026-09-07 review, item 1): the merge inside a PRESERVED
    /// proof is judged by preservation, not by a level restated on the way.
    ///
    /// The fixture would fail every level gate, having no terminator, and it
    /// carries an abbreviation so the merge really runs. A transition that
    /// restated a bar instead of carrying the proof's own judgement refuses it.
    #[test]
    fn a_preserved_proof_is_still_judged_by_preservation_after_the_merge() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\tF B I here\n@End\n";
        let file = parse(text);
        let proof = PostValidated::preserving(
            UtteranceCensus::of(&file),
            file.clone(),
            ReleasedCommand::Compare,
        )
        .expect("nothing was lost: the output IS the input");
        let merged = proof
            .with_abbreviations_merged()
            .expect("the merge must be judged the way the document that reached it was");
        assert!(
            merged.as_str().contains("FBI"),
            "the merge must still have run, got: {}",
            merged.as_str()
        );
    }

    /// RED FIRST (2026-09-07 review, item 5): a merge that breaks the gate
    /// hands the UNMERGED proof back, so a cosmetic transform cannot destroy an
    /// admissible output.
    ///
    /// Reachable rather than contrived: `merge_abbreviations` edits the main
    /// tier and leaves `%mor` alone, so collapsing `F B I` into `FBI` in a
    /// morphotag output leaves four `%mor` items against two words, which is
    /// exactly what `validate_output` refuses. Before this the transition had
    /// consumed the proof and the caller had nothing left to fall back to.
    #[test]
    fn a_merge_that_breaks_the_gate_hands_back_the_unmerged_proof() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\tF B I .\n\
%mor:\tn|F n|B n|I .\n@End\n";
        let proof = PostValidated::gate(
            &parse(text),
            ValidityLevel::StructurallyComplete,
            ReleasedCommand::Morphotag,
        )
        .expect("precondition: the unmerged morphotag output passes its own gate");

        let refused = proof
            .with_abbreviations_merged()
            .expect_err("the merged document desynchronises %mor and must be refused");
        assert!(
            refused.failure.to_string().contains("%mor has"),
            "the failure must name what the merge broke, got: {}",
            refused.failure
        );
        assert!(
            refused.to_string().contains("abbreviation merge"),
            "the refusal must name the merge as the cause, got: {refused}"
        );
        assert!(
            refused.unmerged.as_str().contains("F B I"),
            "the unmerged proof must come back intact, got: {}",
            refused.unmerged.as_str()
        );
    }

    /// RED FIRST: the abbreviation merge is a step INSIDE the gate, so the
    /// bytes it returns are bytes the gate judged. Before this the writer ran
    /// the merge on `as_str()` and wrote the RESULT, which no gate had seen.
    #[test]
    fn merging_abbreviations_returns_a_proof_over_the_merged_bytes() {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\tF B I here .\n@End\n";
        let file = parse(text);
        let proof = PostValidated::gate(
            &file,
            ValidityLevel::StructurallyComplete,
            ReleasedCommand::Utseg,
        )
        .expect("the input is valid at L1");
        assert!(proof.as_str().contains("F B I"), "precondition: unmerged");
        let merged = proof
            .with_abbreviations_merged()
            .expect("the merged document still satisfies its own gate");
        assert!(
            merged.as_str().contains("FBI"),
            "the merge must have been applied to the bytes the proof carries, got: {}",
            merged.as_str()
        );
    }

    /// A pass-through is not re-written by the merge: the command applied
    /// nothing, so there is no output for the merge to be a part of.
    #[test]
    fn merging_abbreviations_leaves_a_pass_through_byte_identical() {
        let text = "@UTF8\n@Begin\n*PAR:\tF B I .\n@End\n";
        let merged = PostValidated::pass_through(text, ReleasedCommand::Utseg)
            .with_abbreviations_merged()
            .expect("a pass-through cannot fail a gate it does not run");
        assert_eq!(merged.as_str(), text);
    }

    /// RED FIRST (2026-09-07 review, item 3): a transition operates on the
    /// MODEL the gate judged, never on a re-parse of the bytes it serialized.
    ///
    /// Both transitions used to run `parse_lenient` over `self.text`, discard
    /// the parse errors into a `_`, transform the RECOVERED model and re-gate
    /// with an empty error list. So the module's own "never re-parsed" claim
    /// was false, the re-gate's L0 rung was vacuous for a reason the module
    /// denied, and wherever the parser reads our bytes back differently from
    /// the model we hold, the parser's reading silently won.
    ///
    /// The fixture makes that disagreement visible rather than arguing about
    /// it. The gated model holds ONE word whose raw text is `F B I`, which
    /// serializes to three tokens: a re-parse sees three single-letter words
    /// and the abbreviation merge collapses them, while the model has nothing
    /// to merge. The bytes that come back therefore say which document the
    /// transition actually ran on.
    #[test]
    fn a_transition_runs_on_the_gated_model_not_on_a_reparse_of_its_bytes() {
        use talkbank_model::model::{TierContentItems, UtteranceContent, Word};

        let mut file = parse(VALID);
        for line in &mut file.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.content = TierContentItems::new(vec![UtteranceContent::Word(
                    Box::new(Word::simple("F B I")),
                )]);
            }
        }

        let proof = PostValidated::gate(
            &file,
            ValidityLevel::StructurallyComplete,
            ReleasedCommand::Utseg,
        )
        .expect("the fixture is valid at L1");
        assert!(
            proof.as_str().contains("F B I"),
            "precondition: the bytes read as three abbreviation tokens, got: {}",
            proof.as_str()
        );

        let merged = proof
            .with_abbreviations_merged()
            .expect("the merged document still satisfies its own gate");
        assert!(
            merged.as_str().contains("F B I"),
            "the merge ran on a RE-PARSE of the proof's bytes, not on the model \
             the gate judged, got: {}",
            merged.as_str()
        );
        assert!(
            !merged.as_str().contains("FBI"),
            "the model holds one word, so there is nothing for the merge to \
             collapse; got: {}",
            merged.as_str()
        );
    }
}
