//! Source-admitted alignment phases. Only active documents can receive UTR.
//!
//! # What align admits
//!
//! Admission decides, once and before any media is resolved or inference
//! runs, whether this source is something align can transform, and records
//! what the output then owes. The admitted states of an actively aligned
//! source are:
//!
//! | `@Media` | timing | admitted as | output owes |
//! | --- | --- | --- | --- |
//! | `name, audio` (linked) | some | ordinary | nothing more |
//! | `name, audio, unlinked` | none | ordinary | nothing; untimed output stays `unlinked` |
//! | `name, audio` (linked) | none | [`TimingObligationOrigin::NeverTimed`] | timing (E544), else no output |
//! | linked, unusable `%wor` removed | none left | [`TimingObligationOrigin::DiscardedWordTiming`] | timing, else no output |
//! | `name, audio` (linked) | only on `[+ diary]` notes, none kept | [`TimingObligationOrigin::OffRecordTimingOnly`] | timing, else no output |
//! | absent, missing, notrans | any | refused ([`AlignmentMediaRefusal`]) | |
//!
//! Either obligation row under `--main-bullets exact` is refused too: `exact`
//! keeps untimed utterances untimed and admits no UTR, so the obligation
//! could never be met. Declarations Chatter's validation already refuses
//! (two `@Media` headers, E501; an unknown type or status, E535/E536; `unlinked`
//! beside timing, E552) stay refused there, with Chatter's message.
//!
//! The third row is CHAT-invalid as input (E544, "linkage declared, no
//! timing"), and it is exactly the condition alignment exists to remove, so
//! refusing it would make a never-aligned file unalignable. It is admitted
//! through Chatter's own timing-regeneration validation, which runs every
//! other rule and returns the E544 requirement as an obligation rather than a
//! finding; nothing here filters diagnostic codes.
//!
//! The fifth row is CHAT-valid as input: Chatter sees the notes' timing. But
//! align never uses timing on an utterance not in the recording and writes
//! it back only as a bullet `--main-bullets keep` or `exact` keeps, so when
//! no such bullet is kept the working document has no timing at all, and the
//! output owes the same timing the third row does. Admission records that
//! obligation itself, after removing the notes' timing, so a file that
//! aligns no speech is refused with a message naming the notes and the
//! remedy, not with the output gate's generic E544.
//!
//! Every admitted active source carries an [`AlignableMedia`]: proof that its
//! one `@Media` declaration admits the media/timing transition. That
//! transition runs in exactly one place, [`FaAdmission::reconcile_output`],
//! so it cannot fail for a reason admission already knew.
use super::FaInputDocument;
use crate::chat_ops::ChatFile;
use crate::error::{
    AlignmentMediaRefusal, ServerError, SuggestedMediaName, TimingObligationOrigin,
};
use crate::types::results::{FaOutput, FaResult};
use batchalign_transform::AdmittedSourceChat;
use batchalign_transform::media_timing::{MediaTimingState, reconcile_media_timing};
use talkbank_model::model::{
    ChatOptionFlag, Header, MediaStatus, MediaType, TranscriptName, TranscriptTimingEvidence,
};
use talkbank_model::validation::{
    MediaTimingObligation, PendingTimingChatFile, TimingRegenerationAdmission,
};
use talkbank_parser::{AdmittedDisposition, ReplacementFailure, ReplacementTiers, WordTimingPlan};

/// Issued only by this source-admission owner. Sibling modules cannot invent
/// complete input admission or detach a pending obligation from its payload.
#[derive(Clone)]
pub(super) struct FaAdmission {
    media: AlignableMedia,
    pending_timing: Option<OutstandingTiming>,
    required_timing: super::completion::RequiredFaTiming,
}

/// A linked-media requirement the output must meet, and why the source owes
/// it: the declaration's location and the origin the user is told.
///
/// Its fields are private to this module and it is built only in
/// [`read_fa_source_named`]: from a Chatter-issued `MediaTimingObligation`,
/// or, for [`TimingObligationOrigin::OffRecordTimingOnly`], from the admitted
/// declaration and what stripping the notes' timing left. The only routes to
/// `MissingTimingRegenerationEvidence` take one, so no other code can assert
/// an obligation from a bare span.
#[derive(Clone)]
pub(crate) struct OutstandingTiming {
    header_span: talkbank_model::Span,
    origin: TimingObligationOrigin,
}

impl OutstandingTiming {
    /// Where the declaration that owes timing is.
    pub(crate) fn header_span(&self) -> talkbank_model::Span {
        self.header_span
    }

    /// Why the source owes it.
    pub(crate) fn origin(&self) -> TimingObligationOrigin {
        self.origin
    }
}

/// Proof that the source's one `@Media` declaration names a usable recording
/// in a state alignment may link: built only by [`AlignableMedia::admit`].
///
/// These are the preconditions of Chatter's `reconcile_media_timing`, decided
/// from the source instead of discovered in the output. The match below is
/// exhaustive over Chatter's media types and statuses, so a new one stops
/// compilation here; the agreement test in this module checks the two
/// decisions coincide. The mirror exists only because the pinned Chatter
/// exposes the transition but not its precondition as a type; when it does,
/// this proof should become that one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AlignableMedia {
    linkage: DeclaredLinkage,
    /// Where the declaration is, for an obligation align establishes itself.
    header_span: talkbank_model::Span,
}

/// What the admitted declaration says about linkage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeclaredLinkage {
    /// `@Media: name, audio|video`.
    Linked,
    /// `@Media: name, audio|video, unlinked`: the transition consumes the
    /// status once the output carries timing (CHAT manual: `unlinked` is for
    /// transcripts not yet linked to media; E552 refuses it beside timing).
    Unlinked,
}

impl AlignableMedia {
    /// Decide the declaration from the admitted source, before any work.
    fn admit(
        document: &ChatFile,
        name: &TranscriptName<'_>,
    ) -> Result<Self, AlignmentMediaRefusal> {
        // The same scan the transition makes: every header line, not only
        // the leading block, so the two cannot disagree about which
        // declarations exist.
        let mut declarations =
            document
                .headers_with_spans()
                .filter_map(|(header, span)| match header {
                    Header::Media(media) => Some((media, span)),
                    _ => None,
                });
        let Some((media, header_span)) = declarations.next() else {
            return Err(AlignmentMediaRefusal::Undeclared {
                suggested: match name {
                    TranscriptName::Named(stem) => {
                        SuggestedMediaName::Transcript(stem.as_str().to_owned())
                    }
                    TranscriptName::Anonymous => SuggestedMediaName::Unnamed,
                },
            });
        };
        let additional = declarations.count();
        if additional != 0 {
            return Err(AlignmentMediaRefusal::Multiple {
                count: additional + 1,
            });
        }
        let declared = || media.filename.as_str().to_owned();
        match &media.media_type {
            MediaType::Audio | MediaType::Video => {}
            MediaType::Missing => {
                return Err(AlignmentMediaRefusal::DeclaredMissing { media: declared() });
            }
            MediaType::Unsupported(token) => {
                return Err(AlignmentMediaRefusal::Unsupported {
                    field: "media type",
                    as_written: token.clone(),
                });
            }
        }
        let linkage = match &media.status {
            None => DeclaredLinkage::Linked,
            Some(MediaStatus::Unlinked) => DeclaredLinkage::Unlinked,
            Some(MediaStatus::Missing) => {
                return Err(AlignmentMediaRefusal::DeclaredMissing { media: declared() });
            }
            Some(MediaStatus::Notrans) => {
                return Err(AlignmentMediaRefusal::NotTranscribed { media: declared() });
            }
            Some(MediaStatus::Unsupported(token)) => {
                return Err(AlignmentMediaRefusal::Unsupported {
                    field: "status",
                    as_written: token.clone(),
                });
            }
        };
        Ok(Self {
            linkage,
            header_span,
        })
    }

    /// The post-alignment media/timing transition, against the declaration
    /// admission proved usable. A refusal here means the alignment changed
    /// the header, which is an internal fault and classified as one.
    fn reconcile(self, output: ChatFile) -> Result<FaOutput, ServerError> {
        let state = reconcile_media_timing(output).map_err(ServerError::MediaTiming)?;
        if let (DeclaredLinkage::Unlinked, MediaTimingState::Timed(_)) = (self.linkage, &state) {
            tracing::debug!("aligned output carries timing: `unlinked` removed from @Media");
        }
        Ok(FaOutput::Processed(ReconciledOutput(state)))
    }
}

/// Aligned output after the media/timing transition, run against the
/// declaration its source was admitted with. Its field is private to this
/// module and [`AlignableMedia::reconcile`] is its only constructor, so a
/// processed FA result cannot carry a document reconciled any other way
/// (Chatter's `reconcile_media_timing` is public; its state alone is not
/// enough to build one of these).
pub(crate) struct ReconciledOutput(MediaTimingState);

impl ReconciledOutput {
    /// Borrow the reconciled document.
    pub(crate) fn as_chat_file(&self) -> &ChatFile {
        self.0.as_chat_file()
    }
}

impl FaAdmission {
    /// Completion consumes the exact attempted result against this source's
    /// private obligations. Callers cannot replace the census or select
    /// success: the result is complete or partial as the census finds it.
    pub(super) fn complete_result(
        &self,
        result: FaResult,
    ) -> Result<super::completion::FaCompletion, ServerError> {
        self.required_timing.admit(result)
    }

    /// Take a draft through the media/timing transition this source's
    /// declaration was admitted for. The only producer of processed FA
    /// output: every FA path hands its draft to `finish`, which calls this.
    pub(super) fn reconcile_output(
        &self,
        draft: FaResult<ChatFile>,
    ) -> Result<FaResult, ServerError> {
        draft.try_map_output(|output| self.media.reconcile(output))
    }

    /// An inference request is not completion. Only the source owner may
    /// discharge its obligation, and it observes the actual output structure.
    pub(super) fn require_output_timing(&self, output: &ChatFile) -> Result<(), ServerError> {
        if let Some(outstanding) = &self.pending_timing
            && let Some(missing) =
                crate::error::MissingTimingRegenerationEvidence::from_untimed_output(
                    outstanding,
                    output.timing_evidence(),
                )
        {
            return Err(ServerError::RequiredEvidenceUnavailable(
                crate::error::MissingRequiredEvidence::TimingRegeneration(missing),
            ));
        }
        Ok(())
    }

    /// Bind this attempt's actual grouping to its source obligation before
    /// inference or empty-plan finalization. No complete output proof is issued.
    pub(super) fn admit_grouping(
        self,
        grouping: crate::chat_ops::fa::Grouping,
        document: &ChatFile,
    ) -> Result<(Self, crate::chat_ops::fa::Grouping), ServerError> {
        if let Some(outstanding) = &self.pending_timing
            && let Some(missing) =
                crate::error::MissingTimingRegenerationEvidence::from_refused_grouping(
                    outstanding,
                    &grouping,
                    document.timing_evidence(),
                )
        {
            return Err(ServerError::RequiredEvidenceUnavailable(
                crate::error::MissingRequiredEvidence::TimingRegeneration(missing),
            ));
        }
        Ok((self, grouping))
    }

    /// Called only after complete checked output admission has succeeded.
    /// That admission includes E544: no outstanding linkage can reach writing.
    pub(super) fn discharge(
        self,
        output: crate::pipeline::post_validate::PostValidated,
    ) -> crate::pipeline::post_validate::PostValidated {
        if let Some(outstanding) = self.pending_timing {
            tracing::debug!(header_span = ?outstanding.header_span,
                origin = ?outstanding.origin,
                "source timing obligation fulfilled by complete output admission");
        }
        output
    }
}

/// The source disposition and model remain one owned value across retries.
pub(crate) struct FaWorkingDocument {
    disposition: FaWorkingDisposition,
    main_bullets: crate::chat_ops::fa::MainBulletAuthority,
}

enum FaWorkingDisposition {
    Preserved(AdmittedSourceChat<'static>),
    Active(ActiveFaDocument),
}

/// Constructible only from Chatter's source-bound admission. Outside callers
/// cannot pair an unchecked model with a free-standing admission token.
pub(crate) struct ActiveFaDocument {
    file: ChatFile,
    admission: FaAdmission,
    anchors: crate::chat_ops::fa::AnchorIndex,
}

impl FaWorkingDocument {
    pub(crate) fn document(&self) -> &ChatFile {
        match &self.disposition {
            FaWorkingDisposition::Preserved(source) => source.document(),
            FaWorkingDisposition::Active(active) => &active.file,
        }
    }

    pub(crate) fn unchanged(&self) -> Option<&AdmittedSourceChat<'static>> {
        match &self.disposition {
            FaWorkingDisposition::Preserved(source) => Some(source),
            FaWorkingDisposition::Active(_) => None,
        }
    }

    pub(crate) fn active_mut(&mut self) -> Option<&mut ActiveFaDocument> {
        match &mut self.disposition {
            FaWorkingDisposition::Preserved(_) => None,
            FaWorkingDisposition::Active(active) => Some(active),
        }
    }

    pub(crate) fn attempt(&self) -> FaInputDocument<'_> {
        match &self.disposition {
            FaWorkingDisposition::Preserved(source) => FaInputDocument::Preserved(source.clone()),
            FaWorkingDisposition::Active(active) => FaInputDocument::Active(super::ActiveFaInput {
                chat_file: active.file.clone(),
                admission: active.admission.clone(),
                main_bullets: self.main_bullets.clone(),
                anchors: &active.anchors,
            }),
        }
    }
}

impl ActiveFaDocument {
    pub(crate) fn document(&self) -> &ChatFile {
        &self.file
    }

    /// UTR mutates this admitted working model and supplies its own anchors.
    /// This transition is unavailable on the preserved-source payload.
    pub(crate) async fn recover_timing(
        &mut self,
        run: impl AsyncFnOnce(&mut ChatFile) -> Result<crate::chat_ops::fa::utr::UtrResult, ServerError>,
    ) -> Result<crate::chat_ops::fa::utr::UtrResult, ServerError> {
        let mut result = run(&mut self.file).await?;
        self.anchors.superseded_by(result.take_anchors());
        Ok(result)
    }
}

#[cfg(test)]
pub(crate) fn read_fa_source(text: &str) -> Result<FaWorkingDocument, ServerError> {
    read_fa_source_named(
        text,
        TranscriptName::Anonymous,
        crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
    )
}

/// What source admission established, before the working document is built.
enum AdmittedAlignmentSource<'source> {
    /// Chatter's planned disposition: preserved, replaced or regenerating.
    Planned(AdmittedDisposition<'source>),
    /// Linked `@Media`, no timing at all: admitted for the alignment that
    /// establishes it (see the module documentation).
    NeverTimed(PendingTimingChatFile),
}

pub(crate) fn read_fa_source_named(
    text: &str,
    name: TranscriptName<'_>,
    policy: crate::chat_ops::fa::MainBulletPolicy,
) -> Result<FaWorkingDocument, ServerError> {
    let source = match crate::chat_parser().admit_word_timing_plan(text, name, |headers| {
        let preserve = headers.iter().any(|header| {
            matches!(header,
                Header::Options { options } if options.iter().any(|flag| match flag {
                    ChatOptionFlag::Ca | ChatOptionFlag::NoAlign => true,
                    ChatOptionFlag::Unsupported(flag) => flag == "dummy",
                })
            )
        });
        if preserve {
            WordTimingPlan::Preserve
        } else {
            WordTimingPlan::PreferRetained
        }
    }) {
        Ok(admitted) => AdmittedAlignmentSource::Planned(admitted.into_disposition()),
        Err(failure) => AdmittedAlignmentSource::NeverTimed(admit_never_timed(failure, &name)?),
    };
    let (mut file, pending_timing) = match source {
        AdmittedAlignmentSource::Planned(AdmittedDisposition::Preserved(source)) => {
            let source = AdmittedSourceChat::from_preservation(source);
            if batchalign_transform::parse::is_no_align(source.document())
                || batchalign_transform::parse::is_dummy(source.document())
            {
                // Written back byte for byte: no timing, so no media needed.
                let main_bullets =
                    crate::chat_ops::fa::MainBulletAuthority::bind(policy, source.document())
                        .map_err(|error| ServerError::Validation(error.to_string()))?;
                return Ok(FaWorkingDocument {
                    disposition: FaWorkingDisposition::Preserved(source.into_owned()),
                    main_bullets,
                });
            }
            (source.into_valid_file().into_unchecked(), None)
        }
        AdmittedAlignmentSource::Planned(AdmittedDisposition::Replaced(replacement)) => {
            // Only a retained word-timing plan reaches replacement here.
            let admitted = replacement.selection();
            if admitted != Some(ReplacementTiers::WordTiming) {
                return Err(ServerError::ReplacementPlanContradicted {
                    planned: Some(ReplacementTiers::WordTiming),
                    admitted,
                });
            }
            (replacement.into_valid_file().into_unchecked(), None)
        }
        AdmittedAlignmentSource::Planned(AdmittedDisposition::Regenerating(pending)) => {
            let (file, obligation) = working_parts(pending.into_pending_file());
            let origin = TimingObligationOrigin::DiscardedWordTiming;
            let header_span = obligation.header_span();
            (
                file,
                Some(OutstandingTiming {
                    header_span,
                    origin,
                }),
            )
        }
        AdmittedAlignmentSource::NeverTimed(pending) => {
            let (file, obligation) = working_parts(pending);
            let origin = TimingObligationOrigin::NeverTimed;
            let header_span = obligation.header_span();
            (
                file,
                Some(OutstandingTiming {
                    header_span,
                    origin,
                }),
            )
        }
    };
    // Every actively aligned source must name the recording its timing will
    // index, decided here rather than discovered after the work.
    let media = AlignableMedia::admit(&file, &name)?;
    // Both bindings read the input AS PARSED: the bullets the policy keeps,
    // including any on an utterance not in the recording, and the
    // obligations, which permit exactly those.
    let main_bullets = crate::chat_ops::fa::MainBulletAuthority::bind(policy, &file)
        .map_err(|error| ServerError::Validation(error.to_string()))?;
    let required_timing = super::completion::RequiredFaTiming::bind(&file, &main_bullets)
        .map_err(super::kept_bullets_failed)?;
    // Then the working model forgets every timing an utterance not in the
    // recording carried, so no stage reads it as an anchor, a window or
    // reusable word timing. `keep` and `exact` restore its given bullet from
    // the binding above, after every phase that orders bullets; `derive`
    // derives none, since it has no aligned word (see `chat_ops::fa::presence`).
    let stripped = crate::chat_ops::fa::strip_off_record_timing(&mut file);
    if stripped.bullets() > 0 {
        tracing::info!(
            removed = stripped.bullets(),
            ?policy,
            "given bullets on utterances not in the recording are not alignment evidence"
        );
    }
    let pending_timing = match pending_timing {
        Some(outstanding) => Some(outstanding),
        None => off_record_timing_only(&file, media, &main_bullets, stripped),
    };
    // An outstanding obligation means the working document has no timing
    // anywhere, main-tier bullets included (that is when E544 fires), and no
    // note keeps one. `exact` keeps every utterance without a bullet untimed
    // and admits no UTR, so the output could never meet the obligation:
    // refuse now, not after FA.
    if pending_timing.is_some() {
        match policy {
            crate::chat_ops::fa::MainBulletPolicy::KeepExact => {
                return Err(AlignmentMediaRefusal::LinkedUntimedUnderExactBullets.into());
            }
            crate::chat_ops::fa::MainBulletPolicy::DeriveFromWords
            | crate::chat_ops::fa::MainBulletPolicy::KeepGiven => {}
        }
    }
    let admission = FaAdmission {
        media,
        pending_timing,
        required_timing,
    };
    Ok(FaWorkingDocument {
        disposition: FaWorkingDisposition::Active(ActiveFaDocument {
            file,
            admission,
            anchors: crate::chat_ops::fa::AnchorIndex::not_observed(),
        }),
        main_bullets,
    })
}

/// The obligation of a linked source whose only timing was on utterances not
/// in the recording, decided on the working document after their timing was
/// removed (see the module documentation's fifth row).
///
/// Owed exactly when the declaration is linked, the strip removed timing
/// from a note, the working document has no timing left, and no note keeps a
/// bullet the output will carry: under `--main-bullets keep` or `exact` a
/// note's given bullet is restored after alignment, which is timing, so
/// nothing beyond it is owed. Decided by the same timing observation E544
/// uses (`timing_evidence`). What was removed names the remedy.
fn off_record_timing_only(
    file: &ChatFile,
    media: AlignableMedia,
    main_bullets: &crate::chat_ops::fa::MainBulletAuthority,
    stripped: crate::chat_ops::fa::StrippedOffRecordTiming,
) -> Option<OutstandingTiming> {
    let notes = stripped.removed()?;
    match (media.linkage, file.timing_evidence()) {
        (DeclaredLinkage::Linked, TranscriptTimingEvidence::Absent)
            if !main_bullets.restores_off_record_bullet() =>
        {
            Some(OutstandingTiming {
                header_span: media.header_span,
                origin: TimingObligationOrigin::OffRecordTimingOnly(notes),
            })
        }
        (DeclaredLinkage::Linked, _) | (DeclaredLinkage::Unlinked, _) => None,
    }
}

/// The working document of a pending-timing admission, and a copy of its
/// obligation for the admission to hold beside it.
///
/// Chatter keeps the obligation attached to its document, so no consumer can
/// discharge it against a different one. Align cannot carry that payload as
/// is: every attempt, retries included, aligns its own copy of the working
/// document, and the payload is not `Clone`. So the admission keeps the
/// obligation's record (its header), and each attempt's output is held to it
/// twice: [`FaAdmission::require_output_timing`] refuses an untimed output
/// with the typed evidence error naming the origin, and complete output
/// admission then runs E544's own check again, so an output that still owes
/// timing can never be written.
fn working_parts(pending: PendingTimingChatFile) -> (ChatFile, MediaTimingObligation) {
    (pending.document().clone(), pending.obligation().clone())
}

/// Admit a source Chatter's complete admission refused, when it is the state
/// before a first alignment: linked `@Media` and no timing anywhere (E544).
///
/// Eligibility is structural, never a diagnostic code: the refusal is a
/// validation finding (not a parse failure, recovered parse or internal
/// fault), the retained model carries no timing, and align will actually
/// change it (a `NoAlign` or dummy file is written back unchanged,
/// so its E544 stands). The document is then re-validated by Chatter's own
/// timing-regeneration admission, which runs every rule and alignment check
/// again and returns only the E544 requirement as an obligation; any other
/// finding refuses exactly as before. An ineligible failure is returned
/// unchanged.
///
/// "Retained model": under the adaptive word-timing plan, a `%wor` tier
/// Chatter could not read is removed before validation, and Chatter grants
/// its own regeneration admission only when a removed tier provably carried
/// timing. A tier that carried bullets but did not lower is therefore seen
/// here as absent timing and admitted under this route. The outcome is the
/// same obligation (the output must carry timing); only the origin's wording
/// differs. Chatter owning a "never timed" disposition in its parser, where
/// the removed tiers are visible, would close this; it is recorded as a
/// follow-up rather than approximated here by reading the source text.
fn admit_never_timed(
    failure: ReplacementFailure,
    name: &TranscriptName<'_>,
) -> Result<PendingTimingChatFile, ServerError> {
    let ReplacementFailure::Validation(validation) = failure else {
        return Err(ServerError::ChatReplacementAdmission(failure));
    };
    let document = validation.document();
    let eligible = !validation.has_internal_failure()
        && !validation.has_incomplete_parse()
        && matches!(document.timing_evidence(), TranscriptTimingEvidence::Absent)
        && !batchalign_transform::parse::is_no_align(document)
        && !batchalign_transform::parse::is_dummy(document);
    if !eligible {
        return Err(ServerError::ChatReplacementAdmission(
            ReplacementFailure::Validation(validation),
        ));
    }
    match validation
        .into_unchecked()
        .validate_for_timing_regeneration(&talkbank_model::ErrorCollector::new(), *name)
    {
        Ok(TimingRegenerationAdmission::Pending(pending)) => Ok(pending),
        // Complete admission refused this document and the same rules, with
        // only E544 deferred, accept it with nothing deferred: the two
        // Chatter admissions disagree, which is a tool fault, not the input's.
        Ok(TimingRegenerationAdmission::Ready(_)) => Err(ServerError::AdmissionDisagreement),
        Err(refused) => Err(ServerError::ChatReplacementAdmission(
            ReplacementFailure::Validation(refused),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY;
    use talkbank_model::WriteChat;

    const SOURCE: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\tsample, audio\n\
*CHI:\thello world . \u{15}0_500\u{15}\n\
%wor:\thello \u{15}0_200\u{15} world \u{15}200_500\u{15} .\n\
@End\n";

    #[test]
    fn pending_timing_reports_the_actual_refused_window_without_write_authority() {
        use crate::chat_ops::fa::coordinates::{Ms, Recording};
        use crate::chat_ops::fa::{AnchorIndex, group_utterances};
        let word_only = SOURCE.replace(" . \u{15}0_500\u{15}", " .");
        let corrupt = pending_word_only(&word_only);
        let pending = read_fa_source(&corrupt).expect("source-bound pending timing");
        let FaInputDocument::Active(active) = pending.attempt() else {
            panic!("pending active input")
        };
        let recording = Recording::of_duration(Ms(20_000)).expect("nonempty recording");
        let grouping = group_utterances(
            &active.chat_file,
            1_000,
            &recording,
            &AnchorIndex::not_observed(),
        );
        assert!(grouping.groups.is_empty());
        let error = match active.admission.admit_grouping(grouping, &active.chat_file) {
            Err(error) => error,
            Ok(_) => panic!("refused acoustic window cannot fulfill timing"),
        };
        let category = crate::runner::util::classify_server_error(&error);
        assert_eq!(
            category,
            crate::scheduling::FailureCategory::EvidenceUnavailable
        );
        let message = error.to_string();
        // The first utterance's estimated window (two of the three words in a
        // 20 s recording) and the budget it exceeds, as measured.
        assert!(message.contains("15333ms"), "{message}");
        assert!(message.contains("1000ms"), "{message}");
        assert!(message.contains("no output was written"), "{message}");
        assert!(!message.contains("internal error"), "{message}");

        let FaInputDocument::Active(active) = pending.attempt() else {
            panic!("pending retry")
        };
        let grouping = group_utterances(
            &active.chat_file,
            30_000,
            &recording,
            &AnchorIndex::not_observed(),
        );
        assert!(!grouping.groups.is_empty());
        let (admission, _) = active
            .admission
            .admit_grouping(grouping, &active.chat_file)
            .expect("in-budget request can proceed");
        let output = crate::types::results::FaResult::without_groups(
            active.chat_file,
            crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
            "test",
            &crate::engine_reports::FaCacheNamespace::for_test("test"),
        );
        assert!(
            admission.finish(output).is_err(),
            "a request plan is not complete timing evidence"
        );

        // A pending source can acquire timing before grouping, from UTR or a
        // checked prior document. Group refusal must not erase that observation.
        let measured = read_fa_source(SOURCE).expect("checked measured source");
        let measured_bullet = measured
            .document()
            .utterances()
            .next()
            .expect("source utterance")
            .main
            .content
            .bullet
            .clone();
        let FaInputDocument::Active(mut active) = pending.attempt() else {
            panic!("pending attempt")
        };
        for line in active.chat_file.lines.as_mut_slice() {
            if let talkbank_model::model::Line::Utterance(utterance) = line {
                utterance.main.content.bullet = measured_bullet.clone();
            }
        }
        let grouping = group_utterances(
            &active.chat_file,
            100,
            &recording,
            &AnchorIndex::not_observed(),
        );
        assert!(
            grouping.groups.is_empty(),
            "known window is over this budget"
        );
        let (admission, _) = active
            .admission
            .admit_grouping(grouping, &active.chat_file)
            .expect("restored timing is not still absent");
        let output = crate::types::results::FaResult::without_groups(
            active.chat_file,
            crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
            "test",
            &crate::engine_reports::FaCacheNamespace::for_test("test"),
        );
        let admitted = admission
            .finish(output)
            .expect("restored linkage is written, its words untimed");
        assert!(
            matches!(
                admitted.into_document().shortfalls.as_slice(),
                [crate::api::OutputShortfallRecord::TimingIncomplete {
                    required_words: 3,
                    untimed_words: 3,
                    ..
                }]
            ),
            "restored media linkage is not complete lexical alignment"
        );
    }

    #[test]
    fn word_only_regeneration_is_pending_across_retries_and_cannot_write_untimed_output() {
        let word_only = SOURCE.replace(" . \u{15}0_500\u{15}", " .");
        let corrupt = pending_word_only(&word_only);
        let working =
            read_fa_source(&corrupt).expect("recorded source timing permits regeneration");
        assert_eq!(word_count(working.document()), 0);
        assert!(working.unchanged().is_none());
        for _ in 0..2 {
            let FaInputDocument::Active(active) = working.attempt() else {
                panic!("pending timing cannot certify unchanged output");
            };
            assert!(active.admission.pending_timing.is_some());
            let output = crate::types::results::FaResult::without_groups(
                active.chat_file,
                crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
                "test",
                &crate::engine_reports::FaCacheNamespace::for_test("test"),
            );
            assert!(
                matches!(
                    active.admission.finish(output),
                    Err(ServerError::RequiredEvidenceUnavailable(
                        crate::error::MissingRequiredEvidence::TimingRegeneration(_)
                    ))
                ),
                "unfulfilled timing is unavailable evidence, not an internal validity failure"
            );
        }
        let FaInputDocument::Active(pending_attempt) = working.attempt() else {
            panic!("active pending input")
        };
        let restored = read_fa_source(&word_only.replace(
            "@End\n",
            "*CHI:\tagain .\n%wor:\tagain \u{15}600_700\u{15} .\n@End\n",
        ))
        .expect("replacement fixture has measured timing");
        let output = crate::types::results::FaResult::without_groups(
            restored.document().clone(),
            crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
            "test",
            &crate::engine_reports::FaCacheNamespace::for_test("test"),
        );
        pending_attempt
            .admission
            .finish(output)
            .expect("complete measured output discharges the pending obligation");
        let valid = read_fa_source(&word_only).expect("valid word-only timing remains admitted");
        let FaInputDocument::Active(active) = valid.attempt() else {
            panic!("ordinary active input")
        };
        assert!(active.admission.pending_timing.is_none());
        assert_eq!(word_count(&active.chat_file), 1);
        let no_align = corrupt.replace("@Begin", "@Begin\n@Options:\tNoAlign");
        assert!(
            read_fa_source(&no_align).is_err(),
            "NoAlign must not defer the obligation"
        );
        let untimed = word_only.replace(
            "%wor:\thello \u{15}0_200\u{15} world \u{15}200_500\u{15} .\n",
            "",
        );
        // No timing was removed, and none was ever there: linked media with
        // no timing is the state before a first alignment. It is admitted
        // with the same obligation under its own origin (it used to be
        // refused, which made a never-aligned file unalignable).
        let never_timed = read_fa_source(&untimed).expect("a never-aligned source is admitted");
        let FaInputDocument::Active(active) = never_timed.attempt() else {
            panic!("a never-aligned source is aligned")
        };
        assert!(matches!(
            active
                .admission
                .pending_timing
                .as_ref()
                .map(|pending| pending.origin),
            Some(TimingObligationOrigin::NeverTimed)
        ));
    }

    /// A word-only source every one of whose word tiers is unreadable, while
    /// the bullets they still carry are source evidence of timing, so
    /// admission removes them all and issues a pending regeneration. Chatter
    /// removes only the tiers it cannot read (a readable `%wor` beside them
    /// is retained), so each utterance's tier carries a bullet only the
    /// grammar can recover. (A zero-duration word bullet is not a fault;
    /// a reversed one is E362, see the regeneration test below.)
    fn pending_word_only(word_only: &str) -> String {
        word_only
            .replace("\u{15}0_200\u{15}", "\u{15}invalid\u{15}")
            .replace(
                "@End\n",
                "*CHI:\tagain .\n%wor:\tagain \u{15}invalid\u{15} .\n@End\n",
            )
    }

    fn word_count(document: &crate::chat_ops::ChatFile) -> usize {
        document
            .lines
            .iter()
            .filter_map(|line| line.as_utterance())
            .flat_map(|utterance| utterance.dependent_tiers.iter())
            .filter(|entry| matches!(&entry.tier, talkbank_model::model::DependentTier::Wor(_)))
            .count()
    }

    #[test]
    fn active_plan_retains_complete_and_partial_valid_word_tiers_across_attempts() {
        for source in [
            SOURCE.to_owned(),
            SOURCE.replace("\u{15}0_200\u{15}", ""),
            SOURCE.replace("hello world .", "hello [/] world ."),
        ] {
            let working = read_fa_source(&source).expect("valid source timing");
            assert_eq!(word_count(working.document()), 1);
            for _ in 0..2 {
                let FaInputDocument::Active(active) = working.attempt() else {
                    panic!("ordinary source must use active alignment");
                };
                assert_eq!(word_count(&active.chat_file), 1);
                assert_eq!(
                    active.chat_file.to_chat_string(),
                    working.document().to_chat_string()
                );
            }
        }
    }

    #[test]
    fn regeneration_removes_corrupt_word_tiers_but_does_not_exempt_retained_faults() {
        let corrupt_words = SOURCE.replace("0_200", "invalid");
        let admitted =
            read_fa_source(&corrupt_words).expect("actual word removal permits regeneration");
        assert_eq!(word_count(admitted.document()), 0);
        insta::assert_snapshot!(format!("active={}\nword_tiers={}",
            admitted.unchanged().is_none(), word_count(admitted.document())), @"
        active=true
        word_tiers=0
        ");
        // A zero-duration word bullet is valid CHAT (CLAN CHECK accepts it on
        // %wor), so it is neither removed nor refused; alignment regenerates
        // the tier from its own evidence.
        let zero_length = SOURCE.replace("0_200", "100_100");
        let working = read_fa_source(&zero_length).expect("a zero-length word bullet is admitted");
        assert_eq!(word_count(working.document()), 1);
        let no_align = zero_length.replace("@ID:", "@Options:\tNoAlign\n@ID:");
        assert!(
            read_fa_source(&no_align).is_ok(),
            "preservation keeps a zero-length word bullet"
        );
        // A reversed one is E362: its tier is removed and regenerated like
        // any other unusable word tier, and preservation, which writes the
        // input back, refuses it.
        let reversed = SOURCE.replace("0_200", "200_0");
        let working = read_fa_source(&reversed).expect("an unusable word tier is regenerated");
        assert_eq!(word_count(working.document()), 0);
        let no_align = reversed.replace("@ID:", "@Options:\tNoAlign\n@ID:");
        assert!(
            read_fa_source(&no_align).is_err(),
            "preservation does not keep a reversed word bullet"
        );
        for bad_retained in [
            corrupt_words.replace("@End", "%mor:\tnoun|hello\n@End"),
            corrupt_words.replace("hello world .", "<hello [/] world ."),
            corrupt_words.replace("@End", "@Media:\tother, audio\n@End"),
            corrupt_words.replace("0_500", "500_0"),
        ] {
            assert!(
                read_fa_source(&bad_retained).is_err(),
                "retained content is not exempt: {bad_retained}"
            );
        }
    }

    #[test]
    fn preservation_has_no_mutation_route_or_word_regeneration_exemption() {
        let no_align = SOURCE
            .replace("@ID:", "@Options:\tNoAlign\n@ID:")
            .replace('\n', "\r\n");
        let mut working = read_fa_source(&no_align).expect("valid preserved source");
        assert!(working.active_mut().is_none());
        assert_eq!(
            working.unchanged().expect("source proof").source(),
            no_align
        );
        assert_eq!(word_count(working.document()), 1);
        assert!(read_fa_source(&no_align.replace("0_200", "invalid")).is_err());
        let ca = SOURCE.replace("@ID:", "@Options:\tCA\n@ID:");
        assert!(read_fa_source(&ca).is_ok());
        assert!(read_fa_source(&ca.replace("0_200", "invalid")).is_err());
    }

    #[test]
    fn named_admission_refuses_media_filename_mismatch_before_any_working_state() {
        let error = read_fa_source_named(
            SOURCE,
            TranscriptName::for_path(std::path::Path::new("other.cha")),
            DEFAULT_MAIN_BULLET_POLICY,
        )
        .err()
        .expect("actual filename participates in admission");
        assert!(matches!(error, ServerError::ChatReplacementAdmission(_)));
        assert!(
            read_fa_source_named(
                SOURCE,
                TranscriptName::for_path(std::path::Path::new("sample.cha")),
                DEFAULT_MAIN_BULLET_POLICY
            )
            .is_ok()
        );
    }

    /// A never-aligned transcript of `sample.cha`: no bullets, no `%wor`,
    /// and whatever `@Media` line (or none) the case supplies.
    fn never_aligned(media_line: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
{media_line}*CHI:\thello world .\n\
*CHI:\tmore words .\n@End\n"
        )
    }

    fn read_named(text: &str) -> Result<FaWorkingDocument, ServerError> {
        read_fa_source_named(
            text,
            TranscriptName::for_path(std::path::Path::new("sample.cha")),
            DEFAULT_MAIN_BULLET_POLICY,
        )
    }

    fn origin(working: &FaWorkingDocument) -> Option<TimingObligationOrigin> {
        let FaInputDocument::Active(active) = working.attempt() else {
            panic!("an aligned source is active")
        };
        active
            .admission
            .pending_timing
            .as_ref()
            .map(|pending| pending.origin)
    }

    /// THE ADMISSION TABLE (module documentation): what align accepts, as
    /// input admission decides it, before any media or inference.
    ///
    /// The regression: a transcript with no `@Media` ran UTR and FA and then
    /// failed "timed CHAT has no @Media declaration", reported as an internal
    /// error to restart; one with `@Media: name, audio` was refused E544, the
    /// condition align exists to remove. Only `, unlinked` got through.
    #[test]
    fn admission_admits_the_pre_alignment_states_and_refuses_the_rest_before_any_work() {
        // Admitted: unlinked owes nothing; linked-without-timing owes timing.
        let unlinked = read_named(&never_aligned("@Media:\tsample, audio, unlinked\n"))
            .expect("an unlinked never-aligned transcript is admitted");
        assert_eq!(origin(&unlinked), None);
        for linked in ["@Media:\tsample, audio\n", "@Media:\tsample, video\n"] {
            let working = read_named(&never_aligned(linked))
                .expect("a linked never-aligned transcript is admitted, not refused E544");
            assert_eq!(origin(&working), Some(TimingObligationOrigin::NeverTimed));
        }

        // Refused, as the input's fault, with the header change named.
        let undeclared = read_named(&never_aligned(""))
            .err()
            .expect("no @Media is refused at admission");
        assert_eq!(
            undeclared.to_string(),
            AlignmentMediaRefusal::Undeclared {
                suggested: SuggestedMediaName::Transcript("sample".to_owned())
            }
            .to_string()
        );
        for (media_line, refusal) in [
            (
                "@Media:\tsample, audio, missing\n",
                AlignmentMediaRefusal::DeclaredMissing {
                    media: "sample".to_owned(),
                },
            ),
            (
                "@Media:\tsample, missing\n",
                AlignmentMediaRefusal::DeclaredMissing {
                    media: "sample".to_owned(),
                },
            ),
            (
                "@Media:\tsample, audio, notrans\n",
                AlignmentMediaRefusal::NotTranscribed {
                    media: "sample".to_owned(),
                },
            ),
        ] {
            match read_named(&never_aligned(media_line)) {
                Err(ServerError::AlignmentMedia(actual)) => assert_eq!(actual, refusal),
                Err(other) => panic!("{media_line}: refused for another reason: {other}"),
                Ok(_) => panic!("{media_line}: admitted"),
            }
        }
        // The rendering the user sees is asserted where the category is
        // decided (`error_classification`); here, that this is that category.
        assert_eq!(
            crate::runner::util::classify_server_error(&undeclared),
            crate::scheduling::FailureCategory::Validation
        );
        assert!(
            undeclared
                .to_string()
                .contains("@Media:\tsample, audio, unlinked"),
            "{undeclared}"
        );

        // An anonymous submission still names what the header must say.
        let anonymous = read_fa_source(&never_aligned("")).err().expect("refused");
        assert!(
            anonymous
                .to_string()
                .contains("<transcript file name without extension>, audio, unlinked"),
            "{anonymous}"
        );
    }

    /// A linked source whose only timing was the bullet on a `[+ diary]`
    /// note: the fifth row of the admission table. Chatter admits it (the
    /// note's bullet is timing), but align never uses that bullet, so under
    /// `derive`, which does not write it back, admission records the
    /// obligation itself, and an output that gains no timing is refused with
    /// a message naming the note and the remedies instead of failing the
    /// output gate's generic E544. Under `keep` and `exact` the note's
    /// bullet is written as given, which is timing: nothing more is owed.
    #[test]
    fn a_linked_source_timed_only_on_a_note_owes_timing_unless_the_note_keeps_it() {
        use crate::chat_ops::fa::MainBulletPolicy;
        use crate::error::OffRecordTiming;
        let source = never_aligned("@Media:\tsample, audio\n").replace(
            "*CHI:\tmore words .\n",
            "*CHI:\tmore words .\n*CHI:\tthe note . [+ diary] \u{15}1000_2000\u{15}\n",
        );
        let named = || TranscriptName::for_path(std::path::Path::new("sample.cha"));
        let derived = read_fa_source_named(&source, named(), MainBulletPolicy::DeriveFromWords)
            .expect("the note's bullet makes the source timed, so it is admitted");
        assert_eq!(
            origin(&derived),
            Some(TimingObligationOrigin::OffRecordTimingOnly(
                OffRecordTiming::BulletNotKept
            ))
        );
        for policy in [MainBulletPolicy::KeepGiven, MainBulletPolicy::KeepExact] {
            let kept = read_fa_source_named(&source, named(), policy)
                .expect("a kept note bullet is admitted");
            assert_eq!(origin(&kept), None, "{policy:?}: the kept bullet is timing");
        }
        // A note with only word timing: no bullet for any policy to keep, so
        // the obligation stands under `keep` too, without the keep remedy;
        // `exact` cannot meet it and is refused before any work.
        let word_timed = never_aligned("@Media:\tsample, audio\n").replace(
            "*CHI:\tmore words .\n",
            "*CHI:\tmore words .\n*CHI:\tthe note . [+ diary]\n\
             %wor:\tthe \u{15}1000_1200\u{15} note \u{15}1200_1500\u{15} .\n",
        );
        for policy in [
            MainBulletPolicy::DeriveFromWords,
            MainBulletPolicy::KeepGiven,
        ] {
            let working = read_fa_source_named(&word_timed, named(), policy)
                .expect("the note's word timing makes the source timed");
            assert_eq!(
                origin(&working),
                Some(TimingObligationOrigin::OffRecordTimingOnly(
                    OffRecordTiming::WordTimingOnly
                )),
                "{policy:?}"
            );
        }
        assert!(matches!(
            read_fa_source_named(&word_timed, named(), MainBulletPolicy::KeepExact),
            Err(ServerError::AlignmentMedia(
                AlignmentMediaRefusal::LinkedUntimedUnderExactBullets
            ))
        ));

        // Speech with timing of its own owes nothing more, note or not.
        let speech_timed = source.replace("hello world .\n", "hello world . \u{15}0_500\u{15}\n");
        assert_eq!(origin(&read_named(&speech_timed).expect("admitted")), None);

        // The output owes timing: one that gains none is refused as
        // unavailable evidence, with the note named and the remedies.
        let FaInputDocument::Active(active) = derived.attempt() else {
            panic!("an ordinary source is aligned")
        };
        let output = crate::types::results::FaResult::without_groups(
            active.chat_file,
            crate::chat_ops::fa::WordGapHealing::PreserveMeasured,
            "test",
            &crate::engine_reports::FaCacheNamespace::for_test("test"),
        );
        let Err(refused) = active.admission.finish(output) else {
            panic!("a linked output with no timing cannot be written");
        };
        assert_eq!(
            crate::runner::util::classify_server_error(&refused),
            crate::scheduling::FailureCategory::EvidenceUnavailable
        );
        let message = refused.to_string();
        for phrase in ["[+ diary]", "--main-bullets keep", ", unlinked", "E544"] {
            assert!(message.contains(phrase), "{phrase}: {message}");
        }
        assert!(message.contains("no output was written"), "{message}");
    }

    /// Admitting linked-without-timing defers E544 and nothing else: every
    /// other refusal of the same source stands, with its own finding and
    /// without E544 (regeneration admission deferred it), and a file align
    /// will not change keeps its E544.
    #[test]
    fn the_never_timed_admission_exempts_only_the_linkage_obligation() {
        let linked = never_aligned("@Media:\tsample, audio\n");
        for (case, source, expected, e544) in [
            (
                "undeclared speaker",
                linked.replace("*CHI:\tmore", "*MOT:\tmore"),
                "E308",
                false,
            ),
            (
                "wrong media name",
                linked.replace("sample, audio", "other, audio"),
                "E531",
                false,
            ),
            (
                "NoAlign writes the source back, so its E544 stands",
                linked.replace("@ID:", "@Options:\tNoAlign\n@ID:"),
                "E544",
                true,
            ),
        ] {
            let failure = match read_named(&source) {
                Err(ServerError::ChatReplacementAdmission(failure)) => failure,
                Err(other) => panic!("{case}: refused for another reason: {other}"),
                Ok(_) => panic!("{case}: admitted"),
            };
            assert!(!failure.has_internal_failure(), "{case}: {failure}");
            let codes: Vec<&str> = failure
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.severity == talkbank_model::Severity::Error)
                .map(|diagnostic| diagnostic.code.as_str())
                .collect();
            assert!(codes.contains(&expected), "{case}: {codes:?}");
            assert_eq!(codes.contains(&"E544"), e544, "{case}: {codes:?}");
        }
        // `unlinked` beside timing contradicts itself (E552) and stays refused.
        let timed_unlinked = SOURCE.replace("sample, audio", "sample, audio, unlinked");
        assert!(matches!(
            read_named(&timed_unlinked),
            Err(ServerError::ChatReplacementAdmission(_))
        ));
    }

    /// `--main-bullets exact` keeps untimed utterances untimed and admits no
    /// UTR, so a linked source with no timing could never be written: it is
    /// refused at admission, as input, instead of after FA. The same source
    /// under `unlinked` owes nothing and is admitted.
    #[test]
    fn exact_bullets_refuse_a_linked_untimed_source_before_any_work() {
        use crate::chat_ops::fa::MainBulletPolicy;
        let named = || TranscriptName::for_path(std::path::Path::new("sample.cha"));
        let refused = read_fa_source_named(
            &never_aligned("@Media:\tsample, audio\n"),
            named(),
            MainBulletPolicy::KeepExact,
        )
        .err()
        .expect("exact cannot meet a linked source's obligation");
        assert!(
            matches!(
                refused,
                ServerError::AlignmentMedia(AlignmentMediaRefusal::LinkedUntimedUnderExactBullets)
            ),
            "{refused}"
        );
        assert_eq!(
            crate::runner::util::classify_server_error(&refused),
            crate::scheduling::FailureCategory::Validation
        );
        for policy in [
            MainBulletPolicy::DeriveFromWords,
            MainBulletPolicy::KeepGiven,
        ] {
            assert!(
                read_fa_source_named(&never_aligned("@Media:\tsample, audio\n"), named(), policy)
                    .is_ok()
            );
        }
        assert!(
            read_fa_source_named(
                &never_aligned("@Media:\tsample, audio, unlinked\n"),
                named(),
                MainBulletPolicy::KeepExact,
            )
            .is_ok()
        );
    }

    /// The admission's media decision and Chatter's transition agree on every
    /// declaration: what admission accepts, the transition takes once the
    /// document is timed, and what admission refuses, the transition refuses
    /// too. This is the mirror's only guard until the pinned Chatter exposes
    /// the transition's precondition as a type (see [`AlignableMedia`]).
    #[test]
    fn admitted_media_is_exactly_what_the_transition_accepts() {
        let parser = crate::chat_parser();
        for media_lines in [
            "",
            "@Media:\tsample, audio\n",
            "@Media:\tsample, video\n",
            "@Media:\tsample, audio, unlinked\n",
            "@Media:\tsample, video, unlinked\n",
            "@Media:\tsample, audio, missing\n",
            "@Media:\tsample, missing\n",
            "@Media:\tsample, audio, notrans\n",
            "@Media:\tsample, audio\n@Media:\tsample, video\n",
            "@Media:\tsample, tape\n",
            "@Media:\tsample, audio, pending\n",
        ] {
            let timed = never_aligned(media_lines)
                .replace("hello world .\n", "hello world . \u{15}0_500\u{15}\n");
            agree(&parser, media_lines, &timed);
        }
        // A declaration after the first utterance: both decisions scan every
        // header line, so both see it.
        let late = never_aligned("")
            .replace("hello world .\n", "hello world . \u{15}0_500\u{15}\n")
            .replace(
                "*CHI:\tmore",
                "@Media:\tsample, audio, unlinked\n*CHI:\tmore",
            );
        agree(&parser, "late @Media", &late);
    }

    /// One case of the agreement: admission accepts exactly when the
    /// transition does, and an admitted declaration leaves the transition
    /// linked, with the admitted linkage's `unlinked` consumed.
    fn agree(parser: &batchalign_transform::parse::TreeSitterParser, case: &str, timed: &str) {
        use batchalign_transform::media_timing::MediaTimingState;
        let (file, _) = batchalign_transform::parse::parse_lenient(parser, timed);
        let admitted = AlignableMedia::admit(
            &file,
            &TranscriptName::for_path(std::path::Path::new("sample.cha")),
        );
        let transition = reconcile_media_timing(file);
        assert_eq!(
            admitted.is_ok(),
            transition.is_ok(),
            "{case:?}: admission {admitted:?}, transition {transition:?}"
        );
        if let (Ok(_), Ok(state)) = (&admitted, &transition) {
            let MediaTimingState::Timed(linked) = state else {
                panic!("{case:?}: a timed document takes the timed transition");
            };
            let media = linked.as_chat_file().media.as_ref().expect("linked media");
            assert_eq!(media.status, None, "{case:?}: the result is linked");
        }
    }
}
