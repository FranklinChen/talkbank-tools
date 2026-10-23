//! Runtime Stanza capability registry: the ONE authority on which languages
//! a Stanza-backed stage can serve, and the proof that it serves one.
//!
//! Built from the catalog a worker reports (`StanzaCatalogReport::Catalog`),
//! which the worker reads from the installed Stanza catalog (`resources.json`)
//! and Rust's ISO 639-1 table (`batchalign/worker/_stanza_capabilities.py`).
//! There is no static fallback list: a server whose workers have not reported
//! a catalog refuses with the reason it has none
//! ([`StanzaRegistryUnavailable`]), and a language becomes available when the
//! installed catalog ships its processors.
//!
//! # The proof graph
//!
//! ```text
//! worker report ──record──▶ StanzaRegistrySlot ──lookup──▶ Arc<StanzaRegistry>
//!                                                              │ admit::<R>
//!                                                              ▼
//!                                       Admitted<R> ──sibling──▶ Admitted<R>
//! ```
//!
//! [`Admitted<R>`] is minted only by [`StanzaRegistry::admit`]: it names a
//! language and holds the registry that admitted it for the requirement `R`
//! (a sealed marker, [`Morphosyntax`] or [`UtteranceSegmentation`]). A stage
//! that needs another language under the same requirement (each utterance's
//! effective language, a secondary `@s` language) asks the proof it already
//! holds, with [`Admitted::sibling`], so "no registry" cannot arise past the
//! first admission and no consumer re-reads the pool's registry. A refusal
//! is a [`StanzaGap<R>`] naming the processors the entry lacks for `R`.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use crate::api::LanguageCode3;
pub use batchalign_types::stanza_catalog::{
    StanzaCatalogReport, StanzaProcessor, StanzaReportReason, StanzaRequirement,
};

/// The sealed set of requirements a proof can be issued for.
mod sealed {
    pub trait Sealed {}
}

/// A Stanza requirement as a type, so a proof says which one it proves.
///
/// Sealed: the markers below are the only requirements a stage is admitted
/// by. `coref` has none: its gate is English-only by construction (it has no
/// `--lang`, analyzes English files only, and its worker always builds the
/// English pipeline), so its processor list in [`StanzaRequirement`] is the
/// loader's, not an admission.
pub trait Requirement: sealed::Sealed + std::fmt::Debug + Send + Sync + 'static {
    /// The processor set this requirement names.
    const REQUIREMENT: StanzaRequirement;
}

/// Morphotag's requirement (and `compare`'s, which morphotags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Morphosyntax {}

/// Utterance segmentation's Stanza constituency route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceSegmentation {}

impl sealed::Sealed for Morphosyntax {}
impl sealed::Sealed for UtteranceSegmentation {}

impl Requirement for Morphosyntax {
    const REQUIREMENT: StanzaRequirement = StanzaRequirement::Morphosyntax;
}

impl Requirement for UtteranceSegmentation {
    const REQUIREMENT: StanzaRequirement = StanzaRequirement::UtteranceSegmentation;
}

/// Why the server has no Stanza registry, as the worker reports said.
///
/// Carried by every refusal that needed a registry, so "no registry" always
/// names its cause: no worker has reported yet, a worker reported that Stanza
/// is not installed, or a worker could not build its report.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StanzaRegistryUnavailable {
    /// No worker has reported its Stanza catalog yet.
    #[error("no worker has reported its Stanza catalog yet")]
    NoReport,
    /// A worker reported that the `stanza` package does not import.
    #[error("worker {worker} reported that Stanza is not installed: {import_error}")]
    NotInstalled {
        /// The reporting worker's key label.
        worker: String,
        /// Why it does not import, as the worker saw it.
        import_error: StanzaReportReason,
    },
    /// A worker imported Stanza but could not build its catalog report.
    #[error("worker {worker} could not build its Stanza catalog report: {reason}")]
    ReportFailed {
        /// The reporting worker's key label.
        worker: String,
        /// The worker's own account of the failure.
        reason: StanzaReportReason,
    },
}

impl StanzaRegistryUnavailable {
    /// Whether this reason should replace `current` as the slot's stated
    /// reason: the most informative one is kept. A failed report says more
    /// than "not installed" (which a worker serving no Stanza task reports
    /// too), and either says more than "no report"; between two of one kind
    /// the latest wins.
    fn supersedes(&self, current: &Self) -> bool {
        match (self, current) {
            (_, Self::NoReport) => true,
            (Self::NoReport, Self::NotInstalled { .. } | Self::ReportFailed { .. }) => false,
            (Self::NotInstalled { .. }, Self::NotInstalled { .. }) => true,
            (Self::NotInstalled { .. }, Self::ReportFailed { .. }) => false,
            (Self::ReportFailed { .. }, Self::NotInstalled { .. } | Self::ReportFailed { .. }) => {
                true
            }
        }
    }
}

/// What a language's catalog entry lacks for a requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapKind {
    /// The installed catalog has no entry serving this language.
    NotInCatalog,
    /// The entry lacks these processors of the requirement (never empty).
    ProcessorsUnavailable {
        /// The requirement's processors the entry does not ship.
        missing: BTreeSet<StanzaProcessor>,
    },
}

/// Why the registry does not admit a language for the requirement `R`,
/// naming what is missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StanzaGap<R: Requirement> {
    kind: GapKind,
    _requirement: PhantomData<fn() -> R>,
}

impl<R: Requirement> StanzaGap<R> {
    fn new(kind: GapKind) -> Self {
        Self {
            kind,
            _requirement: PhantomData,
        }
    }

    /// What the entry lacks.
    pub fn kind(&self) -> &GapKind {
        &self.kind
    }
}

impl<R: Requirement> std::fmt::Display for StanzaGap<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            GapKind::NotInCatalog => {
                f.write_str("the installed Stanza catalog has no model for this language")
            }
            GapKind::ProcessorsUnavailable { missing } => write!(
                f,
                "the installed Stanza catalog lacks the {} processors ({}) for this language: \
                 missing {}",
                R::REQUIREMENT.purpose(),
                R::REQUIREMENT.processor_list(),
                missing
                    .iter()
                    .map(|processor| processor.wire_name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl<R: Requirement> std::error::Error for StanzaGap<R> {}

/// Why a language was not admitted for `R` at the first admission: there is
/// no registry, or the registry does not serve it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StanzaRefusal<R: Requirement> {
    /// The runtime has no admitted registry, for the reason it names.
    #[error("the runtime has no admitted Stanza language registry: {0}")]
    RegistryUnavailable(StanzaRegistryUnavailable),
    /// The admitted registry does not serve this language.
    #[error(transparent)]
    Gap(StanzaGap<R>),
}

/// The registry as a reader finds it: admitted, or why there is none.
pub type StanzaRegistryLookup = Result<Arc<StanzaRegistry>, StanzaRegistryUnavailable>;

/// Runtime Stanza capability registry: each language the reported catalog
/// serves, with the processors of the entry serving it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StanzaRegistry {
    languages: BTreeMap<LanguageCode3, BTreeSet<StanzaProcessor>>,
}

/// Proof that the registry it holds serves the requirement `R` for one
/// language. Minted only by [`StanzaRegistry::admit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted<R: Requirement> {
    registry: Arc<StanzaRegistry>,
    language: LanguageCode3,
    _requirement: PhantomData<fn() -> R>,
}

impl<R: Requirement> Admitted<R> {
    /// The admitted language.
    pub fn language(&self) -> &LanguageCode3 {
        &self.language
    }

    /// Admit another language under the same requirement, by the same
    /// registry: no registry lookup, so no "no registry" refusal.
    ///
    /// # Errors
    ///
    /// The registry does not serve `R` for `language`.
    pub fn sibling(&self, language: &LanguageCode3) -> Result<Admitted<R>, StanzaGap<R>> {
        self.registry.admit(language)
    }

    /// Every language the holding registry serves `R` for, in code order:
    /// what a help message lists.
    pub fn languages(&self) -> impl Iterator<Item = &LanguageCode3> {
        self.registry.languages_for::<R>()
    }
}

impl StanzaRegistry {
    /// The registry of a reported catalog. Keys and processors were parsed
    /// when the report was deserialized, so nothing here can fail. Reached
    /// only through the slot (and test doubles).
    pub(crate) fn from_catalog(
        languages: BTreeMap<LanguageCode3, BTreeSet<StanzaProcessor>>,
    ) -> Self {
        Self { languages }
    }

    /// Admit `language` for the requirement `R`: the one admission every
    /// Stanza-backed stage uses, and the only way to an [`Admitted<R>`].
    ///
    /// # Errors
    ///
    /// The catalog has no entry for the language, or the entry lacks some of
    /// `R`'s processors (named in the gap).
    pub fn admit<R: Requirement>(
        self: &Arc<Self>,
        language: &LanguageCode3,
    ) -> Result<Admitted<R>, StanzaGap<R>> {
        let shipped = self
            .languages
            .get(language)
            .ok_or_else(|| StanzaGap::new(GapKind::NotInCatalog))?;
        let missing: BTreeSet<StanzaProcessor> = R::REQUIREMENT
            .processors()
            .iter()
            .filter(|required| !shipped.contains(required))
            .copied()
            .collect();
        if missing.is_empty() {
            Ok(Admitted {
                registry: Arc::clone(self),
                language: language.clone(),
                _requirement: PhantomData,
            })
        } else {
            Err(StanzaGap::new(GapKind::ProcessorsUnavailable { missing }))
        }
    }

    /// Every language the catalog serves `R` for, in code order.
    pub fn languages_for<R: Requirement>(&self) -> impl Iterator<Item = &LanguageCode3> {
        self.languages.iter().filter_map(|(language, shipped)| {
            R::REQUIREMENT
                .processors()
                .iter()
                .all(|required| shipped.contains(required))
                .then_some(language)
        })
    }

    /// The catalog this registry was built from, for a test pool.
    #[cfg(test)]
    pub(crate) fn into_catalog(self) -> BTreeMap<LanguageCode3, BTreeSet<StanzaProcessor>> {
        self.languages
    }
}

/// Admit `language` for `R` from a lookup: the registry's refusal, or the
/// reason there is no registry.
///
/// # Errors
///
/// No admitted registry, or a gap.
pub fn admit_from<R: Requirement>(
    lookup: &StanzaRegistryLookup,
    language: &LanguageCode3,
) -> Result<Admitted<R>, StanzaRefusal<R>> {
    match lookup {
        Ok(registry) => registry.admit(language).map_err(StanzaRefusal::Gap),
        Err(unavailable) => Err(StanzaRefusal::RegistryUnavailable(unavailable.clone())),
    }
}

/// The pool's Stanza registry: built from the first catalog a worker
/// reports, and until then the most informative reason it has none.
///
/// The one place a report reaches the registry ([`Self::record`]), and the one
/// place it is read ([`Self::current`]). Once a catalog is admitted it stays.
#[derive(Debug)]
pub(crate) struct StanzaRegistrySlot {
    admitted: OnceLock<Arc<StanzaRegistry>>,
    unavailable: Mutex<StanzaRegistryUnavailable>,
}

impl StanzaRegistrySlot {
    /// A slot no worker has reported to.
    pub(crate) fn new() -> Self {
        Self {
            admitted: OnceLock::new(),
            unavailable: Mutex::new(StanzaRegistryUnavailable::NoReport),
        }
    }

    /// Record one worker's report. A catalog becomes the registry if none is
    /// admitted yet; any other report becomes the reason there is none, when
    /// it says more than the reason already held.
    pub(crate) fn record(&self, worker: String, report: StanzaCatalogReport) {
        let reason = match report {
            StanzaCatalogReport::Catalog { languages } => {
                let reported = StanzaRegistry::from_catalog(languages);
                // The first admitted catalog stays. A worker on another
                // installation (a TCP worker on another host) may read a
                // different one; that is said, since the languages admitted
                // here are then not the ones every worker can serve.
                let admitted = self.admitted.get_or_init(|| Arc::new(reported.clone()));
                if **admitted != reported {
                    tracing::warn!(
                        %worker,
                        "A worker reported a Stanza catalog that differs from the admitted one; \
                         admission keeps the first"
                    );
                }
                return;
            }
            StanzaCatalogReport::NotInstalled { import_error } => {
                StanzaRegistryUnavailable::NotInstalled {
                    worker,
                    import_error,
                }
            }
            StanzaCatalogReport::Failed { reason } => {
                StanzaRegistryUnavailable::ReportFailed { worker, reason }
            }
        };
        let mut current = crate::worker::pool::lock_recovered(&self.unavailable);
        if reason.supersedes(&current) {
            *current = reason;
        }
    }

    /// The admitted registry, or why there is none.
    pub(crate) fn current(&self) -> StanzaRegistryLookup {
        self.admitted
            .get()
            .cloned()
            .ok_or_else(|| crate::worker::pool::lock_recovered(&self.unavailable).clone())
    }
}

/// Test doubles for a worker's report.
#[cfg(test)]
impl StanzaRegistry {
    /// A registry whose catalog serves each of `codes` with every processor
    /// morphotag loads, and nothing else.
    pub(crate) fn admitting(codes: &[&str]) -> Self {
        let codes: Vec<(&str, &[StanzaProcessor])> = codes
            .iter()
            .map(|&code| (code, StanzaRequirement::Morphosyntax.processors()))
            .collect();
        Self::serving(&codes)
    }

    /// A registry whose catalog serves each language with exactly the
    /// processors listed for it.
    pub(crate) fn serving(entries: &[(&str, &[StanzaProcessor])]) -> Self {
        Self::from_catalog(
            entries
                .iter()
                .map(|&(code, processors)| {
                    (
                        LanguageCode3::try_new(code).expect("test languages are ISO 639-3"),
                        processors.iter().copied().collect(),
                    )
                })
                .collect(),
        )
    }

    /// This registry, shared, as the slot holds it.
    pub(crate) fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// A test report reason.
#[cfg(test)]
pub(crate) fn reason(text: &str) -> StanzaReportReason {
    StanzaReportReason::try_from(text.to_owned()).expect("test reasons are not blank")
}

#[cfg(test)]
mod tests {
    use super::*;
    use StanzaProcessor::{Constituency, Depparse, Lemma, Mwt, Pos, Tokenize};

    fn language(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("test language is ISO 639-3")
    }

    /// Kazakh as the Stanza 1.15 catalog ships it, Faroese without a
    /// lemmatizer.
    fn registry() -> Arc<StanzaRegistry> {
        StanzaRegistry::serving(&[
            ("kaz", &[Tokenize, Mwt, Pos, Lemma, Depparse]),
            ("fao", &[Tokenize, Mwt, Pos, Depparse]),
        ])
        .shared()
    }

    #[test]
    fn a_language_whose_entry_has_the_morphotag_processors_is_admitted() {
        let admitted = registry()
            .admit::<Morphosyntax>(&language("kaz"))
            .expect("Kazakh ships every morphotag processor");
        assert_eq!(admitted.language(), &language("kaz"));
    }

    #[test]
    fn a_language_with_no_entry_is_not_in_the_catalog() {
        assert_eq!(
            registry()
                .admit::<Morphosyntax>(&language("que"))
                .map(drop)
                .map_err(|gap| gap.kind().clone()),
            Err(GapKind::NotInCatalog)
        );
    }

    /// RED FIRST: a gap named the requirement, not what the entry lacked, so
    /// an operator could not tell which processor to install.
    #[test]
    fn an_entry_missing_a_morphotag_processor_names_what_it_lacks() {
        let gap = registry()
            .admit::<Morphosyntax>(&language("fao"))
            .expect_err("Faroese ships no lemmatizer");
        assert_eq!(
            gap.kind(),
            &GapKind::ProcessorsUnavailable {
                missing: BTreeSet::from([Lemma])
            }
        );
        assert_eq!(
            gap.to_string(),
            "the installed Stanza catalog lacks the morphology processors (tokenize, pos, \
             lemma, depparse) for this language: missing lemma"
        );
    }

    #[test]
    fn the_listed_languages_are_the_admitted_ones() {
        let registry = registry();
        let listed: Vec<&str> = registry
            .languages_for::<Morphosyntax>()
            .map(AsRef::as_ref)
            .collect();
        assert_eq!(listed, ["kaz"]);
    }

    /// A sibling is admitted by the registry the first proof holds, under the
    /// same requirement, whatever the pool's slot holds now.
    #[test]
    fn a_sibling_is_admitted_by_the_same_registry() {
        let admitted = registry()
            .admit::<Morphosyntax>(&language("kaz"))
            .expect("Kazakh");
        assert_eq!(
            admitted
                .sibling(&language("fao"))
                .map(drop)
                .map_err(|gap| gap.kind().clone()),
            Err(GapKind::ProcessorsUnavailable {
                missing: BTreeSet::from([Lemma])
            })
        );
        assert_eq!(
            admitted
                .sibling(&language("kaz"))
                .expect("itself")
                .language(),
            &language("kaz")
        );
    }

    /// Each requirement is admitted by its own processors: utterance
    /// segmentation needs no lemmatizer, and morphotag needs one.
    #[test]
    fn each_requirement_is_admitted_by_its_own_processors() {
        let registry =
            StanzaRegistry::serving(&[("vie", &[Tokenize, Pos, Depparse, Constituency])]).shared();
        registry
            .admit::<UtteranceSegmentation>(&language("vie"))
            .expect("constituency without a lemmatizer segments");
        assert_eq!(
            registry
                .admit::<Morphosyntax>(&language("vie"))
                .map(drop)
                .map_err(|gap| gap.kind().clone()),
            Err(GapKind::ProcessorsUnavailable {
                missing: BTreeSet::from([Lemma])
            })
        );
    }

    /// RED FIRST: a worker whose table build failed (a stale extension
    /// lacking the ISO table) reported an empty catalog, indistinguishable
    /// from "no Stanza", and morphotag refused every language with no reason.
    /// The slot now keeps the worker's reason until a catalog arrives, and a
    /// later "not installed" (which a worker serving no Stanza task reports)
    /// no longer overwrites it: that report used to replace the failure.
    #[test]
    fn the_most_informative_reason_is_kept() {
        let slot = StanzaRegistrySlot::new();
        assert_eq!(
            slot.current().map(drop),
            Err(StanzaRegistryUnavailable::NoReport)
        );
        slot.record(
            "infer:fa:eng".to_owned(),
            StanzaCatalogReport::NotInstalled {
                import_error: reason("No module named 'stanza'"),
            },
        );
        assert_eq!(
            slot.current().map(drop),
            Err(StanzaRegistryUnavailable::NotInstalled {
                worker: "infer:fa:eng".to_owned(),
                import_error: reason("No module named 'stanza'"),
            })
        );
        let failure = StanzaRegistryUnavailable::ReportFailed {
            worker: "infer:morphosyntax:eng".to_owned(),
            reason: reason("ImportError: cannot import name 'ISO_639_1_PAIRS'"),
        };
        slot.record(
            "infer:morphosyntax:eng".to_owned(),
            StanzaCatalogReport::Failed {
                reason: reason("ImportError: cannot import name 'ISO_639_1_PAIRS'"),
            },
        );
        assert_eq!(slot.current().map(drop), Err(failure.clone()));
        slot.record(
            "infer:fa:spa".to_owned(),
            StanzaCatalogReport::NotInstalled {
                import_error: reason("No module named 'stanza'"),
            },
        );
        let refusal = slot.current().map(drop).expect_err("no catalog yet");
        assert_eq!(
            refusal, failure,
            "a later not-installed does not hide the failure"
        );
        assert!(refusal.to_string().contains("ISO_639_1_PAIRS"), "{refusal}");
    }

    /// The first catalog is the registry, and a later non-catalog report
    /// does not take it away.
    #[test]
    fn a_catalog_once_admitted_stays() {
        let slot = StanzaRegistrySlot::new();
        slot.record(
            "infer:morphosyntax:per-file".to_owned(),
            StanzaCatalogReport::Catalog {
                languages: BTreeMap::from([(
                    language("kaz"),
                    StanzaRequirement::Morphosyntax
                        .processors()
                        .iter()
                        .copied()
                        .collect(),
                )]),
            },
        );
        slot.record(
            "infer:fa:eng".to_owned(),
            StanzaCatalogReport::NotInstalled {
                import_error: reason("No module named 'stanza'"),
            },
        );
        let registry = slot.current().expect("the catalog was admitted");
        registry
            .admit::<Morphosyntax>(&language("kaz"))
            .expect("Kazakh stays admitted");
    }
}
