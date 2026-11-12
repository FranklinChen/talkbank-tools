//! Admitted source-candidate cache receipts, before model preparation.

use std::collections::HashMap;

use super::*;
use crate::fa::cache_storage::read_values;
use crate::fa::preparation::RequestScope;

enum CandidateReceipt {
    Admitted(UnitAnswer),
    Miss,
}

pub(in crate::fa) struct CandidateCacheReceipts {
    entries: HashMap<RequestScope, CandidateReceipt>,
}

pub(in crate::fa) enum CandidateCacheState<'a> {
    Admitted(&'a UnitAnswer),
    Miss,
    Unexamined,
}

pub(in crate::fa) enum CandidateCacheAvailability {
    Admitted(UnitAnswer),
    Miss,
}

impl CandidateCacheReceipts {
    /// A child/context proposal crosses the same cache boundary as an initial
    /// candidate before receiving any decoder or model-preparation capability.
    pub(in crate::fa) async fn lookup(
        &mut self,
        request: GroupUnit<'_>,
        context: FaPreparationContext<'_>,
    ) -> CandidateCacheAvailability {
        let scope = RequestScope::of(request.words(), request.window(), context);
        let receipt = match self.entries.entry(scope) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let answer = match context.params.cache_policy {
                    crate::params::CachePolicy::SkipCache => None,
                    crate::params::CachePolicy::UseCache
                    | crate::params::CachePolicy::RequireCache => {
                        let admission = FaCacheUnitAdmission::proposed(request, context);
                        let keys = [admission.key.as_str().to_owned()];
                        let (derived, raw) = read_values(&keys, context.services).await;
                        resolve_admission(
                            admission,
                            context,
                            raw.get(&keys[0]),
                            derived.get(&keys[0]),
                        )
                        .await
                    }
                };
                entry.insert(match answer {
                    Some(answer) => CandidateReceipt::Admitted(answer),
                    None => CandidateReceipt::Miss,
                })
            }
        };
        match receipt {
            CandidateReceipt::Admitted(answer) => {
                CandidateCacheAvailability::Admitted(answer.clone())
            }
            CandidateReceipt::Miss => CandidateCacheAvailability::Miss,
        }
    }
    pub(in crate::fa) fn contains(
        &self,
        words: &[FaWord],
        window: FaWindow,
        context: FaPreparationContext<'_>,
    ) -> bool {
        matches!(
            self.entries.get(&RequestScope::of(words, window, context)),
            Some(CandidateReceipt::Admitted(_))
        )
    }
    pub(in crate::fa) fn state(&self, unit: &DispatchUnit<'_>) -> CandidateCacheState<'_> {
        match self.entries.get(&RequestScope::for_unit(unit)) {
            Some(CandidateReceipt::Admitted(answer)) => CandidateCacheState::Admitted(answer),
            Some(CandidateReceipt::Miss) => CandidateCacheState::Miss,
            None => CandidateCacheState::Unexamined,
        }
    }
}

pub(in crate::fa) async fn collect_candidate_cache(
    groups: &[FaGroup],
    context: FaPreparationContext<'_>,
) -> CandidateCacheReceipts {
    let mut receipts = CandidateCacheReceipts {
        entries: HashMap::new(),
    };
    match context.params.cache_policy {
        crate::params::CachePolicy::SkipCache => return receipts,
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {}
    }
    // Candidate keys describe proposals. They grant no inference capability;
    // the final dispatch plan is built only after resource geometry is chosen.
    let candidates = DispatchPlan::build(groups, super::RequestKeying::of(&context));
    let keys: Vec<_> = candidates
        .units()
        .map(|unit| unit.cache_key().as_str().to_owned())
        .collect();
    let (derived, raw) = read_values(&keys, context.services).await;
    for unit in candidates.units() {
        let scope = RequestScope::for_unit(unit);
        let answer = resolve(
            unit,
            context,
            raw.get(unit.cache_key().as_str()),
            derived.get(unit.cache_key().as_str()),
        )
        .await;
        receipts.entries.insert(
            scope,
            match answer {
                Some(answer) => CandidateReceipt::Admitted(answer),
                None => CandidateReceipt::Miss,
            },
        );
    }
    receipts
}

pub(super) async fn resolve(
    unit: &DispatchUnit<'_>,
    context: FaPreparationContext<'_>,
    raw: Option<&serde_json::Value>,
    derived: Option<&serde_json::Value>,
) -> Option<UnitAnswer> {
    resolve_admission(
        FaCacheUnitAdmission::new(
            unit,
            context.params.engine,
            context.services.cache_namespace,
        ),
        context,
        raw,
        derived,
    )
    .await
}

async fn resolve_admission(
    admission: FaCacheUnitAdmission<'_>,
    context: FaPreparationContext<'_>,
    raw: Option<&serde_json::Value>,
    derived: Option<&serde_json::Value>,
) -> Option<UnitAnswer> {
    if context.services.attention_capture_dir.is_some()
        && context.params.engine == crate::types::engines::FaEngineName::Whisper
        && !crate::fa::attention_capture::cached_available(raw, admission.key.as_str()).await
    {
        return None;
    }
    let resolution = admission.resolve(raw, derived);
    for refusal in resolution.refusals {
        warn!(error = %refusal.error, cache_layer = refusal.layer, key = admission.key.as_str(), "Cached FA evidence was refused");
    }
    let (timings, source) = match resolution.admitted {
        AdmittedFaCacheGroup::RawEvidence(evidence) => (
            evidence.into_timings(),
            FaEvidenceSourceTrace::RawEvidenceReplay,
        ),
        AdmittedFaCacheGroup::DerivedTimings(timings) => {
            (timings.into_timings(), FaEvidenceSourceTrace::Cache)
        }
        AdmittedFaCacheGroup::Miss => return None,
    };
    // Unbound: whoever holds the request binds it (the reply ticket, or the
    // supplied-retry path), and an entry that does not answer its request is
    // refused there and aligned again.
    Some(UnitAnswer::new(timings, source))
}
