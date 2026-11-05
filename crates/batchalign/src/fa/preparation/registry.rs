//! One preparation owner per immutable input scope in a planning epoch.
//!
//! A published reference is reusable evidence, never an execution permit.
//! The registry alone owns materialized inputs, then consumes them at freeze.

use std::collections::{HashMap, HashSet};
use std::future::Future;

use super::{FaResourceConstraintV2, RequestScope, ResourceProbeOutcome};

/// Only an unclaimed scope can issue the capability to perform preparation.
/// No clone or public constructor exists.
pub(in crate::fa) struct PreparationWorkPermit {
    attempt: uuid::Uuid,
}

impl PreparationWorkPermit {
    pub(super) fn attempt(&self) -> uuid::Uuid {
        self.attempt
    }
}

/// Read-only identity of one completed preparation; copying this reference
/// never copies its artifacts or grants another materialization capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PreparedInputRef(uuid::Uuid);

enum PreparationRecord<P> {
    Claimed {
        attempt: uuid::Uuid,
    },
    Ready {
        reference: PreparedInputRef,
        prepared: P,
    },
    Refused(FaResourceConstraintV2),
}

pub(super) struct PreparationRegistry<P> {
    entries: HashMap<RequestScope, PreparationRecord<P>>,
}

#[derive(Debug)]
pub(super) enum PreparationRegistryError<E> {
    Operation(E),
    UnsettledClaim { attempt: uuid::Uuid },
}

type FrozenPreparations<S, P> = (HashMap<RequestScope, P>, ResidualPreparations<S, P>);

impl<P> PreparationRegistry<P> {
    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub(super) async fn prepare<E, F: Future<Output = Result<ResourceProbeOutcome<P>, E>>>(
        &mut self,
        scope: RequestScope,
        work: impl FnOnce(PreparationWorkPermit) -> F,
    ) -> Result<ResourceProbeOutcome<PreparedInputRef>, PreparationRegistryError<E>> {
        use std::collections::hash_map::Entry;
        match self.entries.entry(scope) {
            Entry::Occupied(entry) => match entry.get() {
                PreparationRecord::Ready { reference, .. } => {
                    Ok(ResourceProbeOutcome::Prepared(*reference))
                }
                PreparationRecord::Refused(cause) => {
                    Ok(ResourceProbeOutcome::Limited(cause.clone()))
                }
                // An interrupted operation does not silently recover the right
                // to run again. The planning epoch must be abandoned explicitly.
                PreparationRecord::Claimed { attempt } => {
                    Err(PreparationRegistryError::UnsettledClaim { attempt: *attempt })
                }
            },
            Entry::Vacant(entry) => {
                let attempt = uuid::Uuid::new_v4();
                let slot = entry.insert(PreparationRecord::Claimed { attempt });
                let outcome = work(PreparationWorkPermit { attempt })
                    .await
                    .map_err(PreparationRegistryError::Operation)?;
                match outcome {
                    ResourceProbeOutcome::Prepared(prepared) => {
                        let reference = PreparedInputRef(attempt);
                        *slot = PreparationRecord::Ready {
                            reference,
                            prepared,
                        };
                        Ok(ResourceProbeOutcome::Prepared(reference))
                    }
                    ResourceProbeOutcome::Limited(cause) => {
                        *slot = PreparationRecord::Refused(cause.clone());
                        Ok(ResourceProbeOutcome::Limited(cause))
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn freeze(
        self,
        selected: SelectedPreparations,
    ) -> Result<HashMap<RequestScope, P>, UnknownPreparedSelection> {
        self.freeze_retaining(selected, HashMap::<_, ()>::new())
            .map(|(inputs, _)| inputs)
    }

    /// Transfer selected owners to primary execution and retain only explicitly
    /// admitted source scopes for the optional terminal retry. A selected owner
    /// becomes settled evidence, never a second materialization capability.
    pub(super) fn freeze_retaining<S>(
        self,
        mut selected: SelectedPreparations,
        sources: HashMap<RequestScope, S>,
    ) -> Result<FrozenPreparations<S, P>, UnknownPreparedSelection> {
        let mut inputs = HashMap::new();
        let mut residual: HashMap<_, _> = sources
            .into_iter()
            .map(|(scope, source)| {
                (
                    scope,
                    ScopedPreparation {
                        source,
                        state: ResidualRecord::Unexamined,
                    },
                )
            })
            .collect();
        for (scope, state) in self.entries {
            let retained = residual.get_mut(&scope);
            match state {
                PreparationRecord::Ready {
                    reference,
                    prepared,
                } => {
                    if selected.references.remove(&reference) {
                        if let Some(retained) = retained {
                            retained.state = ResidualRecord::Settled;
                        }
                        inputs.insert(scope, prepared);
                    } else if let Some(retained) = retained {
                        retained.state = ResidualRecord::Ready(prepared);
                    }
                }
                PreparationRecord::Refused(cause) => {
                    if let Some(retained) = retained {
                        retained.state = ResidualRecord::Refused(cause);
                    }
                }
                PreparationRecord::Claimed { .. } => {
                    if let Some(retained) = retained {
                        retained.state = ResidualRecord::Settled;
                    }
                }
            }
        }
        if selected.references.is_empty() {
            Ok((inputs, ResidualPreparations { entries: residual }))
        } else {
            Err(UnknownPreparedSelection)
        }
    }
}

/// The source and its sole preparation/execution capability move together.
/// Neither the source nor a settled record can be looked up and claimed again.
pub(in crate::fa) enum RetainedPreparation<P> {
    Ready(P),
    Refused(FaResourceConstraintV2),
    Settled,
    Fresh(PreparationWorkPermit),
}

struct ScopedPreparation<S, P> {
    source: S,
    state: ResidualRecord<P>,
}

pub(in crate::fa) struct ResidualPreparations<S, P> {
    entries: HashMap<RequestScope, ScopedPreparation<S, P>>,
}

enum ResidualRecord<P> {
    Unexamined,
    Ready(P),
    Refused(FaResourceConstraintV2),
    Settled,
}

impl<S, P> ResidualPreparations<S, P> {
    /// Consuming iteration is the only retry admission route. There is no
    /// claim-by-key operation and no constructor for an unadmitted scope.
    pub(in crate::fa) fn into_scoped(self) -> impl Iterator<Item = (S, RetainedPreparation<P>)> {
        self.entries
            .into_values()
            .map(|ScopedPreparation { source, state }| {
                let capability = match state {
                    ResidualRecord::Unexamined => {
                        RetainedPreparation::Fresh(PreparationWorkPermit {
                            attempt: uuid::Uuid::new_v4(),
                        })
                    }
                    ResidualRecord::Ready(prepared) => RetainedPreparation::Ready(prepared),
                    ResidualRecord::Refused(cause) => RetainedPreparation::Refused(cause),
                    ResidualRecord::Settled => RetainedPreparation::Settled,
                };
                (source, capability)
            })
    }

    pub(in crate::fa) fn settle(&mut self, scope: RequestScope) {
        if let Some(retained) = self.entries.get_mut(&scope) {
            retained.state = ResidualRecord::Settled;
        }
    }
}

pub(super) struct SelectedPreparations {
    references: HashSet<PreparedInputRef>,
}

impl SelectedPreparations {
    pub(super) fn new() -> Self {
        Self {
            references: HashSet::new(),
        }
    }
    pub(super) fn select(&mut self, reference: PreparedInputRef) {
        self.references.insert(reference);
    }
}

#[derive(Debug)]
pub(super) struct UnknownPreparedSelection;

#[cfg(test)]
mod tests;
