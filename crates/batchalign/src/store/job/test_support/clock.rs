//! One owned store and the exact clock that supplies its observations.

use std::sync::Arc;

/// Private fields prevent pairing an unrelated manual clock with a store.
pub struct ManualClockStore {
    store: Arc<crate::store::JobStore>,
    clock: Arc<crate::clock::ManualClock>,
}

impl ManualClockStore {
    pub(crate) fn at(config: crate::config::ServerConfig, start: crate::api::MachineTime) -> Self {
        let clock = Arc::new(crate::clock::ManualClock::at(start));
        let (tx, _) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let store = Arc::new(crate::store::JobStore::for_tests(
            config,
            None,
            tx,
            clock.clone(),
        ));
        Self { store, clock }
    }

    pub(crate) fn store(&self) -> &Arc<crate::store::JobStore> {
        &self.store
    }

    pub(crate) fn now(&self) -> crate::api::MachineTime {
        crate::clock::Clock::now(self.clock.as_ref())
    }

    pub(crate) fn advance(&self, duration: std::time::Duration) {
        self.clock.advance(duration);
    }
}
