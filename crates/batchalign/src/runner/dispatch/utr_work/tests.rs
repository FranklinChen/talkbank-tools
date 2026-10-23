use super::*;
use crate::chat_ops::fa::utr::{GlobalUtr, TwoPassOverlapUtr};
use tokio::sync::oneshot;

fn source() -> ChatFile {
    let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
    batchalign_transform::parse_source_with_parser(&parser,
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n@End\n")
        .admit(talkbank_model::model::TranscriptName::Anonymous, &talkbank_model::NullErrorSink)
        .expect("valid input").into_valid_file().into_unchecked()
}

fn tokens() -> PreparedUtrTokens {
    PreparedUtrTokens::prepare(
        vec![
            AsrTimingToken {
                text: "hello".into(),
                start_ms: 100,
                end_ms: 400,
            },
            AsrTimingToken {
                text: "world".into(),
                start_ms: 400,
                end_ms: 800,
            },
        ],
        &crate::api::LanguageCode3::eng(),
    )
    .expect("English has no script normalization to refuse")
}

#[tokio::test]
async fn completed_candidate_matches_both_synchronous_strategies() {
    for strategy in [
        Box::new(GlobalUtr) as Box<dyn UtrStrategy>,
        Box::new(TwoPassOverlapUtr::new()),
    ] {
        let mut original = source();
        let mut expected = original.clone();
        let expected_result = strategy.inject(&mut expected, &tokens());
        let observed = PendingUtr::new(&mut original, strategy, tokens())
            .run()
            .await
            .expect("completed");
        assert_eq!(
            serde_json::to_value(&original).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(observed.injected(), expected_result.injected());
        assert_eq!(observed.unmatched(), expected_result.unmatched());
        assert_eq!(observed.injected(), 1);
    }
}

struct HeldSearch {
    entered: std::sync::Mutex<Option<oneshot::Sender<std::thread::ThreadId>>>,
    release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    stopped: std::sync::Mutex<Option<oneshot::Sender<bool>>>,
}

#[tokio::test]
async fn long_document_owned_execution_preserves_recovery_and_budget_results() {
    use crate::chat_ops::fa::utr::region_tests::long_turn_fixture;
    for anchored in [false, true] {
        let (mut document, tokens) = long_turn_fixture(anchored);
        let mut expected = document.clone();
        let expected_result = GlobalUtr.inject(
            &mut expected,
            &crate::chat_ops::fa::utr::PreparedUtrTokens::spaced(&tokens),
        );
        let tokens = PreparedUtrTokens::spaced(&tokens);
        let result = PendingUtr::new(&mut document, Box::new(GlobalUtr), tokens)
            .run()
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&document).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(
            (result.injected(), result.skipped(), result.unmatched()),
            (
                expected_result.injected(),
                expected_result.skipped(),
                expected_result.unmatched()
            )
        );
        if anchored {
            assert_eq!(result.injected(), 140);
        } else {
            assert_eq!((result.injected(), result.unmatched()), (280, 0));
        }
    }
}

impl UtrStrategy for HeldSearch {
    fn inject(&self, _: &mut ChatFile, _: &PreparedUtrTokens) -> UtrResult {
        panic!("dispatch must use the controlled entry point")
    }
    fn inject_controlled(
        &self,
        candidate: &mut ChatFile,
        _: &PreparedUtrTokens,
        control: &dyn WorkControl<Error = UtrCancelled>,
    ) -> Result<UtrResult, UtrCancelled> {
        assert_eq!(GlobalUtr.inject(candidate, &tokens()).injected(), 1);
        self.entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(std::thread::current().id())
            .unwrap();
        self.release.lock().unwrap().recv().unwrap();
        let result = control.check();
        self.stopped
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(result.is_err())
            .unwrap();
        result?;
        panic!("cancelled candidate cannot complete")
    }
}

/// The default single-thread executor must be able to receive the entry signal
/// while CPU work is held elsewhere. No latency assertion or sleep is needed.
#[tokio::test]
async fn dropping_pending_work_stops_the_search_and_preserves_source() {
    let mut original = source();
    let before = serde_json::to_value(&original).unwrap();
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let (stopped_tx, stopped_rx) = oneshot::channel();
    let strategy = HeldSearch {
        entered: std::sync::Mutex::new(Some(entered_tx)),
        release: std::sync::Mutex::new(release_rx),
        stopped: std::sync::Mutex::new(Some(stopped_tx)),
    };
    let mut work = Box::pin(PendingUtr::new(&mut original, Box::new(strategy), tokens()).run());
    assert!(futures::poll!(&mut work).is_pending());
    assert_ne!(entered_rx.await.unwrap(), std::thread::current().id());
    drop(work);
    assert_eq!(serde_json::to_value(&original).unwrap(), before);
    release_tx.send(()).unwrap();
    assert!(stopped_rx.await.unwrap());
    assert_eq!(serde_json::to_value(&original).unwrap(), before);
}

struct Panics;
impl UtrStrategy for Panics {
    fn inject(&self, _: &mut ChatFile, _: &PreparedUtrTokens) -> UtrResult {
        panic!("not dispatched")
    }
    fn inject_controlled(
        &self,
        candidate: &mut ChatFile,
        _: &PreparedUtrTokens,
        _: &dyn WorkControl<Error = UtrCancelled>,
    ) -> Result<UtrResult, UtrCancelled> {
        assert_eq!(GlobalUtr.inject(candidate, &tokens()).injected(), 1);
        panic!("test producer failure")
    }
}

#[tokio::test]
async fn producer_panic_is_system_failure_and_cannot_install_candidate() {
    let mut original = source();
    let before = serde_json::to_value(&original).unwrap();
    let error = PendingUtr::new(&mut original, Box::new(Panics), tokens())
        .run()
        .await
        .expect_err("failed task");
    assert!(matches!(error, ServerError::UtrTask(_)));
    assert_eq!(
        crate::runner::util::classify_server_error(&error),
        crate::scheduling::FailureCategory::System
    );
    assert_eq!(
        axum::response::IntoResponse::into_response(error).status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(serde_json::to_value(&original).unwrap(), before);
}

struct StopAt {
    remaining: std::cell::Cell<usize>,
}
impl WorkControl for StopAt {
    type Error = UtrCancelled;
    fn check(&self) -> Result<(), UtrCancelled> {
        match self.remaining.get() {
            0 => Err(UtrCancelled),
            n => {
                self.remaining.set(n - 1);
                Ok(())
            }
        }
    }
}

#[test]
fn actual_strategies_propagate_interruption_instead_of_budget_or_unmatched() {
    for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
        let count = StopAt {
            remaining: std::cell::Cell::new(usize::MAX),
        };
        strategy
            .inject_controlled(&mut source(), &tokens(), &count)
            .unwrap();
        let checkpoints = usize::MAX - count.remaining.get();
        assert!(checkpoints > 2);
        for stop_at in 0..checkpoints {
            let control = StopAt {
                remaining: std::cell::Cell::new(stop_at),
            };
            assert!(
                strategy
                    .inject_controlled(&mut source(), &tokens(), &control)
                    .is_err(),
                "checkpoint {stop_at}"
            );
        }
    }
}

struct NeverStarted(Option<oneshot::Sender<bool>>, AtomicBool);
impl Drop for NeverStarted {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(self.1.load(Ordering::Relaxed));
        }
    }
}
impl UtrStrategy for NeverStarted {
    fn inject(&self, _: &mut ChatFile, _: &PreparedUtrTokens) -> UtrResult {
        self.1.store(true, Ordering::Relaxed);
        panic!("queued work must not start")
    }
    fn inject_controlled(
        &self,
        _: &mut ChatFile,
        _: &PreparedUtrTokens,
        _: &dyn WorkControl<Error = UtrCancelled>,
    ) -> Result<UtrResult, UtrCancelled> {
        self.1.store(true, Ordering::Relaxed);
        panic!("queued work must not start")
    }
}

#[test]
fn cancellation_aborts_queued_work_before_search_entry() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let (occupied_tx, occupied_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let occupied = crate::blocking::spawn_in_span(move || {
            occupied_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        occupied_rx.await.unwrap();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let mut original = source();
        let before = serde_json::to_value(&original).unwrap();
        let mut pending = Box::pin(
            PendingUtr::new(
                &mut original,
                Box::new(NeverStarted(Some(dropped_tx), AtomicBool::new(false))),
                tokens(),
            )
            .run(),
        );
        assert!(futures::poll!(&mut pending).is_pending());
        drop(pending);
        release_tx.send(()).unwrap();
        occupied.await.unwrap();
        assert!(!dropped_rx.await.unwrap(), "queued search never entered");
        assert_eq!(serde_json::to_value(&original).unwrap(), before);
    });
}
