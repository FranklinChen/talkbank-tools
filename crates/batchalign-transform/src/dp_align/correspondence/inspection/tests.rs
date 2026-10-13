use super::*;
struct Count {
    work: Cell<usize>,
    peak: Cell<usize>,
}
impl WorkControl for Count {
    type Error = std::convert::Infallible;
    fn check(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn alignment_work(&self, units: usize) -> Result<(), Self::Error> {
        self.work.set(self.work.get() + units);
        Ok(())
    }
    fn alignment_score_storage(&self, cells: usize) -> Result<(), Self::Error> {
        self.peak.set(self.peak.get().max(cells));
        Ok(())
    }
}
fn inspect(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
    work: usize,
    storage: usize,
) -> InspectionAdmission {
    let mut pending = PendingInspection::bind(payload, reference, mode, &Uninterrupted);
    pending.meter.work_limit = work;
    pending.meter.storage_limit = storage;
    uninterrupted(pending.finish())
}
#[test]
fn exact_work_and_storage_boundaries_preserve_the_original_selected_path() {
    let controls = [
        (vec![], vec!["other".to_owned()]),
        (
            vec!["prefix".to_owned(), "x".to_owned(), "suffix".to_owned()],
            vec!["prefix".to_owned(), "y".to_owned(), "suffix".to_owned()],
        ),
        (
            vec!["source".to_owned(); 128],
            vec!["reference".to_owned(); 128],
        ),
        (
            vec!["a".to_owned(), "b".to_owned(), "a".to_owned()],
            vec!["b".to_owned(), "a".to_owned(), "a".to_owned()],
        ),
    ];
    for (payload, reference) in controls {
        for mode in [
            MatchMode::Exact,
            MatchMode::CaseInsensitive,
            MatchMode::Fuzzy { threshold: 0.85 },
        ] {
            let count = Count {
                work: Cell::new(0),
                peak: Cell::new(0),
            };
            let original = uninterrupted(super::super::super::align_controlled(
                &payload, &reference, mode, &count,
            ));
            let InspectionAdmission::Complete(complete) = inspect(
                &payload,
                &reference,
                mode,
                count.work.get(),
                count.peak.get(),
            ) else {
                panic!("exact measured limits must complete")
            };
            assert_eq!(complete.path(), original);
            let InspectionAdmission::BudgetExhausted(refusal) = inspect(
                &payload,
                &reference,
                mode,
                count.work.get() - 1,
                count.peak.get(),
            ) else {
                panic!("one less work unit must refuse")
            };
            assert_eq!(refusal.kind(), InspectionLimit::Work);
            assert!(refusal.used() <= refusal.limit());
            assert!(refusal.required_at_least() > refusal.limit());
            if count.peak.get() > 0 {
                let InspectionAdmission::BudgetExhausted(refusal) = inspect(
                    &payload,
                    &reference,
                    mode,
                    count.work.get(),
                    count.peak.get() - 1,
                ) else {
                    panic!("one less score cell must refuse")
                };
                assert_eq!(refusal.kind(), InspectionLimit::ScoreStorage);
                assert_eq!(refusal.required_at_least(), count.peak.get());
            }
        }
    }
}
#[test]
fn partial_prefix_is_not_a_completed_selection_and_common_proof_survives() {
    let payload = vec!["same".to_owned(), "payload".to_owned()];
    let reference = vec!["same".to_owned(), "reference".to_owned()];
    let observation = CorrespondenceAnalysis {
        admission: CorrespondenceAdmission::observe(&payload, &reference, MatchMode::Exact),
        inspection: inspect(&payload, &reference, MatchMode::Exact, 4, usize::MAX),
    };
    assert!(observation.selected().is_none());
    let CorrespondenceAdmission::Complete(common) = observation.admission() else {
        panic!("independent proof completes")
    };
    assert_eq!(common.pairs().collect::<Vec<_>>(), vec![(0, 0)]);
}
#[test]
fn interruption_is_never_reclassified_as_inspection_refusal() {
    struct Cancel;
    impl WorkControl for Cancel {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            Err("cancelled")
        }
    }
    let mut pending = PendingInspection::bind(&[], &[], MatchMode::Exact, &Cancel);
    pending.meter.work_limit = 0;
    assert!(matches!(pending.finish(), Err("cancelled")));
    let meter = Meter {
        control: &Cancel,
        work_limit: 0,
        storage_limit: 0,
        work: Cell::new(0),
        peak: Cell::new(0),
    };
    assert!(matches!(
        meter.alignment_work(1),
        Err(Stopped::Interrupted("cancelled"))
    ));
    assert!(matches!(
        meter.alignment_score_storage(1),
        Err(Stopped::Interrupted("cancelled"))
    ));
}
#[test]
fn work_overflow_is_a_refusal_not_a_wrapped_admission() {
    let meter = Meter {
        control: &Uninterrupted,
        work_limit: usize::MAX,
        storage_limit: usize::MAX,
        work: Cell::new(usize::MAX),
        peak: Cell::new(0),
    };
    let Err(Stopped::Budget(refusal)) = meter.alignment_work(1) else {
        panic!("overflow refuses")
    };
    assert_eq!(refusal.used(), usize::MAX);
    assert_eq!(refusal.required_at_least(), usize::MAX);
}
