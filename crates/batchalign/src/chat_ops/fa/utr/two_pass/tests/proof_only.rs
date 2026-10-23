use super::tests::{expected_interval, make_asr_tokens};
use super::*;

#[test]
fn proof_only_overlap_avoids_unused_quadratic_inspection() {
    struct PollAllowance(std::cell::Cell<usize>);
    impl WorkControl for PollAllowance {
        type Error = ();
        fn check(&self) -> Result<(), ()> {
            let remaining = self.0.get().checked_sub(1).ok_or(())?;
            self.0.set(remaining);
            Ok(())
        }
    }
    // Sparse, unique endpoints with a large disjoint middle. Proof still
    // scans the inputs, but overlap recovery has no inspection consumer.
    let mut words = vec!["start".to_owned()];
    words.extend(vec!["payload".to_owned(); 512]);
    words.push("finish".to_owned());
    let mut tokens = make_asr_tokens(&[("start", 100, 200)]);
    tokens.extend(make_asr_tokens(&vec![("reference", 300, 400); 512]));
    tokens.extend(make_asr_tokens(&[("finish", 500, 600)]));
    let reference = super::super::lexical::UtrLexicalStream::within_window(
        &super::super::PreparedUtrTokens::spaced(&tokens),
        0,
        1000,
    )
    .texts();
    let counter = PollAllowance(std::cell::Cell::new(usize::MAX));
    let proof = dp_align::CorrespondenceAdmission::observe_controlled(
        &words,
        &reference,
        MatchMode::Exact,
        &counter,
    )
    .unwrap();
    let dp_align::CorrespondenceAdmission::Complete(common) = proof else {
        panic!("sparse endpoint proof must complete");
    };
    assert_eq!(common.pairs().collect::<Vec<_>>(), vec![(0, 0), (513, 513)]);
    // The caller has one extra entry checkpoint. This is a relative work
    // regression, not a hard-coded poll count or a production search cap.
    let allowance = usize::MAX - counter.0.get() + 1;
    assert!(
        dp_align::CorrespondenceAnalysis::observe_controlled(
            &words,
            &reference,
            MatchMode::Exact,
            &PollAllowance(std::cell::Cell::new(allowance))
        )
        .is_err()
    );
    let result = recover_overlap_timing_controlled(
        &crate::chat_ops::fa::utr::token_text::UtrChatUnits::spaced(&words),
        &crate::chat_ops::fa::utr::PreparedUtrTokens::spaced(&tokens),
        0,
        1000,
        MatchMode::Exact,
        &PollAllowance(std::cell::Cell::new(allowance)),
    );
    assert_eq!(result, Ok(Some(expected_interval(100, 600))));
}
