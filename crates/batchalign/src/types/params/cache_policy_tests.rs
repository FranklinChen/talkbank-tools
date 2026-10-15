use super::*;
use crate::options::CommonOptions;

#[test]
fn required_asr_and_fresh_fa_are_admitted_together() {
    let common = CommonOptions {
        require_media_cache_tasks: vec!["utr_asr".into()],
        override_media_cache_tasks: vec!["forced_alignment".into()],
        ..Default::default()
    };
    let policies = CacheOverrides::from_common(&common).unwrap();
    let utr = UtrCachePolicies::from_overrides(&policies);
    assert_eq!(utr.response(), CachePolicy::RequireCache);
    assert_eq!(utr.rev_evidence(), CachePolicy::RequireCache);
    assert_eq!(
        policies.policy_for(CacheTaskName::ForcedAlignment),
        CachePolicy::SkipCache
    );
    let wire = serde_json::to_value(&common).unwrap();
    let decoded: CommonOptions = serde_json::from_value(wire).unwrap();
    assert_eq!(CacheOverrides::from_common(&decoded).unwrap(), policies);
}

#[test]
fn cache_policy_conflicts_never_construct_selected_policies() {
    for (required, refreshed) in [
        ("utr_asr", "utr_asr"),
        ("utr_asr", "rev_asr_evidence"),
        ("translation", "forced_alignment"),
        ("utr_asrr", "forced_alignment"),
        ("utr_asr", "forced_alignmnt"),
    ] {
        let common = CommonOptions {
            require_media_cache_tasks: vec![required.into()],
            override_media_cache_tasks: vec![refreshed.into()],
            ..Default::default()
        };
        assert!(
            CacheOverrides::from_common(&common).is_err(),
            "{required}/{refreshed}"
        );
    }
    for (require_all, refresh_all) in [(true, false), (false, true), (true, true)] {
        let common = CommonOptions {
            require_media_cache: require_all,
            override_media_cache: refresh_all,
            require_media_cache_tasks: vec!["utr_asr".into()],
            ..Default::default()
        };
        assert!(CacheOverrides::from_common(&common).is_err());
    }
}

#[test]
fn raw_require_survives_derived_refresh_and_legacy_defaults_stay_closed() {
    let common = CommonOptions {
        require_media_cache_tasks: vec!["rev_asr_evidence".into()],
        override_media_cache_tasks: vec!["utr_asr".into()],
        ..Default::default()
    };
    let policies = UtrCachePolicies::from_overrides(&CacheOverrides::from_common(&common).unwrap());
    assert_eq!(policies.response(), CachePolicy::SkipCache);
    assert_eq!(policies.rev_evidence(), CachePolicy::RequireCache);
    let legacy: CommonOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(
        CacheOverrides::from_common(&legacy).unwrap(),
        CacheOverrides::None
    );
    let all = UtrCachePolicies::from_overrides(&CacheOverrides::All);
    assert_eq!(all.response(), CachePolicy::SkipCache);
    assert_eq!(all.rev_evidence(), CachePolicy::SkipCache);
}
