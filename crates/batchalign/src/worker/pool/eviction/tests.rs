use super::*;
/// Guards: cross-group eviction selects the busiest eligible observed group,
/// preserving its carried identity and excluding the requesting or empty groups.
/// Kind: policy
/// Not a type: the highest-idle selection policy has real alternatives after
/// candidate admission. Source association removes the former second map lookup.
/// State: logical group identities and counts require no worker or capacity fixture.
/// Concurrency: none
#[test]
fn eligible_group_selection_keeps_its_observed_source() {
    for (groups, expected) in [
        (vec![("skip", 2), ("other", 0)], None),
        (vec![("skip", 0), ("other", 0)], None),
        (vec![("skip", 0), ("other", 3)], Some("other")),
        (vec![("skip", 0), ("other", 4), ("third", 1)], Some("other")),
        (vec![("skip", 9), ("other", 1)], Some("other")),
    ] {
        let selected = select_eviction_target(
            groups
                .into_iter()
                .map(|(key, n)| (key, format!("source-{key}"), n)),
            &"skip",
        );
        assert_eq!(selected.as_ref().map(|(key, _)| *key), expected);
        if let Some((key, source)) = selected {
            assert_eq!(source, format!("source-{key}"));
        }
    }
}
