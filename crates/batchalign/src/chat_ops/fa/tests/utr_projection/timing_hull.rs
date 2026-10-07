//! The temporal hull belongs to all matched, individually admitted tokens.

use super::*;

#[test]
fn matched_hulls_preserve_nested_and_unordered_provider_extrema() {
    type HullCase<'a> = (&'a str, &'a [(&'a str, u64, u64)], (u64, u64));
    let cases: &[HullCase<'_>] = &[
        (
            "nested",
            &[
                ("hello", 100, 900),
                ("world", 200, 300),
                ("again", 400, 500),
            ],
            (100, 900),
        ),
        (
            "interior",
            &[
                ("hello", 100, 200),
                ("world", 150, 1_200),
                ("again", 300, 400),
            ],
            (100, 1_200),
        ),
        (
            "unordered",
            &[
                ("hello", 500, 600),
                ("world", 100, 200),
                ("again", 300, 400),
            ],
            (100, 600),
        ),
        (
            "unmatched",
            &[
                ("hello", 100, 200),
                ("unrelated", 0, 2_000),
                ("world", 200, 300),
                ("again", 400, 500),
            ],
            (100, 500),
        ),
        ("coarse", &[("hello world again", 100, 900)], (100, 900)),
    ];
    let mut observed = Vec::new();
    for (name, tokens, expected) in cases {
        let mut actual_interval = None;
        for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
            let mut chat = admitted_chat("*PAR:\thello world again .\n", FixtureTiming::Untimed);
            let result = strategy.inject(&mut chat, &make_utr_tokens(tokens));
            actual_interval = get_utterance_bullet(&chat, 0);
            assert_eq!(actual_interval, Some(*expected), "{name}");
            assert_eq!((result.injected(), result.unmatched()), (1, 0), "{name}");
            let plan = serde_json::to_value(selected_plan(&result)).expect("wire evidence");
            assert_eq!(plan["utterances"][0]["proposal"]["start_ms"], expected.0);
            assert_eq!(plan["utterances"][0]["proposal"]["end_ms"], expected.1);
        }
        observed.push(serde_json::json!({"case": name, "interval": actual_interval}));
    }
    insta::assert_json_snapshot!(observed, @r###"
    [
      {
        "case": "nested",
        "interval": [
          100,
          900
        ]
      },
      {
        "case": "interior",
        "interval": [
          100,
          1200
        ]
      },
      {
        "case": "unordered",
        "interval": [
          100,
          600
        ]
      },
      {
        "case": "unmatched",
        "interval": [
          100,
          500
        ]
      },
      {
        "case": "coarse",
        "interval": [
          100,
          900
        ]
      }
    ]
    "###);
}

#[test]
fn invalid_matched_token_cannot_hide_inside_a_positive_outer_hull() {
    for invalid in [(300, 300), (350, 300)] {
        for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
            let mut chat = admitted_chat("*PAR:\thello world again .\n", FixtureTiming::Untimed);
            let tokens = make_utr_tokens(&[
                ("hello", 100, 200),
                ("world", invalid.0, invalid.1),
                ("again", 400, 500),
            ]);
            let result = strategy.inject(&mut chat, &tokens);
            assert_eq!(get_utterance_bullet(&chat, 0), None);
            assert_eq!((result.injected(), result.unmatched()), (0, 1));
            let plan = serde_json::to_value(selected_plan(&result)).expect("wire evidence");
            assert_eq!(
                plan["utterances"][0]["proposal"],
                serde_json::json!({
                    "status": "non_positive", "start_ms": invalid.0, "end_ms": invalid.1,
                })
            );
        }
    }
}

#[test]
fn matched_hull_still_obeys_the_source_bound_following_corridor() {
    let mut chat = admitted_chat(
        "*PAR:\thello world again .\n*PAR:\tnext . \u{15}800_1000\u{15}\n",
        FixtureTiming::PartiallyTimed,
    );
    let result = GlobalUtr.inject(
        &mut chat,
        &make_utr_tokens(&[
            ("hello", 100, 200),
            ("world", 150, 1_200),
            ("again", 300, 400),
            ("next", 800, 1_000),
        ]),
    );
    assert_eq!(get_utterance_bullet(&chat, 0), Some((100, 800)));
    assert_eq!(get_utterance_bullet(&chat, 1), Some((800, 1_000)));
    let plan = serde_json::to_value(selected_plan(&result)).expect("wire evidence");
    assert_eq!(plan["utterances"][0]["proposal"]["end_ms"], 1_200);
}
