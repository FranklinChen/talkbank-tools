//! Compatibility re-exports for canonical utterance-segmentation helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::utseg::{
    UtsegBatchItem, UtsegMisalignmentDiagnostic, UtsegNotApplicableReason, UtsegOutcome,
    UtsegOutcomeKind, UtsegPayloadCollection, UtsegResponse, apply_utseg_results,
    build_word_to_content_map, collect_utseg_payloads, split_utterance, validate_utseg_response,
};

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use talkbank_model::SpeakerCode;
#[cfg(test)]
use talkbank_model::model::{
    ChatFile, DependentTier, Line, Terminator, Utterance, UtteranceContent, WorTier,
};

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::WriteChat;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).unwrap()
    }

    fn get_utterance(chat: &ChatFile, idx: usize) -> &Utterance {
        let mut utt_idx = 0;
        for line in &chat.lines.0 {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    fn count_utterances(chat: &ChatFile) -> usize {
        chat.lines
            .iter()
            .filter(|l| matches!(l, Line::Utterance(_)))
            .count()
    }

    #[test]
    fn test_split_no_change() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_two_groups() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2);

        let out0 = result[0].to_chat_string();
        let out1 = result[1].to_chat_string();
        assert!(out0.contains("I eat cookies"), "First split: {out0}");
        assert!(out1.contains("and he likes cake"), "Second split: {out1}");
    }

    #[test]
    fn test_collect_utseg_payloads() {
        // 3 utterances: 1 single-word, 2 multi-word
        let chat_text = include_str!("../../../test-fixtures/eng_three_utterances.cha");
        let chat = parse_chat(chat_text);
        let collected = collect_utseg_payloads(&chat);
        let payloads = &collected.batch_items;

        // Single-word utterance "hello" should be classified NotApplicable.
        assert_eq!(payloads.len(), 2);
        assert_eq!(collected.not_applicable.len(), 1);
        match &collected.not_applicable[0].kind {
            UtsegOutcomeKind::NotApplicable { reason } => {
                assert_eq!(*reason, UtsegNotApplicableReason::SingleWord);
            }
            other => panic!("expected NotApplicable(SingleWord), got {other:?}"),
        }
        assert_eq!(payloads[0].0, 1); // utt_ordinal of "I eat cookies"
        assert_eq!(payloads[0].1.words, vec!["I", "eat", "cookies"]);
        assert_eq!(payloads[0].1.text, "I eat cookies");
        assert_eq!(payloads[1].0, 2); // utt_ordinal of "he likes cake too"
        assert_eq!(payloads[1].1.words, vec!["he", "likes", "cake", "too"]);
    }

    #[test]
    fn test_apply_utseg_results() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let mut chat = parse_chat(chat_text);
        assert_eq!(count_utterances(&chat), 1);

        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 1, 1, 1, 1]);

        apply_utseg_results(&mut chat, &assignment_map);
        assert_eq!(count_utterances(&chat), 2);

        let out0 = get_utterance(&chat, 0).to_chat_string();
        let out1 = get_utterance(&chat, 1).to_chat_string();
        assert!(out0.contains("I eat cookies"), "First: {out0}");
        assert!(out1.contains("and he likes cake"), "Second: {out1}");
    }

    #[test]
    fn test_apply_utseg_empty_map() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let mut chat = parse_chat(chat_text);
        let original_count = count_utterances(&chat);

        apply_utseg_results(&mut chat, &HashMap::new());
        assert_eq!(count_utterances(&chat), original_count);
    }

    /// After utseg splits, no utterance should start with a Separator node.
    ///
    /// Rev.AI returns "dishes , or she didn't order them" and Stanza puts
    /// the boundary after "dishes". The comma is correctly modeled as
    /// `UtteranceContent::Separator(Separator::Comma)` by build_chat.rs.
    /// But after the split, it lands as the first content item of the second
    /// utterance — which is invalid CHAT. Leading separators must be stripped.
    ///
    /// Bug report: a user, 2026-04-02, 25-3.cha — `*INV: , or she didn't...`
    #[test]
    fn utseg_split_strips_leading_separator() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tINV Investigator\n\
            @ID:\teng|test|INV|||||Investigator|||\n\
            *INV:\tshe's washing dishes , or she didn't order them .\n@End\n";
        let mut chat = parse_chat(chat_text);

        // Stanza boundary: words 0-2 = group 0, words 3-7 = group 1.
        // The comma separator sits between groups.
        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 1, 1, 1, 1, 1]);

        apply_utseg_results(&mut chat, &assignment_map);

        // Verify the split produced two utterances
        let utt_count = count_utterances(&chat);
        assert!(
            utt_count >= 2,
            "expected at least 2 utterances, got {utt_count}"
        );

        // No utterance's first content item should be a Separator
        for (i, line) in chat.lines.iter().enumerate() {
            if let Line::Utterance(u) = line
                && let Some(first) = u.main.content.content.first()
            {
                assert!(
                    !matches!(first, UtteranceContent::Separator(_)),
                    "utterance at line {i} starts with a Separator node \
                         (should have been stripped): {}",
                    u.to_chat_string()
                );
            }
        }
    }

    /// REGRESSION: the parent's main-tier timing bullet must not be silently
    /// dropped when an utterance is split.
    ///
    /// The utseg pipeline produced bullet-less output across 854/885 MOST
    /// corpus files on 2026-04-26 because `split_utterance` constructed each
    /// child's `MainTier` fresh via `MainTier::new(...)` — which sets
    /// `TierContent.bullet = None` — without copying `utt.main.content.bullet`
    /// from the parent. The aggregate signal: 223,277 → 152,192 bullets
    /// (−31.8%) corpus-wide. Files whose only timing came from to-be-split
    /// utterances ended up with no timing at all and tripped E544
    /// (@Media-linkage assertion).
    ///
    /// Conservative invariant tested here: at least one child of a split
    /// must carry the parent's bullet. We attach it to the LAST child —
    /// the original utterance ended at the bullet's end timestamp, and
    /// the last child of the split contains the last words and ends at
    /// that same end timestamp.
    ///
    /// See: docs/postmortems/2026-04-26-utseg-split-bullet-loss.md
    #[test]
    fn utseg_split_preserves_parent_bullet_on_last_child() {
        // Bullet syntax: NAK-delimited "start_end" appended after the
        // terminator. \u{15} is NAK (0x15). Real example from MOST:
        // `*PAR0: ... . 0_668430` (the 0_668430 is the bullet).
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat cookies and he likes cake . \u{15}1000_5000\u{15}\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();

        // Sanity-check the fixture: the parent utterance carries a bullet.
        assert!(
            utt.main.content.bullet.is_some(),
            "fixture pre-condition: parent must have a bullet"
        );
        let parent_bullet = utt.main.content.bullet.as_ref().unwrap().clone();

        // Split into two children: words 0-2 → child 0, words 3-6 → child 1.
        let result = split_utterance(utt, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children from the split");

        // The LAST child must carry the parent's bullet (same start_ms /
        // end_ms). The earlier children may have no bullet — we simply
        // don't know their per-child timing without realignment, and we
        // refuse to fabricate one.
        let last_child_bullet = &result.last().unwrap().main.content.bullet;
        assert!(
            last_child_bullet.is_some(),
            "last child of a split must inherit the parent's bullet \
             (got None — parent timing was dropped). Output: {}",
            result.last().unwrap().to_chat_string()
        );
        let last = last_child_bullet.as_ref().unwrap();
        assert_eq!(
            last.timing.start_ms, parent_bullet.timing.start_ms,
            "last child's bullet start must equal parent's"
        );
        assert_eq!(
            last.timing.end_ms, parent_bullet.timing.end_ms,
            "last child's bullet end must equal parent's"
        );
    }

    /// %wor partitioning: when a parent has %wor with timing for every
    /// main-tier word, splitting must distribute the WorItems to the
    /// children matching their words. F1.5: BA2-equivalent per-word
    /// timing preservation across split.
    #[test]
    fn utseg_split_partitions_wor_tier_across_children() {
        // 4-word utterance with %wor giving each word its own timing.
        // Split 2/2: child 0 gets words "I eat", child 1 gets "the cookies".
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies . \u{15}0_4000\u{15}\n\
            %wor:\tI \u{15}0_500\u{15} eat \u{15}500_1500\u{15} the \u{15}1500_2200\u{15} \
            cookies \u{15}2200_4000\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent has 4 wor items
        let parent_wor = parent
            .dependent_tiers
            .iter()
            .find_map(|t| match t {
                DependentTier::Wor(w) => Some(w),
                _ => None,
            })
            .expect("fixture should parse a %wor tier");
        assert_eq!(parent_wor.word_count(), 4, "fixture sanity check");

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        let wor_of = |u: &Utterance| -> Option<WorTier> {
            u.dependent_tiers.iter().find_map(|t| match t {
                DependentTier::Wor(w) => Some(w.clone()),
                _ => None,
            })
        };
        let child0_wor = wor_of(&result[0]).expect("child 0 must carry %wor");
        let child1_wor = wor_of(&result[1]).expect("child 1 must carry %wor");
        assert_eq!(
            child0_wor.word_count(),
            2,
            "child 0 should carry 2 wor words (I, eat)"
        );
        assert_eq!(
            child1_wor.word_count(),
            2,
            "child 1 should carry 2 wor words (the, cookies)"
        );
    }

    /// %wor partitioning falls back to dropping the tier when item counts
    /// don't match main-tier eligible-word counts (stale %wor). No panic,
    /// no validation error — silent drop matches the rename's intent that
    /// stale %wor is legal.
    #[test]
    fn utseg_split_drops_wor_on_count_mismatch() {
        // Parent has 4 main-tier words, but only 3 wor items (stale).
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %wor:\tI \u{15}0_500\u{15} eat \u{15}500_1500\u{15} cookies \u{15}1500_2000\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            let has_wor = child
                .dependent_tiers
                .iter()
                .any(|t| matches!(t, DependentTier::Wor(_)));
            assert!(
                !has_wor,
                "child {i} should not carry %wor when parent counts mismatched (graceful drop)"
            );
        }
    }

    /// %mor and %gra are dropped on split. Their analysis depends on
    /// utterance-scope context and is invalidated by re-segmentation;
    /// the user reruns morphotag to regenerate.
    #[test]
    fn utseg_split_drops_mor_and_gra() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %mor:\tpron|I v|eat det|the n|cookies\n\
            %gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            for tier in &child.dependent_tiers {
                assert!(
                    !matches!(tier, DependentTier::Mor(_) | DependentTier::Gra(_)),
                    "child {i} should not carry %mor or %gra after split (dropped by policy)"
                );
            }
        }
    }

    /// %com and other free-form / utterance-level annotations attach to
    /// the first child. Strictly better than BA2's silent drop.
    #[test]
    fn utseg_split_attaches_com_to_first_child() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %com:\tchild was excited\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        let first_has_com = result[0]
            .dependent_tiers
            .iter()
            .any(|t| matches!(t, DependentTier::Com(_)));
        let second_has_com = result[1]
            .dependent_tiers
            .iter()
            .any(|t| matches!(t, DependentTier::Com(_)));
        assert!(first_has_com, "first child must inherit the %com");
        assert!(!second_has_com, "second child must not carry %com");
    }

    /// Terminator propagation: the LAST child inherits the parent's
    /// terminator; non-last children get the default `Period`. This
    /// preserves quote-introducer linkage (`+"/.` parent → next-utterance
    /// `+"` quoted speech), interruption markers, and any other
    /// non-default terminator. See spec/errors/E341_auto.md for the
    /// `+"/.` ↔ `+"` validation pairing.
    #[test]
    fn utseg_split_inherits_terminator_on_last_child() {
        // Parent ends with +"/. (quote-introducer). Real shape from
        // childes-eng-na-data/Eng-NA/Kuczaj/030115.cha.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tand he says +\"/.\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent terminator is the quote-introducer, not Period.
        let parent_term = parent
            .main
            .content
            .terminator
            .as_ref()
            .expect("fixture must have a terminator");
        assert!(
            !matches!(parent_term, Terminator::Period { .. }),
            "fixture sanity: parent must end in +\"/. (non-default), got {parent_term:?}"
        );

        let result = split_utterance(parent, &[0, 0, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        let first_term = result[0]
            .main
            .content
            .terminator
            .as_ref()
            .expect("child 0 must have a terminator");
        assert!(
            matches!(first_term, Terminator::Period { .. }),
            "non-last child must default to Period, got {first_term:?}"
        );

        let last_term = result[1]
            .main
            .content
            .terminator
            .as_ref()
            .expect("last child must have a terminator");
        assert!(
            !matches!(last_term, Terminator::Period { .. }),
            "LAST child must inherit the parent's non-default terminator, got {last_term:?}"
        );
    }

    /// Linker propagation: the FIRST child inherits the parent's
    /// linkers; non-first children get none. Linkers describe the
    /// utterance's relationship to the *prior* (different) utterance,
    /// so only the first split-piece is adjacent to that prior turn.
    #[test]
    fn utseg_split_inherits_linkers_on_first_child() {
        // Parent starts with `+,` (SelfCompletion linker). Real shape
        // from clan-info/examples/Adler/adler15a.cha.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tINV Investigator\n\
            @ID:\teng|test|INV|||||Investigator|||\n\
            *INV:\t+, with that letter and one more thing .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent has at least one linker.
        assert!(
            !parent.main.content.linkers.is_empty(),
            "fixture sanity: parent must carry at least one linker"
        );
        let parent_linkers_len = parent.main.content.linkers.0.len();
        assert!(parent_linkers_len > 0);

        let result = split_utterance(parent, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        assert_eq!(
            result[0].main.content.linkers.0.len(),
            parent_linkers_len,
            "FIRST child must inherit the parent's linkers"
        );
        assert!(
            result[1].main.content.linkers.is_empty(),
            "non-first child must have no linkers"
        );
    }

    /// Language code propagation: utterance-level `[- code]` applies
    /// to all of the utterance's words, so every child carries it.
    #[test]
    fn utseg_split_propagates_language_code_to_all_children() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng, spa\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\t[- spa] hola amigo y como estas .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let parent_lang = parent
            .main
            .content
            .language_code
            .clone()
            .expect("fixture must parse [- spa] language code");

        let result = split_utterance(parent, &[0, 0, 1, 1, 1, 1]);
        assert!(result.len() >= 2);

        for (i, child) in result.iter().enumerate() {
            assert_eq!(
                child.main.content.language_code.as_ref(),
                Some(&parent_lang),
                "child {i} must carry the parent's [- spa] language code"
            );
        }
    }

    /// Postcode propagation: utterance-level `[+ exc]` and similar
    /// analysis tags attach to the LAST child only. They describe the
    /// original utterance as a unit; placing them on the last child
    /// (where they serialize after the terminator) keeps each tag
    /// attached exactly once and matches the conventional position.
    #[test]
    fn utseg_split_inherits_postcodes_on_last_child() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tone two three four . [+ exc]\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let parent_postcode_len = parent.main.content.postcodes.0.len();
        assert!(
            parent_postcode_len > 0,
            "fixture sanity: parent must carry at least one postcode"
        );

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        assert!(
            result[0].main.content.postcodes.is_empty(),
            "non-last child must have no postcodes"
        );
        assert_eq!(
            result[1].main.content.postcodes.0.len(),
            parent_postcode_len,
            "LAST child must inherit the parent's postcodes"
        );
    }

    /// Replaced-word handling: a `ReplacedWord(wanna [: want to])` is one
    /// main-tier slot but contributes N replacement words to TierDomain::Mor
    /// (the BERT classifier sees N words). `split_utterance` builds its
    /// word→content mapping with TierDomain::Mor too, so the assignment
    /// vector lengths match. The first-assignment-wins logic in
    /// `split_utterance` correctly attributes each ReplacedWord to ONE
    /// child group regardless of where the boundary lands relative to
    /// the replacement words.
    ///
    /// Coverage gap discovered 2026-04-27 while writing replacements
    /// docs (`book/src/architecture/replacements-handling.md`); analogous
    /// to the FA bug shape from 2026-04-08. The current code is correct
    /// by construction (extract + split both use TierDomain::Mor); these
    /// tests pin that invariant against future drift.
    #[test]
    fn utseg_split_handles_replaced_word_boundary_before() {
        // Boundary BEFORE the ReplacedWord. Mor walks 4 words: I, want, to, go.
        // assignments=[0, 1, 1, 1] → "I" alone in group 0; "wanna" + "go" in group 1.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        // child 0: just "I" — no fragment of the ReplacedWord or "go"
        assert!(
            !s0.contains("wanna") && !s0.contains("want") && !s0.contains("go"),
            "child 0 should be just I, got: {s0}"
        );
        // child 1: ReplacedWord preserved intact, plus "go"
        assert!(
            s1.contains("wanna [: want to]"),
            "child 1 should keep the ReplacedWord intact, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
    }

    #[test]
    fn utseg_split_handles_replaced_word_boundary_after() {
        // Boundary AFTER the ReplacedWord. assignments=[0, 0, 0, 1] →
        // "I wanna" in group 0; "go" alone in group 1. The ReplacedWord
        // (one main-tier slot) lands fully in group 0.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0, 1]);
        assert_eq!(result.len(), 2);
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        assert!(
            s0.contains("wanna [: want to]"),
            "child 0 should keep the ReplacedWord intact, got: {s0}"
        );
        assert!(
            !s1.contains("wanna") && !s1.contains("want"),
            "child 1 should not have any ReplacedWord fragment, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
    }

    #[test]
    fn utseg_split_attributes_replaced_word_atomically_on_inconsistent_split() {
        // BERT puts the boundary BETWEEN the replacement words "want" and
        // "to" (assignments=[0, 0, 1, 1]). Structurally we cannot split
        // inside a single main-tier slot. The first-assignment-wins logic
        // attributes the entire ReplacedWord to "want"'s group (0). Net
        // result: "I wanna" in group 0, "go" in group 1 — same as if the
        // boundary had been after the ReplacedWord. This pins the
        // "ReplacedWord is atomic to splits" invariant.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        // ReplacedWord stayed atomic — went with "want"'s assignment (0).
        assert!(
            s0.contains("wanna [: want to]"),
            "ReplacedWord must be atomic on splits; got s0: {s0}"
        );
        assert!(
            !s1.contains("wanna") && !s1.contains("want"),
            "child 1 should not contain any fragment of the ReplacedWord, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
    }

    #[test]
    fn snapshot_utseg_batch_item() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "words": [
            "I",
            "eat",
            "cookies"
          ],
          "text": "I eat cookies"
        }
        "#);
    }

    #[test]
    fn snapshot_utseg_response() {
        let resp = UtsegResponse {
            assignments: vec![0, 0, 0, 1, 1, 1, 1],
        };
        insta::assert_json_snapshot!(resp, @r#"
        {
          "assignments": [
            0,
            0,
            0,
            1,
            1,
            1,
            1
          ]
        }
        "#);
    }

    // ---------------------------------------------------------------------
    // Wave 5 outcome-classification tests
    // ---------------------------------------------------------------------

    #[test]
    fn validate_utseg_response_aligned_matching_counts() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        let resp = UtsegResponse {
            assignments: vec![0, 0, 0],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::Aligned {
                n_words,
                n_segments,
            } => {
                assert_eq!(n_words, 3);
                assert_eq!(n_segments, 1, "all same group = 1 segment");
            }
            other => panic!("expected Aligned, got {other:?}"),
        }
    }

    #[test]
    fn validate_utseg_response_counts_distinct_segments() {
        let item = UtsegBatchItem {
            words: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            text: "a b c d".into(),
        };
        let resp = UtsegResponse {
            assignments: vec![0, 0, 1, 1],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::Aligned { n_segments, .. } => assert_eq!(n_segments, 2),
            other => panic!("expected Aligned(2 segments), got {other:?}"),
        }
    }

    #[test]
    fn validate_utseg_response_length_mismatch_is_misalignment_bug() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        // Worker returned 2 assignments for 3 input words — contract violation.
        let resp = UtsegResponse {
            assignments: vec![0, 0],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::MisalignmentBug(diag) => {
                assert_eq!(diag.expected_assignments, 3);
                assert_eq!(diag.actual_assignments, 2);
                assert_eq!(diag.words, vec!["I", "eat", "cookies"]);
            }
            other => panic!("expected MisalignmentBug, got {other:?}"),
        }
    }

    #[test]
    fn collect_utseg_emits_not_applicable_for_single_word() {
        let chat_text = include_str!("../../../test-fixtures/eng_three_utterances.cha");
        let chat = parse_chat(chat_text);
        let collected = collect_utseg_payloads(&chat);

        assert_eq!(
            collected.not_applicable.len(),
            1,
            "expected the single-word utterance (\"hello\") to be classified NotApplicable",
        );
        let outcome = &collected.not_applicable[0];
        match &outcome.kind {
            UtsegOutcomeKind::NotApplicable { reason } => {
                assert_eq!(*reason, UtsegNotApplicableReason::SingleWord);
            }
            other => panic!("expected NotApplicable(SingleWord), got {other:?}"),
        }
    }

    #[test]
    fn utseg_outcome_to_decision_record_aligned_is_none() {
        let outcome = UtsegOutcome {
            utt_ordinal: 0,
            speaker: SpeakerCode::new("CHI"),
            kind: UtsegOutcomeKind::Aligned {
                n_words: 3,
                n_segments: 1,
            },
        };
        assert!(outcome.to_decision_record(5).is_none());
    }

    #[test]
    fn utseg_outcome_to_decision_record_misalignment_bug_flags_review() {
        let outcome = UtsegOutcome {
            utt_ordinal: 0,
            speaker: SpeakerCode::new("CHI"),
            kind: UtsegOutcomeKind::MisalignmentBug(UtsegMisalignmentDiagnostic {
                expected_assignments: 3,
                actual_assignments: 2,
                words: vec!["hello".into(), "world".into(), "bye".into()],
            }),
        };
        let record = outcome.to_decision_record(5).expect("record for bug");
        assert!(matches!(
            record.strategy,
            crate::decisions::DecisionStrategy::Utseg(
                crate::decisions::UtsegStrategy::MisalignmentBug
            )
        ));
        assert!(record.needs_review);
        assert!(record.reason.contains("expected_assignments=3"));
        assert!(record.reason.contains("actual_assignments=2"));
    }
}
