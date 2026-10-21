//! Exact repeated n-gram analysis shared by cleanup and segmentation policy.

use super::is_punct_or_terminator;

/// Proven positions from one exact repeated n-gram analysis.
///
/// A protected split is represented as the index of the word that would begin
/// the right-hand segment. Private masks prevent callers from inventing a
/// split guard independently of the retrace analysis that justifies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactRetraceAnalysis {
    retraced_words: Vec<bool>,
    protected_splits_before_words: Vec<bool>,
}

impl ExactRetraceAnalysis {
    /// Whether this original word belongs to a repeated precursor.
    pub fn is_retraced_word(&self, word_index: usize) -> bool {
        self.retraced_words
            .get(word_index)
            .copied()
            .unwrap_or(false)
    }

    /// Whether splitting immediately before this original word would break an
    /// exact repetition that the CHAT cleanup pass can encode as a retrace.
    pub fn protects_split_before(&self, word_index: usize) -> bool {
        self.protected_splits_before_words
            .get(word_index)
            .copied()
            .unwrap_or(false)
    }

    /// Original word indices before which a split is protected.
    pub fn protected_split_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.protected_splits_before_words
            .iter()
            .enumerate()
            .filter_map(|(index, protected)| protected.then_some(index))
    }
}

/// Analyze the exact repeated n-grams recognized by BA3's CHAT cleanup.
///
/// Punctuation is excluded from n-gram comparison. A detected repetition
/// protects every split inside the complete repeated span, including the seam
/// between copies, so segmentation cannot erase the evidence before cleanup.
pub fn analyze_exact_retraces(words: &[String], lang: &str) -> ExactRetraceAnalysis {
    let min_ngram = if lang == "yue" || lang == "zho" { 2 } else { 1 };
    let content_indices: Vec<_> = words
        .iter()
        .enumerate()
        .filter(|(_, word)| !is_punct_or_terminator(word))
        .map(|(index, _)| index)
        .collect();
    let mut retraced_words = vec![false; words.len()];
    let mut protected_splits_before_words = vec![false; words.len()];

    for n in min_ngram..content_indices.len() {
        let mut begin = 0;
        while begin + n < content_indices.len() {
            let mut root = begin;
            while root + 2 * n <= content_indices.len() {
                let next_matches = (0..n).all(|offset| {
                    words[content_indices[begin + offset]]
                        .eq_ignore_ascii_case(&words[content_indices[root + n + offset]])
                });
                if !next_matches {
                    break;
                }

                for content_index in begin..begin + n {
                    retraced_words[content_indices[content_index]] = true;
                }
                let span_start = content_indices[begin];
                let span_end_inclusive = content_indices[root + 2 * n - 1];
                protected_splits_before_words[span_start + 1..=span_end_inclusive].fill(true);
                root += n;
            }
            begin += 1;
        }
    }

    ExactRetraceAnalysis {
        retraced_words,
        protected_splits_before_words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn exact_phrase_retrace_protects_both_copies_from_segmentation() {
        let analysis = analyze_exact_retraces(
            &words(&[
                "How", "can", "I", "take", "it", "off", "blur", "can", "I", "take", "it", "off",
                "blur",
            ]),
            "eng",
        );

        assert!(analysis.is_retraced_word(1));
        assert!(analysis.is_retraced_word(6));
        assert!(!analysis.is_retraced_word(7));
        assert!(analysis.protects_split_before(6));
        assert!(analysis.protects_split_before(12));
        assert!(!analysis.protects_split_before(1));
    }

    #[test]
    fn unrelated_words_have_no_protected_splits() {
        let analysis = analyze_exact_retraces(&words(&["one", "two", "three"]), "eng");
        assert_eq!(analysis.protected_split_indices().count(), 0);
    }
}
