//! Complete word-chunk comparisons over fully admitted CHAT documents.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Serialize, Serializer};
use talkbank_model::alignment::{GraHeadRef, helpers::PositionalDomain};
use talkbank_model::model::{MorChunk, MorTier};
use talkbank_model::validation::ValidChatFile;

use super::super::artifact::{ValidatedArtifactPair, ValidatedMorphotagPlan};
use super::{PairFailureReason, PairOutcome, compare_pairs, correspondence, normalize};
use crate::extract;

/// An analysis difference, not a claim that either side is gold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphotagDifference {
    /// Normalized main-tier text differs.
    Tokenization,
    /// At least one chunk's lemma differs.
    Lemma,
    /// At least one chunk's part of speech differs.
    Pos,
    /// At least one chunk's unordered feature set differs.
    FeatureSet,
    /// The number of chunks in the item differs.
    CliticChunk,
    /// At least one chunk's local dependency head differs.
    DependencyHead,
    /// At least one chunk's dependency relation differs.
    Relation,
}

/// What annotation is actually present at a compared main-tier position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationPresence {
    /// This side has no corresponding main-tier position.
    NoMainToken,
    /// The position exists, but has no morphology.
    Absent,
    /// Morphology exists without dependency analysis.
    MorphologyOnly,
    /// Morphology and a bound dependency for every chunk are present.
    Complete,
}

/// A producer-built row, including annotation coverage as well as differences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MorphotagTokenDifference {
    /// Left speaker code.
    left_speaker: String,
    /// Right speaker code.
    right_speaker: String,
    /// Zero-based utterance ordinal within the mapped speaker.
    utterance: usize,
    /// Zero-based main-token position.
    token: usize,
    /// Normalized left text.
    left_text: Option<String>,
    /// Normalized right text.
    right_text: Option<String>,
    /// Annotation presence on the left.
    left_annotation: AnnotationPresence,
    /// Annotation presence on the right.
    right_annotation: AnnotationPresence,
    /// Every observed difference axis, including all clitic chunks.
    differences: BTreeSet<MorphotagDifference>,
    /// Complete-analysis agreement; absent when either analysis is incomplete.
    analysis_agreement: Option<bool>,
}

impl MorphotagTokenDifference {
    /// Left speaker code.
    pub fn left_speaker(&self) -> &str {
        &self.left_speaker
    }
    /// Right speaker code.
    pub fn right_speaker(&self) -> &str {
        &self.right_speaker
    }
    /// Utterance ordinal within the paired speaker.
    pub fn utterance(&self) -> usize {
        self.utterance
    }
    /// Main-tier position within the utterance.
    pub fn token(&self) -> usize {
        self.token
    }
    /// Normalized left token text.
    pub fn left_text(&self) -> Option<&str> {
        self.left_text.as_deref()
    }
    /// Normalized right token text.
    pub fn right_text(&self) -> Option<&str> {
        self.right_text.as_deref()
    }
    /// Actual left annotation presence.
    pub fn left_annotation(&self) -> AnnotationPresence {
        self.left_annotation
    }
    /// Actual right annotation presence.
    pub fn right_annotation(&self) -> AnnotationPresence {
        self.right_annotation
    }
    /// Observed difference axes.
    pub fn differences(&self) -> &BTreeSet<MorphotagDifference> {
        &self.differences
    }
    /// Whether any observed axis differs.
    pub fn differs(&self) -> bool {
        !self.differences.is_empty()
    }

    /// Agreement is unknown, not true, when complete analysis is unavailable.
    pub fn analysis_agreement(&self) -> Option<bool> {
        self.analysis_agreement
    }
}

/// Sealed result: coverage counts and difference subsets derive from one owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphotagPairResult {
    tokens: Vec<MorphotagTokenDifference>,
}

impl MorphotagPairResult {
    /// Every examined position, including unannotated positions.
    pub fn tokens(&self) -> &[MorphotagTokenDifference] {
        &self.tokens
    }

    /// Positions with observed differences.
    pub fn differences(&self) -> impl Iterator<Item = &MorphotagTokenDifference> {
        self.tokens.iter().filter(|row| row.differs())
    }

    /// Number of examined positions; not a count of successful analyses.
    pub fn compared_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Positions at which both sides have complete annotation.
    pub fn fully_annotated_tokens(&self) -> usize {
        self.tokens
            .iter()
            .filter(|row| row.analysis_agreement.is_some())
            .count()
    }
}

impl Serialize for MorphotagPairResult {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Report<'a> {
            tokens: &'a [MorphotagTokenDifference],
            differences: Vec<&'a MorphotagTokenDifference>,
            compared_tokens: usize,
            fully_annotated_tokens: usize,
        }
        Report {
            tokens: self.tokens(),
            differences: self.differences().collect(),
            compared_tokens: self.compared_tokens(),
            fully_annotated_tokens: self.fully_annotated_tokens(),
        }
        .serialize(serializer)
    }
}

/// Compare admitted artifacts without invoking any inference producer.
pub fn compare_validated_morphotag_plan(
    plan: &ValidatedMorphotagPlan,
) -> Vec<PairOutcome<MorphotagPairResult>> {
    compare_pairs(plan.runs(), plan.artifact_pairs(), compare_morph_pair)
}

#[derive(Clone, PartialEq, Eq)]
struct ChunkAnalysis {
    lemma: String,
    pos: String,
    features: BTreeSet<String>,
}

/// Dependencies are utterance-local; speaker-code spelling is not head identity.
#[derive(Clone, PartialEq, Eq)]
enum HeadIdentity {
    Root,
    WordChunk { item: usize, offset: usize },
    Terminator,
}

#[derive(Clone, PartialEq, Eq)]
struct Dependency {
    head: HeadIdentity,
    relation: String,
}

#[derive(Clone)]
struct AnnotatedChunk {
    analysis: ChunkAnalysis,
    dependency: Dependency,
}

#[derive(Clone)]
enum TokenAnnotation {
    Absent,
    MorphologyOnly(Vec<ChunkAnalysis>),
    Complete(Vec<AnnotatedChunk>),
}

impl TokenAnnotation {
    fn presence(&self) -> AnnotationPresence {
        match self {
            Self::Absent => AnnotationPresence::Absent,
            Self::MorphologyOnly(_) => AnnotationPresence::MorphologyOnly,
            Self::Complete(_) => AnnotationPresence::Complete,
        }
    }

    fn chunks(&self) -> impl Iterator<Item = &ChunkAnalysis> + Clone {
        let morphology: &[ChunkAnalysis] = match self {
            Self::MorphologyOnly(chunks) => chunks,
            Self::Absent | Self::Complete(_) => &[],
        };
        let complete: &[AnnotatedChunk] = match self {
            Self::Complete(chunks) => chunks,
            Self::Absent | Self::MorphologyOnly(_) => &[],
        };
        morphology
            .iter()
            .chain(complete.iter().map(|chunk| &chunk.analysis))
    }

    fn complete_chunks(&self) -> Option<&[AnnotatedChunk]> {
        match self {
            Self::Absent | Self::MorphologyOnly(_) => None,
            Self::Complete(chunks) => Some(chunks),
        }
    }
}

#[derive(Clone)]
struct MorToken {
    text: String,
    annotation: TokenAnnotation,
}

fn binding_failure(side: &str, detail: &str) -> PairFailureReason {
    PairFailureReason::ProducerFailure {
        side: side.to_string(),
        detail: detail.to_string(),
    }
}

/// One canonical traversal owns every semantic chunk's local address.
fn chunk_addresses(tier: &MorTier) -> Vec<HeadIdentity> {
    let mut addresses = Vec::with_capacity(tier.count_chunks());
    let mut next_item = 0;
    let mut offset = 0;
    for chunk in tier.chunks() {
        match chunk {
            MorChunk::Main(_) => {
                addresses.push(HeadIdentity::WordChunk {
                    item: next_item,
                    offset: 0,
                });
                next_item += 1;
                offset = 1;
            }
            MorChunk::PostClitic(_, _) => {
                // The canonical iterator emits post-clitics only after their host.
                addresses.push(HeadIdentity::WordChunk {
                    item: next_item - 1,
                    offset,
                });
                offset += 1;
            }
            MorChunk::Terminator(_) => addresses.push(HeadIdentity::Terminator),
        }
    }
    addresses
}

fn morph_tokens(
    file: &ValidChatFile,
    side: &str,
) -> Result<BTreeMap<String, Vec<Vec<MorToken>>>, PairFailureReason> {
    let chat = file.document();
    let extracted = extract::extract_words(chat, PositionalDomain::Mor);
    let utterances: Vec<_> = chat.utterances().collect();
    let mut per_speaker: BTreeMap<String, Vec<Vec<MorToken>>> = BTreeMap::new();
    for entry in extracted {
        let utterance = utterances
            .get(entry.utterance_index.raw())
            .ok_or_else(|| binding_failure(side, "extracted utterance is absent"))?;
        let annotations: Vec<TokenAnnotation> = match utterance.mor_tier() {
            None => vec![TokenAnnotation::Absent; entry.words.len()],
            Some(tier) => {
                if tier.items().len() != entry.words.len() {
                    return Err(binding_failure(
                        side,
                        "admitted main/mor projection has unequal item counts",
                    ));
                }
                let addresses = chunk_addresses(tier);
                let mut dependencies = BTreeMap::new();
                if let Some(gra) = utterance.gra_tier() {
                    for relation in gra.relations() {
                        let index = relation
                            .index_as_semantic()
                            .map_err(|_| {
                                binding_failure(side, "invalid admitted dependency index")
                            })?
                            .to_chunk_index()
                            .as_usize();
                        let head = match relation.head_ref() {
                            GraHeadRef::Root => HeadIdentity::Root,
                            GraHeadRef::Word(index) => addresses
                                .get(index.to_chunk_index().as_usize())
                                .cloned()
                                .ok_or_else(|| {
                                    binding_failure(side, "dependency head has no chunk")
                                })?,
                        };
                        if dependencies
                            .insert(
                                index,
                                Dependency {
                                    head,
                                    relation: relation.relation.as_str().to_string(),
                                },
                            )
                            .is_some()
                        {
                            return Err(binding_failure(
                                side,
                                "duplicate admitted dependency index",
                            ));
                        }
                    }
                    if dependencies.len() != addresses.len() {
                        return Err(binding_failure(
                            side,
                            "dependency coverage does not match all chunks",
                        ));
                    }
                }
                let mut items = vec![
                    if utterance.gra_tier().is_some() {
                        TokenAnnotation::Complete(Vec::new())
                    } else {
                        TokenAnnotation::MorphologyOnly(Vec::new())
                    };
                    tier.items().len()
                ];
                for (index, chunk) in tier.chunks().enumerate() {
                    if let Some(word) = chunk.word() {
                        let Some(HeadIdentity::WordChunk { item, .. }) = addresses.get(index)
                        else {
                            return Err(binding_failure(side, "word chunk has no host item"));
                        };
                        let analysis = ChunkAnalysis {
                            lemma: word.lemma.as_str().to_string(),
                            pos: word.pos.as_str().to_string(),
                            features: word.features.iter().map(ToString::to_string).collect(),
                        };
                        match &mut items[*item] {
                            TokenAnnotation::MorphologyOnly(chunks) => chunks.push(analysis),
                            TokenAnnotation::Complete(chunks) => {
                                let dependency = dependencies.remove(&index).ok_or_else(|| {
                                    binding_failure(side, "word chunk lacks dependency")
                                })?;
                                chunks.push(AnnotatedChunk {
                                    analysis,
                                    dependency,
                                });
                            }
                            TokenAnnotation::Absent => {
                                return Err(binding_failure(
                                    side,
                                    "annotated item lost its admission state",
                                ));
                            }
                        }
                    }
                }
                items
            }
        };
        let tokens = entry
            .words
            .iter()
            .zip(annotations)
            .map(|(word, annotation)| MorToken {
                text: normalize(word.text.as_str()),
                annotation,
            })
            .collect();
        per_speaker
            .entry(entry.speaker.as_str().to_string())
            .or_default()
            .push(tokens);
    }
    Ok(per_speaker)
}

fn compare_morph_pair(
    left: &ValidChatFile,
    right: &ValidChatFile,
    pair: &ValidatedArtifactPair,
) -> PairOutcome<MorphotagPairResult> {
    let compared = (|| {
        let map = correspondence(left, right, pair)?;
        let left_tokens = morph_tokens(map.left, "left")?;
        let right_tokens = morph_tokens(map.right, "right")?;
        let mut tokens = Vec::new();
        for (left_speaker, right_speaker) in map.assignments() {
            let left_utts = left_tokens
                .get(left_speaker)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let right_utts = right_tokens
                .get(right_speaker)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            for utterance in 0..left_utts.len().max(right_utts.len()) {
                let l = left_utts.get(utterance).map(Vec::as_slice).unwrap_or(&[]);
                let r = right_utts.get(utterance).map(Vec::as_slice).unwrap_or(&[]);
                for token in 0..l.len().max(r.len()) {
                    tokens.push(compare_token(
                        l.get(token),
                        r.get(token),
                        left_speaker,
                        right_speaker,
                        utterance,
                        token,
                    ));
                }
            }
        }
        Ok::<_, PairFailureReason>(MorphotagPairResult { tokens })
    })();
    match compared {
        Ok(result) => PairOutcome::Compared { result },
        Err(reason) => PairOutcome::Unpairable { reason },
    }
}

fn token_chunks(token: Option<&MorToken>) -> impl Iterator<Item = &ChunkAnalysis> + Clone {
    token
        .into_iter()
        .flat_map(|token| token.annotation.chunks())
}

fn projected_differences<T, V: PartialEq>(
    left: Option<&[T]>,
    right: Option<&[T]>,
    project: impl Fn(&T) -> &V,
) -> bool {
    match (left, right) {
        (None, None) => false,
        (Some(left), Some(right)) => left.iter().map(&project).ne(right.iter().map(&project)),
        _ => true,
    }
}

fn compare_token(
    left: Option<&MorToken>,
    right: Option<&MorToken>,
    left_speaker: &str,
    right_speaker: &str,
    utterance: usize,
    token: usize,
) -> MorphotagTokenDifference {
    let mut differences = BTreeSet::new();
    let mut note = |different, kind| {
        if different {
            differences.insert(kind);
        }
    };
    let left_text = left.map(|value| value.text.clone());
    let right_text = right.map(|value| value.text.clone());
    let l = token_chunks(left);
    let r = token_chunks(right);
    note(left_text != right_text, MorphotagDifference::Tokenization);
    note(
        l.clone().map(|c| &c.lemma).ne(r.clone().map(|c| &c.lemma)),
        MorphotagDifference::Lemma,
    );
    note(
        l.clone().map(|c| &c.pos).ne(r.clone().map(|c| &c.pos)),
        MorphotagDifference::Pos,
    );
    note(
        l.clone()
            .map(|c| &c.features)
            .ne(r.clone().map(|c| &c.features)),
        MorphotagDifference::FeatureSet,
    );
    note(l.count() != r.count(), MorphotagDifference::CliticChunk);
    let ld = left.and_then(|value| value.annotation.complete_chunks());
    let rd = right.and_then(|value| value.annotation.complete_chunks());
    note(
        projected_differences(ld, rd, |chunk| &chunk.dependency.head),
        MorphotagDifference::DependencyHead,
    );
    note(
        projected_differences(ld, rd, |chunk| &chunk.dependency.relation),
        MorphotagDifference::Relation,
    );
    let left_annotation = left.map_or(AnnotationPresence::NoMainToken, |value| {
        value.annotation.presence()
    });
    let right_annotation = right.map_or(AnnotationPresence::NoMainToken, |value| {
        value.annotation.presence()
    });
    let analysis_agreement = (left_annotation == AnnotationPresence::Complete
        && right_annotation == AnnotationPresence::Complete)
        .then_some(differences.is_empty());
    MorphotagTokenDifference {
        left_speaker: left_speaker.to_string(),
        right_speaker: right_speaker.to_string(),
        utterance,
        token,
        left_text,
        right_text,
        left_annotation,
        right_annotation,
        differences,
        analysis_agreement,
    }
}

#[cfg(test)]
mod tests;
