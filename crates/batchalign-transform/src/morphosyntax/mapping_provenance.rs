//! Per-chunk provenance for sentence-level `%gra` construction.

use smallvec::{SmallVec, smallvec};
use talkbank_model::model::GrammaticalRelationType;
use talkbank_model::model::dependent_tier::mor::{Mor, MorWord};

use super::{UdHead, UdWord, UdWordId};

/// One `%mor` item ready for a sentence: the item and one provenance entry
/// per chunk (main first, then post-clitics).
///
/// Built a chunk at a time ([`Self::word`], then [`Self::with_post_clitic`]),
/// each chunk with its provenance, so an item cannot have a chunk without an
/// entry or an entry without a chunk.
#[derive(Debug, Clone)]
pub struct MappedItem {
    mor: Mor,
    provenance: MorProvenance,
}

impl MappedItem {
    /// A one-chunk item.
    pub fn word(word: MorWord, chunk: ChunkProvenance) -> Self {
        Self {
            mor: Mor::new(word),
            provenance: smallvec![chunk],
        }
    }

    /// This item with one more post-clitic chunk.
    pub fn with_post_clitic(mut self, clitic: MorWord, chunk: ChunkProvenance) -> Self {
        self.mor = self.mor.with_post_clitic(clitic);
        self.provenance.push(chunk);
        self
    }

    /// The `%mor` item.
    pub fn mor(&self) -> &Mor {
        &self.mor
    }

    /// One entry per chunk of the item, main first.
    pub fn provenance(&self) -> &[ChunkProvenance] {
        &self.provenance
    }

    /// The item and its chunks' provenance.
    pub fn into_parts(self) -> (Mor, MorProvenance) {
        (self.mor, self.provenance)
    }
}

/// Where one `%mor` chunk came from, for the `%gra` builder: which UD words
/// resolve to it, where its head is, and its relation.
#[derive(Debug, Clone)]
pub struct ChunkProvenance {
    source_ud_ids: SmallVec<[UdWordId; 1]>,
    head: ChunkHead,
    deprel: GrammaticalRelationType,
}

/// How a chunk's head resolves to a concrete `%gra` head index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkHead {
    /// The chunk is the sentence root.
    Root,
    /// Head resolves via the original UD head word.
    FromUd(UdWordId),
    /// Head points to the main chunk of this provenance's owning MOR.
    OwningMorMain,
}

impl ChunkHead {
    /// Where a UD word's head is: the root, or another UD word.
    pub fn of_word(word: &UdWord) -> Self {
        match word.head {
            UdHead::Root => Self::Root,
            UdHead::Word(head) => Self::FromUd(head),
        }
    }
}

impl ChunkProvenance {
    /// The chunk of one UD word: its own id (a range row stands for its
    /// first component; an empty node has none), head and relation.
    pub fn of_word(word: &UdWord, deprel: GrammaticalRelationType) -> Self {
        Self {
            source_ud_ids: UdWordId::of_row(word).into_iter().collect(),
            head: ChunkHead::of_word(word),
            deprel,
        }
    }

    /// The one chunk a whole multi-word token collapsed into: every id of
    /// the range resolves to it.
    pub fn collapsed_range(
        source_ud_ids: impl IntoIterator<Item = UdWordId>,
        head: ChunkHead,
        deprel: GrammaticalRelationType,
    ) -> Self {
        Self {
            source_ud_ids: source_ud_ids.into_iter().collect(),
            head,
            deprel,
        }
    }

    /// A post-clitic our tables synthesized: no UD word resolves to it, and
    /// it depends on its item's main chunk.
    pub fn synthetic_post_clitic(deprel: GrammaticalRelationType) -> Self {
        Self {
            source_ud_ids: SmallVec::new(),
            head: ChunkHead::OwningMorMain,
            deprel,
        }
    }

    /// The UD words whose head references resolve to this chunk.
    pub fn source_ud_ids(&self) -> &[UdWordId] {
        &self.source_ud_ids
    }

    /// Where the chunk's head is.
    pub fn head(&self) -> &ChunkHead {
        &self.head
    }

    /// The chunk's relation.
    pub fn deprel(&self) -> &GrammaticalRelationType {
        &self.deprel
    }
}

/// Per-MOR provenance list: chunk 0 is the main, rest are post-clitics.
pub type MorProvenance = SmallVec<[ChunkProvenance; 3]>;
