//! Helper functions for sentence-level UD-to-CHAT mapping.

use super::mor_word::map_ud_mor_word;
use crate::morphosyntax::{
    ChunkProvenance, MappedItem, MappingContext, MappingError, UdWord, is_clitic,
};
use std::borrow::Cow;
use talkbank_model::model::GrammaticalRelationType;

/// Normalize a UD deprel to a validated CHAT `%gra` relation label.
pub fn normalize_deprel(
    raw: &str,
    context_for_error: impl FnOnce() -> String,
) -> Result<GrammaticalRelationType, MappingError> {
    let needs_transform = raw.bytes().any(|b| b.is_ascii_lowercase() || b == b':');
    let relation: Cow<'_, str> = if needs_transform {
        Cow::Owned(raw.to_uppercase().replace(':', "-"))
    } else {
        Cow::Borrowed(raw)
    };
    let bytes = relation.as_bytes();
    if bytes.is_empty()
        || !bytes[0].is_ascii_uppercase()
        || !bytes
            .iter()
            .all(|&b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(MappingError::InvalidDeprel {
            details: format!(
                "{}: deprel {:?} transforms to {:?}, not a valid CHAT %gra relation (must match [A-Z][A-Z0-9-]*)",
                context_for_error(),
                raw,
                relation.as_ref()
            ),
        });
    }
    Ok(GrammaticalRelationType::new(relation.as_ref()))
}

/// Build chunk provenance for a regular UD word that produced one chunk.
pub fn provenance_for_ud_word(ud: &UdWord) -> Result<ChunkProvenance, MappingError> {
    let deprel = normalize_deprel(&ud.deprel, || format!("word {:?}", ud.text))?;
    Ok(ChunkProvenance::of_word(ud, deprel))
}

/// One UD word mapped to a one-chunk `%mor` item with its provenance.
pub(crate) fn map_ud_word_item(
    ud: &UdWord,
    ctx: &MappingContext,
) -> Result<MappedItem, MappingError> {
    Ok(MappedItem::word(
        map_ud_mor_word(ud, ctx)?,
        provenance_for_ud_word(ud)?,
    ))
}

/// Assemble multiple UD tokens into a single CHAT MOR with clitics, each
/// chunk with its provenance.
pub fn assemble_mors(
    components: &[UdWord],
    ctx: &MappingContext,
) -> Result<MappedItem, MappingError> {
    // The first component that is not a clitic is the main word; when every
    // component is one, the first is.
    let (main_idx, main) = components
        .iter()
        .enumerate()
        .find(|(_, comp)| !is_clitic(&comp.text, ctx))
        .or_else(|| components.first().map(|first| (0, first)))
        .ok_or(MappingError::EmptyRangeComponents)?;
    components[..main_idx]
        .iter()
        .chain(&components[main_idx + 1..])
        .try_fold(map_ud_word_item(main, ctx)?, |item, comp| {
            Ok(item.with_post_clitic(map_ud_mor_word(comp, ctx)?, provenance_for_ud_word(comp)?))
        })
}
