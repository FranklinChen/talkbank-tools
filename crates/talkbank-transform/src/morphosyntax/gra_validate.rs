//! Validation of generated `%gra` structures.

use crate::morphosyntax::MappingError;
use std::collections::HashSet;
use talkbank_model::model::GrammaticalRelation;

/// Validate that generated `%gra` relations form a valid dependency tree.
pub fn validate_generated_gra(gras: &[GrammaticalRelation]) -> Result<(), MappingError> {
    if gras.is_empty() {
        return Ok(());
    }

    let mut roots = Vec::new();
    for rel in gras {
        if rel.head == 0 || rel.head == rel.index {
            roots.push(rel.index);
        }
    }

    let non_terminator_roots: Vec<_> = roots
        .iter()
        .filter(|&&idx| idx != gras.len())
        .copied()
        .collect();

    if non_terminator_roots.is_empty() {
        return Err(MappingError::InvalidRoot {
            details: format!("no ROOT relation. GRA: {:?}", gras),
        });
    }

    if non_terminator_roots.len() > 1 {
        return Err(MappingError::InvalidRoot {
            details: format!(
                "multiple ROOT relations: {:?}. GRA: {:?}",
                non_terminator_roots, gras
            ),
        });
    }

    if let Some(word) = has_any_cycle_generated(gras) {
        return Err(MappingError::CircularDependency {
            details: format!("involving word {}. GRA: {:?}", word, gras),
        });
    }

    let max_index = gras.len();
    for rel in gras {
        if rel.head != 0 && rel.head > max_index {
            return Err(MappingError::InvalidHeadReference {
                details: format!(
                    "word {} points to non-existent word {}. GRA: {:?}",
                    rel.index, rel.head, gras
                ),
            });
        }
    }

    Ok(())
}

fn has_any_cycle_generated(gras: &[GrammaticalRelation]) -> Option<usize> {
    let mut safe: HashSet<usize> = HashSet::new();
    for rel in gras {
        if safe.contains(&rel.index) {
            continue;
        }
        let mut path: HashSet<usize> = HashSet::new();
        let mut current = rel.index;
        loop {
            if safe.contains(&current) {
                safe.extend(&path);
                break;
            }
            if path.contains(&current) {
                return Some(current);
            }
            path.insert(current);
            if let Some(r) = gras.iter().find(|r| r.index == current) {
                if r.head == 0 || r.head == current {
                    safe.extend(&path);
                    break;
                }
                current = r.head;
            } else {
                safe.extend(&path);
                break;
            }
        }
    }
    None
}
