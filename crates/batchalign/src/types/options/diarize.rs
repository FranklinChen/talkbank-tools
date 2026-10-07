//! Explicit anonymous-track mappings; acoustic labels never establish identity.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use talkbank_model::SpeakerCode;
use talkbank_model::validation::{Validate, ValidationContext};

/// Syntactically checked anonymous tracks mapped to CHAT participant codes.
///
/// Declaration admission is a separate source-bound transition at execution.
/// Deserialization uses the same constructor as the CLI; malformed persisted
/// mappings cannot reach inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SpeakerTrackMapping(BTreeMap<usize, SpeakerCode>);

/// A mapping cannot be admitted from the supplied option.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid --speaker-map: {0}; expected PAR0=CHI,PAR1=MOT")]
pub struct SpeakerTrackMappingRefusal(String);

impl FromStr for SpeakerTrackMapping {
    type Err = SpeakerTrackMappingRefusal;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut entries = BTreeMap::new();
        for item in value.split(',') {
            let (track, target) = item.split_once('=').ok_or_else(|| {
                SpeakerTrackMappingRefusal(format!("missing track=participant in {item:?}"))
            })?;
            if target.is_empty() {
                return Err(SpeakerTrackMappingRefusal(format!(
                    "missing participant for {track:?}"
                )));
            }
            let index = track
                .strip_prefix("PAR")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .filter(|index| track == format!("PAR{index}"))
                .ok_or_else(|| {
                    SpeakerTrackMappingRefusal(format!("noncanonical track {track:?}"))
                })?;
            let target = SpeakerCode::new(target);
            let errors = talkbank_model::ErrorCollector::new();
            target.validate(&ValidationContext::default(), &errors);
            if !errors.into_vec().is_empty() {
                return Err(SpeakerTrackMappingRefusal(format!(
                    "invalid participant code {:?}",
                    target.as_str()
                )));
            }
            if entries.insert(index, target).is_some() {
                return Err(SpeakerTrackMappingRefusal(format!(
                    "duplicate track {track}"
                )));
            }
        }
        Ok(Self(entries))
    }
}

impl TryFrom<String> for SpeakerTrackMapping {
    type Error = SpeakerTrackMappingRefusal;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl fmt::Display for SpeakerTrackMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, (track, target)) in self.0.iter().enumerate() {
            if position > 0 {
                f.write_str(",")?;
            }
            write!(f, "PAR{track}={target}")?;
        }
        Ok(())
    }
}

impl From<SpeakerTrackMapping> for String {
    fn from(value: SpeakerTrackMapping) -> Self {
        value.to_string()
    }
}

impl SpeakerTrackMapping {
    pub(crate) fn entries(&self) -> impl Iterator<Item = (usize, &SpeakerCode)> {
        self.0.iter().map(|(track, target)| (*track, target))
    }
}

/// Standalone diarization's mutually exclusive source/output contracts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DiarizeOutputMode {
    /// Media input produces anonymous speaker-turn evidence, not rewritten CHAT.
    #[default]
    TurnsJson,
    /// Timed CHAT is rewritten only through an explicit participant mapping.
    MappedChat {
        /// Syntax-checked mapping; all targets must be declared in each source.
        mapping: SpeakerTrackMapping,
    },
}

impl DiarizeOutputMode {
    pub(crate) fn accepts_source_name(&self, name: &str) -> bool {
        let chat = crate::types::request::is_chat_source_name(name);
        match self {
            Self::MappedChat { .. } => chat,
            Self::TurnsJson => {
                !chat
                    && std::path::Path::new(name)
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| {
                            crate::media::MediaExtensions::is_known(&extension.to_ascii_lowercase())
                        })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_has_one_checked_cli_and_wire_constructor() {
        let mapping: SpeakerTrackMapping = "PAR1=MOT,PAR0=CHI".parse().unwrap();
        assert_eq!(mapping.to_string(), "PAR0=CHI,PAR1=MOT");
        let json = serde_json::to_string(&mapping).unwrap();
        assert_eq!(
            serde_json::from_str::<SpeakerTrackMapping>(&json).unwrap(),
            mapping
        );
        // Explicit many-to-one mapping is not an inferred identity claim.
        assert!("PAR0=CHI,PAR1=CHI".parse::<SpeakerTrackMapping>().is_ok());
    }

    #[test]
    fn malformed_or_duplicate_mapping_is_refused_on_both_boundaries() {
        for value in [
            "",
            "PAR0=",
            "PAR0=CHI,",
            "PAR01=CHI",
            "PAR-1=CHI",
            "SPK0=CHI",
            "PAR0=CHI,PAR0=MOT",
            "PAR0=Ä",
            "PAR0=CH:I",
            "PAR0=CHILDREN",
        ] {
            assert!(
                value.parse::<SpeakerTrackMapping>().is_err(),
                "accepted {value:?}"
            );
            assert!(
                serde_json::from_value::<SpeakerTrackMapping>(serde_json::json!(value)).is_err()
            );
        }
    }
}
