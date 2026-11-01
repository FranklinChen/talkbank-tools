//! Recorded local output destinations, projected without inventing a folder
//! from an input path or claiming that the outputs still exist on disk.

use std::collections::BTreeSet;
use std::path::Path;

use batchalign_types::paths::ClientPath;
use serde::Serialize;

/// Snapshot identity and its independent local-output evidence.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobOutputFolders {
    /// The record whose submitted destinations were projected.
    pub job_id: super::domain::JobId,
    /// Absence, refusal and recorded destinations are distinct outcomes.
    pub access: LocalOutputAccess,
}

/// A nonempty, deduplicated population admitted from recorded absolute paths.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct RecordedOutputFolders(Vec<String>);

/// Only paths-mode records carry local filesystem authority.
pub(crate) enum OutputLocationSource<'a> {
    Content,
    Paths {
        count: usize,
        outputs: &'a [ClientPath],
    },
}

/// Directory existence belongs to the native opener at action time. A recorded
/// destination remains useful even when all input files were refused.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub enum LocalOutputAccess {
    /// Content uploads have downloadable results, not client-local folders.
    ContentMode,
    /// A legacy/empty record contains no output-location evidence.
    NotRecorded,
    /// Every entry is the parent of a recorded output path, never source_dir.
    Recorded {
        /// Nonempty destinations admitted together from the stored mapping.
        #[cfg_attr(feature = "server", schema(value_type = Vec<String>, min_items = 1))]
        directories: RecordedOutputFolders,
    },
    /// Malformed stored mappings cannot produce a successful empty response.
    Refused {
        /// Why the stored mapping could not supply folder authority.
        message: String,
    },
}

impl LocalOutputAccess {
    pub(crate) fn admit(source: OutputLocationSource<'_>) -> Self {
        let (count, outputs) = match source {
            OutputLocationSource::Content => return Self::ContentMode,
            OutputLocationSource::Paths { count, outputs } => (count, outputs),
        };
        if outputs.is_empty() {
            return Self::NotRecorded;
        }
        if count != outputs.len() {
            return Self::Refused {
                message: "The stored output mapping does not match the job's input population."
                    .into(),
            };
        }
        let mut directories = BTreeSet::new();
        for output in outputs {
            let path = Path::new(output.as_str());
            let Some(parent) = path
                .parent()
                .filter(|_| path.is_absolute() && path.file_name().is_some())
            else {
                return Self::Refused {
                    message: "The stored output mapping contains a non-absolute file path.".into(),
                };
            };
            let Some(text) = parent.to_str() else {
                return Self::Refused {
                    message: "The stored output folder cannot be represented by the API.".into(),
                };
            };
            directories.insert(text.to_owned());
        }
        // Nonemptiness follows from the admitted nonempty output population;
        // every loop edge either inserts a parent or returns a refusal.
        Self::Recorded {
            directories: RecordedOutputFolders(directories.into_iter().collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_output_folders_preserve_distinct_destinations_and_deduplicate() {
        let root = std::env::temp_dir();
        let outputs = [
            root.join("first/a.cha"),
            root.join("first/b.cha"),
            root.join("second/c.cha"),
        ]
        .into_iter()
        .map(|path| ClientPath::new(path.to_str().unwrap()))
        .collect::<Vec<_>>();
        let access = LocalOutputAccess::admit(OutputLocationSource::Paths {
            count: 3,
            outputs: &outputs,
        });
        let wire = serde_json::to_value(access).unwrap();
        assert_eq!(wire["state"], "recorded");
        assert_eq!(
            wire["directories"],
            serde_json::json!([
                root.join("first").to_str().unwrap(),
                root.join("second").to_str().unwrap()
            ])
        );
    }

    #[test]
    fn output_location_absence_and_malformed_mapping_are_distinct() {
        assert!(matches!(
            LocalOutputAccess::admit(OutputLocationSource::Content),
            LocalOutputAccess::ContentMode
        ));
        assert!(matches!(
            LocalOutputAccess::admit(OutputLocationSource::Paths {
                count: 1,
                outputs: &[]
            }),
            LocalOutputAccess::NotRecorded
        ));
        let relative = [ClientPath::new("relative/result.cha")];
        assert!(matches!(
            LocalOutputAccess::admit(OutputLocationSource::Paths {
                count: 1,
                outputs: &relative
            }),
            LocalOutputAccess::Refused { .. }
        ));
        assert!(matches!(
            LocalOutputAccess::admit(OutputLocationSource::Paths {
                count: 2,
                outputs: &relative
            }),
            LocalOutputAccess::Refused { .. }
        ));
    }
}
