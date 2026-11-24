//! Source-aware output naming for recipe-runner commands.

use std::path::Path;

use crate::api::{ContentType, DisplayPath};

/// Stem rewrite used by commands whose outputs are not a simple extension swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StemRewrite {
    /// Optional suffix to strip from the stem before appending a new one.
    pub strip_suffix: Option<&'static str>,
    /// Suffix to append to the stem before writing the final extension.
    pub append_suffix: &'static str,
    /// Final extension to write.
    pub extension: &'static str,
}

/// Filename policy for primary outputs and sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileNamingPolicy {
    /// Keep the source-relative display path unchanged.
    PreserveInput,
    /// Replace the final extension while keeping directories and stem.
    ReplaceExtension(&'static str),
    /// Rewrite the stem before applying the final extension.
    RewriteStem(StemRewrite),
}

/// Output policy for a released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPolicy {
    /// Source-independent output contract.
    Fixed {
        /// Primary output naming policy.
        primary: FileNamingPolicy,
        /// Primary output content type.
        primary_content_type: ContentType,
    },
    /// A CHAT result and its source-derived comparison metrics CSV.
    ChatMetrics {
        /// Primary CHAT naming policy.
        primary: FileNamingPolicy,
    },
    /// Admitted CHAT sources retain their name; media sources produce turns JSON.
    /// Submission checks the source against the explicit diarization mode first.
    Diarize,
}

/// Catalog declaration: fixed/source-selected outputs or required encoding.
/// A declaration is not an output plan until command options are admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDeclaration {
    /// Existing source-aware text/analysis policy.
    Source(OutputPolicy),
    /// Native export must select its format from submitted typed options.
    AudioExport,
}

impl OutputDeclaration {
    pub(super) const fn select(
        self,
        options: &crate::options::CommandOptions,
    ) -> Result<OutputPolicy, super::planner::PlanningError> {
        match self {
            Self::Source(policy) => Ok(policy),
            Self::AudioExport => {
                let crate::options::CommandOptions::Convert(options) = options else {
                    return Err(super::planner::PlanningError::OutputOptionsMismatch {
                        expected: crate::ReleasedCommand::Convert,
                        observed: options.command(),
                    });
                };
                let content_type = match options.format {
                    crate::media::export::AudioExportFormat::Wav => ContentType::Wav,
                    crate::media::export::AudioExportFormat::Mp3 => ContentType::Mp3,
                };
                Ok(OutputPolicy::Fixed {
                    primary: FileNamingPolicy::RewriteStem(StemRewrite {
                        strip_suffix: None,
                        append_suffix: ".converted",
                        extension: options.format.extension(),
                    }),
                    primary_content_type: content_type,
                })
            }
        }
    }
}

/// Source-specific shape after diarization's input-kind choice is resolved.
enum SelectedOutputPolicy {
    Primary {
        naming: FileNamingPolicy,
        content_type: ContentType,
    },
    ChatMetrics {
        naming: FileNamingPolicy,
    },
}

impl OutputPolicy {
    fn for_source(self, source: &DisplayPath) -> SelectedOutputPolicy {
        match self {
            Self::Fixed {
                primary,
                primary_content_type,
            } => SelectedOutputPolicy::Primary {
                naming: primary,
                content_type: primary_content_type,
            },
            Self::ChatMetrics { primary } => SelectedOutputPolicy::ChatMetrics { naming: primary },
            Self::Diarize => {
                if crate::types::request::is_chat_source_name(source.as_ref()) {
                    SelectedOutputPolicy::Primary {
                        naming: FileNamingPolicy::PreserveInput,
                        content_type: ContentType::Chat,
                    }
                } else {
                    SelectedOutputPolicy::Primary {
                        naming: FileNamingPolicy::RewriteStem(StemRewrite {
                            strip_suffix: None,
                            append_suffix: ".turns",
                            extension: "json",
                        }),
                        content_type: ContentType::Json,
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn primary_naming_for_source(self, source: &DisplayPath) -> FileNamingPolicy {
        match self.for_source(source) {
            SelectedOutputPolicy::Primary {
                naming,
                content_type: _,
            }
            | SelectedOutputPolicy::ChatMetrics { naming } => naming,
        }
    }
}

/// Role of one materialized output artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializedArtifactRole {
    /// Main user-facing output.
    Primary,
    /// Sidecar derived from the same source work unit.
    Sidecar,
}

/// Planned output artifact derived from one source work unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedMaterializedFile {
    /// Relative output path.
    pub display_path: DisplayPath,
    /// Content type surfaced through the result APIs.
    pub content_type: ContentType,
    /// Whether this is the primary artifact or a sidecar.
    pub role: MaterializedArtifactRole,
}

/// A primary result. Transport roles are derived from its type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryArtifact {
    display_path: DisplayPath,
    content_type: ContentType,
}

impl PrimaryArtifact {
    fn for_source(
        source: &DisplayPath,
        naming: FileNamingPolicy,
        content_type: ContentType,
    ) -> Self {
        Self {
            display_path: apply_file_naming_policy(source, naming),
            content_type,
        }
    }

    pub(crate) const fn display_path(&self) -> &DisplayPath {
        &self.display_path
    }
    pub(crate) const fn content_type(&self) -> ContentType {
        self.content_type
    }
    pub(crate) fn into_materialized(self) -> PlannedMaterializedFile {
        PlannedMaterializedFile {
            display_path: self.display_path,
            content_type: self.content_type,
            role: MaterializedArtifactRole::Primary,
        }
    }
}

/// CSV metrics have a fixed sidecar role and content type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsArtifact {
    display_path: DisplayPath,
}

impl MetricsArtifact {
    pub(crate) const fn display_path(&self) -> &DisplayPath {
        &self.display_path
    }
    pub(crate) const fn content_type(&self) -> ContentType {
        ContentType::Csv
    }
    fn into_materialized(self) -> PlannedMaterializedFile {
        let content_type = self.content_type();
        PlannedMaterializedFile {
            display_path: self.display_path,
            content_type,
            role: MaterializedArtifactRole::Sidecar,
        }
    }
}

/// Named comparison outputs derived together from the same source.
/// Fields remain private to the naming producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMetricsArtifacts {
    source: DisplayPath,
    primary: PrimaryArtifact,
    metrics: MetricsArtifact,
}

impl ChatMetricsArtifacts {
    pub(crate) const fn source(&self) -> &DisplayPath {
        &self.source
    }
    pub(crate) const fn primary(&self) -> &PrimaryArtifact {
        &self.primary
    }
    pub(crate) const fn metrics(&self) -> &MetricsArtifact {
        &self.metrics
    }
    pub(crate) fn into_primary(self) -> PrimaryArtifact {
        self.primary
    }
}

/// Source-bound output layout. Every variant owns exactly one primary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedArtifactSet {
    layout: ArtifactLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArtifactLayout {
    Primary {
        source: DisplayPath,
        primary: PrimaryArtifact,
    },
    ChatMetrics(ChatMetricsArtifacts),
}

impl PlannedArtifactSet {
    pub(crate) const fn source(&self) -> &DisplayPath {
        match &self.layout {
            ArtifactLayout::Primary { source, .. } => source,
            ArtifactLayout::ChatMetrics(pair) => pair.source(),
        }
    }

    pub(crate) const fn primary(&self) -> &PrimaryArtifact {
        match &self.layout {
            ArtifactLayout::Primary { primary, .. } => primary,
            ArtifactLayout::ChatMetrics(pair) => pair.primary(),
        }
    }

    pub(crate) const fn single(&self) -> Option<&PrimaryArtifact> {
        match &self.layout {
            ArtifactLayout::Primary { primary, .. } => Some(primary),
            ArtifactLayout::ChatMetrics(_) => None,
        }
    }

    pub(crate) const fn chat_metrics(&self) -> Option<&ChatMetricsArtifacts> {
        match &self.layout {
            ArtifactLayout::Primary { .. } => None,
            ArtifactLayout::ChatMetrics(pair) => Some(pair),
        }
    }

    pub(crate) fn into_chat_metrics(self) -> Result<ChatMetricsArtifacts, Self> {
        match self.layout {
            ArtifactLayout::ChatMetrics(pair) => Ok(pair),
            ArtifactLayout::Primary { .. } => Err(self),
        }
    }

    pub(crate) fn into_primary(self) -> PrimaryArtifact {
        match self.layout {
            ArtifactLayout::Primary { primary, .. } => primary,
            ArtifactLayout::ChatMetrics(pair) => pair.into_primary(),
        }
    }
}

impl IntoIterator for PlannedArtifactSet {
    type Item = PlannedMaterializedFile;
    type IntoIter =
        std::iter::Chain<std::iter::Once<Self::Item>, std::option::IntoIter<Self::Item>>;
    fn into_iter(self) -> Self::IntoIter {
        let (primary, metrics) = match self.layout {
            ArtifactLayout::Primary { primary, .. } => (primary.into_materialized(), None),
            ArtifactLayout::ChatMetrics(pair) => (
                pair.primary.into_materialized(),
                Some(pair.metrics.into_materialized()),
            ),
        };
        std::iter::once(primary).chain(metrics)
    }
}

/// Derive the supported output layout once, retaining its source and roles.
pub fn plan_materialized_files(source: &DisplayPath, policy: OutputPolicy) -> PlannedArtifactSet {
    let layout = match policy.for_source(source) {
        SelectedOutputPolicy::Primary {
            naming,
            content_type,
        } => ArtifactLayout::Primary {
            source: source.clone(),
            primary: PrimaryArtifact::for_source(source, naming, content_type),
        },
        SelectedOutputPolicy::ChatMetrics { naming } => {
            ArtifactLayout::ChatMetrics(ChatMetricsArtifacts {
                source: source.clone(),
                primary: PrimaryArtifact::for_source(source, naming, ContentType::Chat),
                metrics: MetricsArtifact {
                    display_path: apply_file_naming_policy(
                        source,
                        FileNamingPolicy::ReplaceExtension("compare.csv"),
                    ),
                },
            })
        }
    };
    PlannedArtifactSet { layout }
}

fn apply_file_naming_policy(source_path: &DisplayPath, policy: FileNamingPolicy) -> DisplayPath {
    match policy {
        FileNamingPolicy::PreserveInput => source_path.clone(),
        FileNamingPolicy::ReplaceExtension(extension) => DisplayPath::from(
            Path::new(source_path.as_ref())
                .with_extension(extension)
                .to_string_lossy()
                .to_string(),
        ),
        FileNamingPolicy::RewriteStem(rewrite) => rewrite_display_path(source_path, rewrite),
    }
}

fn rewrite_display_path(source_path: &DisplayPath, rewrite: StemRewrite) -> DisplayPath {
    let path = Path::new(source_path.as_ref());
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let rewritten_stem = if let Some(strip_suffix) = rewrite.strip_suffix {
        stem.strip_suffix(strip_suffix).unwrap_or(&stem).to_string()
    } else {
        stem.to_string()
    };
    let file_name = format!(
        "{}{}.{}",
        rewritten_stem, rewrite.append_suffix, rewrite.extension
    );
    let rewritten = path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(file_name)
        .to_string_lossy()
        .to_string();
    DisplayPath::from(rewritten)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards: Media, lowercase CHAT and uppercase CHAT sources select the promised name and content type together.
    /// Kind: policy
    /// Not a type: The closed layout prevents empty or mixed-role results but cannot choose the promised filename and content type for external source spellings.
    /// State: None
    /// Concurrency: None
    #[test]
    fn diarize_names_and_content_types_follow_the_admitted_source_kind_together() {
        for (source, name, content_type) in [
            (
                "nested/session.wav",
                "nested/session.turns.json",
                ContentType::Json,
            ),
            (
                "nested/session.cha",
                "nested/session.cha",
                ContentType::Chat,
            ),
            (
                "nested/session.CHA",
                "nested/session.CHA",
                ContentType::Chat,
            ),
        ] {
            let outputs = plan_materialized_files(&source.into(), OutputPolicy::Diarize);
            let primary = outputs.single().expect("one source-selected output");
            assert_eq!(primary.display_path().as_ref(), name);
            assert_eq!(primary.content_type(), content_type);
        }
    }
}
