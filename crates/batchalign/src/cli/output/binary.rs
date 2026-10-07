//! Source-bound binary delivery: only an admitted plan can receive, and only
//! complete identity-verified bytes can consume no-clobber publication.

use super::{OutputRoot, PlannedOutputPath, ResultDestinations};
use crate::api::{ArtifactDigest, BinaryResultDescriptor, DisplayPath, FileResult};
use crate::cli::error::CliError;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;

/// Why binary delivery refused; never classified by parsing diagnostic prose.
#[derive(Debug, thiserror::Error)]
pub enum BinaryResultRefusal {
    /// HTTP encoding disagrees with the destination's admitted kind.
    #[error("HTTP content type {observed:?} does not match {expected}")]
    MediaType {
        /// Encoding admitted by the destination plan.
        expected: &'static str,
        /// Encoding on the response, absent if no header was supplied.
        observed: Option<String>,
    },
    /// The body ended early or exceeded its declared identity.
    #[error("expected {expected} bytes, received {observed}")]
    Length {
        /// Count declared by the result descriptor.
        expected: u64,
        /// Count actually observed before refusal.
        observed: u64,
    },
    /// Same-length bytes are not the declared artifact.
    #[error("download digest does not match the result descriptor")]
    Digest,
    /// No existing foreign file, including a symlink, may be overwritten.
    #[error("destination already exists and is not the identical artifact: {0}")]
    Existing(PathBuf),
}

pub(super) struct BinaryWritePlan<'a> {
    descriptor: &'a BinaryResultDescriptor,
    destination: PathBuf,
    filename: DisplayPath,
    media_type: &'static str,
    path: PlannedOutputPath,
    root: OutputRoot,
}

impl<'a> BinaryWritePlan<'a> {
    pub(super) fn admit(
        result: &FileResult,
        descriptor: &'a BinaryResultDescriptor,
        destinations: &ResultDestinations,
    ) -> Result<Self, CliError> {
        let planned = destinations
            .artifacts
            .get(result.filename.as_ref())
            .ok_or_else(|| CliError::UnplannedResult(result.filename.clone()))?;
        if planned.content_type != result.content_type {
            return Err(CliError::ResultTypeMismatch(result.filename.clone()));
        }
        let media_type = planned
            .content_type
            .binary_media_type()
            .ok_or_else(|| CliError::ResultTypeMismatch(result.filename.clone()))?;
        planned.path.checked_under(&planned.root)?;
        if let Ok(metadata) = std::fs::symlink_metadata(&planned.path.0)
            && metadata.file_type().is_symlink()
        {
            return Err(CliError::BinaryResult {
                filename: result.filename.clone(),
                refusal: BinaryResultRefusal::Existing(planned.path.0.clone()),
            });
        }
        let root = OutputRoot::prepare(planned.root.as_path())?;
        let destination = planned.path.clone().verified_under(&root)?;
        Ok(Self {
            descriptor,
            destination,
            filename: result.filename.clone(),
            media_type,
            path: planned.path.clone(),
            root,
        })
    }

    fn refusal(&self, refusal: BinaryResultRefusal) -> CliError {
        CliError::BinaryResult {
            filename: self.filename.clone(),
            refusal,
        }
    }

    /// Shared-filesystem jobs may have already published this exact artifact.
    /// That is accepted only by observed byte identity, never by mere existence.
    pub(super) async fn existing_matches(&self) -> Result<bool, CliError> {
        self.check_live_destination()?;
        let metadata = match tokio::fs::symlink_metadata(&self.destination).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() != self.descriptor.byte_len.get()
        {
            return Err(self.refusal(BinaryResultRefusal::Existing(self.destination.clone())));
        }
        let mut file = tokio::fs::File::open(&self.destination).await?;
        let observed = BinaryResultDescriptor::inspect(&mut file).await?;
        if observed != *self.descriptor {
            return Err(self.refusal(BinaryResultRefusal::Existing(self.destination.clone())));
        }
        Ok(true)
    }

    pub(super) async fn receive(
        self,
        mut response: reqwest::Response,
    ) -> Result<VerifiedBinaryDelivery, CliError> {
        let observed = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if observed.as_deref() != Some(self.media_type) {
            return Err(self.refusal(BinaryResultRefusal::MediaType {
                expected: self.media_type,
                observed,
            }));
        }
        self.check_live_destination()?;
        let temporary = NamedTempFile::new_in(self.destination.parent().ok_or_else(|| {
            CliError::InvalidArgument("admitted binary destination has no parent".into())
        })?)?;
        let mut output = tokio::fs::File::from_std(temporary.reopen()?);
        let mut count = 0u64;
        let mut hasher = blake3::Hasher::new();
        while let Some(chunk) = response.chunk().await? {
            count = count.checked_add(chunk.len() as u64).ok_or_else(|| {
                self.refusal(BinaryResultRefusal::Length {
                    expected: self.descriptor.byte_len.get(),
                    observed: u64::MAX,
                })
            })?;
            if count > self.descriptor.byte_len.get() {
                return Err(self.refusal(BinaryResultRefusal::Length {
                    expected: self.descriptor.byte_len.get(),
                    observed: count,
                }));
            }
            hasher.update(&chunk);
            output.write_all(&chunk).await?;
        }
        if count != self.descriptor.byte_len.get() {
            return Err(self.refusal(BinaryResultRefusal::Length {
                expected: self.descriptor.byte_len.get(),
                observed: count,
            }));
        }
        if ArtifactDigest::from_hash(hasher.finalize()) != self.descriptor.digest {
            return Err(self.refusal(BinaryResultRefusal::Digest));
        }
        output.sync_all().await?;
        drop(output);
        Ok(VerifiedBinaryDelivery {
            temporary,
            destination: self.destination,
            filename: self.filename,
            path: self.path,
            root: self.root,
        })
    }

    fn check_live_destination(&self) -> Result<(), CliError> {
        if self.path.checked_under(&self.root)? != self.destination {
            return Err(CliError::PathTraversal(self.filename.to_string()));
        }
        Ok(())
    }
}

/// Publication proof produced only after complete streamed identity admission.
pub(super) struct VerifiedBinaryDelivery {
    temporary: NamedTempFile,
    destination: PathBuf,
    filename: DisplayPath,
    path: PlannedOutputPath,
    root: OutputRoot,
}

impl VerifiedBinaryDelivery {
    pub(super) fn publish(self) -> Result<(), CliError> {
        if self.path.checked_under(&self.root)? != self.destination {
            return Err(CliError::PathTraversal(self.filename.to_string()));
        }
        self.temporary
            .persist_noclobber(&self.destination)
            .map_err(|error| {
                if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                    CliError::BinaryResult {
                        filename: self.filename,
                        refusal: BinaryResultRefusal::Existing(self.destination),
                    }
                } else {
                    CliError::Io(error.error)
                }
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
