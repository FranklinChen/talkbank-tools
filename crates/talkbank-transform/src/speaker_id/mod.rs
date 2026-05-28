//! Speaker identification — rewrite a CHAT file's speaker codes
//! per an operator-supplied mapping, or pick the mapping by text
//! similarity against a reference transcript.
//!
//! Currently implements explicit-mapping mode and the text-similarity
//! "identify" step of reference mode. Override-file mode is defined
//! in the user contract but not yet implemented.
//!
//! See `book/src/chatter/user-guide/speaker-id.md` for the user
//! contract.

mod apply;
mod error;
mod identify;
mod mapping;
mod override_file;
mod types;

pub use apply::{apply_mapping, apply_mapping_chat};
pub use error::SpeakerIdError;
pub use identify::{DEFAULT_CONFIDENCE_THRESHOLD, DonorMatchReport, identify_mapping};
pub use mapping::{MappingSpec, SpeakerAssignment, parse_mapping_spec};
pub use override_file::{
    CURRENT_SCHEMA_VERSION, InsertedRoleSpec, MergeOverride, OverrideFile, OverrideFileError,
    OverrideMode, SpeakerAction,
};
pub use types::{ConfidenceMargin, ConfidenceThreshold, JaccardScore};
