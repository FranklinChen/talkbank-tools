//! Per-file language admission for morphosyntax.

use crate::{api::LanguageCode3, chat_ops::ChatFile, error::ServerError};

/// Resolve the per-file morphotag language from the parsed `@Languages:`
/// header.
///
/// Returns a typed error when the header is absent or the declared
/// language is not a parseable ISO 639-3 code. **No silent fallback to
/// English**: falling back would either tag a non-English file as English
/// (the 2026-05-03 incident) or stamp a falsified `@Languages:` value into
/// the output. The caller (`ParsedFile::parse`) records the error against the
/// file's job-status entry; the file is returned unchanged.
///
/// Earlier BA2 code defaulted missing headers to `["eng"]`. That parity
/// shortcut was a known correctness hazard, see the 2026-05-03 incident.
/// We deliberately diverge.
pub(crate) fn resolve_per_file_lang(chat_file: &ChatFile) -> Result<LanguageCode3, ServerError> {
    let languages = chat_file.language_declaration().map_err(|error| {
        ServerError::Validation(format!(
            "morphotag: unusable `@Languages:` declaration ({error}). Supply one valid header before analysis."
        ))
    })?;
    let raw = languages.primary();
    LanguageCode3::try_from(raw.as_str()).map_err(|err| {
        ServerError::Validation(format!(
            "morphotag: file's `@Languages:` declares '{raw}', which is not a \
             parseable ISO 639-3 code: {err}. Fix the header and re-run."
        ))
    })
}
