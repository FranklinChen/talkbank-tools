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
    let raw = chat_file.languages.first().ok_or_else(|| {
        ServerError::Validation(
            "morphotag: file has no `@Languages:` header. Add the header (e.g. \
             `@Languages: eng`) and re-run; per-file language is required for \
             honest %mor/%gra provenance."
                .to_string(),
        )
    })?;
    LanguageCode3::try_from(raw.as_str()).map_err(|err| {
        ServerError::Validation(format!(
            "morphotag: file's `@Languages:` declares '{raw}', which is not a \
             parseable ISO 639-3 code: {err}. Fix the header and re-run."
        ))
    })
}

/// Returns a per-file error message when the primary `@Languages` code is
/// not in Stanza's supported set.
///
/// Pre-2026-05-10 this function's `Some(...)` return drove a silent
/// pass-through (the file was returned unchanged with no `%mor`/`%gra`
/// injected, and the job reported `completed`). That was dishonest UX:
/// operators submitting a file with a typo'd or unsupported language
/// got back their input unchanged, with no surface signal that nothing
/// happened. The dashboard's failure column never lit up.
///
/// Post-2026-05-10 the caller (`ParsedFile::parse`) converts a `Some(...)`
/// return into a `ServerError::Validation`, which propagates up as a
/// per-file failure with the message visible in the dashboard. The
/// operator can then fix the `@Languages` header and re-run.
///
/// `@Options: CA` files still pass through under the default
/// [`CaMorphotagPolicy::Honor`](crate::options::CaMorphotagPolicy::Honor);
/// that is a legitimate "morphotag not applicable to this transcript
/// convention" case, not a typo to surface. An explicit `Analyze` policy
/// subjects the file to the same language gate as any other input.
pub(crate) fn unsupported_primary_language_error(chat_file: &ChatFile) -> Option<String> {
    if let Some(primary) = chat_file.languages.first()
        && !crate::chat_ops::morphosyntax_ops::is_stanza_supported(primary)
    {
        return Some(format!(
            "morphotag: primary @Languages '{}' is not supported by Stanza. \
             Fix the @Languages header to use a supported ISO-639-3 code and re-run. \
             Supported codes: {}.",
            primary,
            batchalign_transform::morphosyntax::supported_iso3_codes().join(", ")
        ));
    }
    None
}
