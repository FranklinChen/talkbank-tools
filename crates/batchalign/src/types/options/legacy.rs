//! How the job store reads a stored [`CommandOptions`] document written by an
//! earlier version.
//!
//! # The retired `merge_abbrev` key
//!
//! Until the abbreviation merge moved onto the recognizer's tokens, every
//! CHAT-writing command (align, morphotag, utseg, translate, coref, benchmark,
//! compare) accepted `merge_abbrev`, and every job of those commands was
//! stored with the key, almost always as `false`. Transcribe keeps the option;
//! no other command has it now. The server's job store persists options as
//! JSON and reloads every job at startup, so documents carrying the key for a
//! command that no longer has it will be read for as long as such jobs are
//! kept.
//!
//! The rule, applied where the store loads a job (it knows the job's id and
//! status, which a deserializer does not):
//!
//! - **Transcribe** (`transcribe`, `transcribe_s`): the key is the option and
//!   keeps its meaning.
//! - **Any other command, `false` or absent**: dropped silently. It asked for
//!   what is now the only behavior, so nothing about the job changes.
//! - **Any other command, any other value**: dropped, with a warning naming
//!   the job, its status and its command. Loading never fails: one refused
//!   document would stop the server from reloading its whole job table. A job
//!   that runs again runs without the merge, which for align never produced a
//!   file anyway (the merge after alignment changed words the alignment had
//!   admitted, and the write was refused).
//!
//! Dropping is serde's own behavior (the option structs do not deny unknown
//! fields); this module is what makes it stated and logged. A job submission
//! from a client built before the change is read the same way any unknown key
//! is, without a warning; the current command line cannot send the key for
//! these commands at all.

use super::CommandOptions;

/// The key earlier versions accepted on every CHAT-writing command.
const MERGE_ABBREV_KEY: &str = "merge_abbrev";

/// Whether a command has the `merge_abbrev` option. Exhaustive, so a new
/// command must say.
const fn has_merge_abbrev(options: &CommandOptions) -> bool {
    match options {
        CommandOptions::Transcribe(_) | CommandOptions::TranscribeS(_) => true,
        CommandOptions::Align(_)
        | CommandOptions::Translate(_)
        | CommandOptions::Morphotag(_)
        | CommandOptions::Coref(_)
        | CommandOptions::Utseg(_)
        | CommandOptions::Benchmark(_)
        | CommandOptions::Opensmile(_)
        | CommandOptions::Compare(_)
        | CommandOptions::Avqi(_)
        | CommandOptions::Diarize(_)
        | CommandOptions::SpeakerIdentify(_)
        | CommandOptions::Convert(_) => false,
    }
}

/// The retired key's value in a stored document, when dropping it changes
/// what the job asked for: anything but `false` or absence. `None` for a
/// document that is not a JSON object, which the typed read has refused.
fn meaningful_retired_value(document: &str) -> Option<serde_json::Value> {
    let serde_json::Value::Object(mut fields) = serde_json::from_str(document).ok()? else {
        return None;
    };
    match fields.remove(MERGE_ABBREV_KEY)? {
        serde_json::Value::Bool(false) => None,
        value => Some(value),
    }
}

/// Apply the retired-key rule to one stored job whose options were just read
/// as `options` from `document`: warn when the document asked a command that
/// no longer has `merge_abbrev` for something dropping the key changes.
pub(crate) fn report_retired_merge_abbrev(
    job_id: &str,
    status: &str,
    document: &str,
    options: &CommandOptions,
) {
    if has_merge_abbrev(options) {
        return;
    }
    if let Some(value) = meaningful_retired_value(document) {
        tracing::warn!(
            job_id,
            status,
            command = options.command_name(),
            %value,
            "stored options carry `merge_abbrev` from an earlier version; only transcribe \
             merges abbreviations now, so the key is ignored and this job does not merge"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::MergeAbbrevPolicy;

    fn read(json: &str) -> CommandOptions {
        serde_json::from_str(json).expect("legacy options load")
    }

    /// A stored align job from before the change carried the key; it reads
    /// as align and writes back without it.
    #[test]
    fn a_stored_align_job_asking_for_the_merge_loads_without_it() {
        let options = read(r#"{"command":"align","merge_abbrev":true}"#);
        assert!(matches!(options, CommandOptions::Align(_)));
        let written = serde_json::to_value(&options).expect("serialize");
        assert!(written.get(MERGE_ABBREV_KEY).is_none(), "{written}");
    }

    /// The ordinary legacy document: every CHAT-writing command's job was
    /// stored with `merge_abbrev: false`, and even an unreadable value no
    /// longer refuses the document for a command without the option.
    #[test]
    fn every_retired_command_reads_a_stored_value() {
        for command in [
            "align",
            "morphotag",
            "utseg",
            "translate",
            "coref",
            "benchmark",
            "compare",
        ] {
            for value in ["false", "true", r#""yes""#] {
                let options = read(&format!(
                    r#"{{"command":"{command}","merge_abbrev":{value}}}"#
                ));
                assert_eq!(options.command_name(), command);
                assert!(!has_merge_abbrev(&options));
            }
        }
    }

    #[test]
    fn transcribe_keeps_the_option_and_its_meaning() {
        for command in ["transcribe", "transcribe_s"] {
            let options = read(&format!(r#"{{"command":"{command}","merge_abbrev":true}}"#));
            assert!(has_merge_abbrev(&options));
            let (CommandOptions::Transcribe(transcribe) | CommandOptions::TranscribeS(transcribe)) =
                options
            else {
                panic!("{command} read as another command");
            };
            assert_eq!(transcribe.merge_abbrev, MergeAbbrevPolicy::Merge);
        }
        let CommandOptions::Transcribe(absent) = read(r#"{"command":"transcribe"}"#) else {
            panic!("transcribe read as another command");
        };
        assert_eq!(absent.merge_abbrev, MergeAbbrevPolicy::Keep);
    }

    /// For transcribe the key is still the option, so a value it cannot read
    /// is refused as before rather than dropped.
    #[test]
    fn transcribe_still_refuses_an_unreadable_value() {
        assert!(
            serde_json::from_str::<CommandOptions>(
                r#"{"command":"transcribe","merge_abbrev":"yes"}"#
            )
            .is_err()
        );
    }

    /// Only a value whose loss changes the request is reported.
    #[test]
    fn only_a_value_other_than_false_is_worth_reporting() {
        assert_eq!(meaningful_retired_value(r#"{"command":"align"}"#), None);
        assert_eq!(
            meaningful_retired_value(r#"{"command":"align","merge_abbrev":false}"#),
            None
        );
        assert_eq!(
            meaningful_retired_value(r#"{"command":"align","merge_abbrev":true}"#),
            Some(serde_json::Value::Bool(true))
        );
        assert_eq!(
            meaningful_retired_value(r#"{"command":"align","merge_abbrev":1}"#),
            Some(serde_json::json!(1))
        );
    }
}
