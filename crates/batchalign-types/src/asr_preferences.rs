//! Narrow edits to the shared credentials document. Unowned keys and sections
//! remain byte-for-byte intact; callers cannot render a replacement INI file.

use std::io::Write;
use std::path::Path;

/// Engines selectable by the small setup/desktop preference editor.
#[derive(Clone, Copy)]
pub enum PreferenceEngine {
    Rev,
    Whisper,
}

/// An admitted edit owns only an engine preference and an optional key update.
/// Omitting a key means keep the existing credential, including when switching
/// to a local engine. Credential deletion is not a setup operation.
pub struct PreferenceEdit {
    engine: PreferenceEngine,
    key: Option<String>,
}

impl PreferenceEdit {
    /// Admit a single-line credential before it can reach the INI renderer.
    ///
    /// # Errors
    /// Refuses blank keys and INI line/control injection.
    pub fn new(engine: PreferenceEngine, key: Option<String>) -> Result<Self, std::io::Error> {
        if key
            .as_ref()
            .is_some_and(|key| key.trim().is_empty() || key.chars().any(char::is_control))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "The API key must be a nonempty single line",
            ));
        }
        Ok(Self { engine, key })
    }

    /// Apply the edit to an observed document, preserving everything else.
    ///
    /// # Errors
    /// Refuses duplicate ASR sections/owned keys and malformed section headers.
    pub fn render(&self, previous: &str) -> Result<String, std::io::Error> {
        let invalid = || {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Ambiguous or malformed INI configuration; repair it before saving settings",
            )
        };
        let newline = if previous.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let mut output = String::new();
        let mut section = Section::Other;
        let mut asr_count = 0;
        let mut engine_count = 0;
        let mut key_count = 0;
        let engine = match self.engine {
            PreferenceEngine::Rev => "rev",
            PreferenceEngine::Whisper => "whisper",
        };
        let finish = |output: &mut String, engine_count, key_count| {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push_str(newline);
            }
            if engine_count == 0 {
                output.push_str(&format!("engine = {engine}{newline}"));
            }
            if key_count == 0
                && let Some(key) = &self.key
            {
                output.push_str(&format!("engine.rev.key = {key}{newline}"));
            }
        };
        for line in previous.split_inclusive('\n') {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                let name = trimmed
                    .strip_prefix('[')
                    .and_then(|section| section.strip_suffix(']'))
                    .ok_or_else(invalid)?;
                if matches!(section, Section::Asr) {
                    finish(&mut output, engine_count, key_count);
                }
                section = if name.trim().eq_ignore_ascii_case("asr") {
                    asr_count += 1;
                    if asr_count > 1 {
                        return Err(invalid());
                    }
                    Section::Asr
                } else {
                    Section::Other
                };
            } else if matches!(section, Section::Asr)
                && !trimmed.starts_with(['#', ';'])
                && let Some((key, _)) = trimmed.split_once('=')
            {
                match key.trim().to_ascii_lowercase().as_str() {
                    "engine" => {
                        engine_count += 1;
                        if engine_count > 1 {
                            return Err(invalid());
                        }
                        output.push_str(&format!("engine = {engine}{newline}"));
                        continue;
                    }
                    "engine.rev.key" => {
                        key_count += 1;
                        if key_count > 1 {
                            return Err(invalid());
                        }
                        if let Some(value) = &self.key {
                            output.push_str(&format!("engine.rev.key = {value}{newline}"));
                            continue;
                        }
                    }
                    _ => {}
                }
            }
            output.push_str(line);
        }
        if matches!(section, Section::Asr) {
            finish(&mut output, engine_count, key_count);
        }
        if asr_count == 0 {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push_str(newline);
            }
            output.push_str(&format!("[asr]{newline}"));
            finish(&mut output, 0, 0);
        }
        Ok(output)
    }

    /// Read, edit and atomically replace one configuration document. A failed
    /// read/parse/write leaves the old credentials in place.
    ///
    /// # Errors
    /// Propagates I/O and document admission errors, without secret values.
    pub fn write_to(&self, path: &Path) -> Result<(), std::io::Error> {
        let previous = match std::fs::read_to_string(path) {
            Ok(previous) => previous,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(error),
        };
        let rendered = self.render(&previous)?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut staged = tempfile::NamedTempFile::new_in(parent)?;
        staged.write_all(rendered.as_bytes())?;
        staged.as_file().sync_all()?;
        staged.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

enum Section {
    Asr,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_engine_switch_preserves_other_credentials_and_sections() -> std::io::Result<()> {
        let original = "# personal\r\n[asr]\r\nengine = rev\r\nengine.rev.key = retained\r\nengine.tencent.key = other\r\n[diarization]\r\nkey = speaker\r\n";
        let changed = PreferenceEdit::new(PreferenceEngine::Whisper, None)?.render(original)?;
        assert_eq!(
            changed,
            original.replace("engine = rev", "engine = whisper")
        );
        Ok(())
    }

    #[test]
    fn edits_are_inserted_in_the_asr_section_and_ambiguity_refuses() -> std::io::Result<()> {
        let edit = PreferenceEdit::new(PreferenceEngine::Rev, Some("new-key".into()))?;
        assert_eq!(
            edit.render("[asr]\nother = keep\n[other]\nkey = keep\n")?,
            "[asr]\nother = keep\nengine = rev\nengine.rev.key = new-key\n[other]\nkey = keep\n"
        );
        for invalid in [
            "[asr]\nengine=rev\nengine=whisper",
            "[asr]\n[ASR]\n",
            "[asr\n",
        ] {
            assert!(edit.render(invalid).is_err());
        }
        assert!(PreferenceEdit::new(PreferenceEngine::Rev, Some("key\n[other]".into())).is_err());
        Ok(())
    }

    #[test]
    fn atomic_edit_preserves_previous_bytes_when_document_admission_fails() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("settings.ini");
        let original = "[asr]\nengine=rev\nengine=whisper\n[other]\nkey=keep\n";
        std::fs::write(&path, original)?;
        assert!(
            PreferenceEdit::new(PreferenceEngine::Whisper, None)?
                .write_to(&path)
                .is_err()
        );
        assert_eq!(std::fs::read_to_string(&path)?, original);
        Ok(())
    }
}
