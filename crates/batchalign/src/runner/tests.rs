use super::command_requires_chat_infer;
use crate::api::ReleasedCommand;

#[test]
fn chat_backed_commands_require_chat_infer() {
    for cmd in [
        ReleasedCommand::Morphotag,
        ReleasedCommand::Utseg,
        ReleasedCommand::Translate,
        ReleasedCommand::Coref,
    ] {
        assert!(command_requires_chat_infer(cmd));
    }
}

#[test]
fn align_requires_chat_infer() {
    assert!(command_requires_chat_infer(ReleasedCommand::Align));
}

#[test]
fn audio_or_composed_commands_do_not_require_chat_infer() {
    assert!(!command_requires_chat_infer(ReleasedCommand::Opensmile));
    assert!(!command_requires_chat_infer(ReleasedCommand::Avqi));
    assert!(!command_requires_chat_infer(ReleasedCommand::Transcribe));
    assert!(!command_requires_chat_infer(ReleasedCommand::Benchmark));
}
