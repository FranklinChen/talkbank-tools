//! Encoding and new-only publication at the actual ffmpeg/filesystem boundary.

use super::*;

fn tone(path: &Path, rate: u32, channels: u32) {
    let source = format!("sine=frequency=440:sample_rate={rate}:duration=0.25");
    let output = MediaTool::Ffmpeg
        .command()
        .args([
            "-nostdin", "-v", "error", "-f", "lavfi", "-i", &source, "-ac",
        ])
        .arg(channels.to_string())
        .args(["-c:a", "pcm_s16le"])
        .arg(path)
        .output()
        .expect("ffmpeg boundary");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn decoded_pcm(path: &Path) -> Vec<u8> {
    let output = MediaTool::Ffmpeg
        .command()
        .args(["-nostdin", "-v", "error", "-xerror", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-"])
        .output()
        .expect("independent PCM decode");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[tokio::test]
async fn staged_export_is_verified_once_and_published_to_both_roles() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    let staged = dir.path().join("staged.wav");
    tone(&source, 44_100, 2);
    let verified = AudioExportPlan::admit(&source, &primary, AudioExportFormat::Wav)
        .unwrap()
        .with_staged_copy(&staged)
        .unwrap()
        .encode()
        .await
        .unwrap();
    assert!(!primary.exists());
    assert!(!staged.exists());
    verified.publish().unwrap();
    assert_eq!(
        std::fs::read(&primary).unwrap(),
        std::fs::read(&staged).unwrap()
    );
    assert_eq!(decoded_pcm(&source), decoded_pcm(&primary));
}

#[tokio::test]
async fn staged_publication_conflict_refuses_before_user_output_is_published() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    let staged = dir.path().join("staged.wav");
    tone(&source, 44_100, 2);
    let verified = AudioExportPlan::admit(&source, &primary, AudioExportFormat::Wav)
        .unwrap()
        .with_staged_copy(&staged)
        .unwrap()
        .encode()
        .await
        .unwrap();
    // Deterministic interference after encoding, not a timing-dependent race.
    std::fs::write(&staged, b"other writer").unwrap();
    assert!(verified.publish().is_err());
    assert!(!primary.exists());
    assert_eq!(std::fs::read(staged).unwrap(), b"other writer");
}

#[test]
fn staged_role_cannot_alias_source_primary_or_an_existing_recording() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    std::fs::write(&source, b"source").unwrap();
    for refused in [&source, &primary] {
        assert!(
            AudioExportPlan::admit(&source, &primary, AudioExportFormat::Wav)
                .unwrap()
                .with_staged_copy(refused)
                .is_err()
        );
    }
    assert!(!primary.exists());
    assert_eq!(std::fs::read(source).unwrap(), b"source");
}

#[tokio::test]
async fn wav_export_preserves_stereo_rate_and_decoded_pcm() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("résumé.source.wav");
    let destination = dir.path().join("résumé.export.wav");
    tone(&source, 44_100, 2);
    let original = std::fs::read(&source).expect("source bytes");
    let artifact = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("new-only admission")
        .encode()
        .await
        .expect("verified encoding");
    assert!(!destination.exists(), "encoding cannot publish early");
    assert_eq!(artifact.duration().length().0, 250);
    assert_eq!(
        artifact.publish().expect("publication"),
        dir.path()
            .canonicalize()
            .expect("canonical scratch")
            .join("résumé.export.wav")
    );
    assert_eq!(std::fs::read(&source).expect("unchanged source"), original);
    assert_eq!(decoded_pcm(&source), decoded_pcm(&destination));
    let stream = inspect_stream(&destination)
        .await
        .expect("independent inspection");
    assert_eq!(stream.layout.sample_rate.get(), 44_100);
    assert_eq!(stream.layout.channels.get(), 2);
}

#[tokio::test]
async fn mp3_export_is_real_decodable_audio_not_model_mono() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    let destination = dir.path().join("export.mp3");
    tone(&source, 48_000, 2);
    let artifact = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Mp3)
        .expect("admit")
        .encode()
        .await
        .expect("encode");
    assert!(artifact.duration().length().0 >= 250);
    artifact.publish().expect("publish");
    let stream = inspect_stream(&destination).await.expect("inspect");
    assert_eq!(stream.codec, "mp3");
    assert_eq!(stream.layout.sample_rate.get(), 48_000);
    assert_eq!(stream.layout.channels.get(), 2);
    let pcm = decoded_pcm(&destination);
    assert!(!pcm.is_empty());
    assert!(pcm.iter().any(|byte| *byte != 0));
}

#[test]
fn same_source_existing_destination_and_symlink_alias_refuse_without_mutation() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    std::fs::write(&source, b"retained source").expect("source");
    assert!(matches!(
        AudioExportPlan::admit(&source, &source, AudioExportFormat::Wav),
        Err(AudioExportError::ExistingDestination(_))
    ));
    let destination = dir.path().join("existing.wav");
    std::fs::write(&destination, b"retained output").expect("existing output");
    assert!(matches!(
        AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav),
        Err(AudioExportError::ExistingDestination(_))
    ));
    #[cfg(unix)]
    {
        let alias = dir.path().join("alias.wav");
        std::os::unix::fs::symlink(&source, &alias).expect("source alias");
        assert!(matches!(
            AudioExportPlan::admit(&source, &alias, AudioExportFormat::Wav),
            Err(AudioExportError::ExistingDestination(_))
        ));
        let dangling = dir.path().join("dangling.wav");
        std::os::unix::fs::symlink(dir.path().join("absent.wav"), &dangling)
            .expect("dangling alias");
        assert!(matches!(
            AudioExportPlan::admit(&source, &dangling, AudioExportFormat::Wav),
            Err(AudioExportError::ExistingDestination(_))
        ));
    }
    assert_eq!(std::fs::read(&source).expect("source"), b"retained source");
    assert_eq!(
        std::fs::read(&destination).expect("output"),
        b"retained output"
    );
}

#[tokio::test]
async fn destination_appearing_after_encoding_cannot_be_replaced() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    let destination = dir.path().join("raced.wav");
    tone(&source, 16_000, 1);
    let artifact = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("admit")
        .encode()
        .await
        .expect("encode");
    std::fs::write(&destination, b"another writer's output").expect("intervening writer");
    assert!(
        matches!(artifact.publish(), Err(AudioExportError::Io(error))
        if error.kind() == std::io::ErrorKind::AlreadyExists)
    );
    assert_eq!(
        std::fs::read(&destination).expect("existing output"),
        b"another writer's output"
    );
    assert_eq!(std::fs::read_dir(dir.path()).expect("directory").count(), 2);
}

#[tokio::test]
async fn corrupt_audio_cannot_publish_or_leave_a_temporary_output() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("broken.wav");
    let destination = dir.path().join("output.wav");
    std::fs::write(&source, b"not audio").expect("corrupt control");
    let plan = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("filesystem admission is not audio proof");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Refused { .. })
    ));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(dir.path()).expect("directory").count(), 1);
}

#[tokio::test]
async fn mp3_refuses_implicit_rate_and_channel_changes_but_wav_retains_them() {
    let dir = tempfile::tempdir().expect("scratch");
    for (rate, channels) in [(44_100, 3), (96_000, 2)] {
        let source = dir.path().join(format!("source-{rate}-{channels}.wav"));
        tone(&source, rate, channels);
        let destination = source.with_extension("mp3");
        let plan = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Mp3)
            .expect("filesystem admission");
        assert!(matches!(plan.encode().await,
            Err(AudioExportError::Mp3Layout { channels: c, sample_rate: r })
            if c == channels && r == rate));
        assert!(!destination.exists());
        let wav = dir.path().join(format!("export-{rate}-{channels}.wav"));
        AudioExportPlan::admit(&source, &wav, AudioExportFormat::Wav)
            .expect("wav admission")
            .encode()
            .await
            .expect("wav retains layout")
            .publish()
            .expect("wav publication");
        assert_eq!(decoded_pcm(&source), decoded_pcm(&wav));
    }
}

#[tokio::test]
async fn mp3_supported_rate_matrix_retains_the_requested_rate() {
    let dir = tempfile::tempdir().expect("scratch");
    for rate in [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    ] {
        let source = dir.path().join(format!("rate-{rate}.wav"));
        let destination = source.with_extension("mp3");
        tone(&source, rate, 1);
        AudioExportPlan::admit(&source, &destination, AudioExportFormat::Mp3)
            .expect("admit")
            .encode()
            .await
            .expect("supported encoding")
            .publish()
            .expect("publish");
        assert_eq!(
            inspect_stream(&destination)
                .await
                .expect("rate inspection")
                .layout
                .sample_rate
                .get(),
            rate
        );
        assert!(!decoded_pcm(&destination).is_empty());
    }
}

#[tokio::test]
async fn decode_failure_after_stream_admission_discards_private_output() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("truncated.wav");
    let destination = dir.path().join("export.wav");
    tone(&source, 16_000, 1);
    // Preserve an admitted WAV header, but cut its PCM payload through a sample.
    let original_length = source.metadata().expect("metadata").len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&source)
        .expect("controlled mutation")
        .set_len(original_length - 101)
        .expect("truncated PCM");
    inspect_stream(&source)
        .await
        .expect("stream header still admitted");
    let plan = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("path admission");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Refused { tool: "ffmpeg", .. })
    ));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(dir.path()).expect("directory").count(), 1);
}

#[tokio::test]
async fn multiple_audio_streams_require_selection_instead_of_guessing() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("streams.mkv");
    let destination = dir.path().join("export.wav");
    let output = MediaTool::Ffmpeg
        .command()
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.25",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
        ])
        .arg(&source)
        .output()
        .expect("multistream control");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan = AudioExportPlan::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("path admission");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Stream(_))
    ));
    assert!(!destination.exists());
}
