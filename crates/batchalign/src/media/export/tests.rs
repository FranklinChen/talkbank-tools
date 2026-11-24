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

/// Guards: One verified export publishes both role files with identical bytes and the observed duration.
/// Kind: journey
/// Not a type: The filesystem can publish different bytes or expose them before verification despite admitted destinations.
/// State: A generated stereo WAV, two absent targets and an independent PCM comparison.
/// Concurrency: Exclusive scratch paths witness before and after the consuming publish.
#[tokio::test]
async fn staged_export_is_verified_once_and_published_to_both_roles() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    let staged = dir.path().join("staged.wav");
    tone(&source, 44_100, 2);
    let verified = PrimaryAudioExport::admit(&source, &primary, AudioExportFormat::Wav)
        .unwrap()
        .with_staged_copy(&staged)
        .unwrap()
        .encode()
        .await
        .unwrap();
    assert_eq!(verified.duration().length().0, 250);
    assert!(!primary.exists());
    assert!(!staged.exists());
    verified.publish().unwrap();
    assert_eq!(
        std::fs::read(&primary).unwrap(),
        std::fs::read(&staged).unwrap()
    );
    assert_eq!(decoded_pcm(&source), decoded_pcm(&primary));
}

/// Guards: A competing writer at the staged path is preserved before publishing the user-visible output.
/// Kind: contract
/// Not a type: A file can appear at the admitted path after encoding and before atomic publication.
/// State: Real encoded audio and a controlled late writer at the staged name.
/// Concurrency: The staged conflict is injected between two owned transitions.
#[tokio::test]
async fn staged_publication_conflict_refuses_before_user_output_is_published() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    let staged = dir.path().join("staged.wav");
    tone(&source, 44_100, 2);
    let verified = PrimaryAudioExport::admit(&source, &primary, AudioExportFormat::Wav)
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

/// Guards: A staged role cannot target the source or the admitted primary path.
/// Kind: contract
/// Not a type: Distinct destination roles can still name the same actual filesystem entry.
/// State: A real source and canonical destination names test path equality and absence admission.
/// Concurrency: The test owns the filesystem entries and never encodes.
#[test]
fn staged_role_cannot_alias_source_primary_or_an_existing_recording() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.wav");
    let primary = dir.path().join("new.wav");
    std::fs::write(&source, b"source").unwrap();
    for refused in [&source, &primary] {
        assert!(
            PrimaryAudioExport::admit(&source, &primary, AudioExportFormat::Wav)
                .unwrap()
                .with_staged_copy(refused)
                .is_err()
        );
    }
    assert!(!primary.exists());
    assert_eq!(std::fs::read(source).unwrap(), b"source");
}

/// Guards: Published WAV preserves sample rate, stereo geometry and decoded PCM.
/// Kind: journey
/// Not a type: A successful external encoder may still change the written or decoded audio geometry.
/// State: A generated stereo source and real encode, publication, metadata and PCM decode.
/// Concurrency: The private destination is observed after publication.
#[tokio::test]
async fn wav_export_preserves_stereo_rate_and_decoded_pcm() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("résumé.source.wav");
    let destination = dir.path().join("résumé.export.wav");
    tone(&source, 44_100, 2);
    let original = std::fs::read(&source).expect("source bytes");
    let artifact = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav)
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
    assert_eq!(stream.layout().sample_rate.get(), 44_100);
    assert_eq!(stream.layout().channels.get(), 2);
}

/// Guards: Published MP3 is decodable stereo with the requested codec and rate.
/// Kind: journey
/// Not a type: An exit status and container suffix cannot prove the published codec or channel count.
/// State: Generated stereo audio, the actual encoder and inspection and decoded bytes at the destination.
/// Concurrency: The test owns all intermediate and final paths.
#[tokio::test]
async fn mp3_export_is_real_decodable_audio_not_model_mono() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    let destination = dir.path().join("export.mp3");
    tone(&source, 48_000, 2);
    let artifact = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Mp3)
        .expect("admit")
        .encode()
        .await
        .expect("encode");
    assert!(artifact.duration().length().0 >= 250);
    artifact.publish().expect("publish");
    let stream = inspect_stream(&destination).await.expect("inspect");
    assert_eq!(stream.codec(), "mp3");
    assert_eq!(stream.layout().sample_rate.get(), 48_000);
    assert_eq!(stream.layout().channels.get(), 2);
    let pcm = decoded_pcm(&destination);
    assert!(!pcm.is_empty());
    assert!(pcm.iter().any(|byte| *byte != 0));
}

/// Guards: Existing primary destinations and source or dangling symlink aliases refuse without modifying retained bytes.
/// Kind: contract
/// Not a type: A path string can resolve to an existing or dangling symlink on the current filesystem.
/// State: Private source, existing destination and native symlink controls when supported.
/// Concurrency: All named entries are private and remain undisrupted after refusal.
#[test]
fn same_source_existing_destination_and_symlink_alias_refuse_without_mutation() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    std::fs::write(&source, b"retained source").expect("source");
    assert!(matches!(
        PrimaryAudioExport::admit(&source, &source, AudioExportFormat::Wav),
        Err(AudioExportError::ExistingDestination(_))
    ));
    let destination = dir.path().join("existing.wav");
    std::fs::write(&destination, b"retained output").expect("existing output");
    assert!(matches!(
        PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav),
        Err(AudioExportError::ExistingDestination(_))
    ));
    #[cfg(unix)]
    {
        let alias = dir.path().join("alias.wav");
        std::os::unix::fs::symlink(&source, &alias).expect("source alias");
        assert!(matches!(
            PrimaryAudioExport::admit(&source, &alias, AudioExportFormat::Wav),
            Err(AudioExportError::ExistingDestination(_))
        ));
        let dangling = dir.path().join("dangling.wav");
        std::os::unix::fs::symlink(dir.path().join("absent.wav"), &dangling)
            .expect("dangling alias");
        assert!(matches!(
            PrimaryAudioExport::admit(&source, &dangling, AudioExportFormat::Wav),
            Err(AudioExportError::ExistingDestination(_))
        ));
    }
    assert_eq!(std::fs::read(&source).expect("source"), b"retained source");
    assert_eq!(
        std::fs::read(&destination).expect("output"),
        b"retained output"
    );
}

/// Guards: Late primary destination collision preserves the competing bytes and discards private output.
/// Kind: contract
/// Not a type: The kernel can report a new competing path after admission and before publication.
/// State: Actual encode and a file written at the final path just before publish.
/// Concurrency: No timing race is needed to exercise the no-clobber transition.
#[tokio::test]
async fn destination_appearing_after_encoding_cannot_be_replaced() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("source.wav");
    let destination = dir.path().join("raced.wav");
    tone(&source, 16_000, 1);
    let artifact = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav)
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

/// Guards: An unreadable input reaches neither publication nor a leftover temporary output.
/// Kind: contract
/// Not a type: A regular file can hold corrupt bytes that the native tool refuses.
/// State: A deliberately non-audio file with a WAV suffix and private destination directory.
/// Concurrency: No other writer can account for any leftover files.
#[tokio::test]
async fn corrupt_audio_cannot_publish_or_leave_a_temporary_output() {
    let dir = tempfile::tempdir().expect("scratch");
    let source = dir.path().join("broken.wav");
    let destination = dir.path().join("output.wav");
    std::fs::write(&source, b"not audio").expect("corrupt control");
    let plan = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("filesystem admission is not audio proof");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Refused { .. })
    ));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(dir.path()).expect("directory").count(), 1);
}

/// Guards: MP3 refuses incompatible sample geometry while WAV preserves it through PCM.
/// Kind: contract
/// Not a type: Real encoder support and the actual decoded samples are outside the type system.
/// State: Generated three-channel and high-rate recordings, with real WAV and MP3 paths.
/// Concurrency: Each incompatibility uses unique source and destination names.
#[tokio::test]
async fn mp3_refuses_implicit_rate_and_channel_changes_but_wav_retains_them() {
    let dir = tempfile::tempdir().expect("scratch");
    for (rate, channels) in [(44_100, 3), (96_000, 2)] {
        let source = dir.path().join(format!("source-{rate}-{channels}.wav"));
        tone(&source, rate, channels);
        let destination = source.with_extension("mp3");
        let plan = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Mp3)
            .expect("filesystem admission");
        assert!(matches!(plan.encode().await,
            Err(AudioExportError::Mp3Layout { channels: c, sample_rate: r })
            if c == channels && r == rate));
        assert!(!destination.exists());
        let wav = dir.path().join(format!("export-{rate}-{channels}.wav"));
        PrimaryAudioExport::admit(&source, &wav, AudioExportFormat::Wav)
            .expect("wav admission")
            .encode()
            .await
            .expect("wav retains layout")
            .publish()
            .expect("wav publication");
        assert_eq!(decoded_pcm(&source), decoded_pcm(&wav));
    }
}

/// Guards: Every supported MP3 sample rate survives real encoding, publication and decode.
/// Kind: measurement
/// Not a type: The local encoder may silently resample even if the source carries a supported rate.
/// State: One generated source per declared rate and measured published MP3.
/// Concurrency: All fixtures use private filenames and complete in sequence.
#[tokio::test]
async fn mp3_supported_rate_matrix_retains_the_requested_rate() {
    let dir = tempfile::tempdir().expect("scratch");
    for rate in [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    ] {
        let source = dir.path().join(format!("rate-{rate}.wav"));
        let destination = source.with_extension("mp3");
        tone(&source, rate, 1);
        PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Mp3)
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
                .layout()
                .sample_rate
                .get(),
            rate
        );
        assert!(!decoded_pcm(&destination).is_empty());
    }
}

/// Guards: A native decoder failure after stream admission discards the private destination.
/// Kind: contract
/// Not a type: Readable header bytes can precede truncated PCM that fails only during decoding.
/// State: A generated WAV truncated inside its payload and an absent output path.
/// Concurrency: One deterministic edit happens before the only encode attempt.
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
    let plan = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("path admission");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Refused { tool: "ffmpeg", .. })
    ));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(dir.path()).expect("directory").count(), 1);
}

/// Guards: An export with multiple audio tracks refuses to select one implicitly.
/// Kind: contract
/// Not a type: One container path may name multiple independently recorded audio streams.
/// State: A generated two-track file is observed by the real ffprobe selector.
/// Concurrency: All files are private and refusal precedes publication.
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
    let plan = PrimaryAudioExport::admit(&source, &destination, AudioExportFormat::Wav)
        .expect("path admission");
    assert!(matches!(
        plan.encode().await,
        Err(AudioExportError::Stream(_))
    ));
    assert!(!destination.exists());
}
