use super::*;
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::media::probe::MediaProbe;
use crate::media::tools::MediaTool;
use crate::time::FileMs;

const SAMPLE_RATE: SampleRateHzV2 = SampleRateHzV2(16_000);

/// Guards: Audio and text materialization publish correct descriptors and bytes
/// in their runtime-owned directories. Dropping the runtime removes both files.
/// Kind: contract
/// Not a type: Ownership fixes the lifetime of the temporary directory. The
/// actual filesystem bytes, selected paths and final removal are observable I/O.
/// State: The test owns one artifact runtime, with one actual PCM and one JSON write.
/// Concurrency: No shared roots or handles. Both artifacts outlive their writes
/// and remain usable until the runtime is dropped.
#[test]
fn a_scoped_store_writes_audio_and_text_and_removes_them_on_drop() {
    let runtime = PreparedArtifactRuntimeV2::new("artifacts").expect("artifact runtime");
    let store = runtime.store();

    let audio_id = WorkerArtifactIdV2::from("audio-ref");
    let samples = [0.25_f32, -0.5, 1.0, 0.0];
    let audio = store
        .write_prepared_audio_f32le(&audio_id, &samples, SAMPLE_RATE, ChannelCountV2(2))
        .expect("write prepared PCM");
    let audio_path = PathBuf::from(audio.path.as_ref());
    let expected_pcm: Vec<_> = samples.into_iter().flat_map(f32::to_le_bytes).collect();
    assert_eq!(audio.id, audio_id);
    assert_eq!(audio.encoding, PreparedAudioEncodingV2::PcmF32le);
    assert_eq!(audio.sample_rate_hz, SAMPLE_RATE);
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.frame_count, 2);
    assert_eq!(audio.byte_offset, 0);
    assert_eq!(audio.byte_len, expected_pcm.len() as u64);
    assert_eq!(
        audio_path.parent(),
        Some(store.root().join(AUDIO_DIR_NAME).as_path())
    );
    assert_eq!(fs::read(&audio_path).expect("read PCM"), expected_pcm);

    let text_id = WorkerArtifactIdV2::from("text-ref");
    let payload = serde_json::json!({"words": ["hello", "world"]});
    let text = store
        .write_prepared_text_json(&text_id, &payload)
        .expect("write JSON");
    let text_path = PathBuf::from(text.path.as_ref());
    let bytes = fs::read(&text_path).expect("read JSON");
    assert_eq!(text.id, text_id);
    assert_eq!(text.encoding, PreparedTextEncodingV2::Utf8Json);
    assert_eq!(text.byte_offset, 0);
    assert_eq!(text.byte_len, bytes.len() as u64);
    assert_eq!(
        text_path.parent(),
        Some(store.root().join(TEXT_DIR_NAME).as_path())
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).expect("decode JSON"),
        payload
    );

    drop(runtime);
    assert!(!audio_path.exists());
    assert!(!text_path.exists());
}

/// Guards: An interleaved sample population that cannot form complete frames
/// is refused before an artifact is created.
/// Kind: boundary
/// Not a type: The writer accepts raw slice lengths and channel counts. Their
/// divisibility is measured on admission and cannot be inferred from a slice type.
/// State: A private artifact directory observes that refusal publishes no file.
/// Concurrency: The test exclusively owns the scoped store.
#[test]
fn rejects_audio_when_sample_count_does_not_match_channels() {
    let runtime = PreparedArtifactRuntimeV2::new("artifacts").expect("artifact runtime");
    let store = runtime.store();
    let id = WorkerArtifactIdV2::from("misaligned");
    let error = store
        .write_prepared_audio_f32le(&id, &[0.1, 0.2, 0.3], SAMPLE_RATE, ChannelCountV2(2))
        .expect_err("reject an incomplete frame");

    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!store.audio_path_for(&id).exists());
}

/// The one recording observed from the actual fixture path.
struct MeasuredWav {
    path: PathBuf,
    timing: Recording,
}

struct WavSelection<'a> {
    source: &'a MeasuredWav,
    window: FaWindow,
}

impl MeasuredWav {
    async fn generate(path: PathBuf, duration: Ms) -> Self {
        write_tone(&path, duration).await;
        let measured = MediaProbe::new(&path)
            .duration()
            .await
            .expect("probe the fixture WAV");
        let timing =
            Recording::of_duration(Ms(measured.length().0)).expect("positive WAV duration");
        Self { path, timing }
    }

    fn select(&self, start: FileMs, end: FileMs) -> WavSelection<'_> {
        WavSelection {
            source: self,
            window: FaWindow::within(&self.timing, start, end)
                .expect("measured WAV contains window"),
        }
    }
}

impl WavSelection<'_> {
    async fn extract(
        &self,
        store: &PreparedArtifactStoreV2,
        id: &WorkerArtifactIdV2,
    ) -> Result<PreparedAudioRefV2, PreparedArtifactErrorV2> {
        store
            .extract_prepared_audio_segment_f32le(id, &self.source.path, self.window)
            .await
    }
}

async fn write_tone(path: &Path, duration: Ms) {
    let seconds = format!("{:.3}", duration.0 as f64 / 1_000.0);
    let output = MediaTool::Ffmpeg
        .async_command()
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            &seconds,
        ])
        .arg(path)
        .output()
        .await
        .expect("run ffmpeg");
    assert!(
        output.status.success(),
        "ffmpeg should generate the fixture WAV: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Guards: A measured WAV selection decodes the requested frames. If the
/// file is replaced with shorter audio after measurement, a successful decode
/// with zero frames preserves the measured refusal and publishes no descriptor.
/// Kind: contract
/// Not a type: Selection fixes which measured file and window go together.
/// The bytes at that path can change independently after admission.
/// State: One admitted source, an external replacement at its original path,
/// and the real ffprobe/ffmpeg adapters exercise both outcomes.
/// Concurrency: One temporary directory owns both snapshots. Replacement
/// completes between the first and second decode without concurrent edits.
#[tokio::test]
async fn decodes_a_measured_window_and_refuses_empty_bytes_after_source_replacement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = PreparedArtifactStoreV2::new(dir.path().join("artifacts")).expect("artifact store");
    let wav = MeasuredWav::generate(dir.path().join("tone.wav"), Ms(500)).await;
    let selection = wav.select(FileMs::new(200), FileMs::new(300));
    let ok_id = WorkerArtifactIdV2::from("selected-audio");
    let descriptor = selection
        .extract(&store, &ok_id)
        .await
        .expect("extract the original window");

    assert_eq!(descriptor.channels, 1);
    assert_eq!(descriptor.sample_rate_hz, SAMPLE_RATE);
    assert_eq!(descriptor.frame_count, 1_600);
    assert_eq!(descriptor.byte_len, 6_400);
    assert_eq!(
        fs::metadata(Path::new(descriptor.path.as_ref()))
            .expect("actual PCM file")
            .len(),
        6_400
    );

    write_tone(&wav.path, Ms(100)).await;
    let empty_id = WorkerArtifactIdV2::from("empty-after-replacement");
    let result = selection.extract(&store, &empty_id).await;
    let Err(PreparedArtifactErrorV2::EmptyAudioSegment(segment)) = result else {
        panic!("expected a decoder reporting zero frames, got {result:?}");
    };
    assert_eq!(segment.window, selection.window);
    let DecodedFrames::None(reason) = DecodedFrames::measure_f32le(0) else {
        panic!("zero bytes cannot contain an entire sample");
    };
    assert_eq!(segment.reason, reason);
    assert!(!store.audio_path_for(&empty_id).exists());
}

/// Guards: The error describes a partial sample without asserting why the
/// decoder wrote it.
/// Kind: contract
/// Not a type: The formatter can add an unsupported cause to valid measurement
/// evidence. The test checks its actual diagnostic text.
/// State: A pure measurement of three bytes supplies the real empty reason.
/// Concurrency: No file is created or decoded.
#[test]
fn an_empty_segment_reports_what_was_measured_not_a_guessed_cause() {
    let recording = Recording::of_duration(Ms(1_000)).expect("nonempty recording");
    let window = FaWindow::within(&recording, FileMs::new(0), FileMs::new(100)).expect("window");
    let DecodedFrames::None(reason) = DecodedFrames::measure_f32le(3) else {
        panic!("three bytes cannot contain an entire sample");
    };
    let rendered = PreparedArtifactErrorV2::EmptyAudioSegment(EmptySegment {
        path: "clip.wav".to_owned(),
        window,
        reason,
    })
    .to_string();

    assert!(
        rendered.contains("less than one 4-byte sample"),
        "{rendered}"
    );
    assert!(!rendered.contains("past the end"), "{rendered}");
}
