//! Audio decode + 16 kHz resample for the native Whisper path.
//! Cross-platform via symphonia 0.6 (codec) + rubato 4.0 (FFT resampler).
//!
//! Re-ported from the candle-0.10-era Whisper pilot crate (since retired) to
//! the current dependency line: symphonia 0.6 replaced the `AudioBufferRef`
//! enum with `GenericAudioBufferRef` (and removed the `conv`/`sample`/`probe`
//! top-level modules), and rubato 4.0's synchronous FFT resampler is now `Fft`
//! driven through `audioadapter` buffer adapters.

#![cfg(feature = "whisper-rs-backend")]

use std::path::Path;

use crate::whisper_native::error::WhisperNativeError;

/// Whisper's required input sample rate.
pub(super) const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Decode any symphonia-supported audio file to mono f32 PCM (channel 0) at
/// its native sample rate.
pub(super) fn pcm_decode(path: &Path) -> Result<(Vec<f32>, u32), WhisperNativeError> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let decode_err = |reason: String| WhisperNativeError::AudioDecode {
        path: path.to_path_buf(),
        reason,
    };

    let src = std::fs::File::open(path).map_err(|e| decode_err(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();
    let fmt_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .map_err(|e| decode_err(e.to_string()))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| decode_err("no decodable audio track found".to_string()))?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| decode_err("selected track has no audio codec parameters".to_string()))?;
    let sample_rate = audio_params.sample_rate.unwrap_or(0);

    let dec_opts: AudioDecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &dec_opts)
        .map_err(|e| decode_err(format!("unsupported codec: {e}")))?;

    let mut pcm: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    while let Some(packet) = format
        .next_packet()
        .map_err(|e| decode_err(e.to_string()))?
    {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let channels = decoded.num_planes().max(1);
                interleaved.clear();
                decoded.copy_to_vec_interleaved(&mut interleaved);
                // Keep channel 0 only (mono), matching the pilot behavior.
                pcm.extend(interleaved.iter().step_by(channels).copied());
            }
            // Skip a single undecodable packet rather than aborting the file.
            Err(SymError::DecodeError(_)) | Err(SymError::IoError(_)) => continue,
            Err(err) => return Err(decode_err(err.to_string())),
        }
    }

    Ok((pcm, sample_rate))
}

/// Resample mono PCM to a target sample rate using rubato 4.0's synchronous FFT
/// resampler, processing the whole clip in one `process_all_into_buffer` call.
pub(super) fn resample(
    pcm_in: &[f32],
    sr_in: u32,
    sr_out: u32,
) -> Result<Vec<f32>, WhisperNativeError> {
    use audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    let channels = 1usize;
    let input_frames = pcm_in.len();
    if input_frames == 0 {
        return Ok(Vec::new());
    }

    let mut resampler = Fft::<f32>::new(
        sr_in as usize,
        sr_out as usize,
        1024,
        channels,
        FixedSync::Both,
    )
    .map_err(|e| WhisperNativeError::AudioResamplerCtor(e.to_string()))?;

    let ratio = sr_out as f64 / sr_in as f64;
    let output_capacity = (input_frames as f64 * ratio) as usize + 1024;
    let mut output = vec![0f32; output_capacity * channels];

    let input_adapter = InterleavedSlice::new(pcm_in, channels, input_frames)
        .map_err(|e| WhisperNativeError::AudioResample(format!("input adapter: {e}")))?;
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, channels, output_capacity)
        .map_err(|e| WhisperNativeError::AudioResample(format!("output adapter: {e}")))?;

    let (_frames_in, frames_out) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, input_frames, None)
        .map_err(|e| WhisperNativeError::AudioResample(e.to_string()))?;

    output.truncate(frames_out * channels);
    Ok(output)
}
