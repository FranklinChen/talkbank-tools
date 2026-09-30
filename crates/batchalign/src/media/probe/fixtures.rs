//! Media whose container duration is known to be wrong, for tests at the
//! ffprobe boundary, here and in the consumers that bound windows by it.

use std::path::Path;

/// Frames in each unpadded fixture.
pub(crate) const FRAMES: u64 = 11_485;

/// The MPEG-1 fixture's true length in whole milliseconds, rounded UP: 11,485
/// frames of 1152 samples at 44,100 Hz is 300,016.33 ms.
pub(crate) const TRUE_LENGTH_MS: u64 = 300_017;

/// ffprobe's bitrate ESTIMATE for the MPEG-1 fixture, in milliseconds: the
/// number the probe returned before 2026-09-30 (file bytes * 8 / 128,000).
pub(crate) const ESTIMATED_LENGTH_MS: f64 = 299_327.8;

/// Write a constant-bitrate MPEG-1 Layer III stream shaped like the recordings
/// that exposed the defect: 128 kbit/s at 44.1 kHz, no Xing or Info header, no
/// ID3 tag, and every frame UNPADDED (417 bytes).
///
/// At 44.1 kHz a 128 kbit/s frame holds 417.96 bytes on average, so an
/// encoder that honours the bitrate pads about 96% of its frames to 418. These
/// frames never are, so they run at 127.73 kbit/s, and a duration estimated
/// from file size over the DECLARED bitrate falls short by 0.23%: the shortfall
/// measured on the recordings that motivated this.
///
/// The frames are written directly because no encoder ffmpeg ships emits
/// unpadded 44.1 kHz CBR (LAME pads to hold the bitrate). Each is a valid
/// silent frame: a mono header followed by all-zero side information, so
/// `main_data_begin` is 0 and every granule codes no spectral values. ffmpeg
/// decodes each to 1152 samples of silence without a diagnostic, which the
/// tests using this assert rather than assume.
pub(crate) fn write_unpadded_cbr_mp3(path: &Path) {
    // Sync, MPEG-1, Layer III, no CRC | 128 kbit/s, 44.1 kHz, unpadded | mono.
    write_frames(path, [0xFF, 0xFB, 0x90, 0xC0], 417);
}

/// The MPEG-2 sibling: 64 kbit/s at 22.05 kHz, 576 samples per frame, every
/// frame unpadded (208 bytes where the bitrate implies 208.98). The same
/// shortfall on the half-size frame, so a walk that assumed 1152 samples per
/// frame would be off by a factor of two here.
pub(crate) fn write_unpadded_cbr_mpeg2_mp3(path: &Path) {
    // Sync, MPEG-2, Layer III, no CRC | 64 kbit/s, 22.05 kHz, unpadded | mono.
    write_frames(path, [0xFF, 0xF3, 0x80, 0xC0], 208);
}

/// [`FRAMES`] copies of one silent frame: `header`, then zeros to `frame_bytes`.
fn write_frames(path: &Path, header: [u8; 4], frame_bytes: usize) {
    let mut frame = vec![0u8; frame_bytes];
    frame[..header.len()].copy_from_slice(&header);
    let frames = usize::try_from(FRAMES).expect("frame count fits usize");
    std::fs::write(path, frame.repeat(frames)).expect("write fixture");
}
