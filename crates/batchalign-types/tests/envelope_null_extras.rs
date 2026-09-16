//! Envelope-level regression: explicit `null` extras must parse.
//!
// Test code: fixtures use unwrap/panic by convention.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[test]
fn execute_request_envelope_accepts_null_extras() {
    let json = r#"{
        "request_id": "r-1",
        "task": "asr",
        "payload": {"kind": "asr", "lang": "eng", "backend": "local_whisper",
                     "input": {"kind": "prepared_audio", "audio_ref_id": "a-1"},
                     "models": {"engine": "whisper",
                                "asr": {"id": "openai/whisper-large-v3",
                                        "revision": {"kind": "commit",
                                                     "commit": "06f233fe06e710322aca913c1bc4249a0d71fce1"}}},
                     "extras": null},
        "attachments": []
    }"#;
    let value: serde_json::Value = serde_json::from_str(json).unwrap();
    let parsed: Result<batchalign_types::worker_v2::ExecuteRequestV2, _> =
        serde_json::from_value(value);
    if let Err(e) = &parsed {
        panic!("envelope rejected null extras: {e}");
    }
}
