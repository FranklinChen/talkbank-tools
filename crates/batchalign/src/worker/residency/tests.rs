use super::*;
use crate::types::engines::{AsrEngineName, FaEngineName};
fn engines(set: impl FnOnce(&mut EngineOverrides)) -> EngineOverrides {
    let mut values = EngineOverrides::default();
    set(&mut values);
    values
}
/// Guards: measured/estimated footprints retain the exact quarter-margin,
/// rounding and unread-settlement arithmetic used by actual memory leases.
/// Kind: policy
/// Not a type: numeric sizing and rounding remain resource policy after the
/// private footprint representation prevents in-use sizes below built sizes.
/// State: admitted constants and compile-checked footprint constructors need no host ledger.
/// Concurrency: none
#[test]
fn a_lease_reserves_use_and_a_quarter_and_settles_to_the_unread_part() {
    assert_eq!(PYANNOTE_DIARIZATION.margin(), MemoryMb(569));
    assert_eq!(PYANNOTE_DIARIZATION.reserved(), MemoryMb(2_843));
    assert_eq!(PYANNOTE_DIARIZATION.unread(), MemoryMb(1_896));
    assert_eq!(PYANNOTE_DIARIZATION.settled(), MemoryMb(2_465));
    assert_eq!(Footprint::measured(1, 1).margin(), MemoryMb(1));
    assert_eq!(WORKER_BASELINE.reserved(), MemoryMb(344));
    assert_eq!(WORKER_BASELINE.unread(), MemoryMb(0));
    // An estimate has no built figure apart, so nothing is unread.
    assert_eq!(Footprint::estimated(1_000).settled(), MemoryMb(250));
}
/// Guards: each task's selected model keeps the canonical engine estimate,
/// measured speaker footprint, real absence or per-batch coreference residency.
/// Kind: policy
/// Not a type: engine-to-residency choice and lease lifetime are semantic mappings,
/// while every footprint is now opaque and admitted by its private constructor.
/// State: typed task/engine values exercise the actual residency table without loading models.
/// Concurrency: none
#[test]
fn selected_models_keep_their_declared_residency_policy() {
    {
        let whisper = Residency::of(
            InferTask::Asr,
            &engines(|e| e.asr = Some(AsrEngineName::Whisper)),
        );
        assert_eq!(
            whisper,
            Residency::WorkerLife(Footprint::estimated(
                AsrEngineName::Whisper.resident_memory_mb()
            ))
        );
        let wav2vec = Residency::of(
            InferTask::Fa,
            &engines(|e| e.fa = Some(FaEngineName::Wave2Vec)),
        );
        assert_eq!(
            wav2vec.loading(),
            Some(Footprint::estimated(FaEngineName::Wave2Vec.resident_memory_mb()).reserved())
        );
        let nllb = Residency::of(
            InferTask::Translate,
            &engines(|e| e.translate = Some(TranslateEngineName::Nllb)),
        );
        assert_eq!(
            nllb,
            Residency::WorkerLife(Footprint::estimated(
                TranslateEngineName::Nllb.resident_memory_mb()
            ))
        );
    }
    {
        assert_eq!(
            Residency::of(InferTask::Translate, &EngineOverrides::default()),
            Residency::WorkerLife(Footprint::estimated(
                TranslateEngineName::Google.resident_memory_mb()
            ))
        );
        assert_eq!(
            Residency::of(
                InferTask::Asr,
                &engines(|e| e.asr = Some(AsrEngineName::RevAi))
            ),
            Residency::Nothing
        );
        assert_eq!(
            Residency::of(InferTask::Asr, &EngineOverrides::default()),
            Residency::Nothing
        );
        assert_eq!(
            Residency::of(InferTask::Fa, &EngineOverrides::default()),
            Residency::Nothing
        );
    }
    {
        let local = Residency::of(
            InferTask::Speaker,
            &engines(|e| e.speaker = Some(SpeakerEngineName::Pyannote)),
        );
        assert_eq!(local, Residency::WorkerLife(PYANNOTE_DIARIZATION));
        let embedding = Residency::of(
            InferTask::Speaker,
            &engines(|e| e.speaker_embedding = Some(SpeakerEmbeddingBackendV2::Pyannote)),
        );
        assert_eq!(embedding, Residency::WorkerLife(SPEAKER_EMBEDDING));
        assert_eq!(
            Residency::of(
                InferTask::Speaker,
                &engines(|e| e.speaker = Some(SpeakerEngineName::PyannoteAi))
            ),
            Residency::Nothing
        );
        assert_eq!(PYANNOTE_DIARIZATION.basis(), Basis::Measured);
    }
    {
        let coref = Residency::of(InferTask::Coref, &EngineOverrides::default());
        assert_eq!(coref.loading(), Some(COREF.reserved()));
        assert_eq!(coref.resident(), MemoryMb(0));
        assert_eq!(coref.batch(), Some(COREF.reserved()));
        let morph = Residency::of(InferTask::Morphosyntax, &EngineOverrides::default());
        assert_eq!(morph.batch(), None);
        assert_eq!(morph.resident(), MORPHOSYNTAX.settled());
    }
    {
        for &task in InferTask::ALL {
            let residency = Residency::of(task, &EngineOverrides::default());
            match task {
                InferTask::Opensmile | InferTask::Avqi | InferTask::WordSegmentation => {
                    assert_eq!(residency, Residency::Nothing, "{task:?}");
                }
                InferTask::Morphosyntax
                | InferTask::Utseg
                | InferTask::Translate
                | InferTask::Coref => assert!(residency.loading().is_some(), "{task:?}"),
                InferTask::Fa | InferTask::Asr | InferTask::Speaker => {
                    assert_eq!(residency, Residency::Nothing, "{task:?} with no engine");
                }
            }
        }
    }
}
