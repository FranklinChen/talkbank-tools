use super::*;

#[test]
fn morphotag_roundtrip() {
    let opts = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions::default(),
        retokenize: true,

        ..Default::default()
    });
    let json = serde_json::to_string(&opts).unwrap();
    assert!(
        !json.contains("ca_policy"),
        "the default policy stays wire-compatible with older jobs: {json}"
    );
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn morphotag_ca_analyze_policy_roundtrips_explicitly() {
    let opts = CommandOptions::Morphotag(MorphotagOptions {
        ca_policy: CaMorphotagPolicy::Analyze,
        ..Default::default()
    });
    let json = serde_json::to_string(&opts).unwrap();
    assert!(json.contains("\"ca_policy\":\"analyze\""), "json: {json}");
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn align_roundtrip() {
    let opts = CommandOptions::Align(AlignOptions {
        common: CommonOptions::default(),
        fa_engine: FaEngineName::Whisper,
        utr: AlignUtrOptions {
            engine: Some(UtrEngine::RevAi),
            ..Default::default()
        },
        pauses: true,
        boundaries: AlignBoundaryOptions {
            existing_wor_boundaries: Default::default(),
            end_overlap_policy: crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY,
            main_bullets: crate::chat_ops::fa::MainBulletPolicy::KeepGiven,
        },
        wor: true.into(),
        merge_abbrev: false.into(),
        bullet_repair: false,
        review_level: Default::default(),
        media_dir: None,
    });
    let json = serde_json::to_string(&opts).unwrap();
    // The wire name is the CLI's short word, so a stored job reads the
    // way the command line that submitted it did.
    assert!(json.contains("\"main_bullets\":\"keep\""), "json: {json}");
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn align_job_stored_without_main_bullets_reads_as_derive() {
    let json = r#"{"command":"align","existing_wor_boundaries":"preserve"}"#;
    let CommandOptions::Align(align) = serde_json::from_str(json).unwrap() else {
        panic!("expected Align");
    };
    assert_eq!(
        align.boundaries.main_bullets,
        crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY
    );
}

#[test]
fn transcribe_roundtrip() {
    let opts = CommandOptions::Transcribe(TranscribeOptions {
        auto_speakers: false,
        common: CommonOptions::default(),
        asr_engine: AsrEngineName::WhisperX,
        diarize: true,
        wor: false.into(),
        merge_abbrev: false.into(),
        utseg_fallback: false.into(),
        batch_size: 16,
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn transcribe_s_roundtrip() {
    let opts = CommandOptions::TranscribeS(TranscribeOptions {
        auto_speakers: false,
        common: CommonOptions::default(),
        asr_engine: AsrEngineName::RevAi,
        diarize: true,
        wor: false.into(),
        merge_abbrev: false.into(),
        utseg_fallback: false.into(),
        batch_size: 8,
    });
    let json = serde_json::to_string(&opts).unwrap();
    assert!(json.contains("\"command\":\"transcribe_s\""));
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn command_name_matches_tag() {
    let cases: Vec<(CommandOptions, &str)> = vec![
        (
            CommandOptions::Align(AlignOptions {
                common: CommonOptions::default(),
                fa_engine: FaEngineName::Wave2Vec,
                utr: Default::default(),
                pauses: false,
                boundaries: Default::default(),
                wor: true.into(),
                merge_abbrev: false.into(),
                bullet_repair: false,
                review_level: Default::default(),
                media_dir: None,
            }),
            "align",
        ),
        (
            CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
            "morphotag",
        ),
        (
            CommandOptions::Opensmile(OpensmileOptions {
                common: CommonOptions::default(),
                feature_set: "eGeMAPSv02".into(),
            }),
            "opensmile",
        ),
        (
            CommandOptions::Compare(CompareOptions {
                common: CommonOptions::default(),
                merge_abbrev: false.into(),
            }),
            "compare",
        ),
        (
            CommandOptions::Avqi(AvqiOptions {
                common: CommonOptions::default(),
            }),
            "avqi",
        ),
    ];

    for (opts, expected_name) in cases {
        assert_eq!(opts.command_name(), expected_name);
        let json = serde_json::to_string(&opts).unwrap();
        assert!(
            json.contains(&format!("\"command\":\"{expected_name}\"")),
            "JSON should contain command tag '{expected_name}': {json}"
        );
    }
}

#[test]
fn common_accessor() {
    let opts = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_media_cache: true,
            engine_overrides: EngineOverrides::default(),
            mwt: BTreeMap::new(),
            ..Default::default()
        },
        retokenize: true,

        ..Default::default()
    });
    assert!(opts.common().override_media_cache);
}

#[test]
fn required_media_cache_wire_field_is_explicit_and_backward_compatible() {
    let required = CommonOptions {
        require_media_cache: true,
        ..Default::default()
    };
    let json = serde_json::to_string(&required).expect("serialize required policy");
    assert!(json.contains(r#""require_media_cache":true"#));

    let legacy: CommonOptions = serde_json::from_str(r#"{"override_media_cache":false}"#)
        .expect("deserialize legacy common options");
    assert!(!legacy.require_media_cache);
}

#[test]
fn engine_overrides_roundtrip() {
    let overrides = EngineOverrides {
        asr: Some(AsrEngineName::HkTencent),
        fa: Some(FaEngineName::Wav2vecCanto),
        translate: None,
        ..Default::default()
    };

    let opts = CommandOptions::Align(AlignOptions {
        common: CommonOptions {
            override_media_cache: false,
            engine_overrides: overrides.clone(),
            mwt: BTreeMap::new(),
            ..Default::default()
        },
        fa_engine: FaEngineName::Wav2vecCanto,
        ..AlignOptions::default()
    });

    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(back.common().engine_overrides, overrides);
}

/// `CommonOptions::debug_dir` is typed as `Option<PathBuf>`, but the wire
/// format must stay a JSON string so existing clients (and the dashboard
/// schema) keep working. Lock that invariant here.
#[test]
fn debug_dir_serializes_as_json_string() {
    let opts = CommonOptions {
        debug_dir: Some(PathBuf::from("/tmp/some/abs/path")),
        ..Default::default()
    };
    let json = serde_json::to_string(&opts).unwrap();
    assert!(
        json.contains(r#""debug_dir":"/tmp/some/abs/path""#),
        "expected debug_dir as JSON string, got: {json}"
    );
    let back: CommonOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(back.debug_dir, Some(PathBuf::from("/tmp/some/abs/path")));
}

#[test]
fn transcribe_asr_override_effective_engine_prefers_override() {
    let overrides = EngineOverrides {
        asr: Some(AsrEngineName::HkTencent),
        fa: None,
        translate: None,
        ..Default::default()
    };
    let opts = TranscribeOptions {
        auto_speakers: false,
        common: CommonOptions {
            engine_overrides: overrides,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::RevAi,
        diarize: false,
        wor: false.into(),
        merge_abbrev: false.into(),
        utseg_fallback: false.into(),
        batch_size: 8,
    };

    assert_eq!(opts.effective_asr_engine(), AsrEngineName::HkTencent);
}

#[test]
fn benchmark_asr_override_effective_engine_prefers_override() {
    let overrides = EngineOverrides {
        asr: Some(AsrEngineName::HkAliyun),
        fa: None,
        translate: None,
        ..Default::default()
    };
    let opts = BenchmarkOptions {
        common: CommonOptions {
            engine_overrides: overrides,
            ..CommonOptions::default()
        },
        asr_engine: AsrEngineName::RevAi,
        wor: true.into(),
        merge_abbrev: false.into(),
    };

    assert_eq!(opts.effective_asr_engine(), AsrEngineName::HkAliyun);
}

#[test]
fn minimal_json_deserializes_with_defaults() {
    let json = r#"{"command": "morphotag"}"#;
    let opts: CommandOptions = serde_json::from_str(json).unwrap();
    assert_eq!(opts.command_name(), "morphotag");
    if let CommandOptions::Morphotag(m) = &opts {
        assert!(!m.retokenize);
        assert!(!m.skipmultilang);
        assert!(!m.merge_abbrev.should_merge());
    } else {
        panic!("expected Morphotag");
    }
}

#[test]
fn avqi_roundtrip() {
    let opts = CommandOptions::Avqi(AvqiOptions {
        common: CommonOptions::default(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn legacy_diarize_job_defaults_to_historical_local_backend() {
    let options: CommandOptions = serde_json::from_str(r#"{"command":"diarize"}"#)
        .expect("legacy standalone diarize job remains readable");
    let CommandOptions::Diarize(options) = options else {
        panic!("expected diarize options");
    };
    assert_eq!(options.speaker_engine, SpeakerEngineName::Pyannote);
    assert_eq!(options.expected_speakers, None);
    assert_eq!(options.output_mode, DiarizeOutputMode::TurnsJson);
}

#[test]
fn standalone_paid_speaker_backend_roundtrips_by_wire_name() {
    let options = CommandOptions::Diarize(DiarizeOptions {
        common: CommonOptions::default(),
        speaker_engine: SpeakerEngineName::PyannoteAi,
        expected_speakers: Some(
            DiarizationSpeakerCount::try_from(3).expect("three speakers is answerable"),
        ),
        output_mode: DiarizeOutputMode::TurnsJson,
    });
    let json = serde_json::to_string(&options).expect("serialize diarize options");
    assert!(json.contains(r#""speaker_engine":"pyannote_ai""#));
    let back = serde_json::from_str(&json).expect("deserialize diarize options");
    assert_eq!(options, back);
}

/// A diarization count of one has no representation, on either door.
///
/// The ruling is that a count is exactly N (N at least 2) or automatic;
/// automatic is the ABSENCE of a count, so one is neither. Refused at
/// deserialization as well as construction, because a direct HTTP client
/// posts this JSON without passing the CLI's value parser.
#[test]
fn a_diarization_count_below_two_has_no_representation() {
    for refused in [0, 1] {
        assert!(
            DiarizationSpeakerCount::try_from(refused).is_err(),
            "{refused} speakers is not a question a diarizer can answer"
        );
        let json = format!(r#"{{"command":"diarize","expected_speakers":{refused}}}"#);
        let error =
            serde_json::from_str::<CommandOptions>(&json).expect_err("the wire must refuse it too");
        assert!(error.to_string().contains("at least 2"), "{error}");
    }
    assert_eq!(
        DiarizationSpeakerCount::try_from(2)
            .expect("two is the minimum")
            .get(),
        2
    );
    let automatic: CommandOptions = serde_json::from_str(r#"{"command":"diarize"}"#)
        .expect("omitting the count asks for automatic detection");
    let CommandOptions::Diarize(automatic) = automatic else {
        panic!("expected diarize options");
    };
    assert_eq!(automatic.expected_speakers, None);
}

#[test]
fn compare_roundtrip() {
    let opts = CommandOptions::Compare(CompareOptions {
        common: CommonOptions::default(),
        merge_abbrev: true.into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn translate_roundtrip() {
    let opts = CommandOptions::Translate(TranslateOptions {
        common: CommonOptions::default(),
        translate_engine: TranslateEngineName::Google,
        target: crate::api::LanguageCode3::spa(),
        merge_abbrev: true.into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn translate_old_jobs_default_target_but_invalid_targets_are_refused() {
    let mut value = serde_json::to_value(TranslateOptions::default()).unwrap();
    value.as_object_mut().unwrap().remove("target");
    let old: TranslateOptions = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(old.target, crate::api::LanguageCode3::eng());
    value["target"] = serde_json::json!("auto");
    assert!(serde_json::from_value::<TranslateOptions>(value).is_err());
}

#[test]
fn translate_options_default_engine_is_google() {
    // Default preserves the fleet's historical behavior. Operators
    // who want Seamless pass `--translate-engine seamless`
    // explicitly; there is no per-host config-file default.
    let opts = TranslateOptions::default();
    assert_eq!(opts.translate_engine, TranslateEngineName::Google);
}

#[test]
fn translate_options_effective_engine_prefers_explicit_field() {
    let opts = TranslateOptions {
        common: CommonOptions::default(),
        translate_engine: TranslateEngineName::Seamless,
        target: crate::api::LanguageCode3::eng(),
        merge_abbrev: false.into(),
    };
    assert_eq!(
        opts.effective_translate_engine(),
        TranslateEngineName::Seamless,
    );
}

#[test]
fn translate_options_effective_engine_override_wins() {
    // Shared --engine-overrides '{"translate":"seamless"}' beats
    // the dedicated --translate-engine flag, mirroring
    // effective_fa_engine / effective_asr_engine.
    let mut common = CommonOptions::default();
    common.engine_overrides.translate = Some(TranslateEngineName::Seamless);
    let opts = TranslateOptions {
        common,
        translate_engine: TranslateEngineName::Google,
        target: crate::api::LanguageCode3::eng(),
        merge_abbrev: false.into(),
    };
    assert_eq!(
        opts.effective_translate_engine(),
        TranslateEngineName::Seamless,
    );
}

#[test]
fn translate_options_serializes_seamless_engine() {
    let opts = TranslateOptions {
        common: CommonOptions::default(),
        translate_engine: TranslateEngineName::Seamless,
        target: crate::api::LanguageCode3::eng(),
        merge_abbrev: false.into(),
    };
    let json = serde_json::to_string(&opts).unwrap();
    assert!(
        json.contains("\"translate_engine\":\"seamless\""),
        "expected serialized form to contain the Seamless wire token, got: {json}"
    );
}

#[test]
fn coref_roundtrip() {
    let opts = CommandOptions::Coref(CorefOptions {
        common: CommonOptions::default(),
        merge_abbrev: false.into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn utseg_roundtrip() {
    let opts = CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions::default(),
        merge_abbrev: true.into(),
        utseg_fallback: false.into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn benchmark_roundtrip() {
    let opts = CommandOptions::Benchmark(BenchmarkOptions {
        common: CommonOptions::default(),
        asr_engine: AsrEngineName::WhisperOai,
        wor: true.into(),
        merge_abbrev: false.into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn opensmile_roundtrip() {
    let opts = CommandOptions::Opensmile(OpensmileOptions {
        common: CommonOptions::default(),
        feature_set: "ComParE_2016".into(),
    });
    let json = serde_json::to_string(&opts).unwrap();
    let back: CommandOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(opts, back);
}

#[test]
fn utr_engine_roundtrip_preserves_wire_names() {
    let rev_json = serde_json::to_string(&UtrEngine::RevAi).unwrap();
    let whisper_json = serde_json::to_string(&UtrEngine::Whisper).unwrap();
    let custom_json = serde_json::to_string(&UtrEngine::HkTencent).unwrap();

    assert_eq!(rev_json, "\"rev_utr\"");
    assert_eq!(whisper_json, "\"whisper_utr\"");
    assert_eq!(custom_json, "\"tencent_utr\"");

    assert_eq!(
        serde_json::from_str::<UtrEngine>(&rev_json).unwrap(),
        UtrEngine::RevAi
    );
    assert_eq!(
        serde_json::from_str::<UtrEngine>(&whisper_json).unwrap(),
        UtrEngine::Whisper
    );
    assert_eq!(
        serde_json::from_str::<UtrEngine>(&custom_json).unwrap(),
        UtrEngine::HkTencent
    );
}

#[test]
fn fa_engine_roundtrip_preserves_wire_names() {
    let wav2vec_json = serde_json::to_string(&FaEngineName::Wave2Vec).unwrap();
    let whisper_json = serde_json::to_string(&FaEngineName::Whisper).unwrap();
    let custom_json = serde_json::to_string(&FaEngineName::Wav2vecCanto).unwrap();

    assert_eq!(wav2vec_json, "\"wav2vec_fa\"");
    assert_eq!(whisper_json, "\"whisper_fa\"");
    assert_eq!(custom_json, "\"cantonese_fa\"");

    assert_eq!(
        serde_json::from_str::<FaEngineName>(&wav2vec_json).unwrap(),
        FaEngineName::Wave2Vec
    );
    assert_eq!(
        serde_json::from_str::<FaEngineName>(&whisper_json).unwrap(),
        FaEngineName::Whisper
    );
    assert_eq!(
        serde_json::from_str::<FaEngineName>(&custom_json).unwrap(),
        FaEngineName::Wav2vecCanto
    );
}

#[test]
fn asr_engine_roundtrip_preserves_wire_names() {
    let rev_json = serde_json::to_string(&AsrEngineName::RevAi).unwrap();
    let whisperx_json = serde_json::to_string(&AsrEngineName::WhisperX).unwrap();
    let custom_json = serde_json::to_string(&AsrEngineName::HkTencent).unwrap();

    assert_eq!(rev_json, "\"rev\"");
    assert_eq!(whisperx_json, "\"whisperx\"");
    assert_eq!(custom_json, "\"tencent\"");

    assert_eq!(
        serde_json::from_str::<AsrEngineName>(&rev_json).unwrap(),
        AsrEngineName::RevAi
    );
    assert_eq!(
        serde_json::from_str::<AsrEngineName>(&whisperx_json).unwrap(),
        AsrEngineName::WhisperX
    );
    assert_eq!(
        serde_json::from_str::<AsrEngineName>(&custom_json).unwrap(),
        AsrEngineName::HkTencent
    );
}
