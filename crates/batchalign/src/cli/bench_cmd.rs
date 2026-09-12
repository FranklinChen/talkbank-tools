//! `batchalign3 bench`: repeated performance runs for a command.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::ReleasedCommand;
use crate::options::{
    AlignOptions, BenchmarkOptions, CommandOptions, CommonOptions, CompareOptions, CorefOptions,
    MorphotagOptions, OpensmileOptions, TranscribeOptions, TranslateOptions, UtrEngine,
    UtsegOptions,
};
use serde_json::json;

use crate::cli::args::InputKind;
use crate::cli::args::{BenchArgs, BenchTarget, GlobalOpts};
use crate::cli::dispatch;
use crate::cli::error::CliError;

fn metadata(target: BenchTarget) -> (ReleasedCommand, &'static str, u32, InputKind) {
    match target {
        BenchTarget::Align => (ReleasedCommand::Align, "eng", 1, InputKind::Chat),
        BenchTarget::Transcribe => (ReleasedCommand::Transcribe, "eng", 1, InputKind::Media),
        BenchTarget::TranscribeS => (ReleasedCommand::TranscribeS, "eng", 1, InputKind::Media),
        // PerFile commands carry the `"per-file"` wire string, matching
        // `command_profile()` for these same commands. Bench harness uses
        // the same profile so dispatch types stay identical.
        BenchTarget::Morphotag => (ReleasedCommand::Morphotag, "per-file", 1, InputKind::Chat),
        BenchTarget::Translate => (ReleasedCommand::Translate, "per-file", 1, InputKind::Chat),
        BenchTarget::Coref => (ReleasedCommand::Coref, "per-file", 1, InputKind::Chat),
        BenchTarget::Utseg => (ReleasedCommand::Utseg, "eng", 1, InputKind::Chat),
        BenchTarget::Benchmark => (ReleasedCommand::Benchmark, "eng", 1, InputKind::Media),
        BenchTarget::Opensmile => (ReleasedCommand::Opensmile, "eng", 1, InputKind::Media),
        BenchTarget::Compare => (ReleasedCommand::Compare, "eng", 1, InputKind::Chat),
    }
}

fn dataset_label(args: &BenchArgs) -> String {
    if let Some(s) = &args.dataset {
        return s.clone();
    }

    args.in_dir
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| args.in_dir.to_str().unwrap_or("unknown"))
        .to_string()
}

fn build_options(_global: &GlobalOpts, args: &BenchArgs) -> CommandOptions {
    let common = CommonOptions {
        override_media_cache: !args.use_cache,
        ..Default::default()
    };

    match args.command {
        BenchTarget::Align => CommandOptions::Align(AlignOptions {
            common,
            utr: crate::options::AlignUtrOptions {
                engine: Some(UtrEngine::RevAi),
                ..Default::default()
            },
            ..AlignOptions::default()
        }),
        BenchTarget::Transcribe => CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common,
            asr_engine: crate::options::AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
            batch_size: 8,
        }),
        BenchTarget::TranscribeS => CommandOptions::TranscribeS(TranscribeOptions {
            auto_speakers: false,
            common,
            asr_engine: crate::options::AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
            batch_size: 8,
        }),
        BenchTarget::Morphotag => CommandOptions::Morphotag(MorphotagOptions {
            common,

            ..Default::default()
        }),
        BenchTarget::Translate => CommandOptions::Translate(TranslateOptions {
            common,
            merge_abbrev: false.into(),
            ..TranslateOptions::default()
        }),
        BenchTarget::Utseg => CommandOptions::Utseg(UtsegOptions {
            common,
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
        }),
        BenchTarget::Benchmark => CommandOptions::Benchmark(BenchmarkOptions {
            common,
            asr_engine: crate::options::AsrEngineName::RevAi,
            wor: false.into(),
            merge_abbrev: false.into(),
        }),
        BenchTarget::Opensmile => CommandOptions::Opensmile(OpensmileOptions {
            common,
            feature_set: "eGeMAPSv02".into(),
        }),
        BenchTarget::Coref => CommandOptions::Coref(CorefOptions {
            common,
            merge_abbrev: false.into(),
        }),
        BenchTarget::Compare => CommandOptions::Compare(CompareOptions {
            common,
            merge_abbrev: false.into(),
        }),
    }
}

/// Execute repeated benchmark runs and report timing statistics.
pub async fn run(global: &GlobalOpts, args: &BenchArgs) -> Result<(), CliError> {
    if args.runs == 0 {
        return Err(
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "--runs must be >= 1").into(),
        );
    }

    if !Path::new(&args.in_dir).exists() {
        return Err(CliError::InputMissing(args.in_dir.clone()));
    }
    std::fs::create_dir_all(&args.out_dir)?;

    let (command, lang, num_speakers, input_kind) = metadata(args.command);
    let dataset = dataset_label(args);
    let inputs = vec![args.in_dir.clone()];
    let mut elapsed_runs = Vec::with_capacity(args.runs);

    for idx in 0..args.runs {
        let start = Instant::now();

        dispatch::dispatch(
            dispatch::DispatchRequest {
                command,
                lang,
                num_speakers,
                input_kind,
                server_arg: global.server.as_deref(),
                inputs: &inputs,
                out_dir: Some(args.out_dir.as_path()),
                options: Some(build_options(global, args)),
                bank: None,
                subdir: None,
                lexicon: None,
                use_tui: false,
                open_dashboard: global.use_open_dashboard(),
                force_cpu: global.force_cpu,
                allow_mps: global.allow_mps,
                no_server: global.no_server,
                before: None,
                workers: global.workers,
                timeout: global.timeout,
                sequential: global.sequential,
                memory_tier: global.memory_tier,
            },
            &crate::config::RuntimeLayout::from_env(),
        )
        .await?;

        let elapsed = start.elapsed().as_secs_f64();
        elapsed_runs.push(elapsed);

        eprintln!("Run {}/{}: {:.2}s", idx + 1, args.runs, elapsed);
        println!(
            "BENCH_RESULT: {}",
            json!({
                "run": idx + 1,
                "elapsed_s": (elapsed * 100.0).round() / 100.0,
                "command": command.as_wire_name(),
                "dataset": dataset,
                "dispatch_mode": if global.server.is_some() { "server" } else { "auto" },
            })
        );

        if idx + 1 < args.runs {
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    }

    let avg = elapsed_runs.iter().sum::<f64>() / elapsed_runs.len() as f64;
    eprintln!("\nAverage: {:.2}s over {} run(s)", avg, elapsed_runs.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn global() -> GlobalOpts {
        GlobalOpts {
            verbose: 0,
            workers: None,
            force_cpu: false,
            allow_mps: false,
            server: None,
            no_server: false,
            media_cache: crate::cli::args::MediaCacheOpts::default(),
            tui: false,
            no_tui: false,
            open_dashboard: true,
            no_open_dashboard: false,
            debug_dir: None,
            engine_overrides: None,
            timeout: None,
            sequential: false,
            memory_tier: None,
        }
    }

    fn args(target: BenchTarget) -> BenchArgs {
        BenchArgs {
            command: target,
            in_dir: PathBuf::from("/tmp/input"),
            out_dir: PathBuf::from("/tmp/out"),
            runs: 1,
            dataset: None,
            workers: None,
            use_cache: false,
        }
    }

    #[test]
    fn metadata_align() {
        let (cmd, lang, n, exts) = metadata(BenchTarget::Align);
        assert_eq!(cmd, ReleasedCommand::Align);
        assert_eq!(lang, "eng");
        assert_eq!(n, 1);
        assert_eq!(exts, InputKind::Chat);
    }

    #[test]
    fn dataset_uses_path_basename() {
        let a = args(BenchTarget::Align);
        assert_eq!(dataset_label(&a), "input");
    }

    #[test]
    fn options_respect_flags() {
        let g = global();
        let a = args(BenchTarget::Morphotag);
        let opts = build_options(&g, &a);
        // use_cache=false → override_media_cache=true
        assert!(opts.common().override_media_cache);
        assert_eq!(opts.command_name(), "morphotag");
    }
}
