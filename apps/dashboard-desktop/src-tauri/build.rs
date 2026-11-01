//! Admit the window icon's decoded pixel contract before Tauri embeds it.

#[derive(Debug, thiserror::Error)]
enum IconAdmissionError {
    #[error("Cannot read the native window icon: {0}")]
    Read(#[from] std::io::Error),
    #[error("Cannot decode the native window icon: {0}")]
    Decode(#[from] png::DecodingError),
    #[error("Cannot read the native icon configuration: {0}")]
    Configuration(#[from] serde_json::Error),
    #[error("The native bundle must declare at least one PNG window icon")]
    MissingPng,
    #[error("Window icon {} uses {bit_depth:?}/{color_type:?}; an 8-bit RGBA PNG is required. Regenerate icons from app-icon.svg", path.display())]
    UnsupportedEncoding {
        path: std::path::PathBuf,
        bit_depth: png::BitDepth,
        color_type: png::ColorType,
    },
    #[error("Native window icon has invalid dimensions or an incomplete pixel buffer")]
    InvalidPixels,
}

/// Only a fully decoded RGBA8 image can authorize embedding the app context.
struct AdmittedWindowIcon(std::path::PathBuf);

impl AdmittedWindowIcon {
    fn read(path: std::path::PathBuf) -> Result<Self, IconAdmissionError> {
        let image = std::io::BufReader::new(std::fs::File::open(&path)?);
        let mut decoder = png::Decoder::new(image).read_info()?;
        let info = decoder.info();
        if info.bit_depth != png::BitDepth::Eight || info.color_type != png::ColorType::Rgba {
            return Err(IconAdmissionError::UnsupportedEncoding {
                path,
                bit_depth: info.bit_depth,
                color_type: info.color_type,
            });
        }
        let expected = match (info.width, info.height) {
            (width @ 1..=1024, height) if width == height => {
                (width as usize) * (height as usize) * 4
            }
            _ => return Err(IconAdmissionError::InvalidPixels),
        };
        let mut pixels = vec![0; expected];
        let frame = decoder.next_frame(&mut pixels)?;
        if frame.buffer_size() != expected {
            return Err(IconAdmissionError::InvalidPixels);
        }
        Ok(Self(path))
    }
}

#[derive(serde::Deserialize)]
struct IconConfiguration {
    bundle: BundleIcons,
}
#[derive(serde::Deserialize)]
struct BundleIcons {
    icon: Vec<std::path::PathBuf>,
}

/// Nonempty, source-bound icon population admitted from Tauri's own manifest.
struct AdmittedWindowIcons {
    first: AdmittedWindowIcon,
    rest: Vec<AdmittedWindowIcon>,
}

impl AdmittedWindowIcons {
    fn read() -> Result<Self, IconAdmissionError> {
        let config: IconConfiguration =
            serde_json::from_reader(std::fs::File::open("tauri.conf.json")?)?;
        let mut sources = config
            .bundle
            .icon
            .into_iter()
            .filter(|path| path.extension().is_some_and(|extension| extension == "png"));
        let first =
            AdmittedWindowIcon::read(sources.next().ok_or(IconAdmissionError::MissingPng)?)?;
        let rest = sources
            .map(AdmittedWindowIcon::read)
            .collect::<Result<_, _>>()?;
        Ok(Self { first, rest })
    }

    fn embed(self) {
        for AdmittedWindowIcon(source) in std::iter::once(self.first).chain(self.rest) {
            println!("cargo:rerun-if-changed={}", source.display());
        }
        tauri_build::build();
    }
}

fn main() -> std::process::ExitCode {
    println!("cargo:rerun-if-changed=tauri.conf.json");
    match AdmittedWindowIcons::read() {
        Ok(icons) => {
            icons.embed();
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
