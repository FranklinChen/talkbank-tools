//! New execution-kernel seams for recipe-owned command execution.

mod coref;
mod kernel;
pub mod morphotag;
pub mod native_media;
pub mod text_io;
mod translate;
mod utseg;
mod worker_gateway;

pub use coref::dispatch_coref_job;
pub use kernel::dispatch_compare_job;
pub use morphotag::dispatch_morphotag_job;
pub use translate::dispatch_translate_job;
pub use utseg::dispatch_utseg_job;
pub use worker_gateway::{MorphotagRuntimeOptions, PooledWorkerGateway};
