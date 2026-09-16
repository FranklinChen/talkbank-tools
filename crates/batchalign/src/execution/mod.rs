//! New execution-kernel seams for recipe-owned command execution.

mod coref;
mod kernel;
pub(crate) mod morphotag;
mod text_io;
mod translate;
mod utseg;
mod worker_gateway;

pub(crate) use coref::dispatch_coref_job;
pub(crate) use kernel::dispatch_compare_job;
pub(crate) use morphotag::dispatch_morphotag_job;
pub(crate) use translate::dispatch_translate_job;
pub(crate) use utseg::dispatch_utseg_job;
pub(crate) use worker_gateway::{MorphotagRuntimeOptions, PooledWorkerGateway};
