//! Generative fill for PhotoCraft (Photoshop's Generative Fill / Remove / fill empty areas).
//!
//! Pure pixel logic on plain RGBA8 / 8-bit gray buffers, so it has no workspace dependency and
//! can be tested without a document or a network:
//!
//! 1. [`plan::make_plan`] picks the context crop around the selection, snaps it to an aspect
//!    ratio the API accepts and chooses a quality tier from the crop size;
//! 2. [`generate`] sends the crop + mask + prompt through a [`Backend`] and returns a [`Patch`]:
//!    the generated pixels with a feathered alpha, ready to become a new layer;
//! 3. [`empty::find_empty_regions`] finds the transparent areas for "fill empty areas".
//!
//! The only network code is [`openai::UreqBackend`] (OpenAI, Google and Meta), compiled on native targets only.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod empty;
pub mod gemini;
pub mod imgops;
pub mod openai;
pub mod plan;
pub mod prompt;
pub mod provider;
mod run;

pub use imgops::{Gray, Rgba8};
pub use openai::{Backend, BackendError, EditRequest};
pub use plan::{Config, Plan, Tier};
pub use provider::Provider;
pub use run::{Mode, Output, Params, generate, generate_empty};

/// An error with a message ready to show the user.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct GenFillError(pub String);

impl GenFillError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, GenFillError>;
