//! A partial recognizer request owns its window until its response is converted.

use super::*;

struct RecognizedWindows {
    first: UtrTokenConversion,
    remaining: Vec<UtrTokenConversion>,
}

impl RecognizedWindows {
    fn push(&mut self, conversion: UtrTokenConversion) {
        self.remaining.push(conversion);
    }

    fn into_iter(self) -> impl Iterator<Item = UtrTokenConversion> {
        std::iter::once(self.first).chain(self.remaining)
    }
}

struct Attempt {
    planned: Vec<UtrWindowBounds>,
    remaining: std::vec::IntoIter<FaWindow>,
    completed: Option<RecognizedWindows>,
}

pub(super) struct PendingWindow<'a> {
    attempt: Attempt,
    window: FaWindow,
    context: UtrPassContext<'a>,
}

pub(super) struct CompletedPartialRecognition<'a> {
    windows: RecognizedWindows,
    context: UtrPassContext<'a>,
}

impl<'a> CompletedPartialRecognition<'a> {
    pub(super) fn into_parts(
        self,
    ) -> (
        UtrTokenConversion,
        Vec<UtrTokenConversion>,
        UtrPassContext<'a>,
    ) {
        (self.windows.first, self.windows.remaining, self.context)
    }
}

pub(super) enum WindowResult<'a> {
    Recognized(Box<RecognizedWindow<'a>>),
    Abandoned(UtrFullRoute),
}

pub(super) enum RecognizedWindow<'a> {
    Next(PendingWindow<'a>),
    Completed(CompletedPartialRecognition<'a>),
}

impl<'a> PendingWindow<'a> {
    pub(super) fn start(
        windows: Vec<FaWindow>,
        context: UtrPassContext<'a>,
    ) -> Result<Self, UtrFullRoute> {
        let planned = windows.iter().copied().map(UtrWindowBounds::from).collect();
        let mut remaining = windows.into_iter();
        let first = remaining.next();
        let attempt = Attempt {
            planned,
            remaining,
            completed: None,
        };
        match first {
            Some(window) => Ok(Self {
                attempt,
                window,
                context,
            }),
            None => Err(attempt.abandon(PartialExecutionStop::NoWindows)),
        }
    }

    pub(super) async fn run(self) -> Result<WindowResult<'a>, crate::error::ServerError> {
        let context = self.context;
        let window = self.window;
        let (start_ms, end_ms) = (window.audio_start().get(), window.end().get());
        let key = crate::chat_ops::fa::utr_asr_segment_cache_key(
            context.audio_identity,
            context.engine,
            start_ms,
            end_ms,
            context.lang,
        );
        let cached = lookup_utr_asr_cache(
            context.services.pipeline.cache,
            &key,
            &utr_cache_eligibility(context),
            context.cache_policy.response(),
        )
        .await?;
        let response =
            match cached {
                UtrAsrCacheLookup::Hit(cached) => {
                    info!(context.filename, start_ms, end_ms, "UTR segment cache hit");
                    *cached
                }
                UtrAsrCacheLookup::Miss(miss) => {
                    let path =
                        match crate::ensure_wav::extract_audio_segment(context.audio_path, window)
                            .await
                        {
                            Ok(path) => path,
                            Err(error) => {
                                warn!(
                                    context.filename, error = %error, start_ms, end_ms,
                                    "Failed to extract audio segment, falling back to full UTR"
                                );
                                drop(miss);
                                return Ok(WindowResult::Abandoned(self.attempt.abandon(
                                    PartialExecutionStop::SegmentExtraction {
                                        window: window.into(),
                                        error: error.to_string(),
                                    },
                                )));
                            }
                        };
                    let budget = Some(DecodeBudgetSeconds::for_duration_ms(window.len().0));
                    match miss.infer(&path, budget, context).await {
                        Ok(inferred) => {
                            inferred
                                .commit(context.services.pipeline.cache, &key, context.filename)
                                .await
                        }
                        Err(error) => {
                            warn!(
                                context.filename, error = %error,
                                "UTR segment ASR failed, falling back to full-file recovery"
                            );
                            return Ok(WindowResult::Abandoned(self.attempt.abandon(
                                PartialExecutionStop::Recognition {
                                    window: window.into(),
                                    error: error.to_string(),
                                },
                            )));
                        }
                    }
                }
            };
        Ok(WindowResult::Recognized(Box::new(
            self.recognized(response),
        )))
    }

    fn recognized(self, response: crate::transcribe::AsrResponse) -> RecognizedWindow<'a> {
        let conversion = asr_response_to_utr_tokens(
            response,
            self.window,
            &EngineId::new(self.context.engine.as_wire_name()),
        );
        conversion.warn_if_lossy(&self.context);
        let mut attempt = self.attempt;
        let completed = match attempt.completed.take() {
            Some(mut completed) => {
                completed.push(conversion);
                completed
            }
            None => RecognizedWindows {
                first: conversion,
                remaining: Vec::new(),
            },
        };
        match attempt.remaining.next() {
            Some(window) => {
                attempt.completed = Some(completed);
                RecognizedWindow::Next(Self {
                    attempt,
                    window,
                    context: self.context,
                })
            }
            None => RecognizedWindow::Completed(CompletedPartialRecognition {
                windows: completed,
                context: self.context,
            }),
        }
    }
}

impl Attempt {
    fn abandon(self, reason: PartialExecutionStop) -> UtrFullRoute {
        let completed_windows = self
            .completed
            .into_iter()
            .flat_map(RecognizedWindows::into_iter)
            .map(|conversion| {
                let (accepted_tokens, evidence) = conversion.into_evidence();
                AbandonedRecognitionWindow {
                    accepted_tokens,
                    evidence,
                }
            })
            .collect();
        UtrFullRoute::AfterPartial(std::sync::Arc::new(
            AbandonedPartialAttempt::ExecutionStopped {
                planned_windows: self.planned,
                reason,
                completed_windows,
            },
        ))
    }
}

#[cfg(test)]
pub(super) fn abandon_for_test(
    windows: &[FaWindow],
    completed: UtrTokenConversion,
    failed_window: FaWindow,
) -> UtrFullRoute {
    Attempt {
        planned: windows.iter().copied().map(UtrWindowBounds::from).collect(),
        remaining: Vec::new().into_iter(),
        completed: Some(RecognizedWindows {
            first: completed,
            remaining: Vec::new(),
        }),
    }
    .abandon(PartialExecutionStop::Recognition {
        window: failed_window.into(),
        error: "recognition failed".into(),
    })
}
