//! A whole-file request owns its response, recording bound and route.

use super::*;

pub(super) struct FullRecognition<'a> {
    conversion: UtrTokenConversion,
    recording: Recording,
    context: UtrPassContext<'a>,
    route: UtrFullRoute,
}

struct BoundedRequest<'a> {
    context: UtrPassContext<'a>,
    route: UtrFullRoute,
    recording: Recording,
    window: FaWindow,
}

impl<'a> BoundedRequest<'a> {
    async fn admit(
        context: UtrPassContext<'a>,
        route: UtrFullRoute,
    ) -> Result<Self, crate::error::ServerError> {
        let recording = context.recording().await?;
        let window =
            FaWindow::within(&recording, FileMs::new(0), recording.duration()).map_err(|why| {
                crate::error::ServerError::RecordingDuration(
                    crate::error::RecordingDurationError::WholeFileWindow(why),
                )
            })?;
        Ok(Self {
            context,
            route,
            recording,
            window,
        })
    }

    async fn recognize(self) -> Result<FullRecognition<'a>, crate::error::ServerError> {
        let context = self.context;
        let key = crate::chat_ops::fa::utr_asr_cache_key(
            context.audio_identity,
            context.engine,
            context.lang,
        );
        let response = match lookup_utr_asr_cache(
            context.services.pipeline.cache,
            &key,
            &utr_cache_eligibility(context),
            context.cache_policy.response(),
        )
        .await?
        {
            UtrAsrCacheLookup::Hit(cached) => {
                info!(context.filename, "UTR ASR cache hit");
                *cached
            }
            UtrAsrCacheLookup::Miss(miss) => {
                info!(
                    context.filename,
                    engine = context.engine.as_wire_name(),
                    "UTR ASR cache miss, running inference"
                );
                let budget = context
                    .audio_duration
                    .map(|duration| DecodeBudgetSeconds::for_duration_ms(duration.length().0));
                match miss.infer(context.audio_path, budget, context).await {
                    Ok(inferred) => {
                        inferred
                            .commit(context.services.pipeline.cache, &key, context.filename)
                            .await
                    }
                    Err(error) => {
                        warn!(context.filename, error = %error, "UTR ASR inference failed");
                        return Err(error);
                    }
                }
            }
        };
        let conversion = asr_response_to_utr_tokens(
            response,
            self.window,
            &EngineId::new(context.engine.as_wire_name()),
        );
        conversion.warn_if_lossy(&context);
        Ok(FullRecognition {
            conversion,
            recording: self.recording,
            context,
            route: self.route,
        })
    }
}

impl<'a> FullRecognition<'a> {
    pub(super) async fn run(
        context: UtrPassContext<'a>,
        route: UtrFullRoute,
    ) -> Result<Self, crate::error::ServerError> {
        BoundedRequest::admit(context, route)
            .await?
            .recognize()
            .await
    }

    pub(super) fn recording(&self) -> Recording {
        self.recording
    }

    pub(super) fn into_parts(self) -> (UtrTokenConversion, UtrPassContext<'a>, UtrFullRoute) {
        (self.conversion, self.context, self.route)
    }
}
