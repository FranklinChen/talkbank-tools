//! A whole-file request owns its response and route; the recording bound
//! is read once by the pass and handed in.

use super::*;

pub(super) struct FullRecognition<'a> {
    conversion: UtrTokenConversion,
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
    const fn admit(context: UtrPassContext<'a>, recording: Recording, route: UtrFullRoute) -> Self {
        Self {
            context,
            route,
            recording,
            window: recording.whole_window(),
        }
    }

    async fn recognize(self) -> Result<FullRecognition<'a>, crate::error::ServerError> {
        let context = self.context;
        let key = crate::chat_ops::fa::utr_asr_cache_key(
            context.audio.audio_identity,
            context.recognizer.engine(),
            context.lang,
        );
        let response = match lookup_utr_asr_cache(
            context.services.recording_cache,
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
                    engine = context.recognizer.engine().as_wire_name(),
                    "UTR ASR cache miss, running inference"
                );
                let budget = Some(DecodeBudgetSeconds::for_duration_ms(
                    self.recording.duration().get(),
                ));
                match miss.infer_whole(budget, context).await {
                    Ok(inferred) => inferred.commit(&key, context.filename).await,
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
            &EngineId::new(context.recognizer.engine().as_wire_name()),
        );
        conversion.warn_if_lossy(&context);
        Ok(FullRecognition {
            conversion,
            context,
            route: self.route,
        })
    }
}

impl<'a> FullRecognition<'a> {
    pub(super) async fn run(
        context: UtrPassContext<'a>,
        recording: Recording,
        route: UtrFullRoute,
    ) -> Result<Self, crate::error::ServerError> {
        BoundedRequest::admit(context, recording, route)
            .recognize()
            .await
    }

    pub(super) fn into_parts(self) -> (UtrTokenConversion, UtrPassContext<'a>, UtrFullRoute) {
        (self.conversion, self.context, self.route)
    }
}
