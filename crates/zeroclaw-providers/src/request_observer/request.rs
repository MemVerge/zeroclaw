use super::{RequestObserver, RequestSampleOutcome};
use crate::compatible::RequestHeaders;
use reqwest::header::HeaderMap;
use serde::Serialize;
use zeroclaw_api::model_provider::ProjectedMessageSource;

pub(crate) struct RequestHooks<'a> {
    pub(crate) headers: Option<&'a RequestHeaders>,
    pub(crate) observer: Option<&'a RequestObserver>,
}

pub(crate) fn request_with_sources(
    hooks: RequestHooks<'_>,
    reserved: crate::extra_headers::ReservedHeaders,
    additional_reserved: &[&str],
    builder: reqwest::RequestBuilder,
    request: &impl Serialize,
    sources: Vec<ProjectedMessageSource>,
) -> anyhow::Result<reqwest::RequestBuilder> {
    let sample = crate::request_observer::prepare(hooks.observer).and_then(|sample| {
        match sample.check(request, sources.len()) {
            Ok(()) => Some(sample),
            Err(reason) => {
                sample.notify(RequestSampleOutcome::Skipped(reason));
                None
            }
        }
    });
    match (hooks.headers, sample) {
        (None, None) => Ok(builder.json(request)),
        (callback, sample) => {
            let body = match serde_json::to_value(request) {
                Ok(body) => body,
                Err(error) if callback.is_some() => return Err(error.into()),
                Err(_) => {
                    if let Some(sample) = sample {
                        sample.notify(RequestSampleOutcome::Skipped(
                            crate::request_observer::RequestSampleSkipReason::Serialization,
                        ));
                    }
                    return Ok(builder.json(request));
                }
            };
            let mut headers = match callback {
                Some(callback) => callback(&body, &sources)?,
                None => HeaderMap::new(),
            };
            let owned = headers
                .keys()
                .filter(|name| reserved.contains(name.as_str(), additional_reserved))
                .cloned()
                .collect::<Vec<_>>();
            for name in owned {
                headers.remove(name);
            }
            if let Some(sample) = sample {
                sample.notify(RequestSampleOutcome::Captured {
                    body: &body,
                    sources: &sources,
                    generated_headers: &headers,
                });
            }
            Ok(builder.headers(headers).json(request))
        }
    }
}
