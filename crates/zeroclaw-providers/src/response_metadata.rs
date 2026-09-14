//! Opt-in lifecycle metadata for explicit provider HTTP sends.
//!
//! Each provider send gets its own observation, including sends triggered by a
//! caller's retry or repair and provider fallback. Reqwest's internal redirects
//! and protocol retries remain part of that send: only the response returned by
//! reqwest is observed. Installing an observer does not change network behavior.
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMetadataPhase {
    /// The provider is about to send; the request may still fail before reaching the server.
    Started,
    /// The response returned by reqwest is available, before status/body handling.
    Headers,
    /// Processing ended or was cancelled; this does not imply model-call success.
    Completed,
}

/// A local observation ID connects lifecycle events. It is not the provider's
/// request ID; authoritative correlation IDs come from the selected headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseMetadata {
    pub observation_id: String,
    pub phase: ResponseMetadataPhase,
    pub status: Option<u16>,
    pub headers: BTreeMap<String, String>,
}

/// A provider-local observer. Only explicitly selected, valid text headers from
/// the response returned by reqwest are copied. Missing or non-text headers are
/// omitted; interpretation of their values belongs to the caller.
///
/// Callbacks must be quick and non-blocking and may run concurrently. With an
/// unwind panic strategy, callback panics are contained. Under `panic = "abort"`,
/// a panic terminates the process before containment can run. The embedding
/// application's build selects the panic strategy; dependency profiles do not.
#[derive(Clone)]
pub struct ResponseMetadataObserver {
    headers: Arc<[String]>,
    callback: Arc<dyn Fn(ResponseMetadata) + Send + Sync>,
}

impl ResponseMetadataObserver {
    pub fn new(
        headers: impl IntoIterator<Item = impl Into<String>>,
        callback: impl Fn(ResponseMetadata) + Send + Sync + 'static,
    ) -> Self {
        Self {
            headers: headers
                .into_iter()
                .map(|name| name.into().to_ascii_lowercase())
                .collect(),
            callback: Arc::new(callback),
        }
    }

    /// Start one explicit provider send, not each internal reqwest attempt.
    /// Dropping the guard reports completion, including cancellation and
    /// transport failure before headers arrive. Finish this observation before
    /// starting a separate retry, repair, or fallback send.
    pub fn begin_request(&self) -> ResponseMetadataObservation {
        let metadata = ResponseMetadata {
            observation_id: uuid::Uuid::new_v4().to_string(),
            phase: ResponseMetadataPhase::Started,
            status: None,
            headers: BTreeMap::new(),
        };
        self.notify(metadata.clone());
        ResponseMetadataObservation {
            observer: self.clone(),
            metadata,
        }
    }

    fn notify(&self, metadata: ResponseMetadata) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.callback)(metadata)))
            .is_err()
        {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                "Response metadata observer panicked"
            );
        }
    }
}

pub struct ResponseMetadataObservation {
    observer: ResponseMetadataObserver,
    metadata: ResponseMetadata,
}

impl ResponseMetadataObservation {
    /// Call before status handling and before reading any response body.
    pub fn observe_response(&mut self, response: &reqwest::Response) {
        self.metadata.headers = self
            .observer
            .headers
            .iter()
            .filter_map(|name| {
                response
                    .headers()
                    .get(name.as_str())
                    .and_then(|value| value.to_str().ok())
                    .map(|value| (name.clone(), value.to_owned()))
            })
            .collect();
        self.metadata.status = Some(response.status().as_u16());
        self.metadata.phase = ResponseMetadataPhase::Headers;
        self.observer.notify(self.metadata.clone());
    }
}

impl Drop for ResponseMetadataObservation {
    fn drop(&mut self) {
        self.metadata.phase = ResponseMetadataPhase::Completed;
        self.observer.notify(self.metadata.clone());
    }
}

pub(crate) fn begin_request(
    observer: &Option<ResponseMetadataObserver>,
) -> Option<ResponseMetadataObservation> {
    observer
        .as_ref()
        .map(ResponseMetadataObserver::begin_request)
}

pub(crate) fn observe_response(
    observation: &mut Option<ResponseMetadataObservation>,
    response: &reqwest::Response,
) {
    if let Some(observation) = observation {
        observation.observe_response(response);
    }
}

#[cfg(test)]
#[path = "response_metadata_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "response_metadata_fallback_tests.rs"]
mod fallback_tests;

#[cfg(test)]
#[path = "response_metadata_boundary_tests.rs"]
mod boundary_tests;
