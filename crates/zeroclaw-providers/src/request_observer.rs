//! Bounded, opt-in observation of Chat Completions projections.
//!
//! A host reserves its diagnostic capacity in `RequestObserver`, before the SDK
//! creates an observation-only JSON Value. Returning None leaves the existing
//! serialization path alone. This is not an HTTP transport receipt or a request
//! mutation/authorization hook; Responses and native non-compatible APIs do not
//! invoke it. Callbacks must be quick, nonblocking and concurrency-safe.
//!
//! Selected samples add a bounded counting serialization before materialization.
//! The original provider DTOs, source projection and transport body still have
//! their existing allocations; these bounds are not a cap on total request RAM
//! or CPU. Existing RequestHeaders callbacks still require their original Value
//! even when observation is skipped. With neither a legacy callback nor a
//! selected sample, no extra serialization or Value is introduced.
use std::{io, sync::Arc};

use reqwest::header::HeaderMap;
use serde::Serialize;
use serde_json::Value;
use zeroclaw_api::model_provider::ProjectedMessageSource;

pub(crate) mod request;

/// Called once per supported request preparation, including host retries. This
/// may precede a failed send, not necessarily a server-visible request. The host
/// owns cadence, queue capacity and execution allowances, not the SDK. Panics are
/// contained under panic=unwind only, just like response metadata observation.
pub type RequestObserver = Arc<dyn Fn() -> Option<RequestSample> + Send + Sync>;

/// Hard observation ceilings, not model-context limits. They do not restrict the
/// original request or the existing fallible RequestHeaders callback.
pub const MAX_OBSERVED_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_OBSERVED_SOURCE_SPANS: usize = 16_384;

#[derive(Clone, Copy, Debug)]
pub struct RequestSampleBounds {
    pub max_body_bytes: usize,
    pub max_source_spans: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestSampleSkipReason {
    InvalidBounds,
    SourceLimit,
    BodyLimit,
    Serialization,
}

pub enum RequestSampleOutcome<'a> {
    Captured {
        /// Provider-projected body after extra_body merging. Borrowed only for
        /// this notification; SDK serializes the original request for transport.
        body: &'a Value,
        sources: &'a [ProjectedMessageSource],
        /// Filtered callback-generated headers only, not static/auth headers,
        /// the final wire envelope or downstream proxy rewrites.
        generated_headers: &'a HeaderMap,
    },
    Skipped(RequestSampleSkipReason),
}

type SampleCallback = Box<dyn FnOnce(RequestSampleOutcome<'_>) + Send>;

/// A one-send reservation. The callback can own a host queue permit; dropping it
/// releases that reservation even if serialization or a legacy callback fails.
/// Bounds and this callback are created here; no SDK policy cache is maintained.
pub struct RequestSample {
    bounds: RequestSampleBounds,
    callback: SampleCallback,
}

impl RequestSample {
    pub fn new(
        bounds: RequestSampleBounds,
        callback: impl FnOnce(RequestSampleOutcome<'_>) + Send + 'static,
    ) -> Self {
        Self {
            bounds,
            callback: Box::new(callback),
        }
    }

    pub(crate) fn check(
        &self,
        request: &impl Serialize,
        sources: usize,
    ) -> Result<(), RequestSampleSkipReason> {
        let bounds = self.bounds;
        if bounds.max_body_bytes == 0
            || bounds.max_body_bytes > MAX_OBSERVED_BODY_BYTES
            || bounds.max_source_spans > MAX_OBSERVED_SOURCE_SPANS
        {
            return Err(RequestSampleSkipReason::InvalidBounds);
        }
        if sources > bounds.max_source_spans {
            return Err(RequestSampleSkipReason::SourceLimit);
        }
        let mut writer = BoundedCounter {
            remaining: bounds.max_body_bytes,
            exceeded: false,
        };
        serde_json::to_writer(&mut writer, request).map_err(|_| {
            if writer.exceeded {
                RequestSampleSkipReason::BodyLimit
            } else {
                RequestSampleSkipReason::Serialization
            }
        })
    }

    pub(crate) fn notify(self, outcome: RequestSampleOutcome<'_>) {
        contain_panic(|| (self.callback)(outcome));
    }
}

pub(crate) fn prepare(observer: Option<&RequestObserver>) -> Option<RequestSample> {
    contain_panic(|| observer.and_then(|observer| observer())).flatten()
}

fn contain_panic<T>(callback: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)) {
        Ok(value) => Some(value),
        Err(_) => {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                "Request projection observer panicked"
            );
            None
        }
    }
}

struct BoundedCounter {
    remaining: usize,
    exceeded: bool,
}

impl io::Write for BoundedCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self.remaining.checked_sub(bytes.len()) {
            Some(remaining) => {
                self.remaining = remaining;
                Ok(bytes.len())
            }
            None => {
                self.exceeded = true;
                Err(io::Error::other("request observation body limit exceeded"))
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
