//! Optional host-owned authorization of one immutable serialized request.
//! This port does not implement context policy, token counting, source storage,
//! billing, or evidence visibility. Absence retains the existing provider path.
use super::ProjectedMessageSource;
use std::sync::Arc;

/// Body-free facts for the exact authorized request. HTTP response observation
/// is deliberately distinct from a parsed model response. Neither is an effect
/// receipt, a billing settlement or proof that a model comprehended the input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionEvent {
    Dispatched,
    HttpResponseObserved { status: u16 },
    ModelResponseObserved,
    DeliveryUnknown,
}

pub type SubmissionObserver = Arc<dyn Fn(SubmissionEvent) + Send + Sync>;

/// Diagnostic reason only. No prompt, credential, endpoint or host error text
/// crosses this error boundary, and no reason authorizes an automatic retry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("local_request_submission_denied:{self:?}")]
pub enum SubmissionDenied {
    HostUnavailable,
    BindingChanged,
    PermitUnavailable,
    WorkLimit,
    UnsupportedBody,
    UnsupportedTransport,
    InvalidHeaders,
}

/// Safety caps on host inspection, not model windows or execution budgets. The
/// existing transport buffer is already serialized before these checks; these
/// limits do not cap the original DTO/transport allocation or serialization CPU.
#[derive(Clone, Copy, Debug)]
pub struct SubmissionLimits {
    pub max_body_bytes: usize,
    pub max_source_spans: usize,
}

/// Accesses a named final request header without copying/enumerating headers.
/// The edge adapter requires an explicit host-declared name, and rejects
/// duplicate values, the active credential header and common secret headers.
pub trait SubmissionHeaders {
    fn single_value(&self, name: &str) -> Result<Option<&[u8]>, SubmissionDenied>;
}

/// Borrowed actual transport bytes and the original projector's source mapping.
/// Neither bodies nor source mappings can be rewritten through this view.
pub struct SerializedSubmission<'a> {
    pub method: &'a str,
    pub destination: &'a str,
    pub body: &'a [u8],
    pub sources: &'a [ProjectedMessageSource],
    pub headers: &'a dyn SubmissionHeaders,
}

impl std::fmt::Debug for SerializedSubmission<'_> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output
            .debug_struct("SerializedSubmission")
            .field("body_bytes", &self.body.len())
            .field("source_spans", &self.sources.len())
            .finish_non_exhaustive()
    }
}

/// The host resolves live authority and consumes one permit here. A successful
/// callback is the dispatch linearization point: returned ownership must keep
/// its source/account/route/allowance contract valid through transport dispatch.
/// Callbacks must be bounded, nonblocking and concurrency-safe. The SDK holds
/// the lease until response headers or transport failure, and drops it on cancel.
/// A lease does not prove receipt, model visibility, effects or billing success.
pub trait RequestSubmissionAuthorizer: Send + Sync {
    fn limits(&self) -> SubmissionLimits;
    /// Protocol headers needed for this contract; default closed. This is a
    /// borrowed declaration, not permission to inspect credentials.
    fn required_headers(&self) -> &[&str] {
        &[]
    }
    fn authorize(
        &self,
        request: SerializedSubmission<'_>,
    ) -> Result<SubmissionLease, SubmissionDenied>;
}

/// Non-cloneable one-send ownership, not a reusable approval flag.
///
/// ```compile_fail
/// use zeroclaw_api::model_provider::submission::SubmissionLease;
/// let lease = SubmissionLease::hold(());
/// lease.release();
/// lease.release(); // already consumed
/// ```
pub struct SubmissionLease {
    ownership: Box<dyn Send>,
    observer: Option<SubmissionObserver>,
}

impl SubmissionLease {
    pub fn hold(ownership: impl Send + 'static) -> Self {
        Self {
            ownership: Box::new(ownership),
            observer: None,
        }
    }
    pub fn release(self) {
        drop(self.ownership);
    }

    /// Observers must retain metadata only, be bounded/nonblocking, and never
    /// place user data in panic messages. Process panic hooks remain process-owned.
    pub fn with_observer(mut self, observer: SubmissionObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Consumes the one-send lease at an edge transport. The opaque ownership
    /// must survive dispatch; the optional observer follows that same response,
    /// not a guessed last request or a diagnostics sampling sequence.
    pub fn into_parts(self) -> (Box<dyn Send>, Option<SubmissionObserver>) {
        (self.ownership, self.observer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Ownership(Arc<AtomicUsize>);
    impl Drop for Ownership {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn explicit_release_drops_transport_ownership_exactly_once() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let lease = SubmissionLease::hold(Ownership(dropped.clone()));
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        lease.release();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancellation_drop_releases_transport_ownership_without_a_response() {
        let dropped = Arc::new(AtomicUsize::new(0));
        drop(SubmissionLease::hold(Ownership(dropped.clone())));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    struct NoHeaders;
    impl SubmissionHeaders for NoHeaders {
        fn single_value(&self, _: &str) -> Result<Option<&[u8]>, SubmissionDenied> {
            Ok(None)
        }
    }

    #[test]
    fn request_debug_does_not_print_endpoint_prompt_or_headers() {
        let request = SerializedSubmission {
            method: "POST",
            destination: "https://private.invalid",
            body: b"private prompt",
            sources: &[],
            headers: &NoHeaders,
        };
        let diagnostic = format!("{request:?}");
        assert!(!diagnostic.contains("private"));
        assert!(diagnostic.contains("body_bytes"));
    }

    #[test]
    fn local_denial_has_no_remote_status_or_retry_message() {
        let error =
            super::super::StreamError::SubmissionDenied(SubmissionDenied::PermitUnavailable);
        assert_eq!(
            error.to_string(),
            "local_request_submission_denied:PermitUnavailable"
        );
    }
}
