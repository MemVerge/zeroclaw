//! One exact response carries its own body-free receipt in reqwest extensions.
//! No callback sequence, request-body equality or "last request" guess is used.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroclaw_api::model_provider::submission::*;

pub(crate) struct AuthorizedSubmission {
    ownership: Box<dyn Send>,
    receipt: Option<SubmissionReceipt>,
}

#[derive(Clone)]
pub(crate) struct SubmissionReceipt(Arc<ReceiptState>);

struct ReceiptState {
    observer: SubmissionObserver,
    dispatched: AtomicBool,
    http_success: AtomicBool,
    model_observed: AtomicBool,
    invalid: AtomicBool,
}

impl AuthorizedSubmission {
    pub(crate) fn new(lease: SubmissionLease) -> Self {
        let (ownership, observer) = lease.into_parts();
        Self {
            ownership,
            receipt: observer.map(SubmissionReceipt::new),
        }
    }
    pub(crate) fn dispatch(&self) -> Result<(), SubmissionDenied> {
        match &self.receipt {
            Some(receipt) => receipt.dispatch(),
            None => Ok(()),
        }
    }
    pub(crate) fn complete(self, status: u16) -> Option<SubmissionReceipt> {
        let Self { ownership, receipt } = self;
        let receipt = receipt.filter(|receipt| match receipt.http_response(status) {
            Ok(()) => true,
            Err(error) => {
                log_failure(error);
                false
            }
        });
        drop(ownership);
        receipt
    }
}

impl SubmissionReceipt {
    fn new(observer: SubmissionObserver) -> Self {
        Self(Arc::new(ReceiptState {
            observer,
            dispatched: AtomicBool::new(false),
            http_success: AtomicBool::new(false),
            model_observed: AtomicBool::new(false),
            invalid: AtomicBool::new(false),
        }))
    }
    fn dispatch(&self) -> Result<(), SubmissionDenied> {
        if self.0.dispatched.swap(true, Ordering::AcqRel) {
            return Err(SubmissionDenied::PermitUnavailable);
        }
        self.0.emit(SubmissionEvent::Dispatched)
    }
    fn http_response(&self, status: u16) -> Result<(), SubmissionDenied> {
        if !self.0.dispatched.load(Ordering::Acquire) {
            return Err(SubmissionDenied::PermitUnavailable);
        }
        self.0
            .http_success
            .store((200..300).contains(&status), Ordering::Release);
        self.0
            .emit(SubmissionEvent::HttpResponseObserved { status })
    }
    pub(crate) fn model_response(&self) -> Result<(), SubmissionDenied> {
        if !self.0.http_success.load(Ordering::Acquire) || self.0.invalid.load(Ordering::Acquire) {
            return Err(SubmissionDenied::PermitUnavailable);
        }
        if self.0.model_observed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.0.emit(SubmissionEvent::ModelResponseObserved)
    }
}

impl ReceiptState {
    fn emit(&self, event: SubmissionEvent) -> Result<(), SubmissionDenied> {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.observer)(event)));
        if result.is_err() {
            self.invalid.store(true, Ordering::Release);
            return Err(SubmissionDenied::HostUnavailable);
        }
        Ok(())
    }
}

impl Drop for ReceiptState {
    fn drop(&mut self) {
        if (!self.model_observed.load(Ordering::Acquire) || self.invalid.load(Ordering::Acquire))
            && let Err(error) = self.emit(SubmissionEvent::DeliveryUnknown)
        {
            log_failure(error);
        }
    }
}

pub(crate) fn from_response(response: &reqwest::Response) -> Option<SubmissionReceipt> {
    response.extensions().get::<SubmissionReceipt>().cloned()
}

pub(crate) fn observe_model_response(receipt: Option<&SubmissionReceipt>) {
    if let Some(receipt) = receipt
        && let Err(error) = receipt.model_response()
    {
        log_failure(error);
    }
}

fn log_failure(error: SubmissionDenied) {
    ::zeroclaw_log::record!(
        WARN,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(serde_json::json!({"reason":format!("{error:?}")})),
        "Submission event observer failed"
    );
}

#[cfg(test)]
mod tests;
