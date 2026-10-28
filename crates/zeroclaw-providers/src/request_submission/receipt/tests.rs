use super::*;
use std::sync::{Mutex, atomic::AtomicUsize};

struct Ownership(Arc<AtomicUsize>);
impl Drop for Ownership {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

type Events = Arc<Mutex<Vec<SubmissionEvent>>>;
fn fixture() -> (AuthorizedSubmission, Events, Arc<AtomicUsize>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let dropped = Arc::new(AtomicUsize::new(0));
    let lease = SubmissionLease::hold(Ownership(dropped.clone()))
        .with_observer(Arc::new(move |event| captured.lock().unwrap().push(event)));
    (AuthorizedSubmission::new(lease), events, dropped)
}

#[test]
fn cancelling_before_dispatch_releases_ownership_without_visibility() {
    let (submission, events, dropped) = fixture();
    drop(submission);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(*events.lock().unwrap(), [SubmissionEvent::DeliveryUnknown]);
}

#[test]
fn dispatch_is_not_a_model_response_receipt() {
    let (submission, events, _) = fixture();
    submission.dispatch().unwrap();
    drop(submission);
    assert_eq!(
        *events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::DeliveryUnknown
        ]
    );
}

#[test]
fn http_success_without_parsed_model_response_remains_visibility_unknown() {
    let (submission, events, _) = fixture();
    submission.dispatch().unwrap();
    drop(submission.complete(200));
    assert_eq!(
        *events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::DeliveryUnknown
        ]
    );
}

#[test]
fn a_parsed_response_uses_its_own_request_receipt_and_releases_send_ownership() {
    let (submission, events, dropped) = fixture();
    submission.dispatch().unwrap();
    let receipt = submission.complete(200).unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    receipt.model_response().unwrap();
    drop(receipt);
    assert_eq!(
        *events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::ModelResponseObserved
        ]
    );
}

#[test]
fn non_success_http_status_cannot_be_promoted_to_model_visibility() {
    let (submission, events, _) = fixture();
    submission.dispatch().unwrap();
    let receipt = submission.complete(402).unwrap();
    assert_eq!(
        receipt.model_response(),
        Err(SubmissionDenied::PermitUnavailable)
    );
    drop(receipt);
    assert!(
        !events
            .lock()
            .unwrap()
            .contains(&SubmissionEvent::ModelResponseObserved)
    );
}

#[test]
fn repeated_stream_frames_or_receipt_clones_do_not_duplicate_visibility() {
    let (submission, events, _) = fixture();
    submission.dispatch().unwrap();
    let receipt = submission.complete(200).unwrap();
    receipt.model_response().unwrap();
    receipt.clone().model_response().unwrap();
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == SubmissionEvent::ModelResponseObserved)
            .count(),
        1
    );
}

#[test]
fn two_inflight_responses_do_not_borrow_the_last_requests_visibility() {
    let (first, first_events, _) = fixture();
    let (second, second_events, _) = fixture();
    first.dispatch().unwrap();
    second.dispatch().unwrap();
    let first = first.complete(200).unwrap();
    let second = second.complete(200).unwrap();
    first.model_response().unwrap();
    drop((first, second));
    assert!(
        first_events
            .lock()
            .unwrap()
            .contains(&SubmissionEvent::ModelResponseObserved)
    );
    assert!(
        !second_events
            .lock()
            .unwrap()
            .contains(&SubmissionEvent::ModelResponseObserved)
    );
    assert!(
        second_events
            .lock()
            .unwrap()
            .contains(&SubmissionEvent::DeliveryUnknown)
    );
}

#[test]
fn a_second_dispatch_is_denied_before_transport_ownership_can_be_reused() {
    let (submission, _, _) = fixture();
    submission.dispatch().unwrap();
    assert_eq!(
        submission.dispatch(),
        Err(SubmissionDenied::PermitUnavailable)
    );
}

#[test]
fn dispatch_observer_panic_denies_send_and_does_not_leak_the_guard() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let lease =
        SubmissionLease::hold(Ownership(dropped.clone())).with_observer(Arc::new(|event| {
            if event == SubmissionEvent::Dispatched {
                panic!("synthetic observer failure");
            }
        }));
    let submission = AuthorizedSubmission::new(lease);
    assert_eq!(
        submission.dispatch(),
        Err(SubmissionDenied::HostUnavailable)
    );
    drop(submission);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn malformed_observer_state_does_not_authorize_http_before_dispatch() {
    let receipt = SubmissionReceipt::new(Arc::new(|_| {}));
    assert_eq!(
        receipt.http_response(200),
        Err(SubmissionDenied::PermitUnavailable)
    );
    assert_eq!(
        receipt.model_response(),
        Err(SubmissionDenied::PermitUnavailable)
    );
}
