use super::*;
use crate::{
    compatible::{AuthStyle, OpenAiCompatibleModelProvider},
    traits::{ChatMessage, ModelProvider},
};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

fn bounds(bytes: usize) -> RequestSampleBounds {
    RequestSampleBounds {
        max_body_bytes: bytes,
        max_source_spans: 1,
    }
}

#[test]
fn counting_accepts_exact_limit_and_rejects_one_byte_less() {
    let body = serde_json::json!({"text":"汉字\\\""});
    let size = serde_json::to_vec(&body).unwrap().len();
    assert!(
        RequestSample::new(bounds(size), |_| {})
            .check(&body, 1)
            .is_ok()
    );
    assert_eq!(
        RequestSample::new(bounds(size - 1), |_| {}).check(&body, 1),
        Err(RequestSampleSkipReason::BodyLimit)
    );
}

#[test]
fn invalid_bounds_and_source_overflow_do_not_serialize() {
    struct NeverSerialize;
    impl Serialize for NeverSerialize {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            panic!("bounds must be checked first");
        }
    }
    for limit in [0, MAX_OBSERVED_BODY_BYTES + 1] {
        assert_eq!(
            RequestSample::new(bounds(limit), |_| {}).check(&NeverSerialize, 0),
            Err(RequestSampleSkipReason::InvalidBounds)
        );
    }
    assert_eq!(
        RequestSample::new(bounds(100), |_| {}).check(&NeverSerialize, 2),
        Err(RequestSampleSkipReason::SourceLimit)
    );
}

struct CountedRequest<'a>(&'a AtomicUsize);

impl Serialize for CountedRequest<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.fetch_add(1, Ordering::Relaxed);
        "test request".serialize(serializer)
    }
}

fn build_request(
    request: &impl Serialize,
    observer: Option<&RequestObserver>,
) -> anyhow::Result<reqwest::Request> {
    crate::compatible::request_with_sources(
        crate::compatible::RequestHooks {
            headers: None,
            observer,
            submission: None,
        },
        crate::extra_headers::ReservedHeaders::COMPATIBLE,
        &[],
        reqwest::Client::new().post("http://example.invalid"),
        request,
        &[],
    )?
    .build()
    .map_err(Into::into)
}

#[test]
fn selected_sample_adds_only_bounded_counting_and_one_value_materialization() {
    let serializations = AtomicUsize::new(0);
    let observer: RequestObserver = Arc::new(|| {
        Some(RequestSample::new(bounds(100), |outcome| {
            assert!(matches!(outcome, RequestSampleOutcome::Captured { .. }));
        }))
    });
    assert!(build_request(&CountedRequest(&serializations), Some(&observer)).is_ok());
    assert_eq!(serializations.load(Ordering::Relaxed), 3);
}

#[test]
fn preparation_panic_does_not_change_the_original_serialization_path() {
    let serializations = AtomicUsize::new(0);
    let observer: RequestObserver = Arc::new(|| panic!("fixture preparation failure"));
    assert!(build_request(&CountedRequest(&serializations), Some(&observer)).is_ok());
    assert_eq!(serializations.load(Ordering::Relaxed), 1);
}

#[test]
fn notification_panic_does_not_reject_the_original_request() {
    let serializations = AtomicUsize::new(0);
    let observer: RequestObserver = Arc::new(|| {
        Some(RequestSample::new(bounds(100), |_| {
            panic!("fixture notification failure")
        }))
    });
    assert!(build_request(&CountedRequest(&serializations), Some(&observer)).is_ok());
    assert_eq!(serializations.load(Ordering::Relaxed), 3);
}

struct FailOnSerialization<'a> {
    calls: &'a AtomicUsize,
    fail_on: usize,
}

impl Serialize for FailOnSerialization<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if call == self.fail_on {
            return Err(serde::ser::Error::custom("fixture serialization failure"));
        }
        "test request".serialize(serializer)
    }
}

#[test]
fn observation_serialization_failures_do_not_reject_the_transport_serialization() {
    for fail_on in [1, 2] {
        let calls = AtomicUsize::new(0);
        let reported = Arc::new(AtomicUsize::new(0));
        let observer: RequestObserver = Arc::new({
            let reported = Arc::clone(&reported);
            move || {
                let reported = Arc::clone(&reported);
                Some(RequestSample::new(bounds(100), move |outcome| {
                    assert!(matches!(
                        outcome,
                        RequestSampleOutcome::Skipped(RequestSampleSkipReason::Serialization)
                    ));
                    reported.fetch_add(1, Ordering::Relaxed);
                }))
            }
        });
        assert!(
            build_request(
                &FailOnSerialization {
                    calls: &calls,
                    fail_on
                },
                Some(&observer)
            )
            .is_ok()
        );
        assert_eq!(reported.load(Ordering::Relaxed), 1);
        assert_eq!(calls.load(Ordering::Relaxed), fail_on + 1);
    }
}

#[test]
fn legacy_header_errors_propagate_and_drop_the_selected_host_reservation() {
    struct Reservation(Arc<AtomicUsize>);
    impl Drop for Reservation {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let released = Arc::new(AtomicUsize::new(0));
    let observer: RequestObserver = Arc::new({
        let released = Arc::clone(&released);
        move || {
            let reservation = Reservation(Arc::clone(&released));
            Some(RequestSample::new(bounds(100), move |_| {
                drop(reservation);
                panic!("failed legacy callback must not notify capture");
            }))
        }
    });
    let callback: crate::compatible::RequestHeaders =
        Arc::new(|_, _| anyhow::bail!("fixture legacy failure"));
    let result = crate::compatible::request_with_sources(
        crate::compatible::RequestHooks {
            headers: Some(&callback),
            observer: Some(&observer),
            submission: None,
        },
        crate::extra_headers::ReservedHeaders::COMPATIBLE,
        &[],
        reqwest::Client::new().post("http://example.invalid"),
        &"test request",
        &[],
    );
    assert_eq!(result.err().unwrap().to_string(), "fixture legacy failure");
    assert_eq!(released.load(Ordering::Relaxed), 1);
}

#[test]
fn unsampled_observer_leaves_only_the_original_transport_serialization() {
    let serializations = AtomicUsize::new(0);
    let observer: RequestObserver = Arc::new(|| None);
    let builder = crate::compatible::request_with_sources(
        crate::compatible::RequestHooks {
            headers: None,
            observer: Some(&observer),
            submission: None,
        },
        crate::extra_headers::ReservedHeaders::COMPATIBLE,
        &[],
        reqwest::Client::new().post("http://example.invalid"),
        &CountedRequest(&serializations),
        &[],
    )
    .unwrap();
    assert!(builder.build().is_ok());
    assert_eq!(serializations.load(Ordering::Relaxed), 1);
}

#[test]
fn oversized_sample_counts_then_skips_value_materialization_without_rejecting_request() {
    let serializations = AtomicUsize::new(0);
    let completed = Arc::new(AtomicUsize::new(0));
    let observer: RequestObserver = Arc::new({
        let completed = Arc::clone(&completed);
        move || {
            let completed = Arc::clone(&completed);
            Some(RequestSample::new(bounds(1), move |outcome| {
                assert!(matches!(
                    outcome,
                    RequestSampleOutcome::Skipped(RequestSampleSkipReason::BodyLimit)
                ));
                completed.fetch_add(1, Ordering::Relaxed);
            }))
        }
    });
    let builder = crate::compatible::request_with_sources(
        crate::compatible::RequestHooks {
            headers: None,
            observer: Some(&observer),
            submission: None,
        },
        crate::extra_headers::ReservedHeaders::COMPATIBLE,
        &[],
        reqwest::Client::new().post("http://example.invalid"),
        &CountedRequest(&serializations),
        &[],
    )
    .unwrap();
    assert!(builder.build().is_ok());
    assert_eq!(completed.load(Ordering::Relaxed), 1);
    assert_eq!(serializations.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn observer_is_selected_before_serializing_and_does_not_change_wire_bytes() {
    use axum::{Router, extract::State, routing::post};
    use futures_util::StreamExt;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
    let router = Router::new().route("/chat/completions", post(|State(sender): State<tokio::sync::mpsc::Sender<Vec<u8>>>, body: axum::body::Bytes| async move {
        sender.send(body.to_vec()).await.unwrap();
        ([("content-type", "text/event-stream")], "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
    })).with_state(sender);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(Mutex::new(None));
    let selection_count = Arc::clone(&seen);
    let observed_body = Arc::clone(&captured);
    let observer: RequestObserver = Arc::new(move || {
        selection_count.fetch_add(1, Ordering::Relaxed);
        let observed_body = Arc::clone(&observed_body);
        Some(RequestSample::new(bounds(4096), move |outcome| {
            let RequestSampleOutcome::Captured {
                body,
                generated_headers,
                ..
            } = outcome
            else {
                panic!("small body must fit")
            };
            assert!(generated_headers.is_empty());
            *observed_body.lock().unwrap() = Some(body.clone());
        }))
    });
    let client = async {
        let mut baseline = None;
        for observe in [None, Some(observer)] {
            let provider = OpenAiCompatibleModelProvider::builder("test")
                .display_name("test")
                .base_url(&base)
                .auth_style(AuthStyle::Bearer)
                .credential(Some("fake-key"))
                .request_observer(observe)
                .build();
            let results = provider
                .stream_chat_with_history(
                    &[ChatMessage::user("汉字")],
                    "test",
                    None,
                    crate::traits::StreamOptions::default(),
                )
                .collect::<Vec<_>>()
                .await;
            assert!(results.iter().all(Result::is_ok));
            let raw = receiver.recv().await.unwrap();
            if let Some(previous) = baseline.as_ref() {
                assert_eq!(previous, &raw);
            } else {
                baseline = Some(raw);
            }
        }
        assert_eq!(seen.load(Ordering::Relaxed), 1);
        assert_eq!(
            captured.lock().unwrap().as_ref().unwrap(),
            &serde_json::from_slice::<Value>(&baseline.unwrap()).unwrap()
        );
    };
    tokio::select! {
        result = axum::serve(listener, router) => panic!("fixture server ended: {result:?}"),
        result = tokio::time::timeout(std::time::Duration::from_secs(5), client) => result.unwrap(),
    }
}
