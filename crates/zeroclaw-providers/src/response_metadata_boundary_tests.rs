use super::{ResponseMetadata, ResponseMetadataObserver, ResponseMetadataPhase, tests::provider};
use axum::{body::Body, extract::State, http::Response, routing::any};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

const FIRST_ID: &str = "00000000-0000-4000-8000-000000000001";
const SECOND_ID: &str = "00000000-0000-4000-8000-000000000002";
const ROUTES: [&str; 4] = ["openai", "responses", "anthropic", "compatible"];

struct ServerState {
    replies: Mutex<VecDeque<Response<Body>>>,
    requests: Mutex<Vec<Value>>,
}

struct MockServer {
    base: String,
    state: Arc<ServerState>,
    _guard: crate::stream_guard::AbortOnDrop,
}

async fn serve_request(
    State(state): State<Arc<ServerState>>,
    request: axum::extract::Request,
) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let body = axum::body::to_bytes(body, 65536)
        .await
        .expect("request body");
    state.requests.lock().expect("request lock").push(json!({
        "method": parts.method.as_str(),
        "path": parts.uri.path(),
        "authorization": parts.headers.get("authorization").map(|value| value.to_str().expect("authorization")),
        "x-api-key": parts.headers.get("x-api-key").map(|value| value.to_str().expect("API key")),
        "body": serde_json::from_slice::<Value>(&body).expect("request JSON"),
    }));
    state
        .replies
        .lock()
        .expect("reply lock")
        .pop_front()
        .expect("planned response")
}

async fn mock_server(replies: Vec<Response<Body>>) -> MockServer {
    let state = Arc::new(ServerState {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(Vec::new()),
    });
    let app = axum::Router::new()
        .fallback(any(serve_request))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let base = format!("http://{}", listener.local_addr().expect("server address"));
    let task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve mock HTTP") });
    MockServer {
        base,
        state,
        _guard: crate::stream_guard::AbortOnDrop::new(task.abort_handle()),
    }
}

fn response(body: String, request_id: Option<&str>) -> Response<Body> {
    let mut builder = Response::builder().header("content-type", "application/json");
    if let Some(id) = request_id {
        builder = builder.header("x-membox-request-id", id);
    }
    builder.body(Body::from(body)).expect("model response")
}

fn success(route: &str, text: &str, request_id: Option<&str>) -> Response<Body> {
    let body = match route {
        "responses" => json!({"output_text": text}),
        "anthropic" => json!({"content": [{"type": "text", "text": text}]}),
        _ => json!({"choices": [{"message": {"role": "assistant", "content": text}}]}),
    };
    response(body.to_string(), request_id)
}

fn redirect() -> Response<Body> {
    Response::builder()
        .status(307)
        .header("location", "/final")
        .header("x-membox-request-id", FIRST_ID)
        .body(Body::empty())
        .expect("redirect")
}

fn record() -> (ResponseMetadataObserver, mpsc::Receiver<ResponseMetadata>) {
    let (sender, receiver) = mpsc::channel();
    let observer = ResponseMetadataObserver::new(["x-membox-request-id"], move |metadata| {
        sender.send(metadata).expect("metadata receiver");
    });
    (observer, receiver)
}

async fn chat(provider: &dyn crate::traits::ModelProvider) -> anyhow::Result<String> {
    tokio::time::timeout(
        Duration::from_secs(5),
        provider.chat_with_system(None, "test", "test-model", None),
    )
    .await
    .expect("model call timeout")
}

fn assert_lifecycle(events: &[ResponseMetadata], request_id: Option<&str>) {
    use ResponseMetadataPhase::{Completed, Headers, Started};
    assert_eq!(
        events.iter().map(|event| event.phase).collect::<Vec<_>>(),
        [Started, Headers, Completed]
    );
    assert!(
        events
            .iter()
            .all(|event| event.observation_id == events[0].observation_id)
    );
    assert!(events[0].headers.is_empty());
    assert_eq!(
        events[1]
            .headers
            .get("x-membox-request-id")
            .map(String::as_str),
        request_id
    );
    assert_eq!(events[1].headers, events[2].headers);
}

#[tokio::test]
async fn automatic_redirect_preserves_requests_and_observes_only_returned_headers() {
    for route in ROUTES {
        for final_id in [Some(SECOND_ID), None] {
            let (observer, receiver) = record();
            let observed = mock_server(vec![redirect(), success(route, "ok", final_id)]).await;
            let client = provider(route, &observed.base, Some(observer));
            assert_eq!(
                chat(client.as_ref()).await.expect("redirect succeeds"),
                "ok"
            );
            assert_lifecycle(&receiver.try_iter().collect::<Vec<_>>(), final_id);

            let disabled = mock_server(vec![redirect(), success(route, "ok", final_id)]).await;
            let client = provider(route, &disabled.base, None);
            assert_eq!(
                chat(client.as_ref())
                    .await
                    .expect("unobserved redirect succeeds"),
                "ok"
            );
            let requests = observed.state.requests.lock().expect("observed requests");
            assert_eq!(requests.len(), 2);
            assert_eq!(
                *requests,
                *disabled.state.requests.lock().expect("disabled requests")
            );
        }
    }
}

#[tokio::test]
async fn explicit_send_after_parse_failure_gets_a_new_observation_without_stale_headers() {
    for route in ROUTES {
        let server = mock_server(vec![
            response("invalid JSON".into(), Some(FIRST_ID)),
            success(route, "repaired", None),
        ])
        .await;
        let (observer, receiver) = record();
        let client = provider(route, &server.base, Some(observer));
        assert!(chat(client.as_ref()).await.is_err());
        assert_eq!(
            chat(client.as_ref()).await.expect("next send succeeds"),
            "repaired"
        );
        let events = receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(events.len(), 6);
        assert_lifecycle(&events[..3], Some(FIRST_ID));
        assert_lifecycle(&events[3..], None);
        assert_ne!(events[0].observation_id, events[3].observation_id);
    }
}

#[tokio::test]
async fn concurrent_provider_instances_keep_their_own_response_metadata() {
    for route in ROUTES {
        let server = mock_server(vec![
            success(route, FIRST_ID, Some(FIRST_ID)),
            success(route, SECOND_ID, Some(SECOND_ID)),
        ])
        .await;
        let (first_observer, first_events) = record();
        let (second_observer, second_events) = record();
        let first = provider(route, &server.base, Some(first_observer));
        let second = provider(route, &server.base, Some(second_observer));
        let (first_result, second_result) =
            tokio::join!(chat(first.as_ref()), chat(second.as_ref()));
        let first_result = first_result.expect("first call");
        let second_result = second_result.expect("second call");
        assert_ne!(first_result, second_result);
        let first_events = first_events.try_iter().collect::<Vec<_>>();
        let second_events = second_events.try_iter().collect::<Vec<_>>();
        assert_lifecycle(&first_events, Some(&first_result));
        assert_lifecycle(&second_events, Some(&second_result));
        assert_ne!(
            first_events[0].observation_id,
            second_events[0].observation_id
        );
    }
}

#[cfg(panic = "unwind")]
#[tokio::test]
async fn unwinding_callback_panics_leave_real_model_calls_successful() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for route in ROUTES {
        let server = mock_server(vec![success(route, "ok", Some(FIRST_ID))]).await;
        let callbacks = Arc::new(AtomicUsize::new(0));
        let count = callbacks.clone();
        let observer = ResponseMetadataObserver::new(["x-membox-request-id"], move |_| {
            count.fetch_add(1, Ordering::SeqCst);
            panic!("test callback failure");
        });
        let client = provider(route, &server.base, Some(observer));
        assert_eq!(
            chat(client.as_ref())
                .await
                .expect("model response survives"),
            "ok"
        );
        assert_eq!(callbacks.load(Ordering::SeqCst), 3);
    }
}
