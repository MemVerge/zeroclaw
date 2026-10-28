//! Local HTTP boundary proof. Synthetic payloads only; no model service or credentials.
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zeroclaw_api::model_provider::submission::*;
use zeroclaw_api::model_provider::{
    ChatMessage, ChatRequest, MessageContentField, MessageContentSource, ModelProvider,
    StreamOptions,
};
use zeroclaw_providers::compatible::{AuthStyle, OpenAiCompatibleModelProvider};

#[derive(Default)]
struct Capture {
    authorized: Mutex<Vec<Vec<u8>>>,
    received: Mutex<Vec<Vec<u8>>>,
    active: AtomicUsize,
    lease_observed: AtomicBool,
    dropped: AtomicUsize,
    deny: AtomicBool,
    events: Mutex<Vec<SubmissionEvent>>,
    started: tokio::sync::Notify,
    unknown: tokio::sync::Notify,
}

struct HeldOwnership(Arc<Capture>);
impl Drop for HeldOwnership {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct Host(Arc<Capture>);
impl RequestSubmissionAuthorizer for Host {
    fn required_headers(&self) -> &[&str] {
        &["x-fixture-context"]
    }
    fn limits(&self) -> SubmissionLimits {
        SubmissionLimits {
            max_body_bytes: 16_384,
            max_source_spans: 16,
        }
    }
    fn authorize(
        &self,
        request: SerializedSubmission<'_>,
    ) -> Result<SubmissionLease, SubmissionDenied> {
        assert_eq!(
            request.headers.single_value("x-fixture-context"),
            Ok(Some(b"bound".as_slice()))
        );
        self.0
            .authorized
            .lock()
            .unwrap()
            .push(request.body.to_vec());
        if self.0.deny.load(Ordering::SeqCst) {
            return Err(SubmissionDenied::PermitUnavailable);
        }
        self.0.active.fetch_add(1, Ordering::SeqCst);
        let capture = self.0.clone();
        Ok(
            SubmissionLease::hold(HeldOwnership(self.0.clone())).with_observer(Arc::new(
                move |event| {
                    capture.events.lock().unwrap().push(event);
                    if event == SubmissionEvent::DeliveryUnknown {
                        capture.unknown.notify_one();
                    }
                },
            )),
        )
    }
}

#[derive(Clone, Copy)]
enum ResponseMode {
    Json,
    Events,
    Redirect,
    Malformed,
    EmptyChoices,
    EmptyEvents,
    Pending,
}
struct ServerState {
    capture: Arc<Capture>,
    mode: ResponseMode,
}

async fn respond(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    assert_eq!(headers["x-fixture-context"], "bound");
    state.capture.received.lock().unwrap().push(body.to_vec());
    state.capture.lease_observed.store(
        state.capture.active.load(Ordering::SeqCst) == 1,
        Ordering::SeqCst,
    );
    match state.mode {
        ResponseMode::Json => axum::Json(serde_json::json!({"choices":[{"message":{"role":"assistant","content":"synthetic answer"},"finish_reason":"stop"}]})).into_response(),
        ResponseMode::Events => ([("content-type", "text/event-stream")],
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"synthetic answer\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n").into_response(),
        ResponseMode::Redirect => (StatusCode::TEMPORARY_REDIRECT, [("location", "/second-dispatch")]).into_response(),
        ResponseMode::Malformed => "synthetic malformed JSON".into_response(),
        ResponseMode::EmptyChoices => axum::Json(serde_json::json!({"choices":[]})).into_response(),
        ResponseMode::EmptyEvents => ([("content-type", "text/event-stream")], "data: [DONE]\n\n").into_response(),
        ResponseMode::Pending => { state.capture.started.notify_one(); std::future::pending::<Response>().await }
    }
}

struct LocalServer {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for LocalServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn server(capture: &Arc<Capture>, mode: ResponseMode) -> LocalServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new()
        .fallback(respond)
        .with_state(Arc::new(ServerState {
            capture: capture.clone(),
            mode,
        }));
    let task = zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    LocalServer { url, task }
}

fn provider(server: &LocalServer, capture: &Arc<Capture>) -> OpenAiCompatibleModelProvider {
    let headers = Arc::new(
        |_: &serde_json::Value, _: &[zeroclaw_api::model_provider::ProjectedMessageSource]| {
            let mut headers = HeaderMap::new();
            headers.insert("x-fixture-context", "bound".parse().unwrap());
            Ok(headers)
        },
    );
    OpenAiCompatibleModelProvider::builder("fixture")
        .display_name("fixture")
        .base_url(&server.url)
        .auth_style(AuthStyle::Bearer)
        .request_headers(Some(headers))
        .submission_authorizer(Some(Arc::new(Host(capture.clone()))))
        .build()
}

#[derive(Clone, Copy)]
enum Path {
    System,
    History,
    Tools,
    Structured,
    StreamSystem,
    StreamHistory,
    StreamStructured,
}

fn messages() -> Vec<ChatMessage> {
    let mut message = ChatMessage::user("synthetic question");
    message.content_sources = vec![MessageContentSource {
        source_id: 0,
        field: MessageContentField::Text,
        range: 0..message.content.len(),
    }];
    vec![message]
}

fn streaming() -> StreamOptions {
    StreamOptions {
        enabled: true,
        ..Default::default()
    }
}

async fn exercise(provider: &OpenAiCompatibleModelProvider, path: Path) -> anyhow::Result<()> {
    let messages = messages();
    match path {
        Path::System => {
            provider
                .chat_with_system(None, "synthetic question", "fixture", None)
                .await?;
        }
        Path::History => {
            provider
                .chat_with_history(&messages, "fixture", None)
                .await?;
        }
        Path::Tools => {
            provider.chat_with_tools(&messages, &[serde_json::json!({"type":"function","function":{"name":"fixture","parameters":{"type":"object"}}})], "fixture", None).await?;
        }
        Path::Structured => {
            provider
                .chat(
                    ChatRequest {
                        messages: &messages,
                        tools: None,
                        thinking: None,
                    },
                    "fixture",
                    None,
                )
                .await?;
        }
        Path::StreamSystem => {
            let mut stream = provider.stream_chat_with_system(
                None,
                "synthetic question",
                "fixture",
                None,
                streaming(),
            );
            while let Some(event) = stream.next().await {
                event?;
            }
        }
        Path::StreamHistory => {
            let mut stream =
                provider.stream_chat_with_history(&messages, "fixture", None, streaming());
            while let Some(event) = stream.next().await {
                event?;
            }
        }
        Path::StreamStructured => {
            let mut stream = provider.stream_chat(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                "fixture",
                None,
                streaming(),
            );
            while let Some(event) = stream.next().await {
                event?;
            }
        }
    }
    Ok(())
}

async fn assert_wire_binding(path: Path, mode: ResponseMode) {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, mode).await;
    let provider = provider(&server, &capture);
    exercise(&provider, path).await.unwrap();
    let authorized = capture.authorized.lock().unwrap();
    let received = capture.received.lock().unwrap();
    assert_eq!(authorized.len(), 1);
    assert_eq!(*authorized, *received);
    assert!(capture.lease_observed.load(Ordering::SeqCst));
    assert_eq!(capture.active.load(Ordering::SeqCst), 0);
    assert_eq!(capture.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        *capture.events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::ModelResponseObserved
        ]
    );
}

macro_rules! wire_case {
    ($name:ident, $path:ident, $mode:ident) => {
        #[tokio::test]
        async fn $name() {
            assert_wire_binding(Path::$path, ResponseMode::$mode).await;
        }
    };
}
wire_case!(
    system_dispatch_uses_the_exact_authorized_bytes,
    System,
    Json
);
wire_case!(
    history_dispatch_uses_the_exact_authorized_bytes,
    History,
    Json
);
wire_case!(tools_dispatch_uses_the_exact_authorized_bytes, Tools, Json);
wire_case!(
    structured_dispatch_uses_the_exact_authorized_bytes,
    Structured,
    Json
);
wire_case!(
    stream_system_dispatch_uses_the_exact_authorized_bytes,
    StreamSystem,
    Events
);
wire_case!(
    stream_history_dispatch_uses_the_exact_authorized_bytes,
    StreamHistory,
    Events
);
wire_case!(
    stream_structured_dispatch_uses_the_exact_authorized_bytes,
    StreamStructured,
    Events
);

#[tokio::test]
async fn native_tool_local_denial_does_not_fallback_or_send_http() {
    let capture = Arc::new(Capture::default());
    capture.deny.store(true, Ordering::SeqCst);
    let server = server(&capture, ResponseMode::Json).await;
    let provider = provider(&server, &capture);
    let error = exercise(&provider, Path::Tools).await.unwrap_err();
    assert!(zeroclaw_providers::reliable::is_local_submission_denied(
        &error
    ));
    assert_eq!(capture.authorized.lock().unwrap().len(), 1);
    assert!(capture.received.lock().unwrap().is_empty());
    assert_eq!(capture.dropped.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_redirect_cannot_dispatch_a_second_body_under_the_first_permit() {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, ResponseMode::Redirect).await;
    let provider = provider(&server, &capture);
    assert!(exercise(&provider, Path::History).await.is_err());
    assert_eq!(capture.authorized.lock().unwrap().len(), 1);
    assert_eq!(capture.received.lock().unwrap().len(), 1);
    assert_eq!(capture.dropped.load(Ordering::SeqCst), 1);
    assert!(
        !capture
            .events
            .lock()
            .unwrap()
            .contains(&SubmissionEvent::ModelResponseObserved)
    );
}

#[tokio::test]
async fn http_success_with_invalid_json_does_not_establish_model_visibility() {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, ResponseMode::Malformed).await;
    let provider = provider(&server, &capture);
    assert!(exercise(&provider, Path::Structured).await.is_err());
    assert_eq!(
        *capture.events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::DeliveryUnknown
        ]
    );
}

#[tokio::test]
async fn http_success_with_empty_choices_does_not_establish_model_visibility() {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, ResponseMode::EmptyChoices).await;
    let provider = provider(&server, &capture);
    assert!(exercise(&provider, Path::History).await.is_err());
    assert_eq!(
        *capture.events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::DeliveryUnknown
        ]
    );
}

#[tokio::test]
async fn a_bare_stream_done_marker_is_not_a_provider_model_response_frame() {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, ResponseMode::EmptyEvents).await;
    let provider = provider(&server, &capture);
    exercise(&provider, Path::StreamStructured).await.unwrap();
    assert_eq!(
        *capture.events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::HttpResponseObserved { status: 200 },
            SubmissionEvent::DeliveryUnknown
        ]
    );
}

#[tokio::test]
async fn cancelling_a_dispatched_request_releases_its_guard_without_inventing_visibility() {
    let capture = Arc::new(Capture::default());
    let server = server(&capture, ResponseMode::Pending).await;
    let provider = provider(&server, &capture);
    let call = zeroclaw_spawn::spawn!(async move { exercise(&provider, Path::History).await });
    capture.started.notified().await;
    assert_eq!(capture.active.load(Ordering::SeqCst), 1);
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    capture.unknown.notified().await;
    assert_eq!(capture.active.load(Ordering::SeqCst), 0);
    assert_eq!(capture.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        *capture.events.lock().unwrap(),
        [
            SubmissionEvent::Dispatched,
            SubmissionEvent::DeliveryUnknown
        ]
    );
}
