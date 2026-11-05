//! Local HTTP only; response caps apply before whole completion allocation.
use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::Response,
};
use futures_util::stream;
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use zeroclaw_api::model_provider::{ChatMessage, ChatRequest, ModelProvider};
use zeroclaw_providers::compatible::{
    AuthStyle, CompletionBodyLimitExceeded, OpenAiCompatibleModelProvider,
};

const JSON: &str =
    r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;

struct Fixture {
    body: String,
    status: StatusCode,
    chunked: bool,
    requests: AtomicUsize,
}

async fn respond(State(fixture): State<Arc<Fixture>>) -> Response {
    fixture.requests.fetch_add(1, Ordering::SeqCst);
    let body = if fixture.chunked {
        let chunks = fixture
            .body
            .as_bytes()
            .chunks(7)
            .map(|chunk| Ok::<_, std::io::Error>(Bytes::copy_from_slice(chunk)))
            .collect::<Vec<_>>();
        Body::from_stream(stream::iter(chunks))
    } else {
        Body::from(fixture.body.clone())
    };
    Response::builder()
        .status(fixture.status)
        .header("content-type", "application/json")
        .body(body)
        .unwrap()
}

struct LocalServer {
    url: String,
    fixture: Arc<Fixture>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for LocalServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn server(fixture: Fixture) -> LocalServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let fixture = Arc::new(fixture);
    let app = Router::new().fallback(respond).with_state(fixture.clone());
    let task = zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    LocalServer { url, fixture, task }
}

fn fixture(body: &str) -> Fixture {
    Fixture {
        body: body.into(),
        status: StatusCode::OK,
        chunked: false,
        requests: AtomicUsize::new(0),
    }
}

fn provider(server: &LocalServer, maximum: Option<usize>) -> OpenAiCompatibleModelProvider {
    OpenAiCompatibleModelProvider::builder("fixture")
        .display_name("bounded fixture")
        .base_url(&server.url)
        .auth_style(AuthStyle::Bearer)
        .completion_body_limit(maximum.and_then(NonZeroUsize::new))
        .build()
}

#[tokio::test]
async fn known_length_over_limit_fails_before_json_parse_and_returns_no_payload() {
    let server = server(fixture(JSON)).await;
    let error = provider(&server, Some(JSON.len() - 1))
        .chat(
            ChatRequest {
                messages: &[ChatMessage::user("ask")],
                tools: None,
                thinking: None,
            },
            "fixture",
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<CompletionBodyLimitExceeded>(),
        Some(&CompletionBodyLimitExceeded {
            maximum: JSON.len() - 1
        })
    );
    assert!(!error.to_string().contains("choices"));
    assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn chunked_unknown_length_is_checked_cumulatively_before_each_append() {
    let mut response = fixture(JSON);
    response.chunked = true;
    let server = server(response).await;
    let error = provider(&server, Some(JSON.len() - 1))
        .chat_with_history(&[ChatMessage::user("ask")], "fixture", None)
        .await
        .unwrap_err();
    assert!(
        error
            .downcast_ref::<CompletionBodyLimitExceeded>()
            .is_some()
    );
}

#[tokio::test]
async fn exact_serialized_byte_boundary_succeeds_for_structured_completion() {
    let server = server(fixture(JSON)).await;
    let response = provider(&server, Some(JSON.len()))
        .chat(
            ChatRequest {
                messages: &[ChatMessage::user("ask")],
                tools: None,
                thinking: None,
            },
            "fixture",
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.text.as_deref(), Some("ok"));
}

#[tokio::test]
async fn chunked_exact_serialized_boundary_succeeds_without_truncating_utf8() {
    let body = JSON.replace("ok", "中文🙂");
    let mut response = fixture(&body);
    response.chunked = true;
    let server = server(response).await;
    let result = provider(&server, Some(body.len()))
        .chat_with_history(&[ChatMessage::user("ask")], "fixture", None)
        .await
        .unwrap();
    assert_eq!(result, "中文🙂");
}

#[tokio::test]
async fn system_completion_uses_the_same_bounded_reader() {
    let server = server(fixture(JSON)).await;
    let error = provider(&server, Some(1))
        .chat_with_system(None, "ask", "fixture", None)
        .await
        .unwrap_err();
    assert!(
        error
            .downcast_ref::<CompletionBodyLimitExceeded>()
            .is_some()
    );
}

#[tokio::test]
async fn tools_completion_uses_the_same_bounded_reader() {
    let server = server(fixture(JSON)).await;
    let error = provider(&server, Some(1))
        .chat_with_tools(&[ChatMessage::user("ask")], &[], "fixture", None)
        .await
        .unwrap_err();
    assert!(
        error
            .downcast_ref::<CompletionBodyLimitExceeded>()
            .is_some()
    );
}

#[tokio::test]
async fn http_error_bodies_are_bounded_on_every_completion_entrypoint() {
    let mut response = fixture("private oversized response");
    response.status = StatusCode::BAD_REQUEST;
    let server = server(response).await;
    let provider = provider(&server, Some(1));
    let messages = [ChatMessage::user("ask")];
    let errors = [
        provider
            .chat_with_system(None, "ask", "fixture", None)
            .await
            .unwrap_err(),
        provider
            .chat_with_history(&messages, "fixture", None)
            .await
            .unwrap_err(),
        provider
            .chat_with_tools(&messages, &[], "fixture", None)
            .await
            .unwrap_err(),
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
            .await
            .unwrap_err(),
    ];
    assert!(errors.iter().all(|error| {
        error
            .downcast_ref::<CompletionBodyLimitExceeded>()
            .is_some()
    }));
}

#[tokio::test]
async fn none_preserves_the_legacy_unbounded_completion_path() {
    let body = JSON.replace("ok", &"x".repeat(4096));
    let server = server(fixture(&body)).await;
    let response = provider(&server, None)
        .chat_with_history(&[ChatMessage::user("ask")], "fixture", None)
        .await
        .unwrap();
    assert_eq!(response.len(), 4096);
}

#[tokio::test]
async fn tools_free_strict_caller_does_not_implicitly_retry_schema_errors() {
    let mut response = fixture(r#"{"error":{"message":"unknown parameter: tools"}}"#);
    response.status = StatusCode::BAD_REQUEST;
    let server = server(response).await;
    let provider = OpenAiCompatibleModelProvider::builder("fixture")
        .display_name("strict fixture")
        .base_url(&server.url)
        .auth_style(AuthStyle::Bearer)
        .without_completion_fallbacks()
        .build();
    assert!(
        provider
            .chat(
                ChatRequest {
                    messages: &[ChatMessage::user("ask")],
                    tools: None,
                    thinking: None
                },
                "fixture",
                None
            )
            .await
            .is_err()
    );
    assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn legacy_caller_keeps_its_existing_schema_history_fallback() {
    let mut response = fixture(r#"{"error":{"message":"unknown parameter: tools"}}"#);
    response.status = StatusCode::BAD_REQUEST;
    let server = server(response).await;
    assert!(
        provider(&server, None)
            .chat(
                ChatRequest {
                    messages: &[ChatMessage::user("ask")],
                    tools: None,
                    thinking: None
                },
                "fixture",
                None
            )
            .await
            .is_err()
    );
    assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn unknown_length_error_payload_is_bounded_without_a_schema_retry() {
    let mut response = fixture(r#"{"error":{"message":"unknown parameter: tools"}}"#);
    response.status = StatusCode::BAD_REQUEST;
    response.chunked = true;
    let server = server(response).await;
    let error = provider(&server, Some(8))
        .chat(
            ChatRequest {
                messages: &[ChatMessage::user("ask")],
                tools: None,
                thinking: None,
            },
            "fixture",
            None,
        )
        .await
        .unwrap_err();
    assert!(
        error
            .downcast_ref::<CompletionBodyLimitExceeded>()
            .is_some()
    );
    assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
}
