//! Shared completion control is exercised through the public provider trait.
use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::StatusCode,
    response::Response,
};
use futures_util::{StreamExt, stream};
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use zeroclaw_api::model_provider::{
    ChatMessage, ChatRequest, CompletionControl, ModelProvider, StreamOptions,
};
use zeroclaw_providers::{
    anthropic::AnthropicModelProvider,
    compatible::{AuthStyle, CompletionBodyLimitExceeded, OpenAiCompatibleModelProvider},
    openai::{OpenAiModelProvider, OpenAiResponsesModelProvider},
};

#[derive(Clone, Copy, Debug)]
enum Protocol {
    Compatible,
    OpenAi,
    Responses,
    Anthropic,
}

impl Protocol {
    fn response(self) -> &'static str {
        match self {
            Self::Compatible | Self::OpenAi => {
                r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#
            }
            Self::Responses => {
                r#"{"output_text":"ok","output":[],"usage":{"input_tokens":3,"output_tokens":2}}"#
            }
            Self::Anthropic => {
                r#"{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":3,"output_tokens":2}}"#
            }
        }
    }

    fn provider(self, url: &str) -> Box<dyn ModelProvider> {
        match self {
            Self::Compatible => Box::new(
                OpenAiCompatibleModelProvider::builder("test")
                    .display_name("fixture")
                    .base_url(url)
                    .credential(Some("fixture-key"))
                    .auth_style(AuthStyle::Bearer)
                    .build(),
            ),
            Self::OpenAi => Box::new(
                OpenAiModelProvider::builder("test")
                    .base_url(url)
                    .credential(Some("fixture-key"))
                    .build(),
            ),
            Self::Responses => Box::new(
                OpenAiResponsesModelProvider::builder("test")
                    .api_url(url)
                    .credential(Some("fixture-key"))
                    .build(),
            ),
            Self::Anthropic => Box::new(
                AnthropicModelProvider::builder("test")
                    .base_url(url)
                    .credential(Some("fixture-key"))
                    .build(),
            ),
        }
    }
}

const PROTOCOLS: [Protocol; 4] = [
    Protocol::Compatible,
    Protocol::OpenAi,
    Protocol::Responses,
    Protocol::Anthropic,
];

struct Fixture {
    body: String,
    status: StatusCode,
    requests: AtomicUsize,
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

async fn respond(State(fixture): State<Arc<Fixture>>) -> Response {
    fixture.requests.fetch_add(1, Ordering::SeqCst);
    let chunks: Vec<_> = fixture
        .body
        .as_bytes()
        .chunks(7)
        .map(|chunk| Ok::<_, std::io::Error>(Bytes::copy_from_slice(chunk)))
        .collect();
    let mut response = Response::builder()
        .status(fixture.status)
        .header("content-type", "application/json");
    if fixture.status.is_redirection() {
        response = response.header("location", "/redirected");
    }
    response
        .body(Body::from_stream(stream::iter(chunks)))
        .unwrap()
}

async fn server(body: &str, status: StatusCode) -> LocalServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let fixture = Arc::new(Fixture {
        body: body.into(),
        status,
        requests: AtomicUsize::new(0),
    });
    let app = Router::new().fallback(respond).with_state(fixture.clone());
    let task = zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    LocalServer { url, fixture, task }
}

fn controlled(protocol: Protocol, server: &LocalServer, limit: usize) -> Box<dyn ModelProvider> {
    let mut provider = protocol.provider(&server.url);
    provider
        .configure_completion(CompletionControl::bounded_single_call(
            NonZeroUsize::new(limit).unwrap(),
        ))
        .unwrap();
    provider
}

async fn chat(
    provider: &dyn ModelProvider,
) -> anyhow::Result<zeroclaw_api::model_provider::ChatResponse> {
    provider
        .chat(
            ChatRequest {
                messages: &[ChatMessage::user("question")],
                tools: None,
                thinking: None,
            },
            "fixture-model",
            None,
        )
        .await
}

#[tokio::test]
async fn common_control_preserves_the_exact_boundary_and_usage_on_each_protocol() {
    for protocol in PROTOCOLS {
        let server = server(protocol.response(), StatusCode::OK).await;
        let provider = controlled(protocol, &server, protocol.response().len());
        let response = chat(provider.as_ref()).await.unwrap();
        assert_eq!(response.text.as_deref(), Some("ok"), "{protocol:?}");
        assert_eq!(
            response.usage.unwrap().output_tokens,
            Some(2),
            "{protocol:?}"
        );
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn common_control_rejects_one_byte_over_without_truncation_or_fallback() {
    for protocol in PROTOCOLS {
        let server = server(protocol.response(), StatusCode::OK).await;
        let provider = controlled(protocol, &server, protocol.response().len() - 1);
        let error = chat(provider.as_ref()).await.unwrap_err();
        assert!(
            error
                .downcast_ref::<CompletionBodyLimitExceeded>()
                .is_some(),
            "{protocol:?}: {error:#}"
        );
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn common_control_bounds_error_bodies_and_never_retries_schema_errors() {
    for protocol in PROTOCOLS {
        let server = server(
            &"tool schema unsupported".repeat(20),
            StatusCode::BAD_REQUEST,
        )
        .await;
        let provider = controlled(protocol, &server, 32);
        let error = chat(provider.as_ref()).await.unwrap_err();
        assert!(
            error
                .downcast_ref::<CompletionBodyLimitExceeded>()
                .is_some(),
            "{protocol:?}: {error:#}"
        );
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn common_control_bounds_streaming_http_errors_on_supported_stream_routes() {
    for protocol in [
        Protocol::Compatible,
        Protocol::Responses,
        Protocol::Anthropic,
    ] {
        let server = server(&"error payload".repeat(20), StatusCode::BAD_REQUEST).await;
        let provider = controlled(protocol, &server, 32);
        let messages = [ChatMessage::user("question")];
        let events: Vec<_> = provider
            .stream_chat(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                "fixture-model",
                None,
                StreamOptions::new(true),
            )
            .collect()
            .await;
        assert!(
            events.iter().any(|event| event
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("body limit"))),
            "{protocol:?}: {events:?}"
        );
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn common_control_has_an_independent_body_allowance_per_response() {
    for protocol in PROTOCOLS {
        let server = server(protocol.response(), StatusCode::OK).await;
        let provider = controlled(protocol, &server, protocol.response().len());
        chat(provider.as_ref()).await.unwrap();
        let response = chat(provider.as_ref()).await.unwrap();
        assert_eq!(response.text.as_deref(), Some("ok"), "{protocol:?}");
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn common_control_does_not_follow_redirects_or_redispatch_the_completion() {
    for protocol in PROTOCOLS {
        let server = server("redirect", StatusCode::TEMPORARY_REDIRECT).await;
        let provider = controlled(protocol, &server, 32);
        let error = chat(provider.as_ref()).await.unwrap_err();
        assert!(error.to_string().contains("307"), "{protocol:?}: {error:#}");
        assert_eq!(server.fixture.requests.load(Ordering::SeqCst), 1);
    }
}
