use super::*;
use crate::{
    anthropic::AnthropicModelProvider,
    compatible::{AuthStyle, OpenAiCompatibleModelProvider},
    openai::{OpenAiModelProvider, OpenAiResponsesModelProvider},
    traits::{ChatMessage, ChatRequest, ModelProvider, StreamOptions},
};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const REQUEST_ID: &str = "00000000-0000-4000-8000-000000000001";

pub(super) fn provider(
    route: &str,
    base: &str,
    observer: Option<ResponseMetadataObserver>,
) -> Box<dyn ModelProvider> {
    match route {
        "openai" => Box::new(
            OpenAiModelProvider::builder("test")
                .base_url(base)
                .credential(Some("test-key"))
                .response_observer(observer)
                .build(),
        ),
        "responses" => Box::new(
            OpenAiResponsesModelProvider::builder("test")
                .api_url(base)
                .credential(Some("test-key"))
                .response_observer(observer)
                .build(),
        ),
        "anthropic" => Box::new(
            AnthropicModelProvider::builder("test")
                .base_url(base)
                .credential(Some("test-key"))
                .response_observer(observer)
                .build(),
        ),
        _ => Box::new(
            OpenAiCompatibleModelProvider::builder("test")
                .display_name("test")
                .base_url(base)
                .auth_style(AuthStyle::Bearer)
                .credential(Some("test-key"))
                .response_observer(observer)
                .build(),
        ),
    }
}

async fn server(status: u16) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("address"));
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = vec![0; 65536];
        socket.read(&mut request).await.expect("read request");
        // Deliberately provide no body. The observer must run as soon as headers
        // arrive even if parsing would block forever or the consumer cancels.
        let response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: text/event-stream\r\nContent-Length: 100000\r\nx-membox-request-id: {REQUEST_ID}\r\nx-request-id: provider-only\r\nset-cookie: private\r\n\r\n"
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("headers");
        std::future::pending::<()>().await;
    });
    (base, handle)
}

#[tokio::test]
async fn completion_routes_observe_headers_before_parsing_success_and_error_bodies() {
    for route in ["openai", "responses", "anthropic", "compatible"] {
        for status in [200, 402] {
            let (base, server) = server(status).await;
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let observer =
                ResponseMetadataObserver::new(["X-MemBox-Request-ID"], move |metadata| {
                    sender.send(metadata).expect("observer receiver");
                });
            let provider = provider(route, &base, Some(observer));
            let call = tokio::spawn(async move {
                provider
                    .chat_with_system(None, "test", "test-model", None)
                    .await
            });
            let metadata = receive_phase(&mut receiver, ResponseMetadataPhase::Headers).await;
            assert_eq!(metadata.status, Some(status), "{route}");
            assert_eq!(
                metadata.headers,
                BTreeMap::from([("x-membox-request-id".into(), REQUEST_ID.into())])
            );
            call.abort();
            let completed = receive_phase(&mut receiver, ResponseMetadataPhase::Completed).await;
            assert_eq!(completed.observation_id, metadata.observation_id);
            assert_eq!(completed.headers, metadata.headers);
            server.abort();
        }
    }
}

#[tokio::test]
async fn streaming_routes_keep_metadata_when_consumer_drops_before_first_body_chunk() {
    for route in ["responses", "anthropic", "compatible"] {
        let (base, server) = server(200).await;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let observer = ResponseMetadataObserver::new(["x-membox-request-id"], move |metadata| {
            sender.send(metadata).expect("observer receiver");
        });
        let provider = provider(route, &base, Some(observer));
        let messages = vec![ChatMessage::user("test")];
        let request = ChatRequest {
            messages: &messages,
            tools: None,
            thinking: None,
        };
        let mut stream =
            provider.stream_chat(request, "test-model", None, StreamOptions::new(true));
        let metadata = tokio::select! {
            value = receive_phase(&mut receiver, ResponseMetadataPhase::Headers) => value,
            chunk = stream.next() => panic!("body arrived before headers for {route}: {chunk:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("no headers for {route}"),
        };
        drop(stream);
        let completed = receive_phase(&mut receiver, ResponseMetadataPhase::Completed).await;
        assert_eq!(completed.observation_id, metadata.observation_id);
        assert_eq!(
            metadata
                .headers
                .get("x-membox-request-id")
                .map(String::as_str),
            Some(REQUEST_ID)
        );
        server.abort();
    }
}

#[test]
fn absent_invalid_and_unselected_headers_do_not_become_metadata() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let observer = ResponseMetadataObserver::new(
        ["x-membox-request-id", "missing", "invalid"],
        move |metadata| {
            sender.send(metadata).expect("receiver");
        },
    );
    let response: reqwest::Response = axum::http::Response::builder()
        .header("x-membox-request-id", REQUEST_ID)
        .header("set-cookie", "private")
        .header(
            "invalid",
            axum::http::HeaderValue::from_bytes(&[0xff]).expect("raw header"),
        )
        .body(String::new())
        .expect("response")
        .into();
    let mut observation = begin_request(&Some(observer));
    assert_eq!(
        receiver.recv().expect("started").phase,
        ResponseMetadataPhase::Started
    );
    observe_response(&mut observation, &response);
    assert_eq!(receiver.recv().expect("headers").headers.len(), 1);
    observe_response(&mut None, &response);
    assert!(receiver.try_recv().is_err());
}

#[cfg(panic = "unwind")]
#[test]
fn observer_panic_is_contained_under_unwind() {
    let response: reqwest::Response = axum::http::Response::new(String::new()).into();
    let observer = ResponseMetadataObserver::new(["request-id"], |_| panic!("host callback"));
    let mut observation = begin_request(&Some(observer));
    observe_response(&mut observation, &response);
    drop(observation);
    assert!(response.status().is_success());
}

async fn receive_phase(
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<ResponseMetadata>,
    phase: ResponseMetadataPhase,
) -> ResponseMetadata {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let metadata = receiver.recv().await.expect("metadata channel");
            if metadata.phase == phase {
                return metadata;
            }
        }
    })
    .await
    .expect("lifecycle metadata before body")
}

#[test]
fn dropping_before_headers_reports_completion_without_inventing_a_request_id() {
    let (sender, receiver) = std::sync::mpsc::channel();
    let observer = ResponseMetadataObserver::new(["x-membox-request-id"], move |metadata| {
        sender.send(metadata).expect("receiver");
    });
    drop(observer.begin_request());
    drop(observer.begin_request());
    let events: Vec<_> = receiver.try_iter().collect();
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].observation_id, events[1].observation_id);
    assert_ne!(events[0].observation_id, events[2].observation_id);
    assert_eq!(events[1].phase, ResponseMetadataPhase::Completed);
    assert_eq!(events[1].status, None);
    assert!(events[1].headers.is_empty());
}

#[test]
fn disabled_observation_allocates_no_request_guard() {
    assert!(begin_request(&None).is_none());
}
