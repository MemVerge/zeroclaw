//! Exercise the HTTP body boundary with the real provider and its drop guard.
use std::convert::Infallible;

use axum::{Router, body::Body, response::Response, routing::post};
use futures_util::{FutureExt, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use zeroclaw_api::StopReason;
use zeroclaw_api::model_provider::{
    ChatMessage, ChatRequest, ModelProvider, StreamEvent, StreamOptions,
};
use zeroclaw_providers::anthropic::AnthropicModelProvider;

#[tokio::test]
async fn final_waits_for_http_body_eof() {
    tokio::time::timeout(std::time::Duration::from_secs(5), verify_body_completion())
        .await
        .expect("provider must finish after HTTP EOF");
}

async fn verify_body_completion() {
    let (body_tx, body_rx) = mpsc::channel::<Result<&'static str, Infallible>>(4);
    body_tx.send(Ok(concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    ))).await.unwrap();
    let body_rx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(body_rx)));
    let app = Router::new().route(
        "/v1/messages",
        post(move || {
            let body_rx = body_rx.clone();
            async move {
                let reader = body_rx.lock().await.take().expect("exactly one request");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(ReceiverStream::new(reader)))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = ::zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let _server_guard = scopeguard::guard(server, |server| server.abort());
    let provider = AnthropicModelProvider::builder("test")
        .credential(Some("test-key"))
        .base_url(&format!("http://{address}"))
        .build();
    let messages = [ChatMessage::user("Reply with only the word: ok")];
    let mut events = provider.stream_chat(
        ChatRequest {
            messages: &messages,
            tools: None,
            thinking: None,
        },
        "claude-sonnet-4-5",
        None,
        StreamOptions::new(true),
    );

    assert!(
        matches!(events.next().await.unwrap().unwrap(), StreamEvent::TextDelta(chunk) if chunk.delta == "ok")
    );
    assert!(
        matches!(events.next().await.unwrap().unwrap(), StreamEvent::Usage(usage)
        if usage.input_tokens == Some(10) && usage.output_tokens == Some(4))
    );
    assert!(
        events.next().now_or_never().is_none(),
        "message_stop must not finish the consumer while the HTTP body is still open"
    );

    drop(body_tx); // Allow the server to send the terminal HTTP body frame.

    assert!(matches!(
        events.next().await.unwrap().unwrap(),
        StreamEvent::Final {
            stop: StopReason::Complete
        }
    ));
    // The runtime stops polling at Final; dropping here must be safe.
    drop(events);
}
