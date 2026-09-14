use super::{ResponseMetadata, ResponseMetadataObserver, ResponseMetadataPhase};
use crate::{
    compatible::{AuthStyle, OpenAiCompatibleModelProvider},
    traits::{ChatMessage, ChatRequest, ModelProvider},
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

#[derive(Clone, Copy)]
enum InitialFailure {
    UnsupportedTools,
    Transport,
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Value {
    let mut reader = BufReader::new(socket);
    let mut length = 0;
    loop {
        let mut line = String::new();
        assert_ne!(
            reader.read_line(&mut line).await.expect("request headers"),
            0
        );
        if line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().expect("request content length");
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await.expect("request body");
    serde_json::from_slice(&body).expect("request JSON")
}

async fn fallback_server(failure: InitialFailure) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fallback server");
    let base = format!("http://{}", listener.local_addr().expect("server address"));
    let handle = tokio::spawn(async move {
        let mut requests = Vec::new();
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.expect("accept model request");
            requests.push(read_request(&mut socket).await);
            if attempt == 0 && matches!(failure, InitialFailure::Transport) {
                continue;
            }
            let (status, body) = if attempt == 0 {
                (400, json!({"error": "unsupported parameter: tools"}))
            } else {
                (
                    200,
                    json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}),
                )
            };
            let body = body.to_string();
            let response = format!(
                "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nx-membox-request-id: 00000000-0000-4000-8000-00000000000{attempt}\r\n\r\n{body}",
                body.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("send model response");
        }
        requests
    });
    (base, handle)
}

async fn run_fallback(
    failure: InitialFailure,
    observer: Option<ResponseMetadataObserver>,
) -> Vec<Value> {
    let (base, server) = fallback_server(failure).await;
    let guard = crate::stream_guard::AbortOnDrop::new(server.abort_handle());
    let provider = OpenAiCompatibleModelProvider::builder("fallback-test")
        .display_name("fallback-test")
        .base_url(&base)
        .auth_style(AuthStyle::Bearer)
        .credential(Some("test-key"))
        .response_observer(observer)
        .build();
    let messages = [ChatMessage::user("test")];
    let tools = [zeroclaw_api::tool::ToolSpec::new(
        "test",
        "test",
        json!({"type": "object"}),
    )];
    let response = match failure {
        InitialFailure::UnsupportedTools => provider.chat(
            ChatRequest { messages: &messages, tools: Some(&tools), thinking: None },
            "test-model", None,
        ).await,
        InitialFailure::Transport => provider.chat_with_tools(
            &messages,
            &[json!({"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}})],
            "test-model", None,
        ).await,
    }.expect("fallback succeeds");
    assert_eq!(response.text.as_deref(), Some("ok"));
    assert!(response.tool_calls.is_empty());
    let requests = server.await.expect("fallback server finishes");
    drop(guard);
    requests
}

async fn observe_fallback(failure: InitialFailure) -> Vec<ResponseMetadata> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let observer = ResponseMetadataObserver::new(["x-membox-request-id"], move |metadata| {
        sender.send(metadata).expect("metadata receiver");
    });
    let observed = run_fallback(failure, Some(observer)).await;
    let disabled = run_fallback(failure, None).await;
    assert_eq!(
        observed, disabled,
        "observation preserves both request bodies"
    );
    receiver.try_iter().collect()
}

#[tokio::test]
async fn unsupported_tools_completes_before_starting_history_fallback() {
    use ResponseMetadataPhase::{Completed, Headers, Started};
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        observe_fallback(InitialFailure::UnsupportedTools),
    )
    .await
    .expect("fallback timeout");
    assert_eq!(
        events.iter().map(|event| event.phase).collect::<Vec<_>>(),
        [Started, Headers, Completed, Started, Headers, Completed]
    );
    assert_eq!(events[0].observation_id, events[2].observation_id);
    assert_eq!(events[3].observation_id, events[5].observation_id);
    assert_ne!(events[0].observation_id, events[3].observation_id);
    assert_eq!(events[1].headers, events[2].headers);
    assert_ne!(events[2].headers, events[5].headers);
}

#[tokio::test]
async fn transport_failure_completes_before_starting_history_fallback() {
    use ResponseMetadataPhase::{Completed, Headers, Started};
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        observe_fallback(InitialFailure::Transport),
    )
    .await
    .expect("fallback timeout");
    assert_eq!(
        events.iter().map(|event| event.phase).collect::<Vec<_>>(),
        [Started, Completed, Started, Headers, Completed]
    );
    assert_eq!(events[0].observation_id, events[1].observation_id);
    assert_eq!(events[2].observation_id, events[4].observation_id);
    assert_ne!(events[0].observation_id, events[2].observation_id);
    assert_eq!(events[1].status, None);
    assert!(events[1].headers.is_empty());
}
