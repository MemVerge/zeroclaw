use super::*;
use futures_util::FutureExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

const COMPLETED_RESPONSE: &str = concat!(
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":20,\"cache_creation_input_tokens\":30}}}\n\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

fn take_events(
    rx: &mut mpsc::Receiver<StreamResult<StreamEvent>>,
) -> Vec<StreamResult<StreamEvent>> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn message_stop_waits_for_body_eof_before_final() {
    let (mut body, reader) = tokio::io::duplex(4096);
    body.write_all(COMPLETED_RESPONSE.as_bytes()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    let parser =
        AnthropicModelProvider::parse_anthropic_sse_from_reader(BufReader::new(reader), &tx);
    tokio::pin!(parser);

    assert!(parser.as_mut().now_or_never().is_none());
    let before_eof = take_events(&mut rx);
    assert!(
        matches!(before_eof.first(), Some(Ok(StreamEvent::TextDelta(chunk))) if chunk.delta == "ok")
    );
    assert!(
        matches!(before_eof.last(), Some(Ok(StreamEvent::Usage(usage)))
        if usage.input_tokens == Some(60) && usage.cached_input_tokens == Some(20)
            && usage.output_tokens == Some(4))
    );
    assert!(
        !before_eof
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final { .. })))
    );

    body.write_all(b": trailing keepalive\n\n").await.unwrap();
    assert!(parser.as_mut().now_or_never().is_none());
    assert!(rx.try_recv().is_err());
    body.shutdown().await.unwrap();
    parser.await;
    assert!(matches!(
        take_events(&mut rx).as_slice(),
        [Ok(StreamEvent::Final {
            stop: StopReason::Complete
        })]
    ));
}

#[tokio::test]
async fn duplicate_message_stop_emits_usage_and_final_once() {
    let response = format!("{COMPLETED_RESPONSE}{COMPLETED_RESPONSE}");
    let (tx, mut rx) = mpsc::channel(64);
    AnthropicModelProvider::parse_anthropic_sse_from_reader(response.as_bytes(), &tx).await;

    let events = take_events(&mut rx);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::Usage(_))))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::Final { .. })))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::TextDelta(_))))
            .count(),
        1
    );
}

#[tokio::test]
async fn read_error_after_message_stop_does_not_emit_final() {
    let chunks = stream::iter([
        Ok(COMPLETED_RESPONSE.as_bytes()),
        Err(std::io::Error::other("connection reset by peer")),
    ]);
    let reader = tokio_util::io::StreamReader::new(chunks);
    let (tx, mut rx) = mpsc::channel(64);
    AnthropicModelProvider::parse_anthropic_sse_from_reader(reader, &tx).await;

    let events = take_events(&mut rx);
    assert!(
        matches!(events.last(), Some(Err(StreamError::Http(message))) if message.contains("connection reset"))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final { .. })))
    );
}

#[tokio::test(start_paused = true)]
async fn stalled_body_after_message_stop_retains_idle_timeout() {
    let (mut body, reader) = tokio::io::duplex(4096);
    body.write_all(COMPLETED_RESPONSE.as_bytes()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    let parser =
        AnthropicModelProvider::parse_anthropic_sse_from_reader(BufReader::new(reader), &tx);
    tokio::pin!(parser);

    assert!(parser.as_mut().now_or_never().is_none());
    tokio::time::advance(SSE_IDLE_TIMEOUT + std::time::Duration::from_secs(1)).await;
    parser.await;

    let events = take_events(&mut rx);
    assert!(
        matches!(events.last(), Some(Err(StreamError::Http(message))) if message.contains("stalled"))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final { .. })))
    );
}

#[tokio::test(start_paused = true)]
async fn consumer_drop_aborts_parser_waiting_for_body_eof() {
    let (mut body, reader) = tokio::io::duplex(4096);
    body.write_all(COMPLETED_RESPONSE.as_bytes()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(64);
    let parser = ::zeroclaw_spawn::spawn!(async move {
        AnthropicModelProvider::parse_anthropic_sse_from_reader(BufReader::new(reader), &tx).await;
    });
    let guard = AbortOnDrop::new(parser.abort_handle());
    while !matches!(rx.recv().await.unwrap(), Ok(StreamEvent::Usage(_))) {}
    assert!(!parser.is_finished());

    drop(guard);

    assert!(parser.await.unwrap_err().is_cancelled());
    assert_eq!(
        body.read(&mut [0]).await.unwrap(),
        0,
        "reader must be released on cancel"
    );
    assert!(
        !take_events(&mut rx)
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::Final { .. })))
    );
}
