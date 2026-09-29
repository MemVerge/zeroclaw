use super::*;

fn source(
    id: usize,
    field: MessageContentField,
    range: std::ops::Range<usize>,
) -> MessageContentSource {
    MessageContentSource {
        source_id: id,
        field,
        range,
    }
}

fn provider() -> OpenAiCompatibleModelProvider {
    OpenAiCompatibleModelProvider::builder("test")
        .display_name("test")
        .base_url("http://127.0.0.1")
        .auth_style(AuthStyle::Bearer)
        .build()
}

#[test]
fn native_projection_preserves_empty_blank_null_and_reasoning_tool_messages() {
    for value in [
        serde_json::json!(null),
        serde_json::json!(""),
        serde_json::json!(" \r\n "),
        serde_json::json!("查\r\n\"结果\""),
    ] {
        let mut envelope = serde_json::json!({"content":value,"reasoning_content":"private reasoning","tool_calls":[{"id":"call_1","name":"lookup","arguments":"{}"}]});
        for omit in [false, true] {
            if omit {
                envelope.as_object_mut().unwrap().remove("content");
            }
            let mut message = ChatMessage::assistant(envelope.to_string());
            let provider = provider();
            let baseline = provider.build_native_tool_chat_request(
                std::slice::from_ref(&message),
                None,
                "test",
                None,
                true,
            );
            if let Some(text) = envelope.get("content").and_then(serde_json::Value::as_str) {
                message.content_sources =
                    vec![source(4, MessageContentField::JsonContent, 0..text.len())];
            }
            let request =
                provider.build_native_tool_chat_request(&[message], None, "test", None, true);
            assert_eq!(
                serde_json::to_value(&request).unwrap(),
                serde_json::to_value(baseline).unwrap()
            );
            let spans = projected_sources(
                request
                    .messages
                    .iter()
                    .map(|message| &message.content_sources),
            );
            let visible = envelope
                .get("content")
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.trim().is_empty());
            assert_eq!(spans.len(), usize::from(visible.is_some()));
            if let Some(text) = visible {
                assert_eq!(spans[0].range, 0..text.len());
            }
        }
    }
}

#[test]
fn native_projection_preserves_raw_envelope_sources() {
    let text = "查\\\r\n\"结果\"";
    for (role, value) in [
        (
            "assistant",
            serde_json::json!({"content":text,"tool_calls":[]}),
        ),
        (
            "assistant",
            serde_json::json!({"content":text,"reasoning_content":"thinking"}),
        ),
        (
            "tool",
            serde_json::json!({"content":text,"tool_call_id":"call_1"}),
        ),
        (
            "tool",
            serde_json::json!({"status":"done","tool_call_id":"call_1"}),
        ),
    ] {
        let content = value.to_string();
        let expected = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(&content)
            .to_owned();
        let message = ChatMessage {
            content_sources: vec![source(7, MessageContentField::Text, 0..content.len())],
            role: role.into(),
            content,
        };
        let native = provider().convert_messages_for_native(&[message], true);
        assert_eq!(
            serde_json::to_value(&native[0]).unwrap()["content"],
            expected
        );
        assert_eq!(
            projected_sources(native.iter().map(|message| &message.content_sources)),
            vec![ProjectedMessageSource {
                source_id: 7,
                message_index: 0,
                range: 0..expected.len(),
            }]
        );
    }
}

#[test]
fn sources_follow_system_merge_media_conversion_and_assistant_coalescing() {
    let mut current = ChatMessage::user("  问题[IMAGE:https://example.com/a.png]尾巴  ");
    current.content_sources = vec![source(
        2,
        MessageContentField::Text,
        0..current.content.len(),
    )];
    let input = [ChatMessage::system("host"), current];
    let provider = provider();
    let native = provider.convert_messages_for_native(&input, true);
    let spans = projected_sources(native.iter().map(|message| &message.content_sources));
    assert_eq!(
        spans,
        vec![ProjectedMessageSource {
            source_id: 2,
            message_index: 1,
            range: 0.."问题尾巴".len()
        }]
    );
    let merged = OpenAiCompatibleModelProvider::flatten_system_messages(&input, true);
    assert_eq!(merged[0].content_sources[0].range.start, "host\n\n".len());

    let provider = OpenAiCompatibleModelProvider::builder("test")
        .display_name("test")
        .base_url("http://127.0.0.1")
        .auth_style(AuthStyle::Bearer)
        .without_native_tools()
        .build();
    let mut assistant =
        ChatMessage::assistant(serde_json::json!({"content":"答","tool_calls":[]}).to_string());
    assistant.content_sources = vec![source(3, MessageContentField::JsonContent, 0.."答".len())];
    let mut next = ChatMessage::assistant("案");
    next.content_sources = vec![source(4, MessageContentField::Text, 0.."案".len())];
    let messages = provider.strip_native_tool_messages(&[assistant, next]);
    assert_eq!(messages[0].content, "答\n\n案");
    assert_eq!(
        messages[0].content_sources[1].range,
        "答\n\n".len().."答\n\n案".len()
    );
}

#[tokio::test]
async fn request_headers_follow_the_final_body_on_all_chat_paths() {
    use axum::{Router, extract::State, routing::post};
    use std::sync::Arc;
    let (sent, mut received) = tokio::sync::mpsc::unbounded_channel();
    let app = Router::new().route("/chat/completions", post(|State(sender):State<tokio::sync::mpsc::UnboundedSender<(HeaderMap,serde_json::Value,axum::body::Bytes)>>, headers:HeaderMap, raw_body:axum::body::Bytes| async move {
        let body:serde_json::Value=serde_json::from_slice(&raw_body).unwrap();
        let streaming = body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
        sender.send((headers,body,raw_body)).unwrap();
        if streaming {
            ([("content-type","text/event-stream")], "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_owned())
        } else {
            ([("content-type","application/json")], "{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}".to_owned())
        }
    })).with_state(sent);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = ::zeroclaw_spawn::spawn!(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (observed_tx, mut observed_rx) = tokio::sync::mpsc::unbounded_channel();
    let callback: RequestHeaders = Arc::new(move |body, spans| {
        observed_tx.send((body.clone(), spans.to_vec())).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("x-source-projection", HeaderValue::from_static("present"));
        for (name, value) in [
            ("authorization", "Bearer callback"),
            ("x-api-key", "gateway-key"),
            ("x-test-key", "callback-key"),
            ("content-type", "text/plain"),
            ("content-length", "1"),
            ("host", "gateway.example"),
            ("accept", "text/plain"),
        ] {
            headers.insert(name, HeaderValue::from_static(value));
        }
        Ok(headers)
    });
    let mut message = ChatMessage::user("host 当前");
    message.content_sources = vec![source(
        7,
        MessageContentField::Text,
        "host ".len()..message.content.len(),
    )];
    let mut historical = ChatMessage::assistant(
        serde_json::json!({"content":"past \\ 世界", "tool_calls":[]}).to_string(),
    );
    historical.content_sources = vec![source(
        6,
        MessageContentField::JsonContent,
        0.."past \\ 世界".len(),
    )];
    let messages = [ChatMessage::system("instruction"), historical, message];
    let tools = [zeroclaw_api::tool::ToolSpec::new(
        "lookup",
        "lookup",
        serde_json::json!({"type":"object","properties":{}}),
    )];
    for (openai, auth_style, credential) in [
        (false, AuthStyle::Bearer, Some("key")),
        (true, AuthStyle::Bearer, Some("key")),
        (false, AuthStyle::XApiKey, Some("key")),
        (false, AuthStyle::Custom("x-test-key".into()), Some("key")),
        (false, AuthStyle::Bearer, None),
    ] {
        for mode in 0..if openai { 3 } else { 8 } {
            let mut previous = None;
            for callback in [None, Some(callback.clone())] {
                let tagged = callback.is_some();
                let provider: Box<dyn ModelProvider> = if openai {
                    Box::new(
                        crate::openai::OpenAiModelProvider::builder("test")
                            .base_url(&base)
                            .credential(credential)
                            .request_headers(callback)
                            .build(),
                    )
                } else {
                    Box::new(
                        OpenAiCompatibleModelProvider::builder("test")
                            .display_name("test")
                            .base_url(&base)
                            .auth_style(auth_style.clone())
                            .credential(credential)
                            .request_headers(callback)
                            .build(),
                    )
                };
                let request = ProviderChatRequest {
                    messages: &messages,
                    tools: Some(&tools),
                    thinking: None,
                };
                match mode {
                    0 => {
                        provider.chat(request, "test", None).await.unwrap();
                    }
                    1 => {
                        provider
                            .chat_with_tools(&messages, &[], "test", None)
                            .await
                            .unwrap();
                    }
                    2 => {
                        provider
                            .chat_with_history(&messages, "test", None)
                            .await
                            .unwrap();
                    }
                    3 => {
                        let results = provider
                            .stream_chat(request, "test", None, StreamOptions::new(true))
                            .collect::<Vec<_>>()
                            .await;
                        assert!(results.iter().all(Result::is_ok));
                    }
                    4 => {
                        let results = provider
                            .stream_chat(
                                ProviderChatRequest {
                                    tools: None,
                                    ..request
                                },
                                "test",
                                None,
                                StreamOptions::new(true),
                            )
                            .collect::<Vec<_>>()
                            .await;
                        assert!(results.iter().all(Result::is_ok));
                    }
                    5 => {
                        let results = provider
                            .stream_chat_with_history(
                                &messages,
                                "test",
                                None,
                                StreamOptions::new(true),
                            )
                            .collect::<Vec<_>>()
                            .await;
                        assert!(results.iter().all(Result::is_ok));
                    }
                    6 => {
                        provider
                            .chat_with_system(Some("instruction"), "current", "test", None)
                            .await
                            .unwrap();
                    }
                    _ => {
                        let results = provider
                            .stream_chat_with_system(
                                Some("instruction"),
                                "current",
                                "test",
                                None,
                                StreamOptions::new(true),
                            )
                            .collect::<Vec<_>>()
                            .await;
                        assert!(results.iter().all(Result::is_ok));
                    }
                }
                let (headers, body, raw_body) = received.recv().await.unwrap();
                if tagged {
                    assert_eq!(headers.get("x-source-projection").unwrap(), "present");
                    assert_eq!(headers["content-type"], "application/json");
                    assert_eq!(headers["content-length"], raw_body.len().to_string());
                    assert_eq!(
                        headers["host"],
                        if openai {
                            base.strip_prefix("http://").unwrap()
                        } else {
                            "gateway.example"
                        }
                    );
                    assert_eq!(headers.get_all("accept").iter().count(), 1);
                    assert_eq!(
                        headers["accept"],
                        if matches!(mode, 3..=5 | 7) {
                            "text/event-stream"
                        } else {
                            "text/plain"
                        }
                    );
                    for (name, callback_value) in [
                        ("authorization", "Bearer callback"),
                        ("x-api-key", "gateway-key"),
                        ("x-test-key", "callback-key"),
                    ] {
                        let expected = if credential.is_some()
                            && name == auth_style.credential_header_name()
                        {
                            if name == "authorization" {
                                "Bearer key"
                            } else {
                                "key"
                            }
                        } else {
                            callback_value
                        };
                        assert_eq!(headers.get_all(name).iter().count(), 1);
                        assert_eq!(headers[name], expected);
                    }
                    let (observed, spans) = observed_rx.recv().await.unwrap();
                    assert_eq!(observed, body);
                    assert_eq!(previous.as_ref(), Some(&raw_body));
                    if mode >= 6 {
                        assert!(spans.is_empty());
                    } else if openai && mode == 2 {
                        assert_eq!(
                            spans,
                            vec![ProjectedMessageSource {
                                source_id: 7,
                                message_index: 1,
                                range: "host ".len().."host 当前".len()
                            }]
                        );
                    } else {
                        assert_eq!(spans.len(), 2, "mode {mode}");
                        assert_eq!(
                            spans[1],
                            ProjectedMessageSource {
                                source_id: 7,
                                message_index: 2,
                                range: "host ".len().."host 当前".len()
                            }
                        );
                        let historical = body["messages"][1]["content"].as_str().unwrap();
                        let selected = &historical[spans[0].range.clone()];
                        let expected = if matches!(mode, 0 | 1 | 3) {
                            "past \\ 世界"
                        } else {
                            "past \\\\ 世界"
                        };
                        assert_eq!(selected, expected, "mode {mode}");
                    }
                } else {
                    previous = Some(raw_body);
                    assert!(!headers.contains_key("x-source-projection"));
                }
            }
        }
    }
    server.abort();
}

#[test]
fn plain_json_projection_tracks_the_content_field_and_original_escape_spelling() {
    let raw = r#" {"other":"same", "content":"same\n\u6C49\uD83D\uDE00\\\"", "tool_calls":[]} "#;
    let decoded: serde_json::Value = serde_json::from_str(raw).unwrap();
    let text = decoded["content"].as_str().unwrap();
    let mut message = ChatMessage::assistant(raw);
    message.content_sources = vec![source(1, MessageContentField::JsonContent, 0..text.len())];
    let sources = OpenAiCompatibleModelProvider::text_sources(&message, false);
    assert_eq!(sources.len(), 1);
    assert_eq!(
        &raw[sources[0].range.clone()],
        r#"same\n\u6C49\uD83D\uDE00\\\""#
    );
    assert_eq!(
        serde_json::to_value(message).unwrap(),
        serde_json::json!({"role":"assistant","content":raw})
    );
}

#[test]
fn media_replacement_retains_its_source_in_text_and_json_content() {
    for field in [MessageContentField::Text, MessageContentField::JsonContent] {
        let text = "前[AUDIO:/tmp/clip.wav]后";
        let content = if field == MessageContentField::Text {
            text.to_owned()
        } else {
            serde_json::json!({"content":text,"tool_calls":[]}).to_string()
        };
        let mut message = ChatMessage::assistant(content);
        message.content_sources = vec![source(1, field, 0..text.len())];
        let updated = multimodal::strip_media_markers_with_sources(&message);
        let native = provider().convert_messages_for_native(&[updated], false);
        let spans = projected_sources(native.iter().map(|message| &message.content_sources));
        assert_eq!(
            spans,
            vec![ProjectedMessageSource {
                source_id: 1,
                message_index: 0,
                range: 0.."前[media attachment]后".len()
            }]
        );
    }
}

#[test]
fn provider_inserted_separator_stays_inside_one_merged_source() {
    let provider = OpenAiCompatibleModelProvider::builder("test")
        .display_name("test")
        .base_url("http://127.0.0.1")
        .auth_style(AuthStyle::Bearer)
        .without_native_tools()
        .build();
    let input = ["one", "two"].map(|text| ChatMessage {
        content_sources: vec![source(1, MessageContentField::Text, 0..text.len())],
        ..ChatMessage::user(text)
    });
    let merged = provider.strip_native_tool_messages(&input);
    assert_eq!(merged[0].content, "one\n\ntwo");
    assert_eq!(
        projected_sources(merged.iter().map(|message| &message.content_sources)),
        vec![ProjectedMessageSource {
            source_id: 1,
            message_index: 0,
            range: 0..8
        }]
    );
}

#[test]
fn non_native_coalescing_preserves_reasoning_and_legacy_tool_envelope_sources() {
    let provider = OpenAiCompatibleModelProvider::builder("test")
        .display_name("test")
        .base_url("http://127.0.0.1")
        .auth_style(AuthStyle::Bearer)
        .without_native_tools()
        .build();
    let mut reasoning = ChatMessage::assistant(
        serde_json::json!({"content":"past", "reasoning_content":"thinking"}).to_string(),
    );
    reasoning.content_sources = vec![source(1, MessageContentField::JsonContent, 0..4)];
    let raw = reasoning.content.clone();
    let combined =
        provider.strip_native_tool_messages(&[reasoning, ChatMessage::assistant("tail")]);
    assert_eq!(combined[0].content, format!("{raw}\n\ntail"));
    let projected = projected_sources(combined.iter().map(|message| &message.content_sources));
    assert_eq!(&combined[0].content[projected[0].range.clone()], "past");
    for calls in [
        serde_json::Value::Null,
        serde_json::json!({"legacy":"shape"}),
    ] {
        let mut message = ChatMessage::assistant(
            serde_json::json!({"content":"kept", "tool_calls":calls}).to_string(),
        );
        assert_eq!(assistant_text_content(&message.content), None);
        message.content_sources = vec![source(
            2,
            MessageContentField::Text,
            0..message.content.len(),
        )];
        let output = provider.strip_native_tool_messages(&[message]);
        assert_eq!(output[0].content, "kept");
        assert_eq!(
            projected_sources(output.iter().map(|message| &message.content_sources)),
            vec![ProjectedMessageSource {
                source_id: 2,
                message_index: 0,
                range: 0..4
            }]
        );
    }
}
