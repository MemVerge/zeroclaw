//! Caller-supplied request headers for providers whose HTTP client is pooled.
//!
//! `OpenAiCompatibleModelProvider` and the Responses provider bake `extra_headers`
//! into a dedicated client's default headers, which costs them connection reuse
//! whenever the map is non-empty. The Anthropic and OpenAI providers instead share
//! the runtime proxy client cache, and a host can legitimately build one provider
//! per turn with a per-turn correlation header — keying a cached client on that
//! value would grow the cache by one client per turn. So these two validate the map
//! once at build time and stamp the pairs on each request, leaving the pooled
//! client untouched.

use reqwest::header::{HeaderName, HeaderValue};
use std::collections::HashMap;

/// Header names the provider itself owns on every request. A caller-supplied copy
/// is dropped so the built-in credential and framing stay authoritative — the same
/// rule the Responses provider applies to `Authorization`.
const RESERVED: &[&str] = &[
    "authorization",
    "x-api-key",
    "content-type",
    "content-length",
    "host",
];

/// Validate `extra` into typed `(name, value)` pairs. Reserved names and entries
/// that are not valid HTTP header names/values are dropped with a WARN rather than
/// failing the build, mirroring the compatible provider's tolerance.
pub(crate) fn typed_extra_headers(
    extra: &HashMap<String, String>,
) -> Vec<(HeaderName, HeaderValue)> {
    let mut typed = Vec::with_capacity(extra.len());
    for (key, value) in extra {
        if RESERVED
            .iter()
            .any(|reserved| key.eq_ignore_ascii_case(reserved))
        {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "header": key,
                        "reason": "reserved_header_owned_by_provider",
                    })),
                "Dropping reserved entry from extra_headers; the provider's own credential and framing headers are authoritative"
            );
            continue;
        }
        match (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            (Ok(name), Ok(val)) => typed.push((name, val)),
            _ => {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                        .with_attrs(::serde_json::json!({"header": key})),
                    "Skipping invalid extra header name or value"
                );
            }
        }
    }
    typed
}

/// Stamp the validated pairs on one outgoing request. `RequestBuilder::header`
/// appends, so the reserved-name filter in [`typed_extra_headers`] is what keeps
/// a caller from doubling a header the provider sets itself.
pub(crate) fn apply_extra_headers(
    mut request: reqwest::RequestBuilder,
    headers: &[(HeaderName, HeaderValue)],
) -> reqwest::RequestBuilder {
    for (name, value) in headers {
        request = request.header(name.clone(), value.clone());
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn keeps_valid_headers_and_drops_reserved_and_invalid_ones() {
        let typed = typed_extra_headers(&map(&[
            ("x-membox-turn-id", "membox-round-1"),
            ("Authorization", "Bearer stolen"),
            ("X-API-Key", "stolen"),
            ("bad name", "value"),
            ("x-ok", "line\nbreak"),
        ]));
        let mut names: Vec<String> = typed.iter().map(|(n, _)| n.to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["x-membox-turn-id".to_string()]);
    }

    #[test]
    fn applies_every_typed_pair_to_the_request() {
        let typed = typed_extra_headers(&map(&[
            ("x-membox-turn-id", "membox-round-1"),
            ("x-membox-call-kind", "main"),
        ]));
        let request = apply_extra_headers(
            reqwest::Client::new().post("http://127.0.0.1:1/v1/messages"),
            &typed,
        )
        .build()
        .expect("request");
        assert_eq!(
            request.headers().get("x-membox-turn-id").unwrap(),
            "membox-round-1"
        );
        assert_eq!(request.headers().get("x-membox-call-kind").unwrap(), "main");
    }
}
