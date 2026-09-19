//! Caller-supplied request headers for providers whose HTTP client is pooled.
//!
//! A host can legitimately build one provider per turn with a per-turn correlation
//! header. Keying a cached client on that value would grow the cache by one client
//! per turn, so pooled providers stamp caller headers on each request instead.
//! Anthropic and the native OpenAI provider use these helpers to validate the map
//! once at build time while leaving the shared transport client untouched.

use reqwest::header::{HeaderName, HeaderValue};
use std::collections::HashMap;

/// Header names every HTTP client owns for the request it frames. Dropped for
/// every provider; a caller cannot meaningfully set them.
const FRAMING: &[&str] = &["content-type", "content-length", "host"];

/// Single-valued header names a provider sets on every request itself. A
/// caller-supplied copy is dropped so the built-in credential and API version
/// stay authoritative — the Responses provider's `Authorization` rule, widened
/// only to the names *that provider* would otherwise send twice. The set is
/// per provider on purpose: `x-api-key` and `anthropic-version` are Anthropic's,
/// and the OpenAI chat-completions provider forwards them like any other custom
/// header, as its Responses sibling does, so switching `wire_api` does not
/// change which credential or API-version headers a custom gateway receives.
/// (Framing is still dropped on this wire only; the Responses provider forwards
/// it as a request header.) List-valued headers such as
/// `anthropic-beta` are deliberately not reserved: a caller's entry becomes a
/// second field line beside the provider's, which RFC 9110 list-field merging
/// combines, so appending a beta flag is a legitimate caller use.
#[derive(Clone, Copy)]
pub(crate) struct ReservedHeaders {
    /// Which provider's set this is, for the drop WARN — the same name is
    /// reserved by one provider and forwarded by another.
    provider: &'static str,
    names: &'static [&'static str],
}

impl ReservedHeaders {
    /// `x-api-key` / `Authorization` (credential, one or the other per auth
    /// style), `anthropic-version`, and the browser-access flag the OAuth path
    /// sets — every single-valued header the provider stamps itself.
    pub(crate) const ANTHROPIC: Self = Self {
        provider: "anthropic",
        names: &[
            "authorization",
            "x-api-key",
            "anthropic-version",
            "anthropic-dangerous-direct-browser-access",
        ],
    };
    /// `Authorization` only — the bearer credential is the one header the
    /// chat-completions provider sets itself.
    pub(crate) const OPENAI: Self = Self {
        provider: "openai",
        names: &["authorization"],
    };

    fn contains(self, name: &str) -> bool {
        self.names
            .iter()
            .chain(FRAMING)
            .any(|reserved| name.eq_ignore_ascii_case(reserved))
    }
}

/// Validate `extra` into typed `(name, value)` pairs. Names in `reserved` (plus
/// framing) and entries that are not valid HTTP header names/values are dropped
/// with a WARN rather than failing the build, mirroring the compatible
/// provider's tolerance.
pub(crate) fn typed_extra_headers(
    extra: &HashMap<String, String>,
    reserved: ReservedHeaders,
) -> Vec<(HeaderName, HeaderValue)> {
    let mut typed = Vec::with_capacity(extra.len());
    for (key, value) in extra {
        if reserved.contains(key) {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "header": key,
                        "provider": reserved.provider,
                        "reason": "reserved_header_owned_by_provider",
                    })),
                "Dropping reserved entry from extra_headers; the provider's own credential, API-version and framing headers are authoritative"
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

    fn names(typed: &[(HeaderName, HeaderValue)]) -> Vec<String> {
        let mut names: Vec<String> = typed.iter().map(|(n, _)| n.to_string()).collect();
        names.sort();
        names
    }

    #[test]
    fn keeps_valid_headers_and_drops_reserved_and_invalid_ones() {
        let typed = typed_extra_headers(
            &map(&[
                ("x-membox-turn-id", "membox-round-1"),
                ("Authorization", "Bearer stolen"),
                ("X-API-Key", "stolen"),
                ("Anthropic-Version", "2099-01-01"),
                ("anthropic-dangerous-direct-browser-access", "true"),
                ("Content-Length", "0"),
                ("bad name", "value"),
                ("x-ok", "line\nbreak"),
            ]),
            ReservedHeaders::ANTHROPIC,
        );
        assert_eq!(names(&typed), vec!["x-membox-turn-id".to_string()]);
    }

    #[test]
    fn the_reserved_set_is_the_providers_own_not_a_shared_one() {
        // The OpenAI chat-completions provider never sets Anthropic's headers,
        // so a custom gateway that wants them keeps them — same as on the
        // Responses wire. Only its own bearer credential and framing are dropped.
        let typed = typed_extra_headers(
            &map(&[
                ("x-api-key", "gateway-key"),
                ("anthropic-version", "2023-06-01"),
                ("Authorization", "Bearer stolen"),
                ("Host", "evil.example"),
            ]),
            ReservedHeaders::OPENAI,
        );
        let mut pairs: Vec<(String, String)> = typed
            .iter()
            .map(|(n, v)| (n.to_string(), v.to_str().unwrap().to_string()))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
                ("x-api-key".to_string(), "gateway-key".to_string()),
            ]
        );
    }

    #[test]
    fn applies_every_typed_pair_to_the_request() {
        let typed = typed_extra_headers(
            &map(&[
                ("x-membox-turn-id", "membox-round-1"),
                ("x-membox-call-kind", "main"),
            ]),
            ReservedHeaders::ANTHROPIC,
        );
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
