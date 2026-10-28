use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Authorizer {
    calls: AtomicUsize,
    deny: Option<SubmissionDenied>,
}
impl Default for Authorizer {
    fn default() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            deny: None,
        }
    }
}
impl RequestSubmissionAuthorizer for Authorizer {
    fn limits(&self) -> SubmissionLimits {
        SubmissionLimits {
            max_body_bytes: 1024,
            max_source_spans: 8,
        }
    }
    fn authorize(
        &self,
        request: SerializedSubmission<'_>,
    ) -> Result<SubmissionLease, SubmissionDenied> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(request.method, "POST");
        assert_eq!(request.body, b"{\"model\":\"fixture\"}");
        if let Some(reason) = self.deny {
            return Err(reason);
        }
        Ok(SubmissionLease::hold(()))
    }
}

fn request() -> Request {
    reqwest::Client::new()
        .post("http://127.0.0.1/chat/completions")
        .body("{\"model\":\"fixture\"}")
        .build()
        .unwrap()
}
fn input(request: &Request) -> SubmissionAuthorization<'_> {
    SubmissionAuthorization {
        request,
        sources: &[],
        credential_header: "authorization",
    }
}

#[test]
fn actual_serialized_bytes_reach_the_host_without_json_reencoding() {
    let host = Authorizer::default();
    let request = request();
    authorize(&host, input(&request)).unwrap().release();
    assert_eq!(host.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn local_denial_retains_a_structured_non_remote_error() {
    let host = Authorizer {
        deny: Some(SubmissionDenied::BindingChanged),
        ..Default::default()
    };
    let request = request();
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::BindingChanged)
    ));
}

struct PanickingHost;
impl RequestSubmissionAuthorizer for PanickingHost {
    fn limits(&self) -> SubmissionLimits {
        panic!("synthetic unavailable host")
    }
    fn authorize(&self, _: SerializedSubmission<'_>) -> Result<SubmissionLease, SubmissionDenied> {
        panic!("must not reach authorization")
    }
}

#[test]
fn host_panics_fail_closed_instead_of_disabling_the_gate() {
    let request = request();
    assert!(matches!(
        authorize(&PanickingHost, input(&request)),
        Err(SubmissionDenied::HostUnavailable)
    ));
}

#[test]
fn oversized_body_is_denied_before_calling_the_host() {
    let host = Authorizer::default();
    let mut request = request();
    *request.body_mut() = Some(vec![b'x'; 1025].into());
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::WorkLimit)
    ));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn source_span_allowance_is_checked_before_host_authorization() {
    let host = Authorizer::default();
    let request = request();
    let sources: Vec<_> = (0..9)
        .map(|source_id| ProjectedMessageSource {
            source_id,
            message_index: 0,
            range: 0..1,
        })
        .collect();
    let input = SubmissionAuthorization {
        sources: &sources,
        ..input(&request)
    };
    assert!(matches!(
        authorize(&host, input),
        Err(SubmissionDenied::WorkLimit)
    ));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn unsupported_http_method_is_not_sent_under_a_post_contract() {
    let host = Authorizer::default();
    let mut request = request();
    *request.method_mut() = reqwest::Method::GET;
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::UnsupportedTransport)
    ));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn credentials_in_a_destination_are_not_exposed_to_the_host() {
    let host = Authorizer::default();
    let mut request = request();
    *request.url_mut() = "https://fixture:secret@invalid.example/chat"
        .parse()
        .unwrap();
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::UnsupportedTransport)
    ));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn named_routing_header_access_borrows_the_final_value() {
    let mut headers = HeaderMap::new();
    headers.insert("x-current-message", "original-value".parse().unwrap());
    let access = SubmissionHeaderAccess {
        headers: &headers,
        credential_header: "authorization",
        allowed: &["x-current-message", "x-missing"],
    };
    assert_eq!(
        access.single_value("X-Current-Message"),
        Ok(Some(b"original-value".as_slice()))
    );
    assert_eq!(access.single_value("x-missing"), Ok(None));
}

#[test]
fn duplicate_routing_headers_are_not_normalized_into_a_false_binding() {
    let mut headers = HeaderMap::new();
    headers.append("x-current-message", "one".parse().unwrap());
    headers.append("x-current-message", "two".parse().unwrap());
    let access = SubmissionHeaderAccess {
        headers: &headers,
        credential_header: "authorization",
        allowed: &["x-current-message", "x-missing"],
    };
    assert_eq!(
        access.single_value("x-current-message"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn credential_headers_cannot_enter_context_evidence() {
    let headers = HeaderMap::new();
    let access = SubmissionHeaderAccess {
        headers: &headers,
        credential_header: "x-custom-credential",
        allowed: &[
            "authorization",
            "proxy-authorization",
            "cookie",
            "x-api-key",
            "x-custom-credential",
        ],
    };
    for name in [
        "Authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "X-Custom-Credential",
    ] {
        assert_eq!(
            access.single_value(name),
            Err(SubmissionDenied::InvalidHeaders)
        );
    }
}

#[test]
fn invalid_header_name_is_rejected_without_panicking() {
    let headers = HeaderMap::new();
    let access = SubmissionHeaderAccess {
        headers: &headers,
        credential_header: "authorization",
        allowed: &["x-current-message", "x-missing"],
    };
    assert_eq!(
        access.single_value("invalid\nheader"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn stream_error_mapping_keeps_local_denial_out_of_http_error_strings() {
    let error =
        SubmissionSendError::Denied(SubmissionDenied::PermitUnavailable).into_stream_error();
    assert!(matches!(
        error,
        StreamError::SubmissionDenied(SubmissionDenied::PermitUnavailable)
    ));
}

#[test]
fn an_absent_body_is_not_authorized_as_an_empty_json_request() {
    let host = Authorizer::default();
    let mut request = request();
    *request.body_mut() = None;
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::UnsupportedBody)
    ));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn a_uri_fragment_is_not_accepted_as_a_transport_destination_binding() {
    let host = Authorizer::default();
    let mut request = request();
    request.url_mut().set_fragment(Some("not-transmitted"));
    assert!(matches!(
        authorize(&host, input(&request)),
        Err(SubmissionDenied::UnsupportedTransport)
    ));
}

#[test]
fn zero_or_excessive_inspection_bounds_are_not_an_unbounded_opt_out() {
    for body in [0, MAX_SUBMISSION_BODY_BYTES + 1] {
        assert_eq!(
            require_limits(
                SubmissionLimits {
                    max_body_bytes: body,
                    max_source_spans: 0
                },
                1,
                0
            ),
            Err(SubmissionDenied::WorkLimit)
        );
    }
}

#[test]
fn exact_host_inspection_boundary_is_inclusive() {
    assert_eq!(
        require_limits(
            SubmissionLimits {
                max_body_bytes: 1,
                max_source_spans: 1
            },
            1,
            1
        ),
        Ok(())
    );
}

#[test]
fn excessive_source_bound_is_rejected_even_when_no_sources_were_supplied() {
    assert_eq!(
        require_limits(
            SubmissionLimits {
                max_body_bytes: 1,
                max_source_spans: MAX_SUBMISSION_SOURCE_SPANS + 1
            },
            1,
            0
        ),
        Err(SubmissionDenied::WorkLimit)
    );
}

#[test]
fn managed_header_callback_failure_is_a_local_preparation_denial() {
    let host = Authorizer::default();
    let callback: crate::compatible::RequestHeaders =
        std::sync::Arc::new(|_, _| anyhow::bail!("synthetic private preparation failure"));
    let result = crate::compatible::request_with_sources(
        crate::compatible::RequestHooks {
            headers: Some(&callback),
            observer: None,
            submission: Some(&host),
        },
        crate::extra_headers::ReservedHeaders::COMPATIBLE,
        &[],
        reqwest::Client::new().post("http://127.0.0.1/chat/completions"),
        &serde_json::json!({"model":"fixture"}),
        &[],
    );
    let error = preparation_stream_error(result.err().unwrap());
    assert!(matches!(
        error,
        StreamError::SubmissionDenied(SubmissionDenied::UnsupportedBody)
    ));
    assert!(!error.to_string().contains("private"));
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn headers_are_closed_by_default_even_when_the_request_contains_a_value() {
    let mut headers = HeaderMap::new();
    headers.insert("x-unregistered", "synthetic value".parse().unwrap());
    let access = SubmissionHeaderAccess {
        headers: &headers,
        credential_header: "authorization",
        allowed: &[],
    };
    assert_eq!(
        access.single_value("x-unregistered"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn a_host_header_declaration_cannot_expose_credentials() {
    assert_eq!(
        require_header_policy(&["Authorization"], "authorization"),
        Err(SubmissionDenied::InvalidHeaders)
    );
    assert_eq!(
        require_header_policy(&["x-custom"], "X-Custom"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn a_case_variant_cannot_duplicate_a_required_header_declaration() {
    assert_eq!(
        require_header_policy(&["x-context", "X-Context"], "authorization"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn excessive_header_name_length_is_rejected_before_parsing_or_authorization() {
    let name = "x".repeat(MAX_REQUIRED_HEADER_NAME_BYTES + 1);
    assert_eq!(
        require_header_policy(&[&name], "authorization"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}

#[test]
fn excessive_header_declaration_count_cannot_allocate_an_unbounded_policy_set() {
    let names = vec!["x-context"; MAX_REQUIRED_HEADERS + 1];
    assert_eq!(
        require_header_policy(&names, "authorization"),
        Err(SubmissionDenied::InvalidHeaders)
    );
}
