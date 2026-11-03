use super::*;
use crate::request_submission::receipt::{
    AuthorizedSubmission, SubmissionReceipt, observe_model_response,
};
use crate::request_submission::{SubmissionAuthorization, SubmissionSendError, authorize};
use zeroclaw_api::model_provider::submission::SubmissionDenied;

pub(super) fn observe_parsed(receipt: Option<&SubmissionReceipt>, response: &ApiChatResponse) {
    if !response.choices.is_empty() {
        observe_model_response(receipt);
    }
}

pub(super) fn observe_chunks(
    receipt: Option<SubmissionReceipt>,
    input: stream::BoxStream<'static, StreamResult<StreamChunk>>,
) -> stream::BoxStream<'static, StreamResult<StreamChunk>> {
    let Some(receipt) = receipt else {
        return input;
    };
    input
        .map(move |item| {
            if item.as_ref().is_ok_and(is_provider_chunk) {
                observe_model_response(Some(&receipt));
            }
            item
        })
        .boxed()
}

pub(super) fn observe_events(
    receipt: Option<SubmissionReceipt>,
    input: stream::BoxStream<'static, StreamResult<StreamEvent>>,
) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
    let Some(receipt) = receipt else {
        return input;
    };
    input
        .map(move |item| {
            if item.as_ref().is_ok_and(is_provider_event) {
                observe_model_response(Some(&receipt));
            }
            item
        })
        .boxed()
}

fn is_provider_chunk(chunk: &StreamChunk) -> bool {
    !chunk.is_final
        && (!chunk.delta.is_empty()
            || chunk
                .reasoning
                .as_ref()
                .is_some_and(|text| !text.is_empty()))
}

fn is_provider_event(event: &StreamEvent) -> bool {
    match event {
        StreamEvent::TextDelta(chunk) => is_provider_chunk(chunk),
        StreamEvent::ReasoningContent(text) => !text.is_empty(),
        StreamEvent::ToolCall(_) => true,
        _ => false,
    }
}

pub(super) enum SubmissionMode {
    Complete,
    Stream,
}

pub(super) async fn publish_submission_error<Event: Send>(
    sender: &tokio::sync::mpsc::Sender<StreamResult<Event>>,
    error: StreamError,
) {
    if sender.send(Err(error)).await.is_err() {
        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
            "Submission error receiver closed"
        );
    }
}

pub(super) struct SubmissionSendInput<'a> {
    pub(super) builder: reqwest::RequestBuilder,
    pub(super) sources: &'a [ProjectedMessageSource],
}

pub(super) struct CompleteSubmission<'a, Body> {
    pub(super) body: &'a Body,
    pub(super) sources: &'a [ProjectedMessageSource],
    pub(super) credential: Option<&'a str>,
}

pub(super) enum CompleteSubmissionError {
    Preparation(anyhow::Error),
    Send(SubmissionSendError),
}

impl CompleteSubmissionError {
    pub(super) fn is_transport(&self) -> bool {
        matches!(self, Self::Send(SubmissionSendError::Http(_)))
    }
    pub(super) fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Preparation(error) => error,
            Self::Send(SubmissionSendError::Denied(reason)) => reason.into(),
            Self::Send(SubmissionSendError::Http(error)) => error.into(),
        }
    }
}

impl OpenAiCompatibleModelProvider {
    pub(super) async fn send_complete<Body: serde::Serialize + Sync>(
        &self,
        input: CompleteSubmission<'_, Body>,
    ) -> Result<reqwest::Response, CompleteSubmissionError> {
        let client = self
            .submission_request_client(SubmissionMode::Complete)
            .map_err(|reason| CompleteSubmissionError::Send(reason.into()))?;
        let builder = request_with_sources(
            self.request_hooks(),
            crate::extra_headers::ReservedHeaders::COMPATIBLE,
            &self.additional_reserved_headers(input.credential, &[]),
            client.post(self.chat_completions_url()),
            input.body,
            input.sources,
        )
        .map_err(CompleteSubmissionError::Preparation)?;
        let builder = self.apply_auth_header(builder, input.credential);
        self.send_submission(SubmissionSendInput {
            builder,
            sources: input.sources,
        })
        .await
        .map_err(CompleteSubmissionError::Send)
    }

    pub(super) async fn send_submission(
        &self,
        input: SubmissionSendInput<'_>,
    ) -> Result<reqwest::Response, SubmissionSendError> {
        let Some(authorizer) = &self.submission_authorizer else {
            return input
                .builder
                .send()
                .await
                .map_err(SubmissionSendError::Http);
        };
        let (client, request) = input.builder.build_split();
        let request = request.map_err(|_| SubmissionDenied::UnsupportedBody)?;
        let lease = authorize(
            authorizer.as_ref(),
            SubmissionAuthorization {
                request: &request,
                sources: input.sources,
                credential_header: self.credential_header_name(),
            },
        )?;
        let submission = AuthorizedSubmission::new(lease);
        submission.dispatch()?;
        let mut response = client
            .execute(request)
            .await
            .map_err(SubmissionSendError::Http)?;
        if let Some(receipt) = submission.complete(response.status().as_u16()) {
            response.extensions_mut().insert(receipt);
        }
        Ok(response)
    }

    pub(super) fn submission_request_client(
        &self,
        mode: SubmissionMode,
    ) -> Result<Client, SubmissionDenied> {
        if self.submission_authorizer.is_none() {
            return Ok(match mode {
                SubmissionMode::Complete => self.http_client(),
                SubmissionMode::Stream => self.streaming_http_client(),
            });
        }
        self.submission_client(mode)
    }

    fn credential_header_name(&self) -> &str {
        match &self.auth_header {
            AuthStyle::XApiKey => "x-api-key",
            AuthStyle::Custom(name) => name,
            AuthStyle::Bearer | AuthStyle::ZhipuJwt => "authorization",
        }
    }

    fn submission_client(&self, mode: SubmissionMode) -> Result<Client, SubmissionDenied> {
        // The legacy shared client permits automatic redirects/retries. Never
        // let an authorized one-send body silently dispatch a second time.
        let mut builder = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(std::time::Duration::from_secs(10));
        builder = match mode {
            SubmissionMode::Complete => {
                builder.timeout(std::time::Duration::from_secs(self.timeout_secs))
            }
            SubmissionMode::Stream => builder.read_timeout(STREAM_IDLE_TIMEOUT),
        };
        if let Some(certificate) = &self.tls_ca_cert_pem {
            let certificate = reqwest::Certificate::from_pem(certificate)
                .map_err(|_| SubmissionDenied::UnsupportedTransport)?;
            builder = builder.add_root_certificate(certificate);
        }
        let proxy = zeroclaw_config::schema::runtime_proxy_config();
        proxy
            .validate()
            .map_err(|_| SubmissionDenied::UnsupportedTransport)?;
        proxy
            .apply_to_reqwest_builder(builder, "model_provider.compatible")
            .build()
            .map_err(|_| SubmissionDenied::UnsupportedTransport)
    }
}
