use super::OpenAiCompatibleModelProvider;
pub use crate::completion_control::CompletionBodyLimitExceeded;
use crate::completion_control::read_bounded;
use anyhow::Context;

impl OpenAiCompatibleModelProvider {
    pub(super) async fn completion_text(
        &self,
        response: reqwest::Response,
    ) -> anyhow::Result<String> {
        let Some(limit) = self.completion_body_limit else {
            return Ok(response.text().await?);
        };
        let bytes = read_bounded(response, limit).await?;
        String::from_utf8(bytes).context("decode bounded compatible completion as UTF-8")
    }

    pub(super) async fn completion_json<T: serde::de::DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> anyhow::Result<T> {
        let Some(limit) = self.completion_body_limit else {
            return Ok(response.json().await?);
        };
        let bytes = read_bounded(response, limit).await?;
        serde_json::from_slice(&bytes).context("parse bounded compatible completion JSON")
    }

    pub(super) async fn completion_error(&self, response: reqwest::Response) -> anyhow::Error {
        if self.completion_body_limit.is_none() {
            return super::super::api_error(&self.name, response).await;
        }
        let status = response.status();
        match self.completion_text(response).await {
            Ok(text) => self.bounded_api_error(status, &text),
            Err(error) => error,
        }
    }

    fn bounded_api_error(&self, status: reqwest::StatusCode, text: &str) -> anyhow::Error {
        ::zeroclaw_log::record!(
            ERROR,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(serde_json::json!({"http_status": status.as_u16()})),
            "bounded compatible completion HTTP error"
        );
        anyhow::Error::msg(format!(
            "{} API error ({status}): {}",
            self.name,
            super::super::sanitize_api_error(text)
        ))
    }
}
