use super::*;

struct UnsupportedProvider;

impl crate::attribution::Attributable for UnsupportedProvider {
    fn role(&self) -> crate::attribution::Role {
        crate::attribution::Role::Provider(crate::attribution::ProviderKind::Model(
            crate::attribution::ModelProviderKind::OpenAi,
        ))
    }
    fn alias(&self) -> &str {
        "unsupported-fixture"
    }
}

#[async_trait]
impl ModelProvider for UnsupportedProvider {
    async fn chat_with_system(
        &self,
        _: Option<&str>,
        _: &str,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("fixture must not execute")
    }
}

#[test]
fn completion_control_retains_the_declared_decoded_body_ceiling() {
    let maximum = std::num::NonZeroUsize::MIN;
    assert_eq!(
        CompletionControl::bounded_single_call(maximum).maximum_body_bytes(),
        maximum
    );
}

#[test]
fn unsupported_adapters_reject_completion_control_with_a_structured_capability_error() {
    let control = CompletionControl::bounded_single_call(std::num::NonZeroUsize::MIN);
    let error = UnsupportedProvider
        .configure_completion(control)
        .unwrap_err();
    assert_eq!(error.capability, "bounded_single_completion");
    assert_eq!(error.model_provider, "unsupported-fixture");
}
