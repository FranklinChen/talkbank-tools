//! Public constructor, codec, schema, and typed-error boundaries.

use batchalign_types::api::{RevAiJobId, RevAiJobIdError};
use serde::{Deserialize, Serialize};

static_assertions::assert_not_impl_any!(
    RevAiJobId: From<String>, From<&'static str>, Default,
    std::ops::DerefMut, AsMut<String>, AsMut<str>
);

#[derive(Debug, thiserror::Error)]
enum BoundaryError {
    #[error("invalid provider job identifier: {0}")]
    InvalidIdentifier(#[from] RevAiJobIdError),
}

fn admit_identifier(raw: &str) -> Result<RevAiJobId, BoundaryError> {
    Ok(RevAiJobId::try_new(raw)?)
}

#[test]
fn all_public_string_admission_paths_reject_empty_values() -> Result<(), Box<dyn std::error::Error>>
{
    assert!(RevAiJobId::try_new("").is_err());
    assert!(RevAiJobId::try_from(String::new()).is_err());
    assert!(RevAiJobId::try_from("").is_err());
    assert!("".parse::<RevAiJobId>().is_err());
    assert!(serde_json::from_str::<RevAiJobId>(r#""""#).is_err());
    let error = admit_identifier("")
        .err()
        .ok_or("empty identifier was admitted")?;
    assert!(std::error::Error::source(&error).is_some());
    Ok(())
}

#[test]
fn opaque_nonempty_values_preserve_their_exact_wire_representation()
-> Result<(), Box<dyn std::error::Error>> {
    for raw in ["job-1", " MiXeD/provider:id ", " ", "é", "quoted\"id"] {
        let id = admit_identifier(raw)?;
        assert_eq!(id.as_ref(), raw);
        assert_eq!(id.to_string(), raw);
        assert_eq!(id, raw);
        let wire = serde_json::to_string(&id)?;
        assert_eq!(wire, serde_json::to_string(raw)?);
        assert_eq!(serde_json::from_str::<RevAiJobId>(&wire)?, id);
        let inner: String = id.into();
        assert_eq!(inner, raw);
    }
    Ok(())
}

#[test]
fn an_optional_wire_field_distinguishes_absence_from_an_invalid_identifier()
-> Result<(), Box<dyn std::error::Error>> {
    #[derive(Debug, Serialize, Deserialize)]
    struct Envelope {
        rev_job_id: Option<RevAiJobId>,
    }
    for wire in [r#"{}"#, r#"{"rev_job_id":null}"#] {
        assert!(serde_json::from_str::<Envelope>(wire)?.rev_job_id.is_none());
    }
    for wire in [r#"{"rev_job_id":""}"#, r#"{"rev_job_id":17}"#] {
        assert!(serde_json::from_str::<Envelope>(wire).is_err());
    }
    let wire = r#"{"rev_job_id":"job-1"}"#;
    assert_eq!(
        serde_json::to_string(&serde_json::from_str::<Envelope>(wire)?)?,
        wire
    );
    Ok(())
}

#[test]
fn current_schema_stacks_describe_the_same_nonempty_string()
-> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_value(schemars::schema_for!(RevAiJobId))?;
    let openapi = serde_json::to_value(<RevAiJobId as utoipa::PartialSchema>::schema())?;
    for schema in [json, openapi] {
        assert_eq!(schema["type"], "string");
        assert_eq!(schema["minLength"], 1);
    }
    assert_eq!(<RevAiJobId as utoipa::ToSchema>::name(), "RevAiJobId");
    Ok(())
}
