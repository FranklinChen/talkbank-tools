//! Opaque provider identifier with construction-time nonempty validation.

/// A nonempty Rev.AI server-side job identifier.
///
/// This is an opaque value: no trimming, case conversion, or format guessing.
/// Deserialization uses the same admission rule as direct construction.
/// The Rust value does not by itself certify that a provider job exists.
///
/// Infallible string conversion cannot bypass admission:
/// ```compile_fail
/// use batchalign_types::api::RevAiJobId;
/// let invalid = RevAiJobId::from(String::new());
/// ```
/// Nor can a caller construct or mutate the inner field:
/// ```compile_fail
/// use batchalign_types::api::RevAiJobId;
/// let invalid = RevAiJobId(String::new());
/// ```
/// ```compile_fail
/// use batchalign_types::api::RevAiJobId;
/// let mut id = RevAiJobId::try_new("job-1").unwrap();
/// id.0.clear();
/// ```
#[nutype::nutype(
    validate(not_empty),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        Display,
        AsRef,
        Deref,
        Borrow,
        TryFrom,
        Into,
        FromStr,
        Serialize,
        Deserialize
    )
)]
pub struct RevAiJobId(String);

// Schema implementations describe the value but cannot construct it. Keep
// the current schema versions without unchecked derives or a second stack.
impl schemars::JsonSchema for RevAiJobId {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "RevAiJobId".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "string", "minLength": 1 })
    }
}

impl utoipa::PartialSchema for RevAiJobId {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::Type::String)
            .min_length(Some(1))
            .into()
    }
}

impl utoipa::ToSchema for RevAiJobId {
    fn name() -> std::borrow::Cow<'static, str> {
        "RevAiJobId".into()
    }
}

impl PartialEq<&str> for RevAiJobId {
    fn eq(&self, other: &&str) -> bool {
        self.as_ref() == *other
    }
}
