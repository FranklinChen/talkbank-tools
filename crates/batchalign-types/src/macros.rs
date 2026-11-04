//! Helper macros for declaring newtype wrappers.
//!
//! This is a local copy kept for `scheduling.rs` and any future batchalign
//! types. The canonical copy lives in `batchalign-types`.
//!
//! All generated types use `#[serde(transparent)]` so the wire format is
//! unchanged: JSON values remain bare strings or numbers.

/// Declare a `String`-wrapping newtype with serde-transparent serialization.
///
/// Derives: `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Serialize`,
/// `Deserialize`, `ToSchema`, `JsonSchema`, plus `Display`, `From<String>`,
/// `From<&str>`, `Into<String>`, `Deref<Target=str>`, `AsRef<str>`,
/// `PartialEq<&str>`.
macro_rules! string_id {
    ($(#[$meta:meta])* $vis:vis $name:ident) => {
        $(#[$meta])*
        // `PartialOrd`/`Ord` are derived (added 2026-07-29) so a string id can
        // key a `BTreeMap`/`BTreeSet` directly instead of being downgraded to a
        // raw `String` at the boundary that needs ordering. The crate already
        // mandates `BTreeMap` for deterministic JSON, and the first consumer is
        // `BatchInferProgress`, whose language groups are keyed by
        // `LanguageCode3`. The derived order is the underlying `String` order,
        // which agrees with the `Borrow<str>` impl below, so lookups by `&str`
        // stay correct.
        #[derive(
            Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord,
            serde::Serialize, serde::Deserialize,
            utoipa::ToSchema,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        $vis struct $name(pub String);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self { Self(s) }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self { Self(s.to_owned()) }
        }

        impl From<$name> for String {
            fn from(v: $name) -> String { v.0 }
        }

        impl std::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str { &self.0 }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str { &self.0 }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool { self.0 == *other }
        }

        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str { &self.0 }
        }

        impl Default for $name {
            fn default() -> Self { Self(String::new()) }
        }
    };
}

/// Declare a validated `String`-wrapping newtype that rejects empty strings
/// on deserialization.
///
/// **Known compromise:** `From<String>` and `From<&str>` are still infallible
/// and skip validation. They exist for trusted boundaries (HTTP path params
/// extracted by axum, DB rows, test code). Untrusted input goes through the
/// validating `Deserialize` impl. A full `TryFrom` migration is tracked but
/// would touch hundreds of call sites, see AGENTS.md rule 6a.
///
/// Generates: `From<String>`, `From<&str>`, custom `Deserialize` (rejects
/// empty), `Display`, `Deref<Target=str>`, `AsRef<str>`.
/// Does NOT generate `Default` (empty value is never valid).
///
/// For additional validation beyond non-empty, pass a closure:
/// ```ignore
/// validated_string_id!(
///     /// Basename of a file (no path separators).
///     pub DisplayPath
///     |s| !s.contains('/') && !s.contains('\\')
///     "must not contain path separators"
/// );
/// ```
macro_rules! validated_string_id {
    // With custom validation predicate
    ($(#[$meta:meta])* $vis:vis $name:ident |$v:ident| $pred:expr, $msg:literal) => {
        validated_string_id!(@base $(#[$meta])* $vis $name);
        validated_string_id!(@validation $name |$v| { !$v.is_empty() && { let $v = $v; $pred } }, concat!("empty or invalid ", stringify!($name), ": ", $msg));
    };
    // Non-empty only (default)
    ($(#[$meta:meta])* $vis:vis $name:ident) => {
        validated_string_id!(@base $(#[$meta])* $vis $name);
        validated_string_id!(@validation $name |_v| { !_v.is_empty() }, concat!(stringify!($name), " must not be empty"));
    };
    (@base $(#[$meta:meta])* $vis:vis $name:ident) => {
        $(#[$meta])*
        // `PartialOrd`/`Ord` order by the underlying string, so a validated
        // identifier can key a `BTreeMap` or `BTreeSet` without a hand-written
        // comparison.
        #[derive(
            Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord,
            serde::Serialize,
            utoipa::ToSchema,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        $vis struct $name(pub String);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        // From<String> / From<&str> are infallible and skip validation.
        // They exist because validated_string_id types are used at trusted
        // boundaries (HTTP path params from axum, DB rows, test code) where
        // the value is known-good.  Untrusted input goes through the
        // validating `Deserialize` impl or `TryFrom`.
        impl From<String> for $name {
            fn from(s: String) -> Self { Self(s) }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self { Self(s.to_owned()) }
        }

        impl From<$name> for String {
            fn from(v: $name) -> String { v.0 }
        }

        impl std::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str { &self.0 }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str { &self.0 }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool { self.0 == *other }
        }

        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str { &self.0 }
        }

        // No Default impl: an empty identifier is always a bug.
    };
    (@validation $name:ident |$v:ident| $check:block, $msg:expr) => {
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let s = String::deserialize(deserializer)?;
                let $v = s.as_str();
                if $check { Ok(Self(s)) } else { Err(serde::de::Error::custom(format!($msg))) }
            }
        }
    };
}

/// Declare a numeric newtype whose value is proven at construction.
///
/// The field is private and the only route in is `TryFrom<inner>`, which
/// serde also deserializes through, so an out-of-range number is refused at
/// the boundary rather than checked downstream. The schemas are written from
/// the `schema(..)` clause because the derives read the schema off the
/// `try_from` type and drop the range, which would leave the Python
/// conformance check and the OpenAPI document blind to the bound the
/// constructor enforces. One clause feeds both the JSON Schema (schemars, for
/// the worker IPC schema) and the OpenAPI schema (utoipa), so the two cannot
/// disagree; `NonNegativeSeconds` used to restate its schema by hand for
/// utoipa.
///
/// ```ignore
/// validated_numeric!(
///     /// docs
///     pub NonNegativeSeconds(f64 = "f64"), InvalidSeconds,
///     |v| v.is_finite() && v >= 0.0, "seconds must be finite and non-negative",
///     schema(number, "double", minimum 0.0, description "A finite, non-negative number of seconds.")
/// );
/// ```
/// The inner type is written twice, as a type and as the string serde's
/// `try_from`/`into` attributes require. The schema type is `number` or
/// `integer`; `maximum` is optional. Append `[Eq]` after the schema for
/// integer types that also need `Eq` and `Hash`.
macro_rules! validated_numeric {
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty = $inner_str:literal), $error:ident,
     |$v:ident| $pred:expr, $msg:literal,
     schema($ty:ident, $format:literal, minimum $min:literal $(, maximum $max:literal)?, description $desc:literal) [Eq]) => {
        validated_numeric!(@base $(#[$meta])* $vis $name($inner = $inner_str), $error, |$v| $pred, $msg,
            schema($ty, $format, minimum $min $(, maximum $max)?, description $desc));

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty = $inner_str:literal), $error:ident,
     |$v:ident| $pred:expr, $msg:literal,
     schema($ty:ident, $format:literal, minimum $min:literal $(, maximum $max:literal)?, description $desc:literal)) => {
        validated_numeric!(@base $(#[$meta])* $vis $name($inner = $inner_str), $error, |$v| $pred, $msg,
            schema($ty, $format, minimum $min $(, maximum $max)?, description $desc));
    };
    (@utoipa_type number) => { utoipa::openapi::schema::Type::Number };
    (@utoipa_type integer) => { utoipa::openapi::schema::Type::Integer };
    (@base $(#[$meta:meta])* $vis:vis $name:ident($inner:ty = $inner_str:literal), $error:ident,
     |$v:ident| $pred:expr, $msg:literal,
     schema($ty:ident, $format:literal, minimum $min:literal $(, maximum $max:literal)?, description $desc:literal)) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
        #[serde(try_from = $inner_str, into = $inner_str)]
        $vis struct $name($inner);

        #[doc = concat!("Why a value was refused as a `", stringify!($name), "`: ", $msg, ".")]
        #[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
        #[error("{}, got {}", $msg, .0)]
        $vis struct $error(pub $inner);

        impl $name {
            /// The proven value.
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl TryFrom<$inner> for $name {
            type Error = $error;

            fn try_from($v: $inner) -> Result<Self, Self::Error> {
                if $pred { Ok(Self($v)) } else { Err($error($v)) }
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl schemars::JsonSchema for $name {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                stringify!($name).into()
            }

            fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
                schemars::json_schema!({
                    "type": stringify!($ty),
                    "format": $format,
                    "minimum": $min,
                    $("maximum": $max,)?
                    "description": $desc
                })
            }
        }

        impl utoipa::PartialSchema for $name {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(validated_numeric!(@utoipa_type $ty))
                    .format(Some(utoipa::openapi::SchemaFormat::Custom($format.to_owned())))
                    .minimum(Some($min))
                    $(.maximum(Some($max)))?
                    .description(Some($desc))
                    .into()
            }
        }

        impl utoipa::ToSchema for $name {
            fn name() -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed(stringify!($name))
            }
        }
    };
}

/// Declare a numeric newtype with serde-transparent serialization.
///
/// Derives: `Debug`, `Clone`, `Copy`, `PartialEq`, `Serialize`,
/// `Deserialize`, `ToSchema`, plus `Display`, `From<inner>`,
/// `Into<inner>`, `Deref<Target=inner>`, `PartialEq<inner>`.
///
/// Append `[Eq]` for integer types that also need `Eq` and `Hash`:
/// ```ignore
/// numeric_id!(pub DurationMs(u64) [Eq]);
/// ```
macro_rules! numeric_id {
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty) [Eq]) => {
        numeric_id!(@base $(#[$meta])* $vis $name($inner));

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty)) => {
        numeric_id!(@base $(#[$meta])* $vis $name($inner));
    };
    (@base $(#[$meta:meta])* $vis:vis $name:ident($inner:ty)) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, PartialOrd,
            serde::Serialize, serde::Deserialize,
            utoipa::ToSchema,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        $vis struct $name(pub $inner);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$inner> for $name {
            fn from(v: $inner) -> Self { Self(v) }
        }

        impl From<$name> for $inner {
            fn from(v: $name) -> $inner { v.0 }
        }

        impl std::ops::Deref for $name {
            type Target = $inner;
            fn deref(&self) -> &$inner { &self.0 }
        }

        impl PartialEq<$inner> for $name {
            fn eq(&self, other: &$inner) -> bool { self.0 == *other }
        }

        impl Default for $name {
            fn default() -> Self { Self(Default::default()) }
        }
    };
}

/// Declare a closed wire vocabulary: a fieldless enum whose every variant has
/// ONE name, read by its serde form, its JSON Schema constants and `ALL`.
///
/// The variants, `ALL` and `wire_name` used to be written separately for each
/// such enum (six copies in `worker_v2/responses.rs` alone), with nothing but
/// a reader holding `ALL` complete: a variant added to the enum and omitted
/// from `ALL` left every `match` arm accounted for and compiled. Generating all
/// of them from this one invocation removes the possibility, as `infer_tasks!`
/// does for `InferTask`.
///
/// `unknown` is the noun of the deserialization error (`unknown <noun> "x"`);
/// `schema` is the description the JSON Schema carries.
///
/// ```ignore
/// closed_wire_enum! {
///     /// Why a feature value was removed.
///     pub enum UdFeatureRepairKindV2 {
///         unknown = "UD feature repair",
///         schema = "Why a feature value ... Closed: ...",
///         /// Padding as a value.
///         PadFeatureValue => "pad_feature_value",
///     }
/// }
/// ```
macro_rules! closed_wire_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            unknown = $unknown:literal,
            schema = $schema:literal,
            $( $(#[$vmeta:meta])* $variant:ident => $wire:literal ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant, )+
        }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];

            /// The variant's one name: its wire form and its schema constant.
            pub const fn wire_name(self) -> &'static str {
                match self {
                    $( Self::$variant => $wire, )+
                }
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.wire_name())
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let text = <std::borrow::Cow<'de, str> as serde::Deserialize>::deserialize(deserializer)?;
                Self::ALL
                    .iter()
                    .copied()
                    .find(|variant| variant.wire_name() == text)
                    .ok_or_else(|| {
                        serde::de::Error::custom(format!(concat!("unknown ", $unknown, " {:?}"), text))
                    })
            }
        }

        impl schemars::JsonSchema for $name {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                stringify!($name).into()
            }

            fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
                let variants: Vec<serde_json::Value> = Self::ALL
                    .iter()
                    .map(|variant| serde_json::json!({"type": "string", "const": variant.wire_name()}))
                    .collect();
                schemars::json_schema!({
                    "description": $schema,
                    "oneOf": variants,
                })
            }
        }
    };
}
