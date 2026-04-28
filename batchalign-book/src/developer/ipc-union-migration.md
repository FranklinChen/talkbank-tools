# IPC Discriminated Union Migration Plan

**Status:** Draft
**Last updated:** 2026-04-02 20:47 EDT

## Problem

The V2 worker protocol uses **discriminated unions** (`TaskRequestV2`,
`TaskResultV2`, `ExecuteOutcomeV2`, `ArtifactRefV2`, `AsrInputV2`,
`SpeakerInputV2`) that serialize as `{"kind": "variant", "data": {...}}`.

Python represents these as explicit wrapper classes:

```python
class AsrTaskRequestV2(BaseModel):
    kind: Literal["asr"] = "asr"
    data: AsrRequestV2

class FaTaskRequestV2(BaseModel):
    kind: Literal["forced_alignment"] = "forced_alignment"
    data: ForcedAlignmentRequestV2

TaskRequestV2 = Annotated[
    AsrTaskRequestV2 | FaTaskRequestV2 | ...,
    Field(discriminator="kind")
]
```

This is ~32 wrapper classes across `_types_v2.py`. They're verbose, must be
manually kept in sync with Rust, and `datamodel-code-generator` cannot
generate them from JSON Schema (it produces flat unions instead).

**This is the gate for full codegen.** Until solved, `_types_v2.py`
requires hand-written wrapper classes.

## Rust Side: Current

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum TaskRequestV2 {
    #[serde(rename = "asr")]
    Asr(AsrRequestV2),
    #[serde(rename = "forced_alignment")]
    ForcedAlignment(ForcedAlignmentRequestV2),
    ...
}
```

schemars generates `oneOf` with `{"kind": "...", "data": ref}` items.

## Proposed Fix: Explicit Rust Wrapper Structs

Add explicit wrapper structs on the Rust side that schemars can generate
individual schemas for, then let `datamodel-code-generator` handle them:

```rust
/// Wrapper for the `asr` variant of [`TaskRequestV2`].
#[derive(Serialize, Deserialize, JsonSchema)]
pub struct AsrTaskRequestV2 {
    pub kind: AsrTaskKind,
    pub data: AsrRequestV2,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub enum AsrTaskKind {
    #[serde(rename = "asr")]
    Asr,
}
```

Then `TaskRequestV2` becomes:

```rust
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]  // let the inner wrappers handle discriminators
pub enum TaskRequestV2 {
    Asr(AsrTaskRequestV2),
    ForcedAlignment(FaTaskRequestV2),
    ...
}
```

**Pros:**
- schemars emits individual schemas per wrapper → codegen produces Python
  classes that match current hand-written patterns
- Wire format unchanged (still `{"kind": "asr", "data": {...}}`)
- Existing Python code doesn't need to change during transition

**Cons:**
- More Rust code (~2 lines per variant × 30 variants = ~60 lines)
- Two representations in Rust (enum + wrapper structs)

## Alternative: Switch Python to Non-Wrapped Unions

Instead of making Rust emit wrapper structs, switch Python to use
`datamodel-code-generator`'s natural output (non-wrapped unions):

```python
# Current (wrapper pattern):
class AsrTaskRequestV2(BaseModel):
    kind: Literal["asr"] = "asr"
    data: AsrRequestV2

# Alternative (non-wrapped):
TaskRequestV2 = Annotated[
    AsrRequestV2 | ForcedAlignmentRequestV2 | ...,
    Discriminator("kind")  # Pydantic 2.0+ native discriminator
]
```

This requires adding a `kind` field directly to each inner struct
(`AsrRequestV2`, `ForcedAlignmentRequestV2`, etc.) — which changes
their shape. More disruptive but fewer types overall.

## Recommended Approach

**Option 1 (Explicit wrappers in Rust)** is safer:
- No wire format change
- No existing code breakage
- Clear migration path: add wrappers → regenerate → verify → swap imports

**Effort:** 3-4 hours for all 6 discriminated unions (~30 wrapper structs).

## Affected Types

| Union | Variants | Serde Pattern |
|-------|----------|---------------|
| `TaskRequestV2` | 9 | `tag="kind", content="data"` |
| `TaskResultV2` | 11 | `tag="kind", content="data"` |
| `ExecuteOutcomeV2` | 2 | `tag="kind"` (internally tagged) |
| `ArtifactRefV2` | 3 | `tag="kind"` |
| `AsrInputV2` | 3 | `tag="kind", content="data"` |
| `SpeakerInputV2` | 2 | `tag="kind", content="data"` |
| **Total** | **30** | |

## Prerequisites

- [x] schemars derives on all V2 types
- [x] JSON Schema generation pipeline working
- [x] Conformance tests passing (15/15)
- [ ] Phase 2: Add wrapper structs to Rust
- [ ] Phase 2: Regenerate schemas
- [ ] Phase 2: Verify `datamodel-code-generator` produces correct wrappers
- [ ] Phase 3: Replace `_types_v2.py` with generated re-exports + validators
