"""Verify that hand-written Python Pydantic models conform to generated Rust schemas.

This test catches Rust/Python IPC type drift at CI time. When a Rust type
changes shape, the generated schemas update (via ``scripts/generate_ipc_types.sh``)
and this test fails until the hand-written Python model is updated to match.

This is not a bridge to generated Python models; that plan was retired on
2026-08-14 because generation would have cost the domain types and validators
the hand-written models carry. See ``ipc-schema/`` for the JSON Schema files
and the "Rust to Python IPC Type Sync" developer page for why one
representation per language, held to one schema, beats a generated mirror.

Cross-language contract note: this is the Python half of the schema conformance
gate. The Rust half lives in ``crates/batchalign/tests/worker_protocol_v2_compat.rs``.
Both sides must pass independently, a change to the wire format must update both.
The ``Cmd2Task`` constant map (formerly tested in ``test_runtime.py``) is also
covered by the IPC schema drift check in CI (``scripts/check_ipc_type_drift.sh``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest

# Find project root by looking for Cargo.toml
_here = Path(__file__).resolve().parent
ROOT = _here
while ROOT != ROOT.parent:
    if (ROOT / "Cargo.toml").exists() and (ROOT / "ipc-schema").exists():
        break
    ROOT = ROOT.parent
SCHEMA_DIR = ROOT / "ipc-schema"


def _load_schema(layer: str, type_name: str) -> dict:
    """Load a JSON Schema file for an IPC type."""
    path = SCHEMA_DIR / layer / f"{type_name}.json"
    if not path.exists():
        # Not a skip. A missing schema means the Rust type was renamed or
        # dropped without this test being updated, which is exactly the drift
        # the test exists to catch; skipping would report it as a pass.
        pytest.fail(
            f"Schema not found: {path}. Run 'bash scripts/generate_ipc_types.sh', "
            "or update this test if the Rust type was renamed."
        )
    return json.loads(path.read_text())


def _assert_fields_match(
    schema: dict, model_cls: type, *, known_extra_python: frozenset[str] = frozenset()
) -> None:
    """Assert that the schema's required/optional fields match the Pydantic model."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    model_fields = set(model_cls.model_fields.keys())
    schema_fields = set(props.keys())

    # Every schema field must exist in the Python model
    missing_from_python = schema_fields - model_fields
    assert not missing_from_python, (
        f"{model_cls.__name__} is missing fields defined in Rust schema: {missing_from_python}"
    )

    # Required-ness must agree on the fields both sides define. Without this the
    # helper compared only field NAMES, so a field the Rust schema demands and
    # the Python model defaults could round-trip a missing value silently.
    python_required = {
        name for name in schema_fields if model_cls.model_fields[name].is_required()
    }
    assert python_required == required, (
        f"{model_cls.__name__} required-field mismatch vs the Rust schema: "
        f"required only in Python={sorted(python_required - required)}, "
        f"required only in Rust={sorted(required - python_required)}"
    )

    # Extras must be NAMED, not merely permitted. A boolean here allows any
    # field at all, which is how a dead `lang` on UtsegBatchItem survived
    # unnoticed: the flag that admitted the legitimate extra admitted it too.
    unexpected_in_python = model_fields - schema_fields - known_extra_python
    assert not unexpected_in_python, (
        f"{model_cls.__name__} has fields not in the Rust schema and not "
        f"declared as known extras: {unexpected_in_python}"
    )
    stale_allowances = known_extra_python & schema_fields
    assert not stale_allowances, (
        f"{model_cls.__name__} declares {sorted(stale_allowances)} as Python-only, "
        "but the Rust schema now has them: drop the allowance"
    )


def _schema_variants(schema: dict) -> list[dict]:
    """The variant schemas of a generated tagged enum, with ``$ref`` resolved."""
    variants = schema.get("oneOf") or schema.get("anyOf")
    assert variants, f"{schema.get('title')} is not a tagged union in the Rust schema"
    defs = schema.get("$defs", {})
    resolved: list[dict] = []
    for variant in variants:
        if "$ref" in variant:
            variant = defs[variant["$ref"].rsplit("/", 1)[-1]]
        resolved.append(variant)
    return resolved


def _schema_enum_values(schema: dict) -> set[str]:
    """The string values a generated enum admits, in any schemars spelling.

    A single value is a bare ``const`` (how an internally tagged variant's
    ``kind`` is written); several are an ``enum`` or a union of ``const``s.
    """
    if "const" in schema:
        return {schema["const"]}
    if "enum" in schema:
        return set(schema["enum"])
    return {
        value
        for variant in _schema_variants(schema)
        for value in ([variant["const"]] if "const" in variant else variant["enum"])
    }


def _assert_tagged_union_matches(schema: dict, python_union: object) -> None:
    """Assert a Rust internally tagged enum and a pydantic discriminated union agree.

    Both directions: the same set of ``kind`` tags, and for each tag the same
    fields and required-ness. The Python variants are read off the union
    alias itself (``Annotated[A | B | C, Field(discriminator="kind")]``), so
    there is no second hand-kept list of variants here to drift. ``kind`` is
    compared as the tag, not as a field: Rust requires it on every variant,
    while each pydantic variant defaults it to its own literal.
    """
    (union, *_metadata) = get_args(python_union)
    python_by_tag = {
        variant.model_fields["kind"].default: variant for variant in get_args(union)
    }
    schema_by_tag = {
        _schema_enum_values(variant["properties"]["kind"]).pop(): variant
        for variant in _schema_variants(schema)
    }
    assert python_by_tag.keys() == schema_by_tag.keys(), (
        f"tag mismatch vs the Rust schema: only in Python="
        f"{sorted(python_by_tag.keys() - schema_by_tag.keys())}, only in Rust="
        f"{sorted(schema_by_tag.keys() - python_by_tag.keys())}"
    )
    for tag, variant_schema in schema_by_tag.items():
        without_tag = {
            "properties": {
                name: field
                for name, field in variant_schema.get("properties", {}).items()
                if name != "kind"
            },
            "required": [
                name for name in variant_schema.get("required", []) if name != "kind"
            ],
        }
        _assert_fields_match(
            without_tag, python_by_tag[tag], known_extra_python=frozenset({"kind"})
        )


class TestBatchItemConformance:
    """Verify batch item types match Rust schemas."""

    def test_morphosyntax_batch_item(self) -> None:
        from batchalign.inference.morphosyntax import MorphosyntaxBatchItem

        schema = _load_schema("batch_items", "MorphosyntaxBatchItem")
        _assert_fields_match(schema, MorphosyntaxBatchItem)

    def test_utseg_batch_item(self) -> None:
        from batchalign.inference.utseg import UtsegBatchItem

        schema = _load_schema("batch_items", "UtsegBatchItem")
        _assert_fields_match(schema, UtsegBatchItem)

    def test_translate_batch_item(self) -> None:
        from batchalign.inference.translate import TranslateBatchItem

        schema = _load_schema("batch_items", "TranslateBatchItem")
        _assert_fields_match(schema, TranslateBatchItem)

    def test_coref_batch_item(self) -> None:
        from batchalign.inference.coref import CorefBatchItem

        schema = _load_schema("batch_items", "CorefBatchItem")
        _assert_fields_match(schema, CorefBatchItem)

    def test_chain_ref(self) -> None:
        from batchalign.inference.coref import ChainRef

        schema = _load_schema("batch_items", "ChainRef")
        _assert_fields_match(schema, ChainRef)


class TestWorkerV2Conformance:
    """Verify selected V2 protocol types match Rust schemas."""

    def test_execute_request(self) -> None:
        from batchalign.worker._types_v2 import ExecuteRequestV2

        schema = _load_schema("worker_v2", "ExecuteRequestV2")
        _assert_fields_match(schema, ExecuteRequestV2)

    def test_execute_response(self) -> None:
        from batchalign.worker._types_v2 import ExecuteResponseV2

        schema = _load_schema("worker_v2", "ExecuteResponseV2")
        _assert_fields_match(schema, ExecuteResponseV2)

    def test_morphosyntax_item_result(self) -> None:
        from batchalign.worker._types_v2 import MorphosyntaxItemResultV2

        schema = _load_schema("worker_v2", "MorphosyntaxItemResultV2")
        _assert_tagged_union_matches(schema, MorphosyntaxItemResultV2)

    def test_translation_item_result(self) -> None:
        from batchalign.worker._types_v2 import TranslationItemResultV2

        schema = _load_schema("worker_v2", "TranslationItemResultV2")
        _assert_tagged_union_matches(schema, TranslationItemResultV2)

    def test_coref_item_result(self) -> None:
        from batchalign.worker._types_v2 import CorefItemResultV2

        schema = _load_schema("worker_v2", "CorefItemResultV2")
        _assert_tagged_union_matches(schema, CorefItemResultV2)

    def test_morphosyntax_model_identity(self) -> None:
        from batchalign.worker._types_v2 import (
            MorphosyntaxModelIdentityV2,
            MorphosyntaxPipelineV2,
        )

        schema = _load_schema("worker_v2", "MorphosyntaxModelIdentityV2")
        _assert_fields_match(schema, MorphosyntaxModelIdentityV2)
        # The variant vocabulary is a closed set on both sides; a variant
        # added to one language only would be refused by the other at runtime.
        pipeline = schema.get("$defs", {}).get("MorphosyntaxPipelineV2")
        assert pipeline is not None, "schema lost the MorphosyntaxPipelineV2 definition"
        assert _schema_enum_values(pipeline) == {
            variant.value for variant in MorphosyntaxPipelineV2
        }

    def test_ud_relation_repair(self) -> None:
        """The repair an analyzed item carries must read the same on both sides.

        The relation vocabulary is closed in Rust and in Python, so a kind
        added to one language only would be refused by the other at runtime,
        and a repair that could not be read is a repair the file's provenance
        would never count.
        """
        from batchalign.worker._types_v2 import (
            UdRelationRepairKindV2,
            UdRelationRepairV2,
        )

        schema = _load_schema("worker_v2", "UdRelationRepairV2")
        _assert_fields_match(schema, UdRelationRepairV2)
        kind = schema.get("$defs", {}).get("UdRelationRepairKindV2")
        assert kind is not None, "schema lost the UdRelationRepairKindV2 definition"
        assert _schema_enum_values(kind) == {
            variant.value for variant in UdRelationRepairKindV2
        }

    def test_whisper_chunk_span(self) -> None:
        from batchalign.worker._types_v2 import WhisperChunkSpanV2

        schema = _load_schema("worker_v2", "WhisperChunkSpanV2")
        _assert_fields_match(schema, WhisperChunkSpanV2)

    def test_asr_element(self) -> None:
        from batchalign.worker._types_v2 import AsrElementV2

        schema = _load_schema("worker_v2", "AsrElementV2")
        _assert_fields_match(schema, AsrElementV2)

    def test_indexed_word_timing(self) -> None:
        from batchalign.worker._types_v2 import IndexedWordTimingV2

        schema = _load_schema("worker_v2", "IndexedWordTimingV2")
        _assert_fields_match(schema, IndexedWordTimingV2)

    def test_speaker_segment(self) -> None:
        from batchalign.worker._types_v2 import SpeakerSegmentV2

        schema = _load_schema("worker_v2", "SpeakerSegmentV2")
        _assert_fields_match(schema, SpeakerSegmentV2)

    def test_morphosyntax_request(self) -> None:
        from batchalign.worker._types_v2 import MorphosyntaxRequestV2

        schema = _load_schema("worker_v2", "MorphosyntaxRequestV2")
        # Python adds `kind` for Pydantic discrimination; Rust schema doesn't
        # include it (added by the tagged enum wrapper at serialization time).
        _assert_fields_match(
            schema, MorphosyntaxRequestV2, known_extra_python=frozenset({"kind"})
        )

    def test_forced_alignment_request(self) -> None:
        from batchalign.worker._types_v2 import ForcedAlignmentRequestV2

        schema = _load_schema("worker_v2", "ForcedAlignmentRequestV2")
        _assert_fields_match(
            schema, ForcedAlignmentRequestV2, known_extra_python=frozenset({"kind"})
        )

    def test_asr_request(self) -> None:
        from batchalign.worker._types_v2 import AsrRequestV2

        schema = _load_schema("worker_v2", "AsrRequestV2")
        _assert_fields_match(
            schema, AsrRequestV2, known_extra_python=frozenset({"kind"})
        )

    def test_provider_diarization(self) -> None:
        """The two states of "separate speakers?" must agree across the wire.

        This union is what the bridge hands a provider adapter, so a tag spelled
        differently on one side would be a request the other cannot read. It was
        unwitnessed here until 2026-09-16, when the same question stopped being
        an integer on the Python half of the boundary.
        """
        from batchalign.worker._types_v2 import ProviderDiarizationV2

        schema = _load_schema("worker_v2", "ProviderDiarizationV2")
        _assert_tagged_union_matches(schema, ProviderDiarizationV2)

    def test_decode_budget_realtime_factor_matches_rust(self) -> None:
        """Pin Python's realtime factor to Rust's copy of the same constant.

        The two are hand-written in different languages (see the module
        docstring: Python V2 models are deliberately hand-written, so there
        is no codegen seam that would carry a bare numeric constant across
        the boundary the way a schema field carries a type shape). This
        test is the conformance check the field's own doc comments on both
        sides point at; if either constant changes, this fails until the
        other is updated to match.
        """
        from batchalign.inference.languages.cantonese._qwen_chunking import (
            DEADLINE_REALTIME_FACTOR,
        )

        # Mirrors `DEADLINE_REALTIME_FACTOR` in
        # `crates/batchalign-types/src/worker_v2/requests.rs`. Update both
        # together.
        rust_deadline_realtime_factor = 12.8
        assert DEADLINE_REALTIME_FACTOR == rust_deadline_realtime_factor, (
            "Python's DEADLINE_REALTIME_FACTOR "
            f"({DEADLINE_REALTIME_FACTOR}) has drifted from Rust's "
            f"({rust_deadline_realtime_factor}); the two ceilings this "
            "factor drives (Python's decode budget, Rust's transport "
            "ceiling) must be computed from the same number."
        )
