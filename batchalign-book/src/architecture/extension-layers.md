# Extension Layers

Batchalign still needs more than one extension boundary, but the current
release does not ship a Python entry-point plugin system. The important split
today is between built-in inference providers and CHAT-aware pipeline
operations.

## The Three Layers

### 1. Core Primitives

The Rust core owns:

- CHAT parsing and serialization
- AST-safe mutation
- extraction and injection helpers
- alignment and validation invariants

Examples:

- extract utterance words
- inject `%mor/%gra`
- inject timing bullets and `%wor`
- split utterances
- preserve speaker and tier invariants

This is the only layer that should directly own low-level CHAT mutation rules.

### 2. Inference Providers

Inference providers are pure task adapters. In the current release they are
built-in modules such as `batchalign/inference/hk/_tencent_asr.py` or
`batchalign/inference/hk/_cantonese_fa.py`.

They receive typed task payloads and return typed task results. They do not
parse `.cha` files or mutate CHAT directly.

Examples:

- ASR provider: `audio_path -> tagged raw ASR payload`
- FA provider: `FaInferItem -> indexed timings`
- translation provider: `text -> translated text`
- morphosyntax provider: `words/lang -> UD response`

HK/Cantonese engines belong here. Their Cantonese normalization and jyutping
conversion are provider-local preprocessing, not CHAT orchestration.

Provider outputs may still affect CHAT transitively, but only because pipeline
layers later inject those typed results into parsed documents.

### 3. Pipeline Operations

Pipeline operations are CHAT-aware orchestration layers.

They may:

- choose extraction strategy
- batch requests
- call one or more providers
- read or write cache entries
- inject results into the parsed document
- apply task-specific validation and recovery

Examples:

- morphotag-style workflows
- task pipelines that need special pre/postprocessing around inference
- future domain-specific annotations that operate on utterances, tiers, or
  language-routing decisions

Pipeline operations should compose core primitives instead of editing raw CHAT
text themselves.

## Why This Split Exists

HK engines and morphotag both influence final CHAT output, but they do so at
different layers:

- HK engines: provider-level inference modules
- morphotag: pipeline-level orchestration over a parsed CHAT document

That distinction is load-bearing.

If we force providers to understand CHAT:

- simple SDK wrappers become more complex than necessary
- inference adapters become coupled to AST details
- language-agnostic providers become harder to support

If we force CHAT-aware pipeline work into a provider-only interface:

- they cannot safely reuse extraction/injection logic
- they end up reimplementing CHAT logic in Python
- caching and validation policy drift from the core

## Boundary Rules

### Provider Layer Rules

- Inputs and outputs must be typed task payloads.
- No `.cha` parsing or serialization.
- No direct tier editing.
- Implementations may be Python today and Rust-backed later.
- The worker IPC contract is one valid host/runtime for providers.

### Pipeline Layer Rules

- Operates on a parsed document handle or Rust-owned pipeline executor.
- Uses core extraction and injection primitives.
- May depend on providers, but does not expose provider internals as its API.
- Owns orchestration, not low-level AST mutation.

### Core Layer Rules

- Owns structural CHAT invariants.
- Exposes safe primitives upward.
- Should not depend on provider-specific SDK logic.

## Implications For Public APIs

### `batchalign.providers`

This is a narrow public surface that re-exports worker wire types such as
`BatchInferRequest` and `BatchInferResponse`. It is useful for provider-style
modules and tests, but it is not a discovery or descriptor SDK.

### `batchalign.pipeline_api`

This is the supported public surface for CHAT-aware orchestration:

- operation records
- provider lookup/invocation helpers
- `run_pipeline()` as the Rust-forwarding entry point

## Current State

The split is present in code today:

- there is no public `batchalign.plugins` discovery layer or `PluginDescriptor`
  contract in the current release
- built-in provider modules are wired directly into worker loading and dispatch
- `batchalign.pipeline_api` is the supported Python facade, but Rust owns the
  CHAT-aware orchestration through `batchalign_core.run_provider_pipeline()`
- worker execution paths now center on typed `execute_v2` plus narrow
  `infer`/`batch_infer` model-host calls instead of a generic process-path
  orchestration layer

The compatibility promise for these surfaces is defined in
[API Stability](../developer/api-stability.md).
