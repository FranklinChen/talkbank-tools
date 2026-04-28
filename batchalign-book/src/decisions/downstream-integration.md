# Downstream Integration

**Status:** Accepted decision note  
**Last verified:** 2026-03-05

## Scope

This decision defines the integration boundary for downstream consumers such as:

- library wrappers
- batch processing systems
- editor/annotation integrations
- data-conversion pipelines

## Current decision

Downstream integrations should rely on stable, explicit surfaces rather than on
implicit parser internals or fragile text surgery.

The important current expectations are:

- structured parse outcomes with diagnostics
- deterministic serialization behavior
- machine-stable diagnostic codes and severities
- batch-friendly failure handling

## batchalign2-relevant meaning

For BA2-style consumers, the practical public boundary is:

- parse CHAT through explicit APIs
- inspect structured diagnostics instead of scraping free-form error text
- treat serialization and validation behavior as contract surfaces

## Public implication

This page is a current integration decision, not an implementation roadmap.
Any future integrator-specific artifacts should be documented as current
surfaces when they exist, not as speculative “to provide” lists.
