# Arena Allocators (Evaluation: Not Used)

**Status:** Current
> **Status: Evaluated and rejected (2026).** We evaluated `bumpalo` arenas at
> several allocation hot spots and concluded the simpler patterns below
> (scratch buffers, flat tables, dense Vecs) provide equivalent savings with
> less API complexity. Arena allocators are not used in this codebase.

## Why Not `bumpalo`

1. **Lifetime rigidity**: Arena references can't escape the owning function,
   but our processing pipelines return intermediate results across function
   boundaries.
2. **Marginal gains**: The targeted patterns below already eliminate hot spots.
   Adding `bumpalo` on top yielded < 2% additional improvement in benchmarks.
3. **API friction**: `bumpalo::collections::Vec` and `String` are different
   types from `std`, requiring conversion at every boundary.

## Patterns We Use Instead

### Scratch Buffer Reuse

Pre-allocate and reuse buffers instead of allocating fresh each iteration:

```rust,ignore
let mut prev: Vec<usize> = (0..=pay_len).collect();
let mut cur = Vec::with_capacity(pay_len + 1);

for ref_item in reference {
    cur.clear();
    // ... fill cur ...
    std::mem::swap(&mut prev, &mut cur);
}
```

Used in `dp_align/mod.rs` (Hirschberg alignment).

### Prepare Fuzzy Comparison Inputs Once

Word alignment admits fuzzy inputs as `PreparedFuzzyWord`, which retains the
original spelling and computes its Unicode lowercase form once. The DP core
accepts the comparison policy associated with its element type: unprepared
strings and characters cannot request fuzzy comparison. Exact and ASCII-only
paths retain their allocation-free comparisons. Result keys use the original
spelling, and the existing ASCII equality fast path remains authoritative.

This trades storage proportional to the input text for eliminating repeated
lowercase allocations inside DP cells. It does not change the quadratic
comparison count or the alignment tie-breaking rules. Mostly matching inputs
can have little repeated work to save, so this is not a universal speedup.

The ignored `fuzzy_comparison_performance_probe` in the existing library test
binary compares prepared and legacy comparisons in alternating order. Run it
with `cargo test -p batchalign-transform --lib
fuzzy_comparison_performance_probe -- --ignored --nocapture`. It reports time
inside the test, excluding compilation, and has no CI timing threshold. The
separate differential test compares complete plans, including Unicode cases,
original spellings, and threshold edge behavior.

### Flat Table Instead of Vec-of-Vec

```rust,ignore
// 1 allocation instead of rows + 1
let mut dp = vec![(0usize, Action::Start, 0, 0); rows * cols];
let idx = |r: usize, c: usize| r * cols + c;
```

### Dense Index Vec Instead of HashMap

When keys are dense integers `0..N`, a `Vec` is faster than a `HashMap`:

```rust,ignore
let mut mapping: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); num_words];
mapping[word_idx].push(token_idx);
```

### Avoiding Allocation Entirely

The character explosion in retokenization uses `&[char]` directly instead of
converting each character to a `String` for DP alignment. The DP aligner
accepts `&[char]` via the `Alignable` trait.

## Guidelines

1. **Start with the cheapest fix.** Reuse a buffer, use a flat table, avoid
   the allocation entirely.
2. **Don't add an arena for < 10 allocations per call.**
3. **Benchmark with realistic inputs.** The allocator is rarely the bottleneck,
   I/O, parsing, and NLP inference dominate wall-clock time.
