# Arena Allocators (Evaluation — Not Used)

> **Status: Evaluated and rejected (2026).** We evaluated `bumpalo` arenas at
> several allocation hot spots and concluded the simpler patterns below
> (scratch buffers, flat tables, dense Vecs) provide equivalent savings with
> less API complexity. Arena allocators are not used in this codebase.

## Why Not `bumpalo`

1. **Lifetime rigidity** — Arena references can't escape the owning function,
   but our processing pipelines return intermediate results across function
   boundaries.
2. **Marginal gains** — The targeted patterns below already eliminate hot spots.
   Adding `bumpalo` on top yielded < 2% additional improvement in benchmarks.
3. **API friction** — `bumpalo::collections::Vec` and `String` are different
   types from `std`, requiring conversion at every boundary.

## Patterns We Use Instead

### Scratch Buffer Reuse

Pre-allocate and reuse buffers instead of allocating fresh each iteration:

```rust
let mut prev: Vec<usize> = (0..=pay_len).collect();
let mut cur = Vec::with_capacity(pay_len + 1);

for ref_item in reference {
    cur.clear();
    // ... fill cur ...
    std::mem::swap(&mut prev, &mut cur);
}
```

Used in `dp_align.rs` (Hirschberg alignment).

### Flat Table Instead of Vec-of-Vec

```rust
// 1 allocation instead of rows + 1
let mut dp = vec![(0usize, Action::Start, 0, 0); rows * cols];
let idx = |r: usize, c: usize| r * cols + c;
```

### Dense Index Vec Instead of HashMap

When keys are dense integers `0..N`, a `Vec` is faster than a `HashMap`:

```rust
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
3. **Benchmark with realistic inputs.** The allocator is rarely the bottleneck —
   I/O, parsing, and NLP inference dominate wall-clock time.
