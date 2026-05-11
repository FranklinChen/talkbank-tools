# Deliberately Not Implemented

**Status:** Current
## NLP Commands

These commands depend on the legacy CLAN MOR data model -- trie-based lexicons, HMM/Brill taggers, and MaxEnt parsers -- which is incompatible with the UD-style morphological representation used in the current CHAT grammar and data model. Attempting to invoke them produces a clear error message explaining why.

For morphological analysis, POS tagging, and dependency parsing, use batchalign's neural pipeline, which supports more languages with higher accuracy.

| Command | Purpose | Why not ported |
|---------|---------|----------------|
| MOR / MOR_P | Morphological analysis (add `%mor` tiers) | ~11,000 lines of C with trie lexicon and 5 rule engines; output format uses legacy CLAN morphology conventions incompatible with UD-style `%mor` |
| POST | POS disambiguation (disambiguate `%mor`) | Reads ambiguous `%mor` tiers with `^` separators -- a format not supported by the current CHAT grammar |
| MEGRASP | Grammatical relation parsing (add `%gra` tiers) | Requires trained MaxEnt model weights for the legacy CLAN `%mor` format |
| POSTLIST | List POST database entries | Operates on POST's proprietary binary database format |
| POSTMODRULES | Modify POST rules | Operates on POST's proprietary binary database format |
| POSTTRAIN | Train POST database | Produces POST's proprietary binary database format |

## CHECK

| Command | Status |
|---------|--------|
| CHECK | Reimplemented as `chatter clan check` with full flag support |

CHECK is fully ported. `chatter validate` provides additional spec-driven validation with rich diagnostics beyond what legacy CHECK offered.

## Subsumed by existing commands

These commands' functionality is available through other implemented commands:

| Command | Available via |
|---------|---------------|
| GEM (standalone) | `freq --gem`, `gemlist` |
| GEMFREQ | `freq --gem` |

## Truly obsolete

These commands address problems that no longer exist in modern environments:

| Command | Reason |
|---------|--------|
| DOS2UNIX | Modern systems handle line endings natively |
| CP2UTF | UTF-8 is universal; legacy codepage files are rare |
| FIXMP3 | Audio repair, not a CHAT analysis tool |
| SILENCE_P / PID_P | Audio processing, out of scope for transcript analysis |

## Low-priority / specialized

| Command | Reason for low priority |
|---------|------------------------|
| GPS / PP / TEMPLATE | Extremely specialized, no known active users |
