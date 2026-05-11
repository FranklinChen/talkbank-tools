# Cache Management

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

The extension uses a SQLite cache to store validation results, avoiding redundant re-validation of files that have not changed. This chapter covers the cache location, monitoring, and maintenance.

## Cache Location

The cache database is stored in the OS-standard cache directory:

| OS | Path |
|----|------|
| **macOS** | `~/Library/Caches/talkbank-chat/` |
| **Linux** | `~/.cache/talkbank-chat/` |
| **Windows** | `%LocalAppData%\talkbank-chat\` |

The cache directory contains a SQLite database file. It is safe to delete this directory entirely -- the extension will recreate it on the next validation run.

## Status Bar Indicator

A cache indicator appears in the bottom-right of the VS Code status bar, showing the number of cached files (e.g., `Cache: 95,247 files`). This updates every 5 seconds by polling the `chatter cache stats --json` CLI command.

Click the status bar indicator to view detailed cache statistics in a popup.

## View Cache Statistics

Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) and select **TalkBank: View Cache Statistics**.

This displays a popup with:

- **Cache database size** -- the on-disk size of the SQLite file
- **Total cached entries** -- how many files have cached results
- **Valid / Invalid / Expired counts** -- breakdown of cache entry statuses
- **Cache hit rate** -- percentage of validation requests served from cache
- **Last updated timestamp** -- when the cache was last written to

## Clearing the Cache

### Clear a specific file or directory

In the [Validation Explorer](../workflows/corpus-validation.md), right-click on a file or folder and select **Clear Cache**. This removes cached results only for the selected items, forcing revalidation on the next run.

### Clear the entire cache

Open the Command Palette and select **TalkBank: Clear All Validation Cache**. This deletes all cached entries from the database.

You can also use the `--force` flag with the CLI to bypass the cache for a specific run:

```bash
chatter validate path/to/corpus/ --force
```

## When to Clear the Cache

The cache is content-addressed: if a file changes, its cached result is automatically invalidated. You should not need to manually clear the cache under normal circumstances. Clear the cache when:

- **After upgrading chatter** -- a new version of the validator may produce different results for the same input. The cache does not automatically detect binary upgrades.
- **After grammar changes** -- if you are developing the tree-sitter grammar or the parser, cached results from the old grammar are stale.
- **If results seem stale** -- if validation results do not match what you expect after editing a file, clearing the cache rules out staleness as the cause.
- **If the cache becomes corrupted** -- SQLite corruption is rare but possible after a system crash. Deleting and recreating the cache is the simplest fix.

## Cache Design

The cache is managed by the `chatter` CLI binary, not the extension's TypeScript code. The extension's `CacheManager` (`cacheManager.ts`) polls the CLI for statistics and invokes CLI commands for cache operations. The Validation Explorer also uses the CLI (`chatter validate --format json`) for bulk validation, and the CLI handles cache reads and writes transparently.

This design means the cache is shared between the VS Code extension and direct CLI usage. Validating files from the terminal benefits from the same cache, and vice versa.

## Related Chapters

- [Corpus Validation](../workflows/corpus-validation.md) -- the Validation Explorer tree view
- [Settings Reference](settings.md) -- `talkbank.validation.severity` and other settings
- [Troubleshooting: Common Issues](../troubleshooting/common-issues.md) -- resolving stale cache issues
