-- A paths-mode job's earlier version of each file, for incremental
-- processing: a JSON array parallel to `source_paths`, each entry a path or
-- null (no prior for that file). Before this column, priors lived only in
-- memory, so a restart or a recovered job ran without them. Rows written
-- before it read as `[]`: their priors were never stored.
ALTER TABLE jobs ADD COLUMN before_paths TEXT NOT NULL DEFAULT '[]';
