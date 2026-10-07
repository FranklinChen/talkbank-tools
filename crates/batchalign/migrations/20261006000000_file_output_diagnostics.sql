-- A file whose own producer generated output that failed admission is
-- written anyway and recorded as `diagnosed`; this column holds what the
-- admission found, as a JSON object (`findings`, `skipped_stages`). NULL for
-- every other phase, because only a diagnosed file owns it.
ALTER TABLE file_statuses ADD COLUMN diagnostics TEXT;
