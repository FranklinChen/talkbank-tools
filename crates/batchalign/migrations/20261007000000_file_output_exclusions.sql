-- A file whose output was written (`done` or `diagnosed`) may carry what its
-- producer left out of its work on purpose, by a recorded ruling (an
-- utterance align did not look for because the transcript marks it as not in
-- the recording). This column holds them as a JSON array of
-- `OutputExclusionRecord`. Information, never a diagnosis. NULL when there is
-- none, and for every phase that wrote no output.
ALTER TABLE file_statuses ADD COLUMN exclusions TEXT;
