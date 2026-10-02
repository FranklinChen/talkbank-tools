-- `file_statuses.bug_report_id` was never written or read: no code files a
-- bug report, so the column could only ever hold NULL. Dropped so the row
-- holds only what a file's phase and output own.
ALTER TABLE file_statuses DROP COLUMN bug_report_id;
