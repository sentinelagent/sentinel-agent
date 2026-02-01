-- Migration 005: Add unique constraint for pull_requests
-- This constraint is required for ON CONFLICT upsert operations

-- Add unique constraint on (repository_id, pr_number)
-- This ensures one PR record per repository+PR number combination
ALTER TABLE pull_requests
ADD CONSTRAINT uq_repository_pr_number
UNIQUE (repository_id, pr_number);

-- Create index for performance (if not automatically created by the constraint)
CREATE INDEX IF NOT EXISTS idx_pull_requests_repo_pr
ON pull_requests(repository_id, pr_number);
