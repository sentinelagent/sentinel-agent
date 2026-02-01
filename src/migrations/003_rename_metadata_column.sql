-- Migration: 003_rename_metadata_column.sql
-- Purpose: Rename metadata column to event_metadata to avoid conflict with SQLAlchemy Base.metadata
-- Date: 2026-01-31

BEGIN;

-- Check if the column exists under the old name before renaming
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflow_run_events' 
        AND column_name = 'metadata'
    ) THEN
        ALTER TABLE workflow_run_events RENAME COLUMN metadata TO event_metadata;
        RAISE NOTICE 'Renamed column metadata to event_metadata in workflow_run_events';
    ELSE
        RAISE NOTICE 'Column metadata already renamed or does not exist in workflow_run_events';
    END IF;
END $$;

-- Update the comment
COMMENT ON COLUMN workflow_run_events.event_metadata IS
    'Activity-specific data in JSONB format (file counts, SHAs, error details, etc.).';

COMMIT;
