-- ============================================================================
-- Database Migration: Add Ingest Status Tracking
-- ============================================================================
-- 
-- This migration adds ingest_status and quality_gate_result columns to the
-- packages table to track all artifact submissions (approved and rejected).
--
-- Run this migration on your AWS RDS PostgreSQL database before deploying
-- the updated application code.
-- ============================================================================

-- Step 1: Add ingest_status column
ALTER TABLE packages 
ADD COLUMN ingest_status VARCHAR(20) NOT NULL DEFAULT 'approved';

-- Step 2: Add quality_gate_result column (stores JSON)
ALTER TABLE packages 
ADD COLUMN quality_gate_result JSONB;

-- Step 3: Create index for efficient filtering by ingest_status
CREATE INDEX ix_package_ingest_status ON packages(ingest_status);

-- Step 4: Add comments for documentation
COMMENT ON COLUMN packages.ingest_status IS 
  'Status of artifact ingestion: approved, rejected, or pending';

COMMENT ON COLUMN packages.quality_gate_result IS 
  'JSON object containing quality gate evaluation results including pass/fail status, metrics, and evaluation timestamp';

-- Step 5: Set existing packages to 'approved' status
-- (All currently stored packages passed the quality gate)
UPDATE packages 
SET ingest_status = 'approved' 
WHERE ingest_status IS NULL OR ingest_status = '';

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check that columns were added successfully
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'packages' 
    AND column_name IN ('ingest_status', 'quality_gate_result')
ORDER BY column_name;

-- Check the index was created
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'packages' 
    AND indexname = 'ix_package_ingest_status';

-- Verify all existing packages have approved status
SELECT 
    ingest_status,
    COUNT(*) as count
FROM packages
GROUP BY ingest_status;

-- ============================================================================
-- Rollback Script (if needed)
-- ============================================================================

-- CAUTION: This will delete the new columns and their data!
-- Only run this if you need to rollback the migration.

/*
DROP INDEX IF EXISTS ix_package_ingest_status;
ALTER TABLE packages DROP COLUMN IF EXISTS quality_gate_result;
ALTER TABLE packages DROP COLUMN IF EXISTS ingest_status;
*/
