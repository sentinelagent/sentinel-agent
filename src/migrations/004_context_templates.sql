-- Migration: 004_context_templates.sql
-- Purpose: Create context templates feature for reusable review context configuration
-- Date: 2026-02-01
-- Description: Adds tables for context templates and repository-template assignments
--              enabling users to create reusable context configurations for code reviews

BEGIN;

-- ============================================================================
-- CREATE CONTEXT_TEMPLATES TABLE
-- Stores reusable context configurations for code reviews
-- ============================================================================

CREATE TABLE IF NOT EXISTS context_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Ownership (NULL for system/default templates)
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,

    -- Template metadata
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Template content - the actual context configuration
    template_content JSONB NOT NULL DEFAULT '{}',

    -- Visibility: 'private' (only owner), 'organization' (future), 'public' (future)
    visibility VARCHAR(50) NOT NULL DEFAULT 'private',

    -- System/default template indicator (for UI display)
    is_default BOOLEAN NOT NULL DEFAULT FALSE,

    -- Template status for soft delete / archive
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CREATE REPOSITORY_TEMPLATE_ASSIGNMENTS TABLE
-- Junction table linking repositories to templates with priority
-- ============================================================================

CREATE TABLE IF NOT EXISTS repository_template_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign keys
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE NOT NULL,
    template_id UUID REFERENCES context_templates(id) ON DELETE CASCADE NOT NULL,

    -- Priority for ordering when multiple templates are assigned (lower = higher priority)
    priority INTEGER NOT NULL DEFAULT 0,

    -- Assignment status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique assignment per repository-template pair
    CONSTRAINT uq_repository_template UNIQUE (repository_id, template_id)
);

-- ============================================================================
-- CREATE PERFORMANCE INDEXES
-- Essential indexes for efficient queries
-- ============================================================================

-- Index for fetching templates by user (most common query)
CREATE INDEX idx_context_templates_user_id
ON context_templates(user_id)
WHERE is_active = TRUE;

-- Index for fetching templates by visibility (for future public/org templates)
CREATE INDEX idx_context_templates_visibility
ON context_templates(visibility)
WHERE is_active = TRUE;

-- Composite index for user + active templates
CREATE INDEX idx_context_templates_user_active
ON context_templates(user_id, is_active);

-- Index for fetching assignments by repository
CREATE INDEX idx_repo_template_assignments_repository
ON repository_template_assignments(repository_id)
WHERE is_active = TRUE;

-- Index for fetching assignments by template (for cascade operations)
CREATE INDEX idx_repo_template_assignments_template
ON repository_template_assignments(template_id);

-- Index for priority ordering within repository
CREATE INDEX idx_repo_template_assignments_priority
ON repository_template_assignments(repository_id, priority)
WHERE is_active = TRUE;

-- ============================================================================
-- ADD CONSTRAINTS
-- Ensure data integrity
-- ============================================================================

-- Validate visibility values
ALTER TABLE context_templates ADD CONSTRAINT context_templates_visibility_check
    CHECK (visibility IN ('private', 'organization', 'public'));

-- Ensure default templates have NULL user_id and public visibility
ALTER TABLE context_templates ADD CONSTRAINT context_templates_default_check
    CHECK (
        (is_default = FALSE AND user_id IS NOT NULL) OR
        (is_default = TRUE AND user_id IS NULL AND visibility = 'public')
    );

-- Validate priority is non-negative
ALTER TABLE repository_template_assignments ADD CONSTRAINT repo_template_assignments_priority_check
    CHECK (priority >= 0);

-- ============================================================================
-- ADD COMMENTS FOR DOCUMENTATION
-- Document the purpose of each table and column
-- ============================================================================

COMMENT ON TABLE context_templates IS
    'Stores reusable context template configurations for code reviews. Templates can contain review guidelines, coding standards, and other contextual information.';

COMMENT ON COLUMN context_templates.id IS
    'Unique identifier for the template';

COMMENT ON COLUMN context_templates.user_id IS
    'Owner of the template - references the users table. NULL for system/default templates';

COMMENT ON COLUMN context_templates.name IS
    'Human-readable name for the template';

COMMENT ON COLUMN context_templates.description IS
    'Optional description explaining the purpose and usage of the template';

COMMENT ON COLUMN context_templates.template_content IS
    'JSONB field containing the actual template configuration. Structure: {"guidelines": [], "standards": {}, "custom_rules": [], ...}';

COMMENT ON COLUMN context_templates.visibility IS
    'Access level: private (owner only), organization (future), public (future)';

COMMENT ON COLUMN context_templates.is_default IS
    'System/default template indicator. TRUE for built-in templates, FALSE for user-created templates';

COMMENT ON COLUMN context_templates.is_active IS
    'Soft delete flag - FALSE means template is archived/deleted';

COMMENT ON TABLE repository_template_assignments IS
    'Junction table linking repositories to context templates. Supports multiple templates per repository with priority ordering.';

COMMENT ON COLUMN repository_template_assignments.repository_id IS
    'Foreign key to the repositories table';

COMMENT ON COLUMN repository_template_assignments.template_id IS
    'Foreign key to the context_templates table';

COMMENT ON COLUMN repository_template_assignments.priority IS
    'Ordering priority when multiple templates are assigned. Lower values = higher priority (processed first)';

COMMENT ON COLUMN repository_template_assignments.is_active IS
    'Soft delete flag for the assignment';

-- ============================================================================
-- VALIDATION QUERIES
-- Verify the migration was applied correctly
-- ============================================================================

DO $$
BEGIN
    -- Check context_templates table exists with correct columns
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'context_templates'
    ) THEN
        RAISE EXCEPTION 'context_templates table was not created';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'context_templates'
        AND column_name = 'template_content'
        AND data_type = 'jsonb'
    ) THEN
        RAISE EXCEPTION 'template_content column missing or wrong type';
    END IF;

    -- Check repository_template_assignments table exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'repository_template_assignments'
    ) THEN
        RAISE EXCEPTION 'repository_template_assignments table was not created';
    END IF;

    -- Validate unique constraint exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'repository_template_assignments'
        AND constraint_name = 'uq_repository_template'
        AND constraint_type = 'UNIQUE'
    ) THEN
        RAISE EXCEPTION 'uq_repository_template unique constraint missing';
    END IF;

    -- Validate indexes were created
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_context_templates_user_id'
    ) THEN
        RAISE EXCEPTION 'idx_context_templates_user_id index missing';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_repo_template_assignments_repository'
    ) THEN
        RAISE EXCEPTION 'idx_repo_template_assignments_repository index missing';
    END IF;

    RAISE NOTICE 'Migration validation passed - context_templates feature tables created successfully';
END $$;

-- ============================================================================
-- SEED DEFAULT SYSTEM TEMPLATES
-- These are built-in templates available to all users
-- ============================================================================

INSERT INTO context_templates (name, description, template_content, visibility, is_default, is_active, user_id)
VALUES
(
    'Clean Architecture',
    'Domain-driven design with clear separation of concerns. Ideal for enterprise applications with complex business logic.',
    '{"guidelines": ["Focus on dependency inversion - dependencies should point inward toward the domain", "Ensure domain entities are free of infrastructure concerns (no DB annotations, no HTTP)", "Validate that use cases are properly isolated and testable", "Check that adapters properly abstract external dependencies"], "focus_areas": ["code_quality", "maintainability", "testing"], "coding_standards": {"enforce_dependency_direction": true, "separate_domain_from_infrastructure": true}, "additional_context": "This project follows Clean Architecture principles with clear boundaries between domain, application, and infrastructure layers."}',
    'public',
    TRUE,
    TRUE,
    NULL
),
(
    'Security-Focused',
    'Security-first review approach for applications handling sensitive data or requiring compliance (PCI-DSS, HIPAA, SOC2).',
    '{"guidelines": ["Check for SQL injection vulnerabilities - ensure parameterized queries everywhere", "Validate all user inputs at boundaries (API, forms, CLI)", "Ensure proper authentication and authorization checks on all protected endpoints", "Review error messages for information leakage (stack traces, internal paths)", "Check for XSS vulnerabilities in user-generated content", "Verify CSRF protection on state-changing operations", "Ensure secrets are not hardcoded (API keys, passwords, tokens)"], "focus_areas": ["security", "error_handling"], "coding_standards": {"require_input_validation": true, "enforce_parameterized_queries": true, "no_hardcoded_secrets": true}, "custom_rules": [{"rule": "no_hardcoded_secrets", "severity": "critical"}, {"rule": "require_https", "severity": "error"}, {"rule": "validate_all_inputs", "severity": "error"}], "additional_context": "This application handles sensitive user data and must comply with security standards. Focus on OWASP Top 10 vulnerabilities."}',
    'public',
    TRUE,
    TRUE,
    NULL
),
(
    'Performance-Optimized',
    'Performance-critical review for high-throughput systems, real-time applications, and services with strict latency SLAs.',
    '{"guidelines": ["Identify N+1 query problems - use eager loading or batch queries", "Check algorithmic complexity - look for O(n²) or worse in hot paths", "Review async/await usage - ensure non-blocking I/O operations", "Look for potential memory leaks - unclosed resources, growing collections", "Check for unnecessary object allocations in loops", "Verify proper use of caching (Redis, in-memory)", "Review database query plans for inefficient scans"], "focus_areas": ["performance", "code_quality"], "coding_standards": {"max_cyclomatic_complexity": 10, "require_type_hints": true, "enforce_async_io": true}, "custom_rules": [{"rule": "no_blocking_io_in_async", "severity": "error"}, {"rule": "optimize_db_queries", "severity": "warning"}], "additional_context": "This is a high-traffic API service serving millions of requests per day. Performance and scalability are critical."}',
    'public',
    TRUE,
    TRUE,
    NULL
),
(
    'Microservices',
    'Distributed systems review focusing on service boundaries, inter-service communication, and resilience patterns.',
    '{"guidelines": ["Verify service boundaries are well-defined - single responsibility per service", "Check for proper error propagation across service boundaries", "Ensure idempotency for retryable operations (POST, PUT, DELETE)", "Review API contracts for backward compatibility and versioning", "Validate circuit breaker patterns for external dependencies", "Check for proper distributed tracing and correlation IDs", "Review timeout and retry strategies", "Ensure graceful degradation when dependencies fail"], "focus_areas": ["code_quality", "error_handling", "documentation"], "coding_standards": {"require_api_versioning": true, "enforce_correlation_ids": true, "require_circuit_breakers": true}, "custom_rules": [{"rule": "idempotent_operations", "severity": "error"}, {"rule": "api_versioning", "severity": "error"}, {"rule": "circuit_breaker_required", "severity": "warning"}], "additional_context": "This is part of a microservices architecture. Services must be resilient, well-bounded, and properly instrumented."}',
    'public',
    TRUE,
    TRUE,
    NULL
);

-- Verify default templates were created
DO $$
DECLARE
    template_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO template_count
    FROM context_templates
    WHERE is_default = TRUE;

    IF template_count != 4 THEN
        RAISE EXCEPTION 'Expected 4 default templates, but found %', template_count;
    END IF;

    RAISE NOTICE 'Successfully seeded 4 default system templates';
END $$;

COMMIT;

-- ============================================================================
-- MIGRATION COMPLETE
-- Summary: Created Context Templates feature
-- Tables created:
--   - context_templates: Stores template configurations
--   - repository_template_assignments: Links templates to repositories
-- Indexes created: 6 performance indexes
-- Constraints: Visibility check, priority check, unique assignment
-- ============================================================================
