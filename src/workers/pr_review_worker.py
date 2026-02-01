import asyncio
from temporalio.worker import Worker
from src.utils.logging.otel_logger import logger
from src.workflows.pr_review_workflow import PRReviewWorkflow
from src.activities.pr_review_activities import (
    fetch_pr_context_activity,
    clone_pr_head_activity,
    build_seed_set_activity,
    retrieve_kg_candidates_activity,
    retrieve_and_assemble_context_activity,
    fetch_context_template_activity,
    generate_review_activity,
    persist_pr_review_metadata_activity,
    anchor_and_publish_activity,
    cleanup_pr_clone_activity,
)
from src.core.config import settings
from src.core.temporal_client import connect_to_temporal_with_retry
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions


async def initialize_neo4j_pool():
    """Initialize Neo4j connection pool with retry logic."""
    from src.services.kg.connection_pool import Neo4jPoolConfig, initialize_connection_pool
    from src.core.pr_review_config import pr_review_settings

    logger.info("Initializing Neo4j connection pool...")

    try:
        neo4j_config = Neo4jPoolConfig(
            uri=pr_review_settings.neo4j_uri,
            username=pr_review_settings.neo4j_username,
            password=pr_review_settings.neo4j_password,
            database=pr_review_settings.neo4j_database,
            max_connection_pool_size=pr_review_settings.neo4j_max_pool_size,
            max_connection_lifetime=pr_review_settings.neo4j_max_connection_lifetime,
            connection_acquisition_timeout=pr_review_settings.timeouts.neo4j_connection_timeout,
            max_init_retries=5,
            init_retry_delay=2.0,
            health_check_failure_threshold=3,
        )

        pool = await initialize_connection_pool(neo4j_config)
        logger.info(f"Neo4j pool initialized: {pool.get_diagnostic_info()}")
        return pool

    except Exception as e:
        logger.error(f"Failed to initialize Neo4j pool: {e}. Worker will continue but KG queries may fail.")
        return None


async def main():
    target_host = settings.TEMPORAL_SERVER_URL
    logger.info(f"Connecting to Temporal server at {target_host}...")

    # Use retry logic to wait for Temporal to be available
    client = await connect_to_temporal_with_retry(
        target_host=target_host,
        namespace='default'
    )

    # Initialize Neo4j Connection Pool
    neo4j_pool = await initialize_neo4j_pool()

    # Create custom sandbox restrictions with passthrough modules
    # These modules do I/O at import time (read .env files, create thread locals, etc.)
    # but are not actually used inside workflow code - only in activities
    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "pydantic_settings",
        "dotenv",
        "httpx",
        "sniffio",
        "src",  # Pass through entire src package - activities do I/O, workflows don't
    )
    worker = Worker(
        client,
        task_queue="pr-review-pipeline",
        workflows=[PRReviewWorkflow],
        activities=[
            fetch_pr_context_activity,
            clone_pr_head_activity,
            build_seed_set_activity,
            retrieve_kg_candidates_activity,
            retrieve_and_assemble_context_activity,
            fetch_context_template_activity,
            generate_review_activity,
            persist_pr_review_metadata_activity,
            anchor_and_publish_activity,
            cleanup_pr_clone_activity,
        ],
        workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
    )
    logger.info("Temporal worker started")
    logger.info(f"Connected to temporal host ${target_host}. Polling for task queue ${worker.task_queue}")
    try:
        await worker.run()
    except Exception as e:
        logger.error(f"Error starting temporal worker: {e}")
        raise
    finally:
        # Clean up Neo4j connection pool
        logger.info("Shutting down, closing Neo4j pool...")
        try:
            from src.services.kg.connection_pool import close_connection_pool
            await close_connection_pool()
            logger.info("Neo4j pool closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j pool: {e}")


if __name__ == "__main__":
    asyncio.run(main())