"""
Comment Assist Temporal Workflow

Orchestrates @sentinel inline comment assistance.
"""

from datetime import timedelta
from typing import Optional
from temporalio import workflow
from temporalio.common import RetryPolicy

from src.models.schemas.pr_review.comment_assist import (
    PRCommentAssistRequest,
    PRCommentAssistResult,
)
from src.models.schemas.pr_review.pr_request import PRReviewRequest
from src.core.pr_review_config import pr_review_settings
from src.utils.logging import get_logger

from src.activities.comment_assist_activities import (
    fetch_comment_context_activity,
    build_focused_seed_set_activity,
    assemble_comment_context_activity,
    generate_comment_response_activity,
    publish_comment_reply_activity,
)
from src.activities.pr_review_activities import (
    clone_pr_head_activity,
    retrieve_kg_candidates_activity,
    cleanup_pr_clone_activity,
)

logger = get_logger(__name__)


@workflow.defn
class PRCommentAssistWorkflow:
    """Workflow for answering @sentinel comment mentions."""

    @workflow.run
    async def run(self, request: PRCommentAssistRequest) -> PRCommentAssistResult:
        workflow_id = workflow.info().workflow_id
        start_time = workflow.now()
        logger.info(
            f"Starting comment assist workflow {workflow_id} for "
            f"{request.github_repo_name}#{request.pr_number} comment {request.comment_id}"
        )

        standard_retry = RetryPolicy(
            maximum_attempts=pr_review_settings.timeouts.max_retry_attempts,
            initial_interval=timedelta(seconds=3),
            maximum_interval=timedelta(seconds=30),
            backoff_coefficient=pr_review_settings.timeouts.retry_backoff_factor,
        )
        no_retry = RetryPolicy(maximum_attempts=1)

        clone_path: Optional[str] = None
        response_text: Optional[str] = None
        reply_comment_id: Optional[int] = None

        try:
            comment_context = await workflow.execute_activity(
                fetch_comment_context_activity,
                {
                    "repo_name": request.github_repo_name,
                    "pr_number": request.pr_number,
                    "comment_id": request.comment_id,
                    "installation_id": request.installation_id,
                },
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=standard_retry,
            )

            diff_hunk = comment_context.get("diff_hunk")
            file_path = comment_context.get("file_path")

            review_request = PRReviewRequest(
                installation_id=request.installation_id,
                repo_id=request.repo_id,
                github_repo_id=request.github_repo_id,
                github_repo_name=request.github_repo_name,
                pr_number=request.pr_number,
                head_sha=request.head_sha,
                base_sha=request.base_sha,
            )

            clone_result = await workflow.execute_activity(
                clone_pr_head_activity,
                review_request,
                start_to_close_timeout=timedelta(
                    seconds=pr_review_settings.timeouts.clone_pr_head_timeout
                ),
                retry_policy=standard_retry,
            )
            clone_path = clone_result.get("clone_path")

            seed_set_payload = None
            context_pack_payload = {"context_items": [], "total_context_characters": 0}
            patches_payload = []
            kg_candidates = {"candidates": []}

            if diff_hunk and file_path:
                seed_set_payload = await workflow.execute_activity(
                    build_focused_seed_set_activity,
                    {
                        "diff_hunk": diff_hunk,
                        "file_path": file_path,
                        "clone_path": clone_path,
                        "change_type": "modified",
                    },
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=standard_retry,
                )

                patches_payload = seed_set_payload.get("patches", [])
                seed_set = seed_set_payload.get("seed_set")

                kg_candidates = await workflow.execute_activity(
                    retrieve_kg_candidates_activity,
                    {
                        "repo_id": str(request.repo_id),
                        "pr_head_sha": request.head_sha,
                        "seed_set": seed_set,
                    },
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=standard_retry,
                )

                context_result = await workflow.execute_activity(
                    assemble_comment_context_activity,
                    {
                        "repo_id": str(request.repo_id),
                        "github_repo_name": request.github_repo_name,
                        "pr_number": request.pr_number,
                        "pr_head_sha": request.head_sha,
                        "pr_base_sha": request.base_sha,
                        "seed_set": seed_set,
                        "patches": patches_payload,
                        "kg_candidates": kg_candidates,
                        "clone_path": clone_path,
                    },
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=standard_retry,
                )
                context_pack_payload = context_result.get("context_pack", context_pack_payload)

            response_result = await workflow.execute_activity(
                generate_comment_response_activity,
                {
                    "question": comment_context.get("question", ""),
                    "diff_hunk": diff_hunk,
                    "file_path": file_path,
                    "thread_comments": comment_context.get("thread_comments", []),
                    "context_pack": context_pack_payload,
                },
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=standard_retry,
            )
            response_text = response_result.get("response_text")

            publish_result = await workflow.execute_activity(
                publish_comment_reply_activity,
                {
                    "repo_name": request.github_repo_name,
                    "pr_number": request.pr_number,
                    "comment_id": request.comment_id,
                    "installation_id": request.installation_id,
                    "response_text": response_text,
                },
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=no_retry,
            )
            reply_comment_id = publish_result.get("reply_comment_id")

            duration_ms = int((workflow.now() - start_time).total_seconds() * 1000)

            return PRCommentAssistResult(
                status="completed",
                comment_id=request.comment_id,
                reply_comment_id=reply_comment_id,
                response_text=response_text,
                processing_duration_ms=duration_ms,
                completed_at=workflow.now(),
            )
        except Exception as exc:
            logger.error(
                f"Comment assist workflow failed for {request.github_repo_name}#{request.pr_number}: {exc}",
                exc_info=True,
            )
            duration_ms = int((workflow.now() - start_time).total_seconds() * 1000)
            return PRCommentAssistResult(
                status="failed",
                comment_id=request.comment_id,
                reply_comment_id=reply_comment_id,
                response_text=response_text,
                processing_duration_ms=duration_ms,
                error_message=str(exc),
                completed_at=workflow.now(),
            )
        finally:
            if clone_path:
                try:
                    await workflow.execute_activity(
                        cleanup_pr_clone_activity,
                        {"clone_path": clone_path},
                        start_to_close_timeout=timedelta(seconds=60),
                        retry_policy=no_retry,
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup clone directory {clone_path}: {cleanup_error}"
                    )


# ============================================================================
# WORKFLOW HELPER FUNCTIONS
# ============================================================================

def create_comment_assist_workflow_id(comment_id: int) -> str:
    """Create deterministic workflow ID for comment assistance."""
    return f"comment-assist:{comment_id}"


def create_comment_assist_task_queue() -> str:
    """Get the task queue name for comment assist workflows."""
    return pr_review_settings.temporal_task_queue


async def start_comment_assist_workflow(
    temporal_client,
    request: PRCommentAssistRequest,
    workflow_id: Optional[str] = None,
) -> str:
    """Start comment assist workflow with consistent configuration."""
    if not workflow_id:
        workflow_id = create_comment_assist_workflow_id(request.comment_id)

    await temporal_client.start_workflow(
        PRCommentAssistWorkflow.run,
        request,
        id=workflow_id,
        task_queue=create_comment_assist_task_queue(),
        execution_timeout=timedelta(minutes=5),
        run_timeout=timedelta(minutes=10),
    )

    logger.info(
        f"Started comment assist workflow {workflow_id} for "
        f"{request.github_repo_name}#{request.pr_number} comment {request.comment_id}"
    )
    return workflow_id
