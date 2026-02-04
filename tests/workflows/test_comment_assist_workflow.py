"""Tests for comment assist workflow helpers."""

import uuid
import pytest
from unittest.mock import AsyncMock

from src.models.schemas.pr_review.comment_assist import PRCommentAssistRequest
from src.workflows.comment_assist_workflow import (
    create_comment_assist_workflow_id,
    create_comment_assist_task_queue,
    start_comment_assist_workflow,
)


@pytest.mark.unit
def test_create_comment_assist_workflow_id():
    assert create_comment_assist_workflow_id(123) == "comment-assist:123"


@pytest.mark.unit
def test_create_comment_assist_task_queue_uses_settings():
    assert isinstance(create_comment_assist_task_queue(), str)


@pytest.mark.asyncio
async def test_start_comment_assist_workflow_calls_temporal_client():
    request = PRCommentAssistRequest(
        installation_id=123,
        repo_id=uuid.uuid4(),
        github_repo_id=456,
        github_repo_name="owner/test-repo",
        pr_number=10,
        comment_id=999,
        head_sha="1234567890abcdef1234567890abcdef12345678",
        base_sha="abcdef1234567890abcdef1234567890abcdef12",
        in_reply_to_id=None,
    )
    temporal_client = AsyncMock()
    temporal_client.start_workflow.return_value = AsyncMock()

    workflow_id = await start_comment_assist_workflow(temporal_client, request)

    assert workflow_id == f"comment-assist:{request.comment_id}"
    temporal_client.start_workflow.assert_called_once()
