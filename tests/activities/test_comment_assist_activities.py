"""
Tests for Comment Assist Activities.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.schemas.pr_review.seed_set import SeedSetS0
from src.activities.comment_assist_activities import (
    fetch_comment_context_activity,
    build_focused_seed_set_activity,
    generate_comment_response_activity,
    publish_comment_reply_activity,
)


@pytest.fixture
def sample_comment_input():
    return {
        "repo_name": "owner/test-repo",
        "pr_number": 12,
        "comment_id": 101,
        "installation_id": 999,
    }


@pytest.mark.asyncio
async def test_fetch_comment_context_activity_uses_parent_hunk(sample_comment_input):
    comment = {
        "id": 101,
        "body": "@sentinel what can be other alternative to this?",
        "in_reply_to_id": 100,
        "diff_hunk": None,
        "path": None,
        "commit_id": "abc123",
        "user": {"type": "User"},
    }
    parent_comment = {
        "id": 100,
        "body": "Original inline comment",
        "diff_hunk": "@@ -1,2 +1,2 @@\n-foo\n+bar",
        "path": "src/app.py",
        "commit_id": "abc123",
        "user": {"type": "User"},
    }
    thread_comments = [parent_comment, comment]

    with patch("src.activities.comment_assist_activities.PRApiClient") as mock_client:
        client = AsyncMock()
        mock_client.return_value = client
        client.get_review_comment.side_effect = [comment, parent_comment]
        client.list_review_comments.return_value = thread_comments

        result = await fetch_comment_context_activity(sample_comment_input)

    assert result["diff_hunk"] == parent_comment["diff_hunk"]
    assert result["file_path"] == parent_comment["path"]
    assert "alternative" in result["question"].lower()
    assert result["thread_comments"] == thread_comments


@pytest.mark.asyncio
async def test_build_focused_seed_set_activity_returns_seed_set():
    diff_hunk = "@@ -1,1 +1,1 @@\n-foo\n+bar"
    input_data = {
        "diff_hunk": diff_hunk,
        "file_path": "src/app.py",
        "clone_path": "/tmp/repo",
        "change_type": "modified",
    }

    seed_set = SeedSetS0(
        seed_symbols=[],
        seed_files=[],
        extraction_timestamp="2024-01-01T00:00:00Z",
        ast_parser_version="tree-sitter-0.21",
    )
    build_stats = MagicMock(
        files_processed=0,
        files_with_symbols=0,
        files_skipped=0,
        total_symbols_extracted=0,
        total_symbols_overlapping=0,
        parse_errors=0,
        unsupported_languages=0,
    )

    with patch("src.activities.comment_assist_activities.SeedSetBuilder") as mock_builder:
        builder = MagicMock()
        builder.build_seed_set.return_value = (seed_set, build_stats)
        mock_builder.return_value = builder

        result = await build_focused_seed_set_activity(input_data)

    assert result["seed_set"]["ast_parser_version"] == "tree-sitter-0.21"
    assert result["patches"][0]["file_path"] == "src/app.py"
    assert result["build_stats"]["files_processed"] == build_stats.files_processed


@pytest.mark.asyncio
async def test_generate_comment_response_activity_returns_llm_content():
    input_data = {
        "question": "What can be other alternative to this?",
        "diff_hunk": "@@ -1,1 +1,1 @@\n-foo\n+bar",
        "file_path": "src/app.py",
        "thread_comments": [],
        "context_pack": {"context_items": [], "total_context_characters": 0},
    }

    mock_response = {
        "content": "Consider using a different approach.",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "model": "test-model",
    }

    with patch("src.activities.comment_assist_activities.get_llm_client") as mock_client:
        llm = AsyncMock()
        llm.generate_completion.return_value = mock_response
        mock_client.return_value = llm

        result = await generate_comment_response_activity(input_data)

    assert "response_text" in result
    assert result["response_text"] == "Consider using a different approach."
    assert result["usage"]["output_tokens"] == 20


@pytest.mark.asyncio
async def test_publish_comment_reply_activity_posts_reply():
    input_data = {
        "repo_name": "owner/test-repo",
        "pr_number": 12,
        "comment_id": 101,
        "installation_id": 999,
        "response_text": "Reply content",
    }

    with patch("src.activities.comment_assist_activities.PRApiClient") as mock_client:
        client = AsyncMock()
        mock_client.return_value = client
        client.create_review_comment_reply.return_value = {"id": 555}

        result = await publish_comment_reply_activity(input_data)

    client.create_review_comment_reply.assert_called_once()
    assert result["reply_comment_id"] == 555
