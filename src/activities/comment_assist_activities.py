"""
Comment Assist Activities

Temporal activities for @sentinel inline comment assistance.
"""

from typing import Any, Dict, List, Optional
import re
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.services.github.pr_api_client import PRApiClient
from src.services.diff_parsing.unified_diff_parser import UnifiedDiffParser
from src.services.seed_generation import SeedSetBuilder
from src.models.schemas.pr_review.pr_patch import PRFilePatch, ChangeType
from src.models.schemas.pr_review.seed_set import SeedSetS0
from src.langgraph.context_assembly import ContextAssemblyService, AssemblyConfig
from src.models.schemas.pr_review.context_pack import ContextPackLimits
from src.core.pr_review_config import pr_review_settings
from src.services.llm import get_llm_client
from src.utils.logging import get_logger
from src.exceptions.pr_review_exceptions import (
    GitHubPRNotFoundException,
    GitHubPermissionException,
    GitHubAuthenticationException,
    CommentNotFoundException,
    InvalidDiffFormatException,
    PRHunkParsingException,
)

logger = get_logger(__name__)

MENTION_PATTERN = re.compile(r"@sentinel\b", re.IGNORECASE)


def _extract_question(body: str) -> str:
    """Remove @sentinel mention and return the remaining question text."""
    cleaned = MENTION_PATTERN.sub("", body or "")
    return cleaned.strip()


def _filter_thread_comments(
    comments: List[Dict[str, Any]],
    root_id: int,
) -> List[Dict[str, Any]]:
    """Filter review comments to the same thread root."""
    filtered = []
    for comment in comments:
        if comment.get("id") == root_id or comment.get("in_reply_to_id") == root_id:
            filtered.append(comment)
    return filtered


def _count_line_changes(hunk_lines: List[str]) -> Dict[str, int]:
    additions = 0
    deletions = 0
    for line in hunk_lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return {"additions": additions, "deletions": deletions}


@activity.defn
async def fetch_comment_context_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch review comment details and assemble thread context.
    """
    repo_name = input_data["repo_name"]
    pr_number = input_data["pr_number"]
    comment_id = input_data["comment_id"]
    installation_id = input_data["installation_id"]

    api_client = PRApiClient()

    try:
        comment = await api_client.get_review_comment(
            repo_name=repo_name,
            comment_id=comment_id,
            installation_id=installation_id,
        )

        parent_comment: Optional[Dict[str, Any]] = None
        diff_hunk = comment.get("diff_hunk")
        file_path = comment.get("path")
        commit_id = comment.get("commit_id")

        if comment.get("in_reply_to_id") and (not diff_hunk or not file_path):
            parent_comment = await api_client.get_review_comment(
                repo_name=repo_name,
                comment_id=comment.get("in_reply_to_id"),
                installation_id=installation_id,
            )
            diff_hunk = parent_comment.get("diff_hunk")
            file_path = parent_comment.get("path")
            commit_id = parent_comment.get("commit_id")

        all_comments = await api_client.list_review_comments(
            repo_name=repo_name,
            pr_number=pr_number,
            installation_id=installation_id,
        )
        thread_root_id = comment.get("in_reply_to_id") or comment.get("id")
        thread_comments = _filter_thread_comments(all_comments, thread_root_id)

        question = _extract_question(comment.get("body", ""))

        return {
            "comment": comment,
            "parent_comment": parent_comment,
            "thread_comments": thread_comments,
            "question": question,
            "diff_hunk": diff_hunk,
            "file_path": file_path,
            "commit_id": commit_id,
            "in_reply_to_id": comment.get("in_reply_to_id"),
        }

    except (
        GitHubPRNotFoundException,
        GitHubPermissionException,
        GitHubAuthenticationException,
        CommentNotFoundException,
    ) as exc:
        logger.error(f"Non-retryable GitHub error fetching comment context: {exc}")
        raise ApplicationError(str(exc), non_retryable=True) from exc
    except Exception as exc:
        logger.error(f"Failed to fetch comment context: {exc}", exc_info=True)
        raise ApplicationError(str(exc)) from exc


@activity.defn
async def build_focused_seed_set_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a focused seed set from a single diff hunk.
    """
    diff_hunk = input_data.get("diff_hunk") or ""
    file_path = input_data.get("file_path")
    clone_path = input_data.get("clone_path")
    change_type = input_data.get("change_type", ChangeType.MODIFIED.value)

    if not file_path or not diff_hunk:
        raise ApplicationError("Missing diff hunk or file path", non_retryable=True)

    try:
        parser = UnifiedDiffParser()
        hunks = parser.parse_patch_to_hunks(diff_hunk, file_path)
        if not hunks:
            raise InvalidDiffFormatException("No hunks parsed from diff_hunk")

        line_stats = _count_line_changes(hunks[0].lines)

        patch = PRFilePatch(
            file_path=file_path,
            change_type=ChangeType(change_type),
            hunks=hunks,
            additions=line_stats["additions"],
            deletions=line_stats["deletions"],
            patch=diff_hunk,
        )

        builder = SeedSetBuilder(clone_path=clone_path)
        seed_set, build_stats = builder.build_seed_set([patch])

        seed_set_data = seed_set.model_dump() if isinstance(seed_set, SeedSetS0) else seed_set
        return {
            "seed_set": seed_set_data,
            "patches": [patch.model_dump()],
            "build_stats": {
                "files_processed": build_stats.files_processed,
                "files_with_symbols": build_stats.files_with_symbols,
                "files_skipped": build_stats.files_skipped,
                "total_symbols_extracted": build_stats.total_symbols_extracted,
                "total_symbols_overlapping": build_stats.total_symbols_overlapping,
                "parse_errors": build_stats.parse_errors,
                "unsupported_languages": build_stats.unsupported_languages,
            },
        }

    except (InvalidDiffFormatException, PRHunkParsingException) as exc:
        logger.error(f"Invalid diff format for comment assist: {exc}")
        raise ApplicationError(str(exc), non_retryable=True) from exc
    except Exception as exc:
        logger.error(f"Failed to build focused seed set: {exc}", exc_info=True)
        raise ApplicationError(str(exc)) from exc


@activity.defn
async def assemble_comment_context_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a focused context pack for comment assistance."""
    repo_id = input_data["repo_id"]
    github_repo_name = input_data["github_repo_name"]
    pr_number = input_data["pr_number"]
    pr_head_sha = input_data["pr_head_sha"]
    pr_base_sha = input_data.get("pr_base_sha", "")
    seed_set_data = input_data["seed_set"]
    patches_data = input_data["patches"]
    kg_candidates = input_data.get("kg_candidates") or {"candidates": []}

    limits = pr_review_settings.comment_assist_limits
    context_limits = ContextPackLimits(
        max_context_items=limits.max_context_items,
        max_total_characters=limits.max_total_characters,
        max_lines_per_snippet=limits.max_lines_per_snippet,
        max_chars_per_item=limits.max_chars_per_item,
        max_hops=pr_review_settings.limits.max_hops,
        max_neighbors_per_seed=pr_review_settings.limits.max_callers_per_seed,
    )

    service_config = AssemblyConfig(
        failure_threshold=3,
        recovery_timeout=30,
        operation_timeout_seconds=120,
        max_context_items=context_limits.max_context_items,
        max_total_characters=context_limits.max_total_characters,
        max_lines_per_snippet=context_limits.max_lines_per_snippet,
        max_chars_per_item=context_limits.max_chars_per_item,
        max_hops=context_limits.max_hops,
        max_neighbors_per_seed=context_limits.max_neighbors_per_seed,
    )

    try:
        from src.models.schemas.pr_review.seed_set import SeedSetS0
        from src.models.schemas.pr_review.pr_patch import PRFilePatch
        import uuid

        seed_set = SeedSetS0(**seed_set_data) if isinstance(seed_set_data, dict) else seed_set_data
        patches = [PRFilePatch(**p) if isinstance(p, dict) else p for p in patches_data]

        context_service = ContextAssemblyService(config=service_config)
        context_pack = await context_service.assemble_context(
            repo_id=uuid.UUID(repo_id),
            github_repo_name=github_repo_name,
            pr_number=pr_number,
            head_sha=pr_head_sha,
            base_sha=pr_base_sha,
            seed_set=seed_set,
            patches=patches,
            kg_candidates=kg_candidates,
            limits=context_limits,
            clone_path=input_data.get("clone_path"),
            kg_commit_sha=input_data.get("kg_commit_sha"),
        )

        return {
            "context_pack": context_pack.model_dump(),
            "assembly_stats": {
                "context_items_generated": len(context_pack.context_items),
                "total_characters": context_pack.total_context_characters,
                "execution_time_seconds": context_pack.assembly_duration_ms / 1000.0 if context_pack.assembly_duration_ms else 0,
            },
            "warnings": [],
        }

    except Exception as exc:
        logger.error(f"Failed to assemble comment context: {exc}", exc_info=True)
        raise ApplicationError(str(exc)) from exc


@activity.defn
async def generate_comment_response_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response text for a comment assist request."""
    question = input_data.get("question", "").strip()
    diff_hunk = input_data.get("diff_hunk") or ""
    file_path = input_data.get("file_path") or ""
    thread_comments = input_data.get("thread_comments") or []
    context_pack = input_data.get("context_pack") or {}

    if not question:
        return {
            "response_text": "Can you clarify what you need help with? Please include the specific concern or goal.",
            "usage": {},
            "model": None,
        }

    prompt_parts = [
        "User question:",
        question,
        "",
    ]

    if file_path:
        prompt_parts.extend([f"File: {file_path}", ""])

    if diff_hunk:
        prompt_parts.extend(["Diff hunk:", diff_hunk, ""])

    if thread_comments:
        recent_thread = thread_comments[-5:]
        prompt_parts.append("Thread context:")
        for comment in recent_thread:
            author = comment.get("user", {}).get("login", "unknown")
            body = comment.get("body", "")
            prompt_parts.append(f"- {author}: {body}")
        prompt_parts.append("")

    context_items = context_pack.get("context_items", [])
    if context_items:
        prompt_parts.append("Relevant context:")
        for item in context_items:
            label = item.get("title") or item.get("path") or "context"
            content = item.get("content") or item.get("snippet") or ""
            prompt_parts.append(f"[{label}]\n{content}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    system_prompt = (
        "You are Sentinel, an AI code review assistant. "
        "Answer concisely (under 500 words). "
        "Cite relevant files/functions when possible. "
        "Ask for clarification if the question is ambiguous."
    )

    try:
        client = get_llm_client()
        response = await client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return {
            "response_text": response.get("content", "").strip(),
            "usage": response.get("usage", {}),
            "model": response.get("model"),
        }
    except Exception as exc:
        logger.error(f"Failed to generate comment response: {exc}", exc_info=True)
        raise ApplicationError(str(exc)) from exc


@activity.defn
async def publish_comment_reply_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Publish comment reply to GitHub."""
    repo_name = input_data["repo_name"]
    pr_number = input_data["pr_number"]
    comment_id = input_data["comment_id"]
    installation_id = input_data["installation_id"]
    response_text = input_data["response_text"]

    api_client = PRApiClient()

    try:
        response = await api_client.create_review_comment_reply(
            repo_name=repo_name,
            pr_number=pr_number,
            comment_id=comment_id,
            body=response_text,
            installation_id=installation_id,
        )
        return {
            "reply_comment_id": response.get("id"),
            "response": response,
        }
    except (GitHubPRNotFoundException, GitHubPermissionException, GitHubAuthenticationException) as exc:
        logger.error(f"Non-retryable GitHub error publishing reply: {exc}")
        raise ApplicationError(str(exc), non_retryable=True) from exc
    except Exception as exc:
        logger.error(f"Failed to publish comment reply: {exc}", exc_info=True)
        raise ApplicationError(str(exc)) from exc
