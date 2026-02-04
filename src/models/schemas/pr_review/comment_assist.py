"""
Comment Assist Workflow Schemas

Pydantic schemas for @sentinel comment assist workflow.
"""

from datetime import datetime
from typing import Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field, validator


class PRCommentAssistRequest(BaseModel):
    """Input contract for comment assist workflow."""

    installation_id: int = Field(..., description="GitHub installation ID", gt=0)

    repo_id: UUID = Field(..., description="Internal repository UUID")
    github_repo_id: int = Field(..., description="GitHub repository ID", gt=0)
    github_repo_name: str = Field(
        ...,
        description="Repository name in owner/repo format",
        pattern=r"^[^/]+/[^/]+$",
    )

    pr_number: int = Field(..., description="Pull request number", ge=1)
    comment_id: int = Field(..., description="GitHub review comment ID", ge=1)

    head_sha: str = Field(
        ...,
        description="PR head commit SHA",
        min_length=40,
        max_length=40,
    )
    base_sha: str = Field(
        ...,
        description="PR base commit SHA",
        min_length=40,
        max_length=40,
    )

    in_reply_to_id: Optional[int] = Field(
        None,
        description="Parent comment ID if this is a reply",
        ge=1,
    )

    @validator("head_sha", "base_sha")
    def validate_sha_format(cls, v: str) -> str:
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("SHA must be a valid 40-character hexadecimal string")
        return v.lower()

    @validator("github_repo_name")
    def validate_repo_name_format(cls, v: str) -> str:
        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("Repository name must be in owner/repo format")
        if not all(part.strip() for part in parts):
            raise ValueError("Owner and repository name cannot be empty")
        return v


class PRCommentAssistResult(BaseModel):
    """Output contract for comment assist workflow."""

    status: Literal["completed", "failed", "cancelled"] = Field(
        ..., description="Final workflow status"
    )
    comment_id: int = Field(..., description="Original GitHub review comment ID", ge=1)
    reply_comment_id: Optional[int] = Field(
        None, description="GitHub reply comment ID", ge=1
    )
    response_text: Optional[str] = Field(None, description="Response body posted to GitHub")
    processing_duration_ms: Optional[int] = Field(
        None, description="Total processing time in milliseconds", ge=0
    )
    error_message: Optional[str] = Field(None, description="Error message if workflow failed")
    completed_at: datetime = Field(..., description="Workflow completion timestamp")
