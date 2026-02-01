"""
Pydantic schemas for context templates feature.

This module defines all request/response schemas for the context templates API,
including enums for visibility, template CRUD operations, and repository assignments.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================

class TemplateVisibility(str, Enum):
    """
    Visibility levels for context templates.

    - PRIVATE: Only the owner can view and use the template
    - ORGANIZATION: Visible to all users in the same organization (future)
    - PUBLIC: Visible to all users (future)
    """
    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class TemplateContentSchema(BaseModel):
    """
    Schema for the template content structure.

    This defines the expected structure of the JSONB template_content field.
    The content is flexible but should generally contain review guidelines
    and configuration options.
    """
    guidelines: Optional[List[str]] = Field(
        default=None,
        description="List of review guidelines to apply"
    )
    coding_standards: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Coding standards configuration"
    )
    custom_rules: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Custom rules for code review"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Areas to focus on during review (e.g., security, performance)"
    )
    ignore_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to ignore during review"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context or instructions for the reviewer"
    )

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


# =============================================================================
# CONTEXT TEMPLATE SCHEMAS
# =============================================================================

class ContextTemplateBase(BaseModel):
    """Base schema with common template fields."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the template"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description explaining the template's purpose"
    )


class ContextTemplateCreate(ContextTemplateBase):
    """Schema for creating a new context template."""
    template_content: Dict[str, Any] = Field(
        default_factory=dict,
        description="The template configuration content"
    )
    visibility: TemplateVisibility = Field(
        default=TemplateVisibility.PRIVATE,
        description="Access level for the template"
    )

    @field_validator('template_content')
    @classmethod
    def validate_template_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure template_content is a valid dictionary."""
        if not isinstance(v, dict):
            raise ValueError("template_content must be a dictionary")
        return v


class ContextTemplateUpdate(BaseModel):
    """
    Schema for updating an existing context template.

    All fields are optional to support partial updates.
    """
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated template name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Updated description"
    )
    template_content: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated template configuration"
    )
    visibility: Optional[TemplateVisibility] = Field(
        default=None,
        description="Updated visibility level"
    )
    is_active: Optional[bool] = Field(
        default=None,
        description="Updated active status"
    )

    @field_validator('template_content')
    @classmethod
    def validate_template_content(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ensure template_content is a valid dictionary if provided."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("template_content must be a dictionary")
        return v


class ContextTemplateRead(ContextTemplateBase):
    """Schema for reading a context template (response)."""
    id: UUID
    user_id: Optional[UUID]  # NULL for system/default templates
    template_content: Dict[str, Any]
    visibility: TemplateVisibility
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ContextTemplateList(BaseModel):
    """Schema for paginated template list response."""
    templates: List[ContextTemplateRead]
    total: int
    page: int
    page_size: int
    has_more: bool


class ContextTemplateSummary(BaseModel):
    """Lightweight schema for template summaries (used in lists/dropdowns)."""
    id: UUID
    name: str
    description: Optional[str]
    visibility: TemplateVisibility
    is_default: bool
    is_active: bool

    class Config:
        from_attributes = True


# =============================================================================
# REPOSITORY TEMPLATE ASSIGNMENT SCHEMAS
# =============================================================================

class RepositoryTemplateAssignmentCreate(BaseModel):
    """Schema for creating a new repository-template assignment."""
    template_id: UUID = Field(
        ...,
        description="ID of the template to assign"
    )
    priority: int = Field(
        default=0,
        ge=0,
        description="Priority order (lower = higher priority)"
    )


class RepositoryTemplateAssignmentUpdate(BaseModel):
    """Schema for updating an assignment."""
    priority: Optional[int] = Field(
        default=None,
        ge=0,
        description="Updated priority"
    )
    is_active: Optional[bool] = Field(
        default=None,
        description="Updated active status"
    )


class RepositoryTemplateAssignmentRead(BaseModel):
    """Schema for reading an assignment (response)."""
    id: UUID
    repository_id: UUID
    template_id: UUID
    priority: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RepositoryTemplateAssignmentWithTemplate(RepositoryTemplateAssignmentRead):
    """Assignment with embedded template details."""
    template: ContextTemplateSummary


class RepositoryTemplatesResponse(BaseModel):
    """Response schema for getting all templates assigned to a repository."""
    repository_id: UUID
    assignments: List[RepositoryTemplateAssignmentWithTemplate]
    total: int


# =============================================================================
# BULK OPERATIONS SCHEMAS
# =============================================================================

class BulkAssignTemplatesRequest(BaseModel):
    """Request schema for bulk assigning templates to a repository."""
    template_ids: List[UUID] = Field(
        ...,
        min_length=1,
        description="List of template IDs to assign"
    )
    replace_existing: bool = Field(
        default=False,
        description="If true, removes existing assignments first"
    )


class BulkAssignTemplatesResponse(BaseModel):
    """Response schema for bulk assignment operation."""
    repository_id: UUID
    assigned_count: int
    assignments: List[RepositoryTemplateAssignmentRead]


class ReorderAssignmentsRequest(BaseModel):
    """Request schema for reordering template assignments."""
    assignment_ids: List[UUID] = Field(
        ...,
        min_length=1,
        description="Ordered list of assignment IDs (first = highest priority)"
    )


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    # Enums
    'TemplateVisibility',
    # Content schema
    'TemplateContentSchema',
    # Template schemas
    'ContextTemplateBase',
    'ContextTemplateCreate',
    'ContextTemplateUpdate',
    'ContextTemplateRead',
    'ContextTemplateList',
    'ContextTemplateSummary',
    # Assignment schemas
    'RepositoryTemplateAssignmentCreate',
    'RepositoryTemplateAssignmentUpdate',
    'RepositoryTemplateAssignmentRead',
    'RepositoryTemplateAssignmentWithTemplate',
    'RepositoryTemplatesResponse',
    # Bulk operations
    'BulkAssignTemplatesRequest',
    'BulkAssignTemplatesResponse',
    'ReorderAssignmentsRequest',
]
