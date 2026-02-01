"""
API Routes for Context Templates.

This module provides RESTful endpoints for managing context templates,
including CRUD operations with proper authentication and authorization.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse

from src.api.fastapi.middlewares.auth import get_current_user
from src.models.db.users import User
from src.models.schemas.context_templates import (
    ContextTemplateCreate,
    ContextTemplateRead,
    ContextTemplateUpdate,
    ContextTemplateList,
)
from src.services.context_templates import ContextTemplateService
from src.utils.logging.otel_logger import logger


router = APIRouter(
    prefix="/templates",
    tags=["Context Templates"],
)


@router.post(
    "",
    response_model=ContextTemplateRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new context template",
    description="Create a new context template for use in code reviews. "
                "The template will be owned by the authenticated user."
)
async def create_template(
    data: ContextTemplateCreate,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> ContextTemplateRead:
    """
    Create a new context template.

    Args:
        data: Template creation data including name, description, and content
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        The created template
    """
    logger.info(f"Creating template '{data.name}' for user {current_user.email}")
    return template_service.create_template(current_user, data)


@router.get(
    "",
    response_model=ContextTemplateList,
    summary="List context templates",
    description="Get a paginated list of context templates owned by the authenticated user."
)
async def list_templates(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    include_inactive: bool = Query(False, description="Include inactive/archived templates"),
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> ContextTemplateList:
    """
    List templates with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        include_inactive: Whether to include inactive templates
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        Paginated list of templates
    """
    logger.info(f"Listing templates for user {current_user.email}, page {page}")
    return template_service.list_templates(
        current_user,
        page=page,
        page_size=page_size,
        include_inactive=include_inactive
    )


@router.get(
    "/{template_id}",
    response_model=ContextTemplateRead,
    summary="Get a specific context template",
    description="Retrieve a specific context template by its ID. "
                "Only the owner can view private templates."
)
async def get_template(
    template_id: UUID,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> ContextTemplateRead:
    """
    Get a specific template by ID.

    Args:
        template_id: UUID of the template
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        The requested template

    Raises:
        404: Template not found
        401: Unauthorized access
    """
    logger.info(f"Getting template {template_id} for user {current_user.email}")
    return template_service.get_template(template_id, current_user)


@router.patch(
    "/{template_id}",
    response_model=ContextTemplateRead,
    summary="Update a context template",
    description="Update an existing context template. Only the owner can update. "
                "Supports partial updates - only provided fields will be changed."
)
async def update_template(
    template_id: UUID,
    data: ContextTemplateUpdate,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> ContextTemplateRead:
    """
    Update a template.

    Args:
        template_id: UUID of the template to update
        data: Update data (partial updates supported)
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        The updated template

    Raises:
        404: Template not found
        401: Unauthorized (not the owner)
    """
    logger.info(f"Updating template {template_id} by user {current_user.email}")
    return template_service.update_template(template_id, current_user, data)


@router.delete(
    "/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a context template",
    description="Delete a context template. By default performs a soft delete "
                "(archive). Use hard_delete=true to permanently delete."
)
async def delete_template(
    template_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete instead of archive"),
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> None:
    """
    Delete a template.

    Args:
        template_id: UUID of the template to delete
        hard_delete: If True, permanently delete; otherwise soft delete
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        204 No Content on success

    Raises:
        404: Template not found
        401: Unauthorized (not the owner)
    """
    logger.info(
        f"{'Hard' if hard_delete else 'Soft'} deleting template {template_id} "
        f"by user {current_user.email}"
    )
    template_service.delete_template(template_id, current_user, hard_delete)
    return None
