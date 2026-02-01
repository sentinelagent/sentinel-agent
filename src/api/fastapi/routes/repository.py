"""
API Routes for Repository Management.

This module provides RESTful endpoints for repository operations,
including template assignments for context configuration.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from src.api.fastapi.middlewares.auth import get_current_user
from src.models.db.users import User
from src.models.schemas.context_templates import (
    RepositoryTemplateAssignmentCreate,
    RepositoryTemplateAssignmentRead,
    RepositoryTemplateAssignmentUpdate,
    RepositoryTemplatesResponse,
    BulkAssignTemplatesRequest,
    BulkAssignTemplatesResponse,
    ReorderAssignmentsRequest,
    ContextTemplateRead,
)
from src.services.context_templates import ContextTemplateService
from src.services.repository.repository_service import RepositoryService
from src.utils.logging.otel_logger import logger


router = APIRouter(
    prefix="/repository",
    tags=["Repository"],
)


# =============================================================================
# EXISTING REPOSITORY ENDPOINTS
# =============================================================================

@router.get("/all")
async def get_all_repositories(
    current_user: User = Depends(get_current_user),
    repository_service: RepositoryService = Depends(RepositoryService)
):
    """Get a list of all repositories"""
    return await repository_service.get_all_repositories(current_user)


@router.get("/user-selected")
async def get_user_selected_repositories(
    current_user: User = Depends(get_current_user),
    repository_service: RepositoryService = Depends(RepositoryService)
):
    """Get a list of user's repositories"""
    return repository_service.get_user_selected_repositories(current_user)


# =============================================================================
# TEMPLATE ASSIGNMENT ENDPOINTS
# =============================================================================

@router.get(
    "/{repository_id}/templates",
    response_model=RepositoryTemplatesResponse,
    summary="Get templates assigned to a repository",
    description="Retrieve all context templates assigned to a repository, "
                "ordered by priority (highest priority first)."
)
async def get_repository_templates(
    repository_id: UUID,
    include_inactive: bool = Query(False, description="Include inactive assignments"),
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> RepositoryTemplatesResponse:
    """
    Get all templates assigned to a repository.

    Args:
        repository_id: UUID of the repository
        include_inactive: Whether to include inactive assignments
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        List of template assignments with embedded template details
    """
    logger.info(f"Getting templates for repository {repository_id}")
    return template_service.get_repository_templates(
        repository_id,
        current_user,
        include_inactive=include_inactive
    )


@router.post(
    "/{repository_id}/templates",
    response_model=RepositoryTemplateAssignmentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Assign a template to a repository",
    description="Assign a context template to a repository. The template will be "
                "used to provide context during code reviews."
)
async def assign_template_to_repository(
    repository_id: UUID,
    data: RepositoryTemplateAssignmentCreate,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> RepositoryTemplateAssignmentRead:
    """
    Assign a template to a repository.

    Args:
        repository_id: UUID of the repository
        data: Assignment data including template_id and priority
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        The created assignment

    Raises:
        404: Repository or template not found
        409: Template already assigned
    """
    logger.info(
        f"Assigning template {data.template_id} to repository {repository_id}"
    )
    return template_service.assign_template_to_repository(
        repository_id,
        current_user,
        data
    )


@router.patch(
    "/{repository_id}/templates/{assignment_id}",
    response_model=RepositoryTemplateAssignmentRead,
    summary="Update a template assignment",
    description="Update an existing template assignment (e.g., change priority)."
)
async def update_template_assignment(
    repository_id: UUID,
    assignment_id: UUID,
    data: RepositoryTemplateAssignmentUpdate,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> RepositoryTemplateAssignmentRead:
    """
    Update a template assignment.

    Args:
        repository_id: UUID of the repository (for URL consistency)
        assignment_id: UUID of the assignment to update
        data: Update data
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        The updated assignment
    """
    logger.info(f"Updating assignment {assignment_id} for repository {repository_id}")
    return template_service.update_assignment(repository_id, assignment_id, current_user, data)


@router.delete(
    "/{repository_id}/templates/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a template from a repository",
    description="Remove a template assignment from a repository. By default "
                "performs a soft delete (archive)."
)
async def remove_template_from_repository(
    repository_id: UUID,
    template_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete the assignment"),
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> None:
    """
    Remove a template from a repository.

    Args:
        repository_id: UUID of the repository
        template_id: UUID of the template to remove
        hard_delete: If True, permanently delete; otherwise soft delete
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        204 No Content on success
    """
    logger.info(
        f"{'Hard' if hard_delete else 'Soft'} removing template {template_id} "
        f"from repository {repository_id}"
    )
    template_service.remove_template_from_repository(
        repository_id,
        template_id,
        current_user,
        hard_delete=hard_delete
    )
    return None


@router.post(
    "/{repository_id}/templates/bulk",
    response_model=BulkAssignTemplatesResponse,
    summary="Bulk assign templates to a repository",
    description="Assign multiple templates to a repository at once. "
                "Optionally replace all existing assignments."
)
async def bulk_assign_templates(
    repository_id: UUID,
    data: BulkAssignTemplatesRequest,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> BulkAssignTemplatesResponse:
    """
    Bulk assign templates to a repository.

    Args:
        repository_id: UUID of the repository
        data: Bulk assignment request with list of template IDs
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        Result of the bulk operation
    """
    logger.info(
        f"Bulk assigning {len(data.template_ids)} templates to repository {repository_id}"
    )
    return template_service.bulk_assign_templates(repository_id, current_user, data)


@router.post(
    "/{repository_id}/templates/reorder",
    response_model=List[RepositoryTemplateAssignmentRead],
    summary="Reorder template assignments",
    description="Reorder template assignments by providing an ordered list "
                "of assignment IDs. First in list = highest priority."
)
async def reorder_template_assignments(
    repository_id: UUID,
    data: ReorderAssignmentsRequest,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> List[RepositoryTemplateAssignmentRead]:
    """
    Reorder template assignments.

    Args:
        repository_id: UUID of the repository
        data: Reorder request with ordered list of assignment IDs
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        Updated list of assignments in new order
    """
    logger.info(f"Reordering {len(data.assignment_ids)} assignments for repository {repository_id}")
    return template_service.reorder_assignments(repository_id, current_user, data)


@router.get(
    "/{repository_id}/effective-templates",
    response_model=List[ContextTemplateRead],
    summary="Get effective templates for a repository",
    description="Get the full template content for all active templates assigned "
                "to a repository, in priority order. Useful for building review context."
)
async def get_effective_templates(
    repository_id: UUID,
    current_user: User = Depends(get_current_user),
    template_service: ContextTemplateService = Depends(ContextTemplateService)
) -> List[ContextTemplateRead]:
    """
    Get effective templates for a repository.

    This endpoint returns the full template content (not just summaries)
    for all active templates assigned to the repository, ordered by priority.

    Args:
        repository_id: UUID of the repository
        current_user: The authenticated user (injected)
        template_service: Template service (injected)

    Returns:
        List of full template objects in priority order
    """
    logger.info(f"Getting effective templates for repository {repository_id}")
    return template_service.get_effective_templates_for_repository(
        repository_id,
        current_user
    )