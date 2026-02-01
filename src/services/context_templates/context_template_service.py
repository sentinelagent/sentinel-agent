"""
Context Template Service.

This service implements the business logic for managing context templates
and repository-template assignments. It follows SOLID principles with
single responsibility for template operations and proper dependency injection.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

from fastapi import Depends, status
from sqlalchemy import and_, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from src.core.database import get_db
from src.models.db.context_templates import ContextTemplate
from src.models.db.repositories import Repository
from src.models.db.repository_template_assignments import RepositoryTemplateAssignment
from src.models.db.users import User
from src.models.schemas.context_templates import (
    ContextTemplateCreate,
    ContextTemplateRead,
    ContextTemplateUpdate,
    ContextTemplateList,
    ContextTemplateSummary,
    RepositoryTemplateAssignmentCreate,
    RepositoryTemplateAssignmentRead,
    RepositoryTemplateAssignmentUpdate,
    RepositoryTemplateAssignmentWithTemplate,
    RepositoryTemplatesResponse,
    BulkAssignTemplatesRequest,
    BulkAssignTemplatesResponse,
    ReorderAssignmentsRequest,
    TemplateVisibility,
)
from src.utils.exception import (
    AppException,
    NotFoundException,
    BadRequestException,
    DuplicateResourceException,
    ForbiddenException,
)
from src.utils.logging.otel_logger import logger


class ContextTemplateNotFoundError(NotFoundException):
    """Raised when a context template is not found."""
    def __init__(self, template_id: UUID):
        super().__init__(message=f"Context template with ID {template_id} not found")


class TemplateAssignmentNotFoundError(NotFoundException):
    """Raised when a template assignment is not found."""
    def __init__(self, assignment_id: UUID):
        super().__init__(message=f"Template assignment with ID {assignment_id} not found")


class TemplateAlreadyAssignedError(DuplicateResourceException):
    """Raised when trying to assign an already-assigned template."""
    def __init__(self, repository_id: UUID, template_id: UUID):
        super().__init__(
            message=f"Template {template_id} is already assigned to repository {repository_id}"
        )


class ContextTemplateService:
    """
    Service class for managing context templates and assignments.

    This service handles all CRUD operations for context templates,
    including authorization checks to ensure users can only access
    their own templates (for private visibility).

    Attributes:
        db: SQLAlchemy database session
    """

    def __init__(self, db: Session = Depends(get_db)):
        """
        Initialize the service with a database session.

        Args:
            db: SQLAlchemy session injected via FastAPI's Depends()
        """
        self.db = db

    # =========================================================================
    # AUTHORIZATION HELPERS
    # =========================================================================

    def _verify_template_ownership(
        self,
        template: ContextTemplate,
        user: User,
        action: str = "access"
    ) -> None:
        """
        Verify that a user has permission to access/modify a template.

        Args:
            template: The template to check
            user: The user making the request
            action: Description of the action for error messages

        Raises:
            ForbiddenException: If user doesn't have permission
        """
        # For private templates, only the owner can access
        if template.visibility == TemplateVisibility.PRIVATE.value:
            if template.user_id != user.user_id:
                logger.warning(
                    f"User {user.user_id} attempted to {action} template {template.id} "
                    f"owned by {template.user_id}"
                )
                raise ForbiddenException(
                    f"You do not have permission to {action} this template"
                )
        # Future: Add organization and public visibility checks here

    def _verify_repository_access(
        self,
        repository: Repository,
        user: User
    ) -> None:
        """
        Verify that a user has access to a repository.

        Args:
            repository: The repository to check
            user: The user making the request

        Raises:
            ForbiddenException: If user doesn't have access
        """
        # Check if user has a GitHub installation that owns this repository
        user_installation_ids = [
            inst.installation_id for inst in user.github_installations
        ]

        if repository.installation_id not in user_installation_ids:
            logger.warning(
                f"User {user.user_id} attempted to access repository {repository.id} "
                f"without proper installation access"
            )
            raise ForbiddenException(
                "You do not have access to this repository"
            )

    # =========================================================================
    # TEMPLATE CRUD OPERATIONS
    # =========================================================================

    def create_template(
        self,
        user: User,
        data: ContextTemplateCreate
    ) -> ContextTemplateRead:
        """
        Create a new context template.

        Args:
            user: The authenticated user creating the template
            data: Template creation data

        Returns:
            The created template

        Raises:
            AppException: On database errors
        """
        logger.info(f"Creating context template '{data.name}' for user {user.user_id}")

        try:
            template = ContextTemplate(
                user_id=user.user_id,
                name=data.name,
                description=data.description,
                template_content=data.template_content,
                visibility=data.visibility.value,
                is_active=True,
            )

            self.db.add(template)
            self.db.commit()
            self.db.refresh(template)

            logger.info(f"Created context template {template.id} for user {user.user_id}")
            return ContextTemplateRead.model_validate(template)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating template: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to create template due to a database error"
            )

    def get_template(
        self,
        template_id: UUID,
        user: User
    ) -> ContextTemplateRead:
        """
        Get a specific template by ID.

        Args:
            template_id: UUID of the template
            user: The authenticated user

        Returns:
            The template if found and accessible

        Raises:
            ContextTemplateNotFoundError: If template not found
            ForbiddenException: If user doesn't have access
        """
        template = self.db.query(ContextTemplate).filter(
            ContextTemplate.id == template_id,
            ContextTemplate.is_active == True
        ).first()

        if not template:
            raise ContextTemplateNotFoundError(template_id)

        self._verify_template_ownership(template, user, "view")

        return ContextTemplateRead.model_validate(template)

    def list_templates(
        self,
        user: User,
        page: int = 1,
        page_size: int = 20,
        include_inactive: bool = False
    ) -> ContextTemplateList:
        """
        List templates accessible to the user with pagination.

        Args:
            user: The authenticated user
            page: Page number (1-indexed)
            page_size: Number of items per page
            include_inactive: Whether to include inactive templates

        Returns:
            Paginated list of templates
        """
        query = self.db.query(ContextTemplate).filter(
            ContextTemplate.user_id == user.user_id
        )

        if not include_inactive:
            query = query.filter(ContextTemplate.is_active == True)

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        templates = query.order_by(ContextTemplate.created_at.desc()) \
            .offset(offset) \
            .limit(page_size) \
            .all()

        has_more = (offset + len(templates)) < total

        return ContextTemplateList(
            templates=[ContextTemplateRead.model_validate(t) for t in templates],
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )

    def update_template(
        self,
        template_id: UUID,
        user: User,
        data: ContextTemplateUpdate
    ) -> ContextTemplateRead:
        """
        Update an existing template.

        Args:
            template_id: UUID of the template to update
            user: The authenticated user
            data: Update data (partial updates supported)

        Returns:
            The updated template

        Raises:
            ContextTemplateNotFoundError: If template not found
            ForbiddenException: If user doesn't own the template
            AppException: On database errors
        """
        template = self.db.query(ContextTemplate).filter(
            ContextTemplate.id == template_id
        ).first()

        if not template:
            raise ContextTemplateNotFoundError(template_id)

        self._verify_template_ownership(template, user, "update")

        try:
            # Apply updates only for provided fields
            update_data = data.model_dump(exclude_unset=True)

            for field, value in update_data.items():
                if value is not None:
                    # Handle enum conversion for visibility
                    if field == 'visibility' and isinstance(value, TemplateVisibility):
                        value = value.value
                    setattr(template, field, value)

            template.updated_at = datetime.now(timezone.utc)

            self.db.commit()
            self.db.refresh(template)

            logger.info(f"Updated context template {template_id} by user {user.user_id}")
            return ContextTemplateRead.model_validate(template)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating template {template_id}: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to update template due to a database error"
            )

    def delete_template(
        self,
        template_id: UUID,
        user: User,
        hard_delete: bool = False
    ) -> None:
        """
        Delete a template (soft delete by default).

        Args:
            template_id: UUID of the template to delete
            user: The authenticated user
            hard_delete: If True, permanently delete; otherwise soft delete

        Raises:
            ContextTemplateNotFoundError: If template not found
            ForbiddenException: If user doesn't own the template
            AppException: On database errors
        """
        template = self.db.query(ContextTemplate).filter(
            ContextTemplate.id == template_id
        ).first()

        if not template:
            raise ContextTemplateNotFoundError(template_id)

        self._verify_template_ownership(template, user, "delete")

        try:
            if hard_delete:
                # Hard delete - cascades to assignments due to FK constraint
                self.db.delete(template)
                logger.info(f"Hard deleted template {template_id} by user {user.user_id}")
            else:
                # Soft delete
                template.is_active = False
                template.updated_at = datetime.now(timezone.utc)

                # Also deactivate all assignments
                self.db.query(RepositoryTemplateAssignment).filter(
                    RepositoryTemplateAssignment.template_id == template_id
                ).update({
                    RepositoryTemplateAssignment.is_active: False,
                    RepositoryTemplateAssignment.updated_at: datetime.now(timezone.utc)
                })

                logger.info(f"Soft deleted template {template_id} by user {user.user_id}")

            self.db.commit()

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deleting template {template_id}: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to delete template due to a database error"
            )

    # =========================================================================
    # REPOSITORY TEMPLATE ASSIGNMENT OPERATIONS
    # =========================================================================

    def assign_template_to_repository(
        self,
        repository_id: UUID,
        user: User,
        data: RepositoryTemplateAssignmentCreate
    ) -> RepositoryTemplateAssignmentRead:
        """
        Assign a template to a repository.

        Args:
            repository_id: UUID of the repository
            user: The authenticated user
            data: Assignment data including template_id and priority

        Returns:
            The created assignment

        Raises:
            NotFoundException: If repository or template not found
            ForbiddenException: If user doesn't have access
            TemplateAlreadyAssignedError: If template already assigned
            AppException: On database errors
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Verify template exists and user has access
        template = self.db.query(ContextTemplate).filter(
            ContextTemplate.id == data.template_id,
            ContextTemplate.is_active == True
        ).first()

        if not template:
            raise ContextTemplateNotFoundError(data.template_id)

        self._verify_template_ownership(template, user, "assign")

        # Check if assignment already exists
        existing = self.db.query(RepositoryTemplateAssignment).filter(
            RepositoryTemplateAssignment.repository_id == repository_id,
            RepositoryTemplateAssignment.template_id == data.template_id
        ).first()

        if existing:
            if existing.is_active:
                raise TemplateAlreadyAssignedError(repository_id, data.template_id)
            else:
                # Reactivate existing assignment
                existing.is_active = True
                existing.priority = data.priority
                existing.updated_at = datetime.now(timezone.utc)
                self.db.commit()
                self.db.refresh(existing)
                logger.info(
                    f"Reactivated template assignment {existing.id} for "
                    f"repository {repository_id}"
                )
                return RepositoryTemplateAssignmentRead.model_validate(existing)

        try:
            assignment = RepositoryTemplateAssignment(
                repository_id=repository_id,
                template_id=data.template_id,
                priority=data.priority,
                is_active=True,
            )

            self.db.add(assignment)
            self.db.commit()
            self.db.refresh(assignment)

            logger.info(
                f"Created template assignment {assignment.id}: "
                f"template {data.template_id} -> repository {repository_id}"
            )
            return RepositoryTemplateAssignmentRead.model_validate(assignment)

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity error creating assignment: {str(e)}")
            raise TemplateAlreadyAssignedError(repository_id, data.template_id)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating assignment: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to create assignment due to a database error"
            )

    def get_repository_templates(
        self,
        repository_id: UUID,
        user: User,
        include_inactive: bool = False
    ) -> RepositoryTemplatesResponse:
        """
        Get all templates assigned to a repository.

        Args:
            repository_id: UUID of the repository
            user: The authenticated user
            include_inactive: Whether to include inactive assignments

        Returns:
            List of assignments with embedded template details

        Raises:
            NotFoundException: If repository not found
            ForbiddenException: If user doesn't have access
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Query assignments with template details
        query = self.db.query(RepositoryTemplateAssignment).options(
            joinedload(RepositoryTemplateAssignment.template)
        ).filter(
            RepositoryTemplateAssignment.repository_id == repository_id
        )

        if not include_inactive:
            query = query.filter(RepositoryTemplateAssignment.is_active == True)

        assignments = query.order_by(
            RepositoryTemplateAssignment.priority.asc()
        ).all()

        # Build response with embedded template summaries
        assignment_responses = []
        for assignment in assignments:
            template_summary = ContextTemplateSummary.model_validate(assignment.template)
            assignment_with_template = RepositoryTemplateAssignmentWithTemplate(
                id=assignment.id,
                repository_id=assignment.repository_id,
                template_id=assignment.template_id,
                priority=assignment.priority,
                is_active=assignment.is_active,
                created_at=assignment.created_at,
                updated_at=assignment.updated_at,
                template=template_summary
            )
            assignment_responses.append(assignment_with_template)

        return RepositoryTemplatesResponse(
            repository_id=repository_id,
            assignments=assignment_responses,
            total=len(assignment_responses)
        )

    def update_assignment(
        self,
        repository_id: UUID,
        assignment_id: UUID,
        user: User,
        data: RepositoryTemplateAssignmentUpdate
    ) -> RepositoryTemplateAssignmentRead:
        """
        Update an existing assignment.

        Args:
            repository_id: UUID of the repository (for URL consistency validation)
            assignment_id: UUID of the assignment
            user: The authenticated user
            data: Update data

        Returns:
            The updated assignment

        Raises:
            TemplateAssignmentNotFoundError: If assignment not found
            BadRequestException: If assignment doesn't belong to the specified repository
            ForbiddenException: If user doesn't have access
            AppException: On database errors
        """
        assignment = self.db.query(RepositoryTemplateAssignment).options(
            joinedload(RepositoryTemplateAssignment.repository)
        ).filter(
            RepositoryTemplateAssignment.id == assignment_id
        ).first()

        if not assignment:
            raise TemplateAssignmentNotFoundError(assignment_id)

        # Validate that the assignment belongs to the specified repository
        if assignment.repository_id != repository_id:
            raise BadRequestException(
                f"Assignment {assignment_id} does not belong to repository {repository_id}"
            )

        self._verify_repository_access(assignment.repository, user)

        try:
            update_data = data.model_dump(exclude_unset=True)

            for field, value in update_data.items():
                if value is not None:
                    setattr(assignment, field, value)

            assignment.updated_at = datetime.now(timezone.utc)

            self.db.commit()
            self.db.refresh(assignment)

            logger.info(f"Updated assignment {assignment_id} by user {user.user_id}")
            return RepositoryTemplateAssignmentRead.model_validate(assignment)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating assignment {assignment_id}: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to update assignment due to a database error"
            )

    def remove_template_from_repository(
        self,
        repository_id: UUID,
        template_id: UUID,
        user: User,
        hard_delete: bool = False
    ) -> None:
        """
        Remove a template assignment from a repository.

        Args:
            repository_id: UUID of the repository
            template_id: UUID of the template
            user: The authenticated user
            hard_delete: If True, permanently delete; otherwise soft delete

        Raises:
            NotFoundException: If assignment not found
            ForbiddenException: If user doesn't have access
            AppException: On database errors
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Find the assignment
        assignment = self.db.query(RepositoryTemplateAssignment).filter(
            RepositoryTemplateAssignment.repository_id == repository_id,
            RepositoryTemplateAssignment.template_id == template_id
        ).first()

        if not assignment:
            raise NotFoundException(
                f"Template {template_id} is not assigned to repository {repository_id}"
            )

        try:
            if hard_delete:
                self.db.delete(assignment)
                logger.info(
                    f"Hard deleted assignment for template {template_id} "
                    f"from repository {repository_id}"
                )
            else:
                assignment.is_active = False
                assignment.updated_at = datetime.now(timezone.utc)
                logger.info(
                    f"Soft deleted assignment for template {template_id} "
                    f"from repository {repository_id}"
                )

            self.db.commit()

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error removing assignment: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to remove assignment due to a database error"
            )

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    def bulk_assign_templates(
        self,
        repository_id: UUID,
        user: User,
        data: BulkAssignTemplatesRequest
    ) -> BulkAssignTemplatesResponse:
        """
        Bulk assign multiple templates to a repository.

        Args:
            repository_id: UUID of the repository
            user: The authenticated user
            data: Bulk assignment request

        Returns:
            Result of the bulk operation

        Raises:
            NotFoundException: If repository or templates not found
            ForbiddenException: If user doesn't have access
            AppException: On database errors
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Verify all templates exist and user has access
        templates = self.db.query(ContextTemplate).filter(
            ContextTemplate.id.in_(data.template_ids),
            ContextTemplate.is_active == True
        ).all()

        if len(templates) != len(data.template_ids):
            found_ids = {t.id for t in templates}
            missing_ids = [tid for tid in data.template_ids if tid not in found_ids]
            raise NotFoundException(f"Templates not found: {missing_ids}")

        for template in templates:
            self._verify_template_ownership(template, user, "assign")

        try:
            if data.replace_existing:
                # Remove existing assignments
                self.db.query(RepositoryTemplateAssignment).filter(
                    RepositoryTemplateAssignment.repository_id == repository_id
                ).delete()

            # Create new assignments with priority based on order
            created_assignments = []
            for priority, template_id in enumerate(data.template_ids):
                # Check if assignment already exists
                existing = self.db.query(RepositoryTemplateAssignment).filter(
                    RepositoryTemplateAssignment.repository_id == repository_id,
                    RepositoryTemplateAssignment.template_id == template_id
                ).first()

                if existing:
                    existing.is_active = True
                    existing.priority = priority
                    existing.updated_at = datetime.now(timezone.utc)
                    created_assignments.append(existing)
                else:
                    assignment = RepositoryTemplateAssignment(
                        repository_id=repository_id,
                        template_id=template_id,
                        priority=priority,
                        is_active=True,
                    )
                    self.db.add(assignment)
                    created_assignments.append(assignment)

            self.db.commit()

            # Refresh all assignments
            for assignment in created_assignments:
                self.db.refresh(assignment)

            logger.info(
                f"Bulk assigned {len(created_assignments)} templates to "
                f"repository {repository_id}"
            )

            return BulkAssignTemplatesResponse(
                repository_id=repository_id,
                assigned_count=len(created_assignments),
                assignments=[
                    RepositoryTemplateAssignmentRead.model_validate(a)
                    for a in created_assignments
                ]
            )

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error in bulk assign: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to bulk assign templates due to a database error"
            )

    def reorder_assignments(
        self,
        repository_id: UUID,
        user: User,
        data: ReorderAssignmentsRequest
    ) -> List[RepositoryTemplateAssignmentRead]:
        """
        Reorder template assignments for a repository.

        Args:
            repository_id: UUID of the repository
            user: The authenticated user
            data: Reorder request with ordered list of assignment IDs

        Returns:
            Updated list of assignments

        Raises:
            NotFoundException: If repository or assignments not found
            ForbiddenException: If user doesn't have access
            BadRequestException: If assignment IDs don't match repository
            AppException: On database errors
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Fetch all specified assignments
        assignments = self.db.query(RepositoryTemplateAssignment).filter(
            RepositoryTemplateAssignment.id.in_(data.assignment_ids),
            RepositoryTemplateAssignment.repository_id == repository_id,
            RepositoryTemplateAssignment.is_active == True
        ).all()

        if len(assignments) != len(data.assignment_ids):
            raise BadRequestException(
                "Some assignment IDs are invalid or don't belong to this repository"
            )

        try:
            # Create a mapping of id to assignment
            assignment_map = {a.id: a for a in assignments}

            # Update priorities based on order in the request
            for priority, assignment_id in enumerate(data.assignment_ids):
                assignment = assignment_map[assignment_id]
                assignment.priority = priority
                assignment.updated_at = datetime.now(timezone.utc)

            self.db.commit()

            # Refresh and return
            result = []
            for assignment_id in data.assignment_ids:
                assignment = assignment_map[assignment_id]
                self.db.refresh(assignment)
                result.append(RepositoryTemplateAssignmentRead.model_validate(assignment))

            logger.info(f"Reordered {len(result)} assignments for repository {repository_id}")
            return result

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error reordering assignments: {str(e)}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to reorder assignments due to a database error"
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_effective_templates_for_repository(
        self,
        repository_id: UUID,
        user: User
    ) -> List[ContextTemplateRead]:
        """
        Get the effective list of templates for a repository in priority order.

        This is useful when building the context for a code review.

        Args:
            repository_id: UUID of the repository
            user: The authenticated user

        Returns:
            List of templates in priority order (highest priority first)

        Raises:
            NotFoundException: If repository not found
            ForbiddenException: If user doesn't have access
        """
        # Verify repository exists and user has access
        repository = self.db.query(Repository).filter(
            Repository.id == repository_id
        ).first()

        if not repository:
            raise NotFoundException(f"Repository with ID {repository_id} not found")

        self._verify_repository_access(repository, user)

        # Fetch assignments with templates in a single query using joinedload
        assignments = self.db.query(RepositoryTemplateAssignment).options(
            joinedload(RepositoryTemplateAssignment.template)
        ).filter(
            RepositoryTemplateAssignment.repository_id == repository_id,
            RepositoryTemplateAssignment.is_active == True
        ).order_by(
            RepositoryTemplateAssignment.priority.asc()
        ).all()

        # Extract active templates from the pre-loaded assignments
        templates = []
        for assignment in assignments:
            if assignment.template and assignment.template.is_active:
                templates.append(ContextTemplateRead.model_validate(assignment.template))

        return templates
