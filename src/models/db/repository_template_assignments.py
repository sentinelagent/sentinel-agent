"""
SQLAlchemy model for repository_template_assignments table.

This junction table links repositories to context templates,
supporting multiple templates per repository with priority ordering.
"""

from sqlalchemy import Column, TIMESTAMP, text, ForeignKey, Boolean, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base


class RepositoryTemplateAssignment(Base):
    """
    SQLAlchemy model representing a repository-template assignment.

    This is a junction table that links repositories to context templates.
    Each repository can have multiple templates assigned with different priorities,
    allowing for layered context configuration during code reviews.

    Attributes:
        id: Unique identifier (UUID)
        repository_id: Foreign key to repositories table
        template_id: Foreign key to context_templates table
        priority: Ordering priority (lower = higher priority)
        is_active: Soft delete flag for the assignment
        created_at: Timestamp when assignment was created
        updated_at: Timestamp when last modified
    """

    __tablename__ = 'repository_template_assignments'

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )

    # Foreign keys
    repository_id = Column(
        UUID(as_uuid=True),
        ForeignKey('repositories.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    template_id = Column(
        UUID(as_uuid=True),
        ForeignKey('context_templates.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # Priority for ordering (lower value = higher priority)
    priority = Column(
        Integer,
        nullable=False,
        server_default=text("0")
    )

    # Soft delete / archive flag
    is_active = Column(Boolean, nullable=False, server_default=text("TRUE"))

    # Audit timestamps
    created_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )

    # Unique constraint for repository-template pair
    __table_args__ = (
        UniqueConstraint('repository_id', 'template_id', name='uq_repository_template'),
    )

    # Relationships
    repository = relationship("Repository", back_populates="template_assignments")
    template = relationship("ContextTemplate", back_populates="repository_assignments")

    def __repr__(self) -> str:
        return (
            f"<RepositoryTemplateAssignment("
            f"id={self.id}, "
            f"repository_id={self.repository_id}, "
            f"template_id={self.template_id}, "
            f"priority={self.priority}"
            f")>"
        )
