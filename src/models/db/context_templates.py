"""
SQLAlchemy model for context_templates table.

Context templates store reusable context configurations for code reviews,
including review guidelines, coding standards, and custom rules.
"""

from sqlalchemy import Column, String, Text, TIMESTAMP, text, ForeignKey, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from src.core.database import Base


class ContextTemplate(Base):
    """
    SQLAlchemy model representing a context template.

    A context template is a reusable configuration containing review guidelines,
    coding standards, and other contextual information that can be applied to
    repositories during code reviews.

    Attributes:
        id: Unique identifier (UUID)
        user_id: Owner of the template (NULL for system/default templates)
        name: Human-readable template name
        description: Optional description of the template
        template_content: JSONB containing the actual configuration
        visibility: Access level (private, organization, public)
        is_default: System/default template indicator (TRUE for built-in templates)
        is_active: Soft delete flag
        created_at: Timestamp when created
        updated_at: Timestamp when last modified
    """

    __tablename__ = 'context_templates'

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )

    # Ownership - references users table (NULL for system/default templates)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.user_id', ondelete='CASCADE'),
        nullable=True,
        index=True
    )

    # Template metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Template content - JSONB for flexible structure
    template_content = Column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb")
    )

    # Visibility control
    visibility = Column(
        String(50),
        nullable=False,
        server_default=text("'private'")
    )

    # System/default template indicator
    is_default = Column(Boolean, nullable=False, server_default=text("FALSE"))

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

    # Relationships
    user = relationship("User", back_populates="context_templates")
    repository_assignments = relationship(
        "RepositoryTemplateAssignment",
        back_populates="template",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ContextTemplate(id={self.id}, name='{self.name}', user_id={self.user_id})>"
