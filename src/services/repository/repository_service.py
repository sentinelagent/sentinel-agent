import datetime
from src.models.db.users import User
from src.models.schemas.repositories import RepositoryRead, RepositoryCreate
from src.utils.logging.otel_logger import logger
from typing import Any, Dict, List
from sqlalchemy.exc import SQLAlchemyError
import httpx
from src.services.repository.helpers import RepositoryHelpers
from src.utils.exception import AppException, UserNotFoundError
from sqlalchemy.orm import Session
from fastapi import Depends
from src.core.database import get_db
from src.models.db.repositories import Repository

class RepositoryService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.helpers = RepositoryHelpers(db)

    async def get_all_repositories(self, current_user: User) -> List[RepositoryRead]:
        """Get a list of all repositories from GitHub for the user's installation."""
        if not current_user.github_installations:
            raise UserNotFoundError("No GitHub installation found for the current user.")
            
        installation = current_user.github_installations[0]
        installation_id: int = installation.installation_id
        
        try:
            installation_token: str = await self.helpers.generate_installation_token(installation_id)
            repositories_data = await self._fetch_all_repositories_from_github(installation_token)
            return repositories_data
        except Exception as e:
            logger.error(f"Error getting all repositories for user {current_user.email}: {str(e)}")
            if not isinstance(e, AppException):
                raise AppException(status_code=500, message="An unexpected error occurred while fetching repositories.")
            raise e
        
    async def _fetch_all_repositories_from_github(self, installation_token: str) -> List[Dict[str, Any]]:
        """Get a list of all repositories from the GitHub API."""
        repos_url: str = "https://api.github.com/installation/repositories"
        
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {installation_token}",
                "Accept": "application/vnd.github+json"
            }
            response = await client.get(repos_url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get all repositories from GitHub API: {response.status_code} {response.text}")
                raise AppException(
                    status_code=response.status_code,
                    message="Failed to fetch repositories from GitHub."
                )
            
            repositories_data = response.json()
            return repositories_data.get("repositories", [])
        
    def get_user_selected_repositories(self, current_user: User) -> List[RepositoryRead]:
        """Get a list of user's repositories from our database."""
        if not current_user.github_installations:
            raise UserNotFoundError("No GitHub installation found for the current user.")
            
        installation = current_user.github_installations[0]
        installation_id: int = installation.installation_id
        try:
            result = self.db.query(Repository).filter(Repository.installation_id == installation_id).all()
            return result
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error getting repositories for user {current_user.email}: {str(e)}")
            raise AppException(status_code=500, message="A database error occurred while fetching repositories.")
        except Exception as e:
            logger.error(f"Unexpected error getting repositories for user {current_user.email}: {str(e)}")
            raise AppException(status_code=500, message="An unexpected error occurred while fetching repositories.")

    def upsert_repository(self, installation_id: int, repo_data: Dict[str, Any]) -> Repository:
        """
        Create or update a repository record in the local database.
        
        Args:
            installation_id: The GitHub installation ID
            repo_data: Dictionary containing repository metadata
            
        Returns:
            The Repository DB model instance
        """
        github_repo_id = repo_data["github_repo_id"]
        full_name = repo_data["github_repo_name"]
        
        # Split owner/repo if possible
        if "/" in full_name:
            owner_login, name = full_name.split("/", 1)
        else:
            owner_login = "unknown"
            name = full_name

        try:
            # Check if exists
            repo = self.db.query(Repository).filter(Repository.github_repo_id == github_repo_id).first()
            
            if repo:
                # Update existing
                repo.name = name
                repo.full_name = full_name
                repo.owner_login = owner_login
                repo.default_branch = repo_data.get("default_branch", repo.default_branch)
                repo.installation_id = installation_id
                repo.updated_at = datetime.datetime.utcnow()
            else:
                # Create new
                repo = Repository(
                    github_repo_id=github_repo_id,
                    installation_id=installation_id,
                    name=name,
                    full_name=full_name,
                    owner_login=owner_login,
                    private=repo_data.get("private", False),
                    default_branch=repo_data.get("default_branch", "main"),
                )
                self.db.add(repo)
            
            self.db.commit()
            self.db.refresh(repo)
            return repo
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to upsert repository {full_name}: {e}")
            raise AppException(status_code=500, message=f"Database error during repository upsert: {str(e)}")