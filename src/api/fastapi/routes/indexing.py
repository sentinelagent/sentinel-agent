from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from temporalio.client import Client
from src.api.fastapi.middlewares.auth import get_current_user
from src.core.temporal_client import temporal_client
from src.models.db.users import User
from src.utils.response import IndexRepoResponse, IndexRepoResponseItem
from src.utils.requests import IndexRepoRequest
from src.workflows.repo_indexing_workflow import RepoIndexingWorkflow
from src.services.repository.repository_service import RepositoryService
from src.models.schemas.context_templates import BulkAssignTemplatesRequest

router = APIRouter(prefix="/indexing", tags=["Indexing"])

@router.post("/index", response_model = IndexRepoResponse)
async def index_repo(
    repo_request: IndexRepoRequest,
    temporal_client: Client = Depends(temporal_client.get_client),
    current_user: User = Depends(get_current_user),
    repo_service: RepositoryService = Depends(RepositoryService)
):
    """
    Trigger repository indexing workflows for multiple repositories.

    This endpoint:
    1. Validates user has access to the installation
    2. Starts a Temporal workflow for each repository in the list
    3. Returns workflow handles and SSE event URLs for tracking all started workflows
    """

    try:
        repo_list = repo_request.repositories
        responses = []

        for repo in repo_list:
            # Step 1: Ensure repository exists in our local DB and get its UUID
            db_repo = repo_service.upsert_repository(
                installation_id=repo_request.installation_id,
                repo_data=repo.model_dump()
            )

            # Step 1.5: Handle template assignments if provided
            if repo.template_ids:
                from src.services.context_templates.context_template_service import ContextTemplateService
                template_service = ContextTemplateService(repo_service.db)
                template_service.bulk_assign_templates(
                    repository_id=db_repo.id,
                    user=current_user,
                    data=BulkAssignTemplatesRequest(
                        template_ids=repo.template_ids,
                        replace_existing=True
                    )
                )
            
            # Step 2: Use the local DB UUID for repo_id in the workflow
            workflow_repo_data = repo.model_dump(mode="json")
            workflow_repo_data["repo_id"] = str(db_repo.id)
            
            workflow_id = f"repo-index-{repo.github_repo_id}-{repo.default_branch}"
            input_data = {
                "installation_id": repo_request.installation_id,
                "user_id": str(current_user.user_id),
                "repository": workflow_repo_data,
            }
            handle = await temporal_client.start_workflow(
                RepoIndexingWorkflow.run,
                input_data,
                id=workflow_id,
                task_queue="repo-indexing-queue",
            )
            responses.append(
                IndexRepoResponseItem(
                    workflow_id=handle.id,
                    run_id=str(handle.first_execution_run_id or ""),
                    message=f"Indexing started for repository {repo.github_repo_name}",
                    repo_name=repo.github_repo_name,
                    events_url=f"/api/workflows/{handle.id}/events"
                )
            )

        return IndexRepoResponse(
            repositories=responses,
            total_count=len(responses)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index repositories: {str(e)}")
    
