from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import json
from typing import Optional
from src.api.fastapi.middlewares.github import GithubMiddleware
from src.utils.logging.otel_logger import logger
from src.services.github.github_service import GithubService
from src.core.config import settings

router = APIRouter(
    prefix="/github",
    tags=["Github"],
)

@router.get("/auth")
async def github_auth(
    github_service: GithubService = Depends(GithubService)
):
    """Handle GitHub OAuth authentication"""
    return github_service.handle_auth()

@router.get("/callback")
async def github_callback(
    request: Request,
    code: Optional[str] = None, 
    state: Optional[str] = None,
    github_service: GithubService = Depends(GithubService)
):
    """Handle GitHub OAuth callback"""
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    return await github_service.handle_callback(code, state)

@router.get("/setup")
async def github_setup(
    installation_id: Optional[str] = None,
    setup_action: Optional[str] = None,
    state: Optional[str] = None
):
    """
    Handle GitHub App installation setup callback
    
    This endpoint is called by GitHub after a user completes the installation.
    Redirects to the frontend's settings page with the installation_id.
    """
    from fastapi.responses import RedirectResponse
    from src.core.config import settings
    
    if not installation_id:
        logger.warning("No installation_id provided in setup callback")
        # Redirect to settings with error
        frontend_url = settings.FRONTEND_URL or "http://localhost:3000"
        return RedirectResponse(url=f"{frontend_url}/settings?error=no_installation_id")
        
    # Redirect to frontend settings page with installation_id
    frontend_url = settings.FRONTEND_URL or "http://localhost:3000"
    return RedirectResponse(url=f"{frontend_url}/settings?connected=true&installation_id={installation_id}")

@router.post("/events")
async def github_webhook(
    request: Request,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None),
    github_service: GithubService = Depends(GithubService)
):
    """Handle GitHub webhook events"""
    github_middleware = GithubMiddleware()
    try:
        body_bytes = await request.body()
        
        webhook_secret =settings.GITHUB_WEBHOOK_SECRET if settings.GITHUB_WEBHOOK_SECRET else None

        if webhook_secret and not github_middleware.verify_webhook_signature(body_bytes, x_hub_signature_256, webhook_secret):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        try:
            body = json.loads(body_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        if not x_github_event:
            raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")
        
        logger.info(f"GitHub webhook received - event: {x_github_event}")
        
        result = await github_service.process_webhook(body, x_github_event)
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")