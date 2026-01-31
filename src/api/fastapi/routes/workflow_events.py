"""
SSE endpoint for streaming workflow progress events.

This endpoint provides real-time updates on workflow execution progress
via Server-Sent Events. The frontend connects using EventSource and receives
events as they are emitted by Temporal activities.

Key features:
- JWT authentication via query parameter (EventSource doesn't support headers)
- Reconnection support via last_event_id
- Heartbeats every 30 seconds
- Auto-closes on workflow completion/failure
- Authorization check ensures users can only access their own workflows
"""

import asyncio
import json
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.api.fastapi.middlewares.auth import get_current_user
from src.models.db.users import User
from src.services.workflow_events import WorkflowEventService, WorkflowEventType
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflow_events"])


@router.get("/{workflow_id}/events")
async def stream_workflow_events(
    workflow_id: str,
    run_id: Optional[str] = Query(None, description="Workflow run ID to filter events for current execution"),
    last_event_id: Optional[int] = Query(None, description="Last received sequence number for reconnection"),
    current_user: User = Depends(get_current_user),
):
    """
    Stream workflow progress events via Server-Sent Events (SSE).
    """
    user_id = str(current_user.user_id)
    user_email = current_user.email
    logger.info(f"SSE connection initiated by {user_email} for workflow_id={workflow_id}, run_id={run_id}")

    # Step 3: Create event generator
    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generate SSE events by polling the database.

        Polls every 500ms for new events and streams them to the client.
        Includes heartbeats every 30 seconds to keep connection alive.
        """
        service = WorkflowEventService()
        last_seq = last_event_id or 0
        heartbeat_counter = 0
        poll_interval = 0.5  # 500ms
        heartbeat_interval = 30  # 30 seconds
        polls_per_heartbeat = int(heartbeat_interval / poll_interval)

        # Send connection established event
        yield f"event: connected\ndata: {json.dumps({'workflow_id': workflow_id, 'sequence': last_seq})}\n\n"

        while True:
            try:
                # Fetch new events
                events = await service.get_events_since(
                    workflow_id=workflow_id,
                    user_id=user_id,
                    workflow_run_id=run_id,
                    since_sequence=last_seq,
                    limit=50,  # Fetch up to 50 events per poll
                )

                # Stream events
                for event in events:
                    # Format: event: <type>\nid: <seq>\ndata: <json>\n\n
                    event_data = event.model_dump(mode="json")
                    yield f"event: activity\nid: {event.sequence_number}\ndata: {json.dumps(event_data)}\n\n"

                    last_seq = event.sequence_number

                    # Check for terminal events
                    if event.event_type in [
                        WorkflowEventType.WORKFLOW_COMPLETED.value,
                        WorkflowEventType.WORKFLOW_FAILED.value,
                    ]:
                        logger.info(
                            f"Terminal event received for workflow_id={workflow_id}: {event.event_type}. "
                            f"Closing SSE connection."
                        )
                        # Send final event and close
                        yield f"event: close\ndata: {json.dumps({'reason': event.event_type})}\n\n"
                        return

                # Heartbeat to keep connection alive
                heartbeat_counter += 1
                if heartbeat_counter >= polls_per_heartbeat:
                    yield ": heartbeat\n\n"
                    heartbeat_counter = 0

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error in SSE event generator for workflow_id={workflow_id}: {e}")
                # Send error event and close
                error_data = {"error": str(e), "workflow_id": workflow_id}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                return

    # Step 4: Return streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )
