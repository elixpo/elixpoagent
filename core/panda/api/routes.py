"""FastAPI routes — health, sessions, webhook placeholder."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from panda.agent.session import SessionStore
from panda.config import settings

router = APIRouter()

_session_store = SessionStore(settings.agent.session_storage_path)


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/api/v1/sessions")
async def list_sessions(limit: int = 50):
    sessions = _session_store.list_sessions(limit=limit)
    return {"sessions": sessions}


@router.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    session = _session_store.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


class CreateSessionRequest(BaseModel):
    task: str
    repo: str | None = None
    workspace_path: str | None = None


@router.post("/api/v1/sessions")
async def create_session(req: CreateSessionRequest):
    """Create and run a new agent session.

    For now, this is a synchronous placeholder. Full async streaming
    will be implemented with WebSocket support in Phase 3.
    """
    import os
    from panda.agent.engine import AgentEngine
    from panda.agent.session import Session, SessionTrigger
    from panda.llm.client import LLMClient
    from panda.mcp.registry import create_default_registry

    session = Session(
        trigger=SessionTrigger.WEB_DASHBOARD,
        repo_full_name=req.repo,
    )

    workspace = req.workspace_path or os.path.join(settings.agent.workspace_path, session.id)
    os.makedirs(workspace, exist_ok=True)

    llm = LLMClient()
    tools = create_default_registry()
    engine = AgentEngine(llm=llm, tools=tools, session_store=_session_store)

    events = []
    async for event in engine.run(session, task=req.task, workspace_path=workspace):
        events.append(event.to_dict())

    await llm.close()

    return {
        "session_id": session.id,
        "status": session.status.value,
        "events": events,
    }


@router.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = _session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@router.post("/webhook/github")
async def github_webhook():
    """GitHub webhook endpoint — placeholder for Phase 2."""
    return {"status": "received"}
