"""Core agent loop — plan, think, act, observe."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from panda.agent.context import ContextManager
from panda.agent.prompt import build_system_prompt
from panda.agent.session import Session, SessionStatus, SessionStore, TokenUsage
from panda.config import settings
from panda.llm.client import LLMClient, assemble_tool_result
from panda.llm.models import Message, ToolCall
from panda.mcp.registry import ToolRegistry

log = structlog.get_logger()


class AgentEvent:
    """An event emitted by the agent loop for streaming to clients."""

    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}


class AgentEngine:
    """Runs the agentic loop: receive task -> plan -> (think -> act -> observe)* -> done."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        session_store: SessionStore,
        context_manager: ContextManager | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.session_store = session_store
        self.context = context_manager or ContextManager(
            max_tokens=settings.llm.max_context_tokens,
        )

    async def run(
        self,
        session: Session,
        task: str,
        workspace_path: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop, yielding events as they occur."""
        session.status = SessionStatus.RUNNING
        session.workspace_path = workspace_path

        # Build initial messages
        system_prompt = build_system_prompt(
            task=task,
            repo_name=session.repo_full_name,
            plan=session.plan,
        )

        if not session.messages:
            session.messages.append(Message(role="system", content=system_prompt))
            session.messages.append(Message(role="user", content=task))

        tool_defs = self.tools.list_tool_defs()

        yield AgentEvent("session_start", {"session_id": session.id, "task": task})

        while session.current_step < session.max_steps:
            session.current_step += 1

            # Compress context if needed
            if self.context.needs_compression(session.messages):
                session.messages = self.context.compress(session.messages)
                yield AgentEvent("context_compressed", {"step": session.current_step})

            # Think: send messages to LLM
            yield AgentEvent("thinking", {"step": session.current_step})

            try:
                response = await self.llm.chat(
                    messages=session.messages,
                    tools=tool_defs,
                )
            except Exception as e:
                log.error("agent.llm_error", error=str(e), step=session.current_step)
                session.status = SessionStatus.FAILED
                yield AgentEvent("error", {"error": str(e)})
                break

            # Update token usage
            if response.usage:
                session.token_usage.prompt_tokens += response.usage.prompt_tokens
                session.token_usage.completion_tokens += response.usage.completion_tokens
                session.token_usage.total_tokens += response.usage.total_tokens

            choice = response.choices[0] if response.choices else None
            if not choice:
                session.status = SessionStatus.FAILED
                yield AgentEvent("error", {"error": "No response from LLM"})
                break

            assistant_msg = choice.message
            session.messages.append(assistant_msg)

            # If the assistant has text content, yield it
            if assistant_msg.content:
                yield AgentEvent("assistant_message", {
                    "content": assistant_msg.content,
                    "step": session.current_step,
                })

            # If there are tool calls, execute them
            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    yield AgentEvent("tool_call", {
                        "tool": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "step": session.current_step,
                    })

                    result = await self.tools.execute(tool_call, workspace_path)

                    yield AgentEvent("tool_result", {
                        "tool": tool_call.function.name,
                        "success": result.success,
                        "output_preview": result.output[:500] if result.output else "",
                        "error": result.error,
                        "step": session.current_step,
                    })

                    # Add tool result to conversation
                    tool_msg = assemble_tool_result(
                        tool_call_id=tool_call.id,
                        content=result.output if result.success else f"ERROR: {result.error}",
                    )
                    session.messages.append(tool_msg)

            # If finish_reason is "stop" and no tool calls, the agent is done
            elif choice.finish_reason == "stop":
                session.status = SessionStatus.COMPLETED
                import time
                session.completed_at = time.time()
                yield AgentEvent("session_complete", {
                    "session_id": session.id,
                    "steps": session.current_step,
                    "token_usage": session.token_usage.model_dump(),
                })
                break

            # Check token budget
            if session.token_usage.total_tokens >= settings.agent.max_tokens_per_session:
                log.warning("agent.token_budget_exceeded", session_id=session.id)
                session.status = SessionStatus.FAILED
                yield AgentEvent("error", {"error": "Token budget exceeded"})
                break

            # Persist after each step
            self.session_store.save(session)

        else:
            # Max steps reached
            session.status = SessionStatus.FAILED
            yield AgentEvent("error", {"error": f"Max steps ({session.max_steps}) reached"})

        # Final save
        self.session_store.save(session)
        yield AgentEvent("session_end", {"session_id": session.id, "status": session.status.value})
