"""CLI entrypoint — interactive agent in your terminal."""

from __future__ import annotations

import asyncio
import os
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from panda_cli.config import get_api_key, get_api_url, get_model, load_config, save_config

app = typer.Typer(
    name="panda",
    help="Panda AI Agent — autonomous software engineering in your terminal.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    task: str = typer.Argument(None, help="Task to execute. If omitted, enters interactive mode."),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory for the agent."),
    model: str = typer.Option(None, "--model", "-m", help="LLM model override."),
    api_url: str = typer.Option(None, "--api-url", help="LLM API URL override."),
):
    """Start an agent session — give it a task or chat interactively."""
    api_key = get_api_key()
    if not api_key:
        console.print(
            "[red]No API key configured.[/red]\n"
            "Set PANDA_LLM_API_KEY env var or run: panda config --api-key YOUR_KEY"
        )
        raise typer.Exit(1)

    resolved_model = model or get_model()
    resolved_url = api_url or get_api_url()
    resolved_workspace = os.path.abspath(workspace)

    console.print(Panel(
        f"[bold green]Panda Agent[/bold green] v0.1.0\n"
        f"Model: {resolved_model}\n"
        f"Workspace: {resolved_workspace}",
        title="panda",
        border_style="green",
    ))

    if task:
        asyncio.run(_run_task(task, resolved_workspace, resolved_url, api_key, resolved_model))
    else:
        asyncio.run(_interactive_loop(resolved_workspace, resolved_url, api_key, resolved_model))


async def _run_task(task: str, workspace: str, api_url: str, api_key: str, model: str):
    """Run a single task through the agent."""
    from panda.agent.engine import AgentEngine
    from panda.agent.session import Session, SessionStore, SessionTrigger
    from panda.llm.client import LLMClient
    from panda.mcp.registry import create_default_registry

    llm = LLMClient(api_url=api_url, api_key=api_key, model=model)
    tools = create_default_registry()
    store = SessionStore(os.path.join(workspace, ".panda_sessions"))
    engine = AgentEngine(llm=llm, tools=tools, session_store=store)

    session = Session(trigger=SessionTrigger.CLI, max_steps=50)

    async for event in engine.run(session, task=task, workspace_path=workspace):
        _render_event(event)

    await llm.close()


async def _interactive_loop(workspace: str, api_url: str, api_key: str, model: str):
    """Interactive chat loop — keep sending tasks until user exits."""
    from panda.agent.engine import AgentEngine
    from panda.agent.session import Session, SessionStore, SessionTrigger
    from panda.llm.client import LLMClient
    from panda.mcp.registry import create_default_registry

    llm = LLMClient(api_url=api_url, api_key=api_key, model=model)
    tools = create_default_registry()
    store = SessionStore(os.path.join(workspace, ".panda_sessions"))

    console.print("[dim]Type your task, or 'exit' to quit.[/dim]\n")

    while True:
        try:
            task = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if task.strip().lower() in ("exit", "quit", "q"):
            break

        if not task.strip():
            continue

        engine = AgentEngine(llm=llm, tools=tools, session_store=store)
        session = Session(trigger=SessionTrigger.CLI, max_steps=50)

        async for event in engine.run(session, task=task, workspace_path=workspace):
            _render_event(event)

        console.print()

    await llm.close()
    console.print("[dim]Goodbye.[/dim]")


def _render_event(event):
    """Render an agent event to the terminal."""
    etype = event.type
    data = event.data

    if etype == "thinking":
        console.print(f"[dim]Step {data['step']}...[/dim]", end=" ")

    elif etype == "assistant_message":
        console.print()
        console.print(Markdown(data["content"]))

    elif etype == "tool_call":
        tool = data["tool"]
        console.print(f"\n[yellow]> {tool}[/yellow]", end="")
        # Show brief args
        args = data.get("arguments", "")
        if len(args) > 100:
            args = args[:100] + "..."
        console.print(f"[dim]({args})[/dim]")

    elif etype == "tool_result":
        if data.get("success"):
            preview = data.get("output_preview", "")
            if preview:
                lines = preview.split("\n")
                shown = "\n".join(lines[:5])
                if len(lines) > 5:
                    shown += f"\n  ... ({len(lines)} lines total)"
                console.print(f"[green]{shown}[/green]")
        else:
            console.print(f"[red]Error: {data.get('error', 'unknown')}[/red]")

    elif etype == "session_complete":
        usage = data.get("token_usage", {})
        console.print(Panel(
            f"[green]Done[/green] in {data['steps']} steps | "
            f"Tokens: {usage.get('total_tokens', 0):,}",
            border_style="green",
        ))

    elif etype == "error":
        console.print(f"\n[red bold]Error:[/red bold] {data.get('error', 'unknown')}")

    elif etype == "context_compressed":
        console.print("[dim](context compressed)[/dim]")


@app.command()
def config(
    api_key: str = typer.Option(None, "--api-key", help="Set your LLM API key."),
    api_url: str = typer.Option(None, "--api-url", help="Set the LLM API URL."),
    model_name: str = typer.Option(None, "--model", help="Set the default model."),
    show: bool = typer.Option(False, "--show", help="Show current configuration."),
):
    """Configure Panda CLI settings."""
    cfg = load_config()

    if show:
        console.print(Panel(
            f"API URL: {cfg.get('api_url', '(default)')}\n"
            f"API Key: {'***' + cfg['api_key'][-4:] if cfg.get('api_key') else '(not set)'}\n"
            f"Model:   {cfg.get('model', '(default)')}",
            title="Panda Config",
        ))
        return

    changed = False
    if api_key:
        cfg["api_key"] = api_key
        changed = True
    if api_url:
        cfg["api_url"] = api_url
        changed = True
    if model_name:
        cfg["model"] = model_name
        changed = True

    if changed:
        save_config(cfg)
        console.print("[green]Configuration saved.[/green]")
    else:
        console.print("No changes. Use --api-key, --api-url, or --model to set values.")


@app.command()
def version():
    """Show version info."""
    console.print("panda-cli v0.1.0")
