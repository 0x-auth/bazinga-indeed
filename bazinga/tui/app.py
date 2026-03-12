"""
BAZINGA TUI - Main Application

Full-screen terminal interface like Claude Code.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Input, RichLog, Rule
from textual.binding import Binding
from textual.reactive import reactive

from rich.text import Text
from rich.panel import Panel

import asyncio
from typing import Optional
from datetime import datetime

# Try to import agent loop
try:
    from ..agent.loop import AgentLoop
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class StatusBar(Static):
    """Status bar showing current state."""

    coherence = reactive(0.0)
    source = reactive("ready")
    mode = reactive("")

    def render(self) -> Text:
        text = Text()
        text.append(" φ=", style="bold yellow")
        text.append(f"{self.coherence:.3f}", style="yellow")
        text.append(" │ ", style="dim")
        text.append("source: ", style="dim")
        text.append(self.source, style="cyan")
        if self.mode:
            text.append(" │ ", style="dim")
            text.append(self.mode, style="green bold")
        text.append(" │ ", style="dim")
        text.append("1.618033988749895", style="dim yellow")
        return text


class BazingaApp(App):
    """BAZINGA Full-Screen TUI Application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #header-bar {
        dock: top;
        height: 1;
        background: #238636;
        padding: 0 1;
        color: white;
    }

    #chat-log {
        height: 1fr;
        border: solid #30363d;
        padding: 1;
        background: #0d1117;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: #21262d;
        padding: 0 1;
    }

    #input-box {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: #161b22;
    }

    Input {
        border: tall #30363d;
    }

    Input:focus {
        border: tall #58a6ff;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+k", "kb_search", "KB Search", show=True),
        Binding("ctrl+m", "multi_ai", "Multi-AI", show=True),
        Binding("ctrl+a", "agent_mode", "Agent", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, bazinga_instance=None, mode: str = "chat"):
        super().__init__()
        self.bazinga = bazinga_instance
        self.mode = mode
        self.is_processing = False
        self.multi_ai_mode = False
        self.agent_mode = False
        self.agent_loop = None

        # Conversation history for multi-turn chat
        self.conversation_history = []

        # Initialize agent if available
        if AGENT_AVAILABLE:
            try:
                self.agent_loop = AgentLoop(verbose=False)
            except Exception:
                self.agent_loop = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static(
            "[bold]BAZINGA[/bold] v5.4.1 │ The first AI you actually own │"
            "[dim]Ctrl+K: KB │ Ctrl+M: Multi-AI │ Ctrl+A: Agent │ Ctrl+C: Quit[/dim]",
            id="header-bar"
        )

        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)

        yield StatusBar(id="status-bar")

        yield Input(
            placeholder="Ask anything... (Enter to send)",
            id="input-box"
        )

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one("#input-box", Input).focus()

        # Welcome message
        log = self.query_one("#chat-log", RichLog)
        log.write(Panel(
            "[bold green]Welcome to BAZINGA![/bold green]\n\n"
            "Ask me anything. I'll search memory, quantum patterns, knowledge base, "
            "and LLMs to find the best answer.\n\n"
            "[bold cyan]Agent Mode (Ctrl+A):[/bold cyan] I can read/edit files and run commands!\n"
            "Try: 'Check my script.py and fix the bug on line 10'\n\n"
            "[dim]Shortcuts: Ctrl+K (KB) │ Ctrl+M (Multi-AI) │ Ctrl+A (Agent) │ Ctrl+L (Clear)[/dim]",
            title="[yellow]φ = 1.618033988749895[/yellow]",
            border_style="green"
        ))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if self.is_processing:
            return

        query = event.value.strip()
        if not query:
            return

        # Clear input
        input_box = self.query_one("#input-box", Input)
        input_box.value = ""

        # Add user message
        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold cyan]You:[/bold cyan] {query}")

        # Handle special commands
        if query.startswith("/"):
            await self._handle_command(query, log)
            return

        # Process query
        self.is_processing = True
        status = self.query_one("#status-bar", StatusBar)
        status.source = "thinking..."

        try:
            response, metadata = await self._process_query(query)

            # Display response
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            if metadata:
                coherence = metadata.get('coherence', 0)
                source = metadata.get('source', 'unknown')
                log.write(f"[dim]  φ={coherence:.3f} │ source: {source}[/dim]")

            # Update status
            status.coherence = metadata.get('coherence', 0)
            status.source = metadata.get('source', 'ready')
        except Exception as e:
            log.write(f"\n[bold red]Error:[/bold red] {str(e)}")
            status.source = "error"
        finally:
            self.is_processing = False
            if self.multi_ai_mode:
                status.mode = "[MULTI-AI]"
            else:
                status.mode = ""

    async def _handle_command(self, cmd: str, log: RichLog) -> None:
        """Handle slash commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            log.write(Panel(
                "[bold]Commands:[/bold]\n"
                "  /agent <task>  - Agent mode (read/edit files, run commands)\n"
                "  /kb <query>    - Search knowledge base\n"
                "  /multi <query> - Multi-AI consensus\n"
                "  /chain         - Show blockchain status\n"
                "  /stats         - Show session stats\n"
                "  /clear         - Clear chat\n"
                "  /quit          - Exit BAZINGA\n\n"
                "[bold]Keyboard Shortcuts:[/bold]\n"
                "  Ctrl+A - Toggle Agent mode\n"
                "  Ctrl+K - KB search prefix\n"
                "  Ctrl+M - Toggle Multi-AI\n"
                "  Ctrl+L - Clear chat",
                title="Help",
                border_style="blue"
            ))
        elif command == "/clear":
            log.clear()
            self.conversation_history = []
            log.write("[dim]Chat cleared. Conversation history reset.[/dim]")
        elif command == "/quit":
            self.exit()
        elif command == "/kb" and args:
            status = self.query_one("#status-bar", StatusBar)
            status.source = "searching KB..."
            log.write(f"\n[bold cyan]You:[/bold cyan] [KB Search] {args}")
            response, metadata = await self._process_query(f"[KB SEARCH] {args}")
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            status.source = "ready"
        elif command == "/multi" and args:
            self.multi_ai_mode = True
            status = self.query_one("#status-bar", StatusBar)
            status.mode = "[MULTI-AI]"
            status.source = "consensus..."
            log.write(f"\n[bold cyan]You:[/bold cyan] [Multi-AI] {args}")
            response, metadata = await self._process_query(args)
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            self.multi_ai_mode = False
            status.mode = ""
            status.source = "ready"
        elif command == "/agent":
            if not args:
                # Toggle agent mode
                self.action_agent_mode()
            else:
                # Run agent for this task
                if not AGENT_AVAILABLE or not self.agent_loop:
                    log.write("[yellow]Agent mode not available.[/yellow]")
                    return
                self.agent_mode = True
                status = self.query_one("#status-bar", StatusBar)
                status.mode = "[AGENT]"
                status.source = "working..."
                log.write(f"\n[bold cyan]You:[/bold cyan] [Agent] {args}")
                response, metadata = await self._process_query(args)
                log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
                self.agent_mode = False
                status.mode = ""
                status.source = "ready"
        elif command == "/stats":
            # Show session stats
            stats_text = (
                f"[bold]Session Stats:[/bold]\n"
                f"  Agent available: {AGENT_AVAILABLE}\n"
                f"  Agent mode: {self.agent_mode}\n"
                f"  Multi-AI mode: {self.multi_ai_mode}\n"
                f"  BAZINGA connected: {self.bazinga is not None}"
            )
            if self.bazinga:
                try:
                    bz_stats = self.bazinga.get_stats() if hasattr(self.bazinga, 'get_stats') else {}
                    if bz_stats:
                        stats_text += f"\n  Queries: {bz_stats.get('queries', 0)}"
                        stats_text += f"\n  Cache hits: {bz_stats.get('cache_hits', 0)}"
                except Exception:
                    pass
            log.write(Panel(stats_text, title="Stats", border_style="blue"))
        elif command == "/chain":
            # Show blockchain status
            if self.bazinga and hasattr(self.bazinga, 'chain'):
                try:
                    chain = self.bazinga.chain
                    chain_text = (
                        f"[bold]Blockchain Status:[/bold]\n"
                        f"  Length: {len(chain.chain) if hasattr(chain, 'chain') else 'N/A'}\n"
                        f"  Valid: {chain.is_valid() if hasattr(chain, 'is_valid') else 'N/A'}"
                    )
                    log.write(Panel(chain_text, title="Chain", border_style="green"))
                except Exception as e:
                    log.write(f"[yellow]Chain error: {e}[/yellow]")
            else:
                log.write("[dim]Blockchain not initialized.[/dim]")
        else:
            log.write(f"[yellow]Unknown command: {command}. Type /help for commands.[/yellow]")

    async def _process_query(self, query: str) -> tuple:
        """Process user query through BAZINGA."""
        if self.bazinga is None:
            # Demo mode without actual BAZINGA instance
            await asyncio.sleep(0.5)  # Simulate processing
            return (
                f"[Demo Mode] You asked: {query}\n\n"
                "BAZINGA instance not connected. Run with actual instance for real responses.",
                {"coherence": 0.618, "source": "demo"}
            )

        try:
            # Agent mode - use ReAct loop with tools
            if self.agent_mode and self.agent_loop:
                return await self._process_agent_query(query)

            # Build context from conversation history (last 3 exchanges = 6 turns)
            full_query = query
            if self.conversation_history:
                context = "Previous conversation:\n"
                for turn in self.conversation_history[-6:]:
                    context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
                context += "\nCurrent question: "
                full_query = context + query

            # Multi-AI mode
            if self.multi_ai_mode:
                result = await self.bazinga.multi_ai_ask(full_query)
            else:
                result = await self.bazinga.ask(full_query, fresh=True)

            # Extract response text
            if isinstance(result, dict):
                response_text = result.get('response', str(result))
                metadata = {
                    'coherence': result.get('coherence', 0),
                    'source': result.get('source', 'unknown')
                }
            else:
                response_text = str(result)
                metadata = {'coherence': 0.618, 'source': 'llm'}

            # Store in conversation history
            self.conversation_history.append({
                'user': query,
                'assistant': response_text
            })

            return (response_text, metadata)
        except Exception as e:
            return (f"Error: {str(e)}", {'coherence': 0, 'source': 'error'})

    async def _process_agent_query(self, query: str) -> tuple:
        """Process query using agent loop (ReAct pattern)."""
        import os
        log = self.query_one("#chat-log", RichLog)
        status = self.query_one("#status-bar", StatusBar)

        # Provide current directory context to the agent
        cwd = os.getcwd()
        context = f"Current working directory: {cwd}\nUser's home: {os.path.expanduser('~')}"

        try:
            final_answer = ""
            step_count = 0

            async for step in self.agent_loop.run(query, context=context):
                step_count += 1

                # Show thought process
                if step.thought:
                    log.write(f"[dim]💭 {step.thought}[/dim]")

                # Show tool usage
                if step.tool:
                    status.source = f"using {step.tool}..."
                    log.write(f"[cyan]🔧 Tool: {step.tool}[/cyan]")
                    if step.tool_args:
                        args_str = str(step.tool_args)[:100]
                        log.write(f"[dim]   Args: {args_str}[/dim]")

                # Show observation
                if step.observation:
                    obs_preview = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
                    log.write(f"[dim]👁 Observation: {obs_preview}[/dim]")

                # Final answer
                if step.is_final:
                    final_answer = step.final_answer
                    break

            return (final_answer or "Agent completed but no answer provided.",
                    {'coherence': 0.85, 'source': 'agent'})

        except Exception as e:
            return (f"Agent error: {str(e)}", {'coherence': 0, 'source': 'error'})

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_kb_search(self) -> None:
        """Trigger KB search mode."""
        input_box = self.query_one("#input-box", Input)
        if not input_box.value.startswith("/kb "):
            input_box.value = "/kb " + input_box.value
        input_box.focus()

    def action_multi_ai(self) -> None:
        """Toggle multi-AI mode."""
        self.multi_ai_mode = not self.multi_ai_mode
        self.agent_mode = False  # Disable agent mode
        status = self.query_one("#status-bar", StatusBar)
        if self.multi_ai_mode:
            status.mode = "[MULTI-AI]"
        else:
            status.mode = ""

    def action_agent_mode(self) -> None:
        """Toggle agent mode (can read/edit files, run commands)."""
        import os
        if not AGENT_AVAILABLE:
            log = self.query_one("#chat-log", RichLog)
            log.write("[yellow]Agent mode not available. Missing agent module.[/yellow]")
            return

        self.agent_mode = not self.agent_mode
        self.multi_ai_mode = False  # Disable multi-AI mode
        status = self.query_one("#status-bar", StatusBar)
        log = self.query_one("#chat-log", RichLog)

        if self.agent_mode:
            status.mode = "[AGENT]"
            cwd = os.getcwd()
            log.write(Panel(
                "[bold cyan]Agent Mode ON[/bold cyan]\n\n"
                f"[dim]Working directory: {cwd}[/dim]\n\n"
                "I can now:\n"
                "- Read and edit files\n"
                "- Run bash commands\n"
                "- Search your codebase\n"
                "- Fix bugs in your code\n\n"
                "Try: 'Read my config.py and explain what it does'\n"
                "Or: 'Fix the syntax error in main.py'\n\n"
                "[dim]Tip: Use absolute paths like ~/project/file.py[/dim]",
                border_style="cyan"
            ))
        else:
            status.mode = ""
            log.write("[dim]Agent mode OFF[/dim]")

    def action_clear_chat(self) -> None:
        """Clear chat history."""
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        self.conversation_history = []
        log.write("[dim]Chat cleared. Conversation history reset. Ready for new conversation.[/dim]")

    def action_cancel(self) -> None:
        """Cancel current operation."""
        if self.is_processing:
            self.is_processing = False
            status = self.query_one("#status-bar", StatusBar)
            status.source = "cancelled"


def run_tui(bazinga_instance=None, mode: str = "chat"):
    """Run the BAZINGA TUI application (synchronous)."""
    app = BazingaApp(bazinga_instance=bazinga_instance, mode=mode)
    app.run()


async def run_tui_async(bazinga_instance=None, mode: str = "chat"):
    """Run the BAZINGA TUI application (async)."""
    app = BazingaApp(bazinga_instance=bazinga_instance, mode=mode)
    await app.run_async()


if __name__ == "__main__":
    # Demo mode
    run_tui()
