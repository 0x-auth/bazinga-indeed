"""
BAZINGA TUI - Main Application

Full-screen terminal interface inspired by Claude Code.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Static, Input, RichLog, Rule, TextArea
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message
from textual.events import Key
from textual.timer import Timer

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

import asyncio
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import agent loop
try:
    from ..agent.loop import AgentLoop
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class ChatInput(TextArea):
    """Multi-line input that submits on Enter, newline on Shift+Enter."""

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False),
    ]

    class Submitted(Message):
        """Posted when user presses Enter to submit."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, **kwargs):
        super().__init__(
            language=None,
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="indent",
            **kwargs,
        )

    def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
                self.clear()

    def action_submit(self) -> None:
        pass


class StatusBar(Static):
    """Status bar showing current state — inspired by Claude Code's bottom bar."""

    coherence = reactive(0.0)
    source = reactive("ready")
    mode = reactive("")
    token_count = reactive(0)

    def render(self) -> Text:
        text = Text()
        # Mode indicator (left)
        if self.mode:
            text.append(f" {self.mode} ", style="bold white on #238636")
            text.append(" ", style="dim")

        # Coherence
        text.append(" φ=", style="bold yellow")
        phi_color = "green" if self.coherence > 0.618 else "yellow" if self.coherence > 0.3 else "red"
        text.append(f"{self.coherence:.3f}", style=phi_color)
        text.append(" │ ", style="dim")

        # Source
        source_colors = {
            "ready": "green", "thinking...": "yellow", "searching KB...": "cyan",
            "consensus...": "magenta", "working...": "yellow", "error": "red",
            "cancelled": "red", "demo": "dim",
        }
        text.append(self.source, style=source_colors.get(self.source, "cyan"))

        # Right side: cwd
        text.append(" │ ", style="dim")
        cwd = os.path.basename(os.getcwd())
        text.append(f"📂 {cwd}", style="dim")

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
        scrollbar-color: #30363d;
        scrollbar-color-hover: #58a6ff;
        overflow-y: scroll;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: #21262d;
        padding: 0 1;
    }

    #input-container {
        dock: bottom;
        height: auto;
        min-height: 3;
        max-height: 12;
        padding: 0 1;
        background: #161b22;
    }

    #input-hint {
        height: 1;
        background: #161b22;
        color: #484f58;
        padding: 0 2;
    }

    ChatInput {
        border: tall #30363d;
        background: #0d1117;
        height: auto;
        min-height: 1;
        max-height: 8;
    }

    ChatInput:focus {
        border: tall #58a6ff;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+k", "kb_search", "KB Search", show=True),
        Binding("ctrl+m", "multi_ai", "Multi-AI", show=True),
        Binding("ctrl+a", "agent_mode", "Agent", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+p", "show_commands", "Commands", show=True),
        Binding("ctrl+o", "attach_file", "Attach", show=True),
        Binding("ctrl+u", "scroll_up", "Scroll Up", show=False),
        Binding("ctrl+d", "scroll_down", "Scroll Down", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, bazinga_instance=None, mode: str = "chat", mesh_query=None):
        super().__init__()
        self.bazinga = bazinga_instance
        self.mode = mode
        self.is_processing = False
        self.multi_ai_mode = False
        self.agent_mode = False
        self.agent_loop = None
        self.mesh_query = mesh_query
        self.conversation_history = []
        self.last_response = ""
        self.attached_files = []  # Files attached to next message
        self._thinking_timer: Optional[Timer] = None
        self._thinking_dots = 0

        if AGENT_AVAILABLE:
            try:
                self.agent_loop = AgentLoop(verbose=False)
            except Exception:
                self.agent_loop = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        try:
            from .. import __version__
            version = __version__
        except Exception:
            version = "6.0.0"

        yield Static(
            f"[bold]BAZINGA[/bold] v{version} │ "
            "[dim]Ctrl+P: Commands │ Ctrl+K: KB │ Ctrl+M: Multi-AI │ Ctrl+A: Agent │ Ctrl+O: Attach[/dim]",
            id="header-bar"
        )

        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)

        yield StatusBar(id="status-bar")

        with Container(id="input-container"):
            yield Static(
                "Enter to send │ Shift+Enter for newline │ /help for commands",
                id="input-hint"
            )
            yield ChatInput(id="input-box")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one("#input-box", ChatInput).focus()

        log = self.query_one("#chat-log", RichLog)
        log.write(Panel(
            "[bold green]Welcome to BAZINGA![/bold green]\n\n"
            "Ask me anything. I search memory, quantum patterns, knowledge base, "
            "and LLMs to find the best answer.\n\n"
            "[bold]Quick start:[/bold]\n"
            "  Just type a question and press Enter\n"
            "  [cyan]Ctrl+A[/cyan] — Agent mode (read/edit files, run commands)\n"
            "  [cyan]Ctrl+M[/cyan] — Multi-AI consensus\n"
            "  [cyan]Ctrl+O[/cyan] — Attach a file to your message\n"
            "  [cyan]Ctrl+P[/cyan] — Show all commands\n"
            "  [cyan]Ctrl+U/D[/cyan] — Scroll up/down\n"
            "  [cyan]/help[/cyan]  — Full command reference\n\n"
            "[dim]v6.0: Evolution Engine, Constitutional Safety, Graduated Autonomy[/dim]",
            title="[yellow]φ = 1.618033988749895[/yellow]",
            border_style="green"
        ))

    def _start_thinking(self) -> None:
        """Start thinking animation in status bar."""
        self._thinking_dots = 0
        status = self.query_one("#status-bar", StatusBar)
        status.source = "thinking"

    def _stop_thinking(self) -> None:
        """Stop thinking animation."""
        pass

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user input submission."""
        if self.is_processing:
            return

        query = event.value.strip()
        if not query:
            return

        log = self.query_one("#chat-log", RichLog)

        # Show attached files if any
        if self.attached_files:
            file_list = ", ".join([os.path.basename(f) for f in self.attached_files])
            log.write(f"[dim]📎 Attached: {file_list}[/dim]")

        # Show user message
        if "\n" in query:
            log.write(f"\n[bold cyan]You:[/bold cyan]")
            for line in query.split("\n"):
                log.write(f"  {line}")
        else:
            log.write(f"\n[bold cyan]You:[/bold cyan] {query}")

        # Handle slash commands
        if query.startswith("/"):
            await self._handle_command(query, log)
            return

        # Process query
        self.is_processing = True
        self._start_thinking()

        try:
            # Attach file content to query if files are attached
            if self.attached_files:
                file_context = self._read_attached_files()
                if file_context:
                    query = f"{query}\n\n[Attached files]\n{file_context}"
                self.attached_files = []
                self._update_hint()

            response, metadata = await self._process_query(query)

            self.last_response = response
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            if metadata:
                coherence = metadata.get('coherence', 0)
                source = metadata.get('source', 'unknown')
                extras = []
                if metadata.get('mesh_peers'):
                    extras.append(f"mesh({metadata['mesh_peers']})")
                extra_str = " │ " + " │ ".join(extras) if extras else ""
                log.write(f"[dim]  φ={coherence:.3f} │ {source}{extra_str}[/dim]")

            status = self.query_one("#status-bar", StatusBar)
            status.coherence = metadata.get('coherence', 0)
            status.source = metadata.get('source', 'ready')
        except Exception as e:
            log.write(f"\n[bold red]Error:[/bold red] {str(e)}")
            status = self.query_one("#status-bar", StatusBar)
            status.source = "error"
        finally:
            self.is_processing = False
            self._stop_thinking()
            status = self.query_one("#status-bar", StatusBar)
            if self.multi_ai_mode:
                status.mode = "MULTI-AI"
            elif self.agent_mode:
                status.mode = "AGENT"
            else:
                status.mode = ""

    def _read_attached_files(self) -> str:
        """Read attached files and return their content."""
        parts = []
        for filepath in self.attached_files:
            try:
                path = Path(filepath).expanduser().resolve()
                if not path.exists():
                    parts.append(f"--- {path.name} (not found) ---")
                    continue
                if path.stat().st_size > 100_000:
                    parts.append(f"--- {path.name} (too large, {path.stat().st_size:,} bytes) ---")
                    continue
                content = path.read_text(errors='replace')
                parts.append(f"--- {path.name} ---\n{content}")
            except Exception as e:
                parts.append(f"--- {filepath} (error: {e}) ---")
        return "\n\n".join(parts)

    def _update_hint(self) -> None:
        """Update input hint based on state."""
        hint = self.query_one("#input-hint", Static)
        if self.attached_files:
            names = ", ".join([os.path.basename(f) for f in self.attached_files])
            hint.update(f"📎 {names} │ Type your question about these files")
        elif self.agent_mode:
            hint.update("🤖 Agent mode — I can read/edit files and run commands")
        elif self.multi_ai_mode:
            hint.update("🌐 Multi-AI mode — Multiple AIs will reach consensus")
        else:
            hint.update("Enter to send │ Shift+Enter for newline │ /help for commands")

    async def _handle_command(self, cmd: str, log: RichLog) -> None:
        """Handle slash commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("/help", "/h", "/?"):
            table = Table(title="BAZINGA Commands", border_style="blue", show_header=True)
            table.add_column("Command", style="green", width=25)
            table.add_column("Shortcut", style="cyan", width=12)
            table.add_column("Description", style="white")

            table.add_row("/help", "", "Show this help")
            table.add_row("/agent <task>", "Ctrl+A", "Toggle agent mode / run task")
            table.add_row("/kb <query>", "Ctrl+K", "Search knowledge base")
            table.add_row("/multi <query>", "Ctrl+M", "Multi-AI consensus query")
            table.add_row("/attach <path>", "Ctrl+O", "Attach a file to next message")
            table.add_row("/detach", "", "Remove attached files")
            table.add_row("/chain", "", "Show blockchain status")
            table.add_row("/constitution", "", "Show safety bounds")
            table.add_row("/evolution", "", "Show evolution status")
            table.add_row("/copy", "", "Copy last response to clipboard")
            table.add_row("/copyall", "", "Copy entire chat to clipboard")
            table.add_row("/stats", "", "Show session statistics")
            table.add_row("/clear", "Ctrl+L", "Clear chat history")
            table.add_row("/commands", "Ctrl+P", "Show command palette")
            table.add_row("/quit", "Ctrl+C", "Exit BAZINGA")

            log.write(table)

        elif command == "/copy":
            if self.last_response:
                try:
                    import subprocess, re
                    clean = re.sub(r'\[/?[^\]]*\]', '', self.last_response)
                    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                    process.communicate(clean.encode('utf-8'))
                    log.write("[green]✓ Copied to clipboard.[/green]")
                except FileNotFoundError:
                    try:
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                        process.communicate(clean.encode('utf-8'))
                        log.write("[green]✓ Copied to clipboard.[/green]")
                    except Exception:
                        log.write(f"[yellow]Clipboard not available. Use terminal selection.[/yellow]")
                except Exception as e:
                    log.write(f"[yellow]Copy failed: {e}[/yellow]")
            else:
                log.write("[dim]No response to copy yet.[/dim]")

        elif command == "/copyall":
            if self.conversation_history:
                try:
                    import subprocess
                    full_text = ""
                    for turn in self.conversation_history:
                        full_text += f"You: {turn['user']}\n\nBAZINGA: {turn['assistant']}\n\n---\n\n"
                    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                    process.communicate(full_text.encode('utf-8'))
                    log.write(f"[green]✓ Copied entire chat ({len(self.conversation_history)} turns) to clipboard.[/green]")
                except FileNotFoundError:
                    try:
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                        process.communicate(full_text.encode('utf-8'))
                        log.write(f"[green]✓ Copied entire chat to clipboard.[/green]")
                    except Exception:
                        log.write("[yellow]Clipboard not available.[/yellow]")
                except Exception as e:
                    log.write(f"[yellow]Copy failed: {e}[/yellow]")
            else:
                log.write("[dim]No conversation to copy.[/dim]")

        elif command == "/clear":
            log.clear()
            self.conversation_history = []
            log.write("[dim]✓ Chat cleared.[/dim]")

        elif command in ("/quit", "/exit", "/q"):
            self.exit()

        elif command == "/attach":
            if args:
                path = Path(args).expanduser().resolve()
                if path.exists():
                    self.attached_files.append(str(path))
                    size = path.stat().st_size
                    log.write(f"[green]📎 Attached: {path.name} ({size:,} bytes)[/green]")
                    self._update_hint()
                else:
                    log.write(f"[red]File not found: {args}[/red]")
            else:
                log.write("[yellow]Usage: /attach <file_path>[/yellow]")

        elif command == "/detach":
            self.attached_files = []
            self._update_hint()
            log.write("[dim]✓ Attachments cleared.[/dim]")

        elif command == "/kb" and args:
            status = self.query_one("#status-bar", StatusBar)
            status.source = "searching KB..."
            response, metadata = await self._process_query(f"[KB SEARCH] {args}")
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            status.source = "ready"

        elif command == "/multi" and args:
            self.multi_ai_mode = True
            status = self.query_one("#status-bar", StatusBar)
            status.mode = "MULTI-AI"
            status.source = "consensus..."
            response, metadata = await self._process_query(args)
            log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
            self.multi_ai_mode = False
            status.mode = ""
            status.source = "ready"

        elif command == "/agent":
            if not args:
                self.action_agent_mode()
            else:
                if not AGENT_AVAILABLE or not self.agent_loop:
                    log.write("[yellow]Agent module not available.[/yellow]")
                    return
                self.agent_mode = True
                status = self.query_one("#status-bar", StatusBar)
                status.mode = "AGENT"
                status.source = "working..."
                response, metadata = await self._process_query(args)
                log.write(f"\n[bold green]BAZINGA:[/bold green] {response}")
                self.agent_mode = False
                status.mode = ""
                status.source = "ready"

        elif command == "/stats":
            stats_table = Table(title="Session Stats", border_style="blue")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("Agent available", str(AGENT_AVAILABLE))
            stats_table.add_row("Agent mode", "ON" if self.agent_mode else "OFF")
            stats_table.add_row("Multi-AI mode", "ON" if self.multi_ai_mode else "OFF")
            stats_table.add_row("BAZINGA connected", "YES" if self.bazinga else "NO")
            stats_table.add_row("Conversation turns", str(len(self.conversation_history)))
            stats_table.add_row("Attached files", str(len(self.attached_files)))
            if self.bazinga:
                try:
                    bz_stats = self.bazinga.get_stats() if hasattr(self.bazinga, 'get_stats') else {}
                    if bz_stats:
                        stats_table.add_row("Total queries", str(bz_stats.get('queries', 0)))
                        stats_table.add_row("Cache hits", str(bz_stats.get('cache_hits', 0)))
                except Exception:
                    pass
            log.write(stats_table)

        elif command == "/chain":
            if self.bazinga and hasattr(self.bazinga, 'chain'):
                try:
                    chain = self.bazinga.chain
                    chain_table = Table(title="Blockchain", border_style="green")
                    chain_table.add_column("Metric", style="cyan")
                    chain_table.add_column("Value", style="white")
                    chain_table.add_row("Chain length", str(len(chain.chain) if hasattr(chain, 'chain') else 'N/A'))
                    chain_table.add_row("Valid", str(chain.is_valid() if hasattr(chain, 'is_valid') else 'N/A'))
                    log.write(chain_table)
                except Exception as e:
                    log.write(f"[yellow]Chain error: {e}[/yellow]")
            else:
                log.write("[dim]Blockchain not initialized.[/dim]")

        elif command == "/constitution":
            try:
                from ..evolution.constitution import ConstitutionEnforcer
                enforcer = ConstitutionEnforcer()
                table = Table(title="Constitutional Bounds (IMMUTABLE)", border_style="red")
                table.add_column("Bound", style="bold red", width=30)
                table.add_column("Description", style="white")
                for b in enforcer.list_bounds():
                    table.add_row(b['name'], b['description'])
                log.write(table)
                log.write("[dim red]These bounds cannot be modified by any proposal.[/dim red]")
            except Exception as e:
                log.write(f"[yellow]Error loading constitution: {e}[/yellow]")

        elif command == "/evolution":
            try:
                from ..evolution.engine import EvolutionEngine
                engine = EvolutionEngine()
                stats = engine.get_stats()
                auto = stats['autonomy_status']
                table = Table(title="Evolution Status", border_style="magenta")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("Autonomy Level", f"{auto['level_name']} (Level {auto['current_level']})")
                table.add_row("Successful Proposals", str(auto['successful_proposals']))
                table.add_row("Success Rate", f"{auto['success_rate']:.0%}")
                table.add_row("Reverts", str(auto['reverts']))
                table.add_row("Node Age", f"{auto['age_days']:.1f} days")
                log.write(table)
            except Exception as e:
                log.write(f"[yellow]Error loading evolution status: {e}[/yellow]")

        elif command == "/commands":
            self.action_show_commands()

        else:
            log.write(f"[yellow]Unknown command: {command}. Type /help for commands.[/yellow]")

    async def _process_query(self, query: str) -> tuple:
        """Process user query through BAZINGA."""
        if self.bazinga is None:
            await asyncio.sleep(0.5)
            return (
                f"[Demo Mode] You asked: {query}\n\n"
                "BAZINGA instance not connected. Run with actual instance for real responses.",
                {"coherence": 0.618, "source": "demo"}
            )

        try:
            # Agent mode — use ReAct loop with tools
            if self.agent_mode and self.agent_loop:
                return await self._process_agent_query(query)

            # Build context from conversation history
            # Truncate assistant responses to prevent context bloat (same fix as CLI chat)
            full_query = query
            if self.conversation_history:
                context = "Previous conversation:\n"
                for turn in self.conversation_history[-4:]:
                    assistant_text = turn['assistant']
                    if len(assistant_text) > 300:
                        assistant_text = assistant_text[:300] + "..."
                    context += f"User: {turn['user']}\nAssistant: {assistant_text}\n"
                context += "\nCurrent question: "
                full_query = context + query

            # Multi-AI mode
            if self.multi_ai_mode:
                result = await self.bazinga.multi_ai_ask(full_query)
            else:
                result = await self.bazinga.ask(full_query, fresh=True)

            # Extract response
            if isinstance(result, dict):
                response_text = result.get('response', str(result))
                metadata = {
                    'coherence': result.get('coherence', 0),
                    'source': result.get('source', 'unknown')
                }
            else:
                response_text = str(result)
                metadata = {'coherence': 0.618, 'source': 'llm'}

            # Mesh query: fan out to peers
            if self.mesh_query and not self.agent_mode:
                try:
                    mesh_context = ""
                    if self.conversation_history:
                        ctx_lines = []
                        for turn in self.conversation_history[-4:]:
                            ctx_lines.append(f"User: {turn['user']}")
                            assistant_short = turn['assistant'][:300]
                            ctx_lines.append(f"Assistant: {assistant_short}")
                        mesh_context = "\n".join(ctx_lines)

                    mesh_result = await self.mesh_query.query_mesh(
                        question=query,
                        local_answer=response_text,
                        local_source=metadata.get('source', 'llm'),
                        context=mesh_context,
                    )
                    if mesh_result.has_peers:
                        response_text = mesh_result.merged_answer
                        metadata['mesh_peers'] = mesh_result.peer_count
                        metadata['mesh_coherence'] = mesh_result.coherence
                        metadata['source'] = f"{metadata.get('source', 'llm')}+mesh({mesh_result.peer_count})"
                except Exception:
                    pass

            # Store in history
            self.conversation_history.append({
                'user': query,
                'assistant': response_text
            })

            return (response_text, metadata)
        except Exception as e:
            return (f"Error: {str(e)}", {'coherence': 0, 'source': 'error'})

    async def _process_agent_query(self, query: str) -> tuple:
        """Process query using agent loop (ReAct pattern)."""
        log = self.query_one("#chat-log", RichLog)
        status = self.query_one("#status-bar", StatusBar)

        cwd = os.getcwd()
        context = f"Current working directory: {cwd}\nUser's home: {os.path.expanduser('~')}"

        try:
            final_answer = ""
            step_count = 0

            async for step in self.agent_loop.run(query, context=context):
                step_count += 1

                if step.thought:
                    log.write(f"[dim]💭 {step.thought}[/dim]")

                if step.tool:
                    status.source = f"using {step.tool}..."
                    log.write(f"[cyan]🔧 {step.tool}[/cyan]")
                    if step.tool_args:
                        args_str = str(step.tool_args)[:100]
                        log.write(f"[dim]   {args_str}[/dim]")

                if step.observation:
                    obs_preview = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
                    log.write(f"[dim]👁 {obs_preview}[/dim]")

                if step.is_final:
                    final_answer = step.final_answer
                    break

            return (final_answer or "Agent completed but no answer provided.",
                    {'coherence': 0.85, 'source': 'agent'})

        except Exception as e:
            return (f"Agent error: {str(e)}", {'coherence': 0, 'source': 'error'})

    def action_quit(self) -> None:
        self.exit()

    def action_kb_search(self) -> None:
        input_box = self.query_one("#input-box", ChatInput)
        current = input_box.text
        if not current.startswith("/kb "):
            input_box.clear()
            input_box.insert("/kb " + current)
        input_box.focus()

    def action_multi_ai(self) -> None:
        self.multi_ai_mode = not self.multi_ai_mode
        self.agent_mode = False
        status = self.query_one("#status-bar", StatusBar)
        log = self.query_one("#chat-log", RichLog)
        if self.multi_ai_mode:
            status.mode = "MULTI-AI"
            log.write("[green]✓ Multi-AI mode ON — queries go to multiple AIs for consensus[/green]")
        else:
            status.mode = ""
            log.write("[dim]Multi-AI mode OFF[/dim]")
        self._update_hint()

    def action_agent_mode(self) -> None:
        if not AGENT_AVAILABLE:
            log = self.query_one("#chat-log", RichLog)
            log.write("[yellow]Agent module not available.[/yellow]")
            return

        self.agent_mode = not self.agent_mode
        self.multi_ai_mode = False
        status = self.query_one("#status-bar", StatusBar)
        log = self.query_one("#chat-log", RichLog)

        if self.agent_mode:
            status.mode = "AGENT"
            cwd = os.getcwd()
            log.write(Panel(
                "[bold cyan]Agent Mode ON[/bold cyan]\n\n"
                f"[dim]Working directory: {cwd}[/dim]\n\n"
                "I can now:\n"
                "  [green]•[/green] Read and edit files\n"
                "  [green]•[/green] Run bash commands\n"
                "  [green]•[/green] Search your codebase\n"
                "  [green]•[/green] Fix bugs in your code\n\n"
                "[dim]Try: 'Read config.py and explain it' or 'Fix the bug in main.py'[/dim]",
                border_style="cyan"
            ))
        else:
            status.mode = ""
            log.write("[dim]Agent mode OFF[/dim]")
        self._update_hint()

    def action_clear_chat(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        self.conversation_history = []
        log.write("[dim]✓ Chat cleared.[/dim]")

    def action_cancel(self) -> None:
        if self.is_processing:
            self.is_processing = False
            status = self.query_one("#status-bar", StatusBar)
            status.source = "cancelled"

    def action_show_commands(self) -> None:
        """Show command palette."""
        log = self.query_one("#chat-log", RichLog)
        table = Table(title="Command Palette", border_style="magenta")
        table.add_column("Shortcut", style="bold cyan", width=12)
        table.add_column("Action", style="white")
        table.add_row("Ctrl+A", "Toggle agent mode (read/edit files)")
        table.add_row("Ctrl+M", "Toggle multi-AI consensus")
        table.add_row("Ctrl+K", "Prefix input with /kb for KB search")
        table.add_row("Ctrl+O", "Attach a file")
        table.add_row("Ctrl+L", "Clear chat")
        table.add_row("Ctrl+P", "Show this palette")
        table.add_row("Ctrl+U", "Scroll chat up")
        table.add_row("Ctrl+D", "Scroll chat down")
        table.add_row("Ctrl+C", "Quit")
        table.add_row("Enter", "Send message")
        table.add_row("Shift+Enter", "New line")
        table.add_row("Escape", "Cancel current operation")
        log.write(table)

    def action_scroll_up(self) -> None:
        """Scroll chat log up (half page)."""
        log = self.query_one("#chat-log", RichLog)
        log.scroll_up(animate=False)
        log.scroll_up(animate=False)
        log.scroll_up(animate=False)
        log.scroll_up(animate=False)
        log.scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll chat log down (half page)."""
        log = self.query_one("#chat-log", RichLog)
        log.scroll_down(animate=False)
        log.scroll_down(animate=False)
        log.scroll_down(animate=False)
        log.scroll_down(animate=False)
        log.scroll_down(animate=False)

    def action_attach_file(self) -> None:
        """Prompt user to type /attach command."""
        input_box = self.query_one("#input-box", ChatInput)
        input_box.clear()
        input_box.insert("/attach ")
        input_box.focus()


def run_tui(bazinga_instance=None, mode: str = "chat"):
    """Run the BAZINGA TUI application (synchronous)."""
    app = BazingaApp(bazinga_instance=bazinga_instance, mode=mode)
    app.run()


async def run_tui_async(bazinga_instance=None, mode: str = "chat", mesh_query=None):
    """Run the BAZINGA TUI application (async)."""
    app = BazingaApp(bazinga_instance=bazinga_instance, mode=mode, mesh_query=mesh_query)
    await app.run_async()


if __name__ == "__main__":
    run_tui()
