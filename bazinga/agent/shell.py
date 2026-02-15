"""
BAZINGA Agent Shell - Interactive REPL for the agent.

"The first AI you actually own"

Features:
- Session memory (remembers conversation)
- Streaming output
- Tool execution feedback
- RAG context injection
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from .loop import AgentLoop
from .tools import SearchTool


class SessionMemory:
    """
    Remembers conversation history within a session.

    This gives the agent context about what you've been doing.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict] = []
        self.session_start = datetime.now()
        self.files_read: List[str] = []
        self.commands_run: List[str] = []

    def add_interaction(self, user_input: str, agent_response: str, tools_used: List[str] = None):
        """Add an interaction to history."""
        self.history.append({
            "user": user_input,
            "agent": agent_response,
            "tools": tools_used or [],
            "time": datetime.now().isoformat()
        })

        # Track files read
        if tools_used:
            for tool in tools_used:
                if tool.startswith("read:"):
                    filepath = tool.split(":", 1)[1]
                    if filepath not in self.files_read:
                        self.files_read.append(filepath)
                elif tool.startswith("bash:"):
                    cmd = tool.split(":", 1)[1]
                    self.commands_run.append(cmd)

        # Trim if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_summary(self) -> str:
        """Get a summary of session context for the agent."""
        if not self.history:
            return ""

        lines = ["## Session Context (what we've done so far):"]

        # Recent interactions
        recent = self.history[-5:]  # Last 5
        for h in recent:
            lines.append(f"- User asked: {h['user'][:100]}...")
            if h['tools']:
                lines.append(f"  Tools used: {', '.join(h['tools'][:3])}")

        # Files we've looked at
        if self.files_read:
            lines.append(f"\nFiles examined: {', '.join(self.files_read[-5:])}")

        return "\n".join(lines)

    def clear(self):
        """Clear session memory."""
        self.history = []
        self.files_read = []
        self.commands_run = []


class AgentShell:
    """
    Interactive shell for the BAZINGA agent.

    Features:
    - REPL interface
    - Session memory
    - Tool execution feedback
    - RAG context injection
    """

    BANNER = '''
╔══════════════════════════════════════════════════════════════╗
║                    BAZINGA AGENT v0.2                        ║
║            "The first AI you actually own"                   ║
╚══════════════════════════════════════════════════════════════╝

  Commands:
    /help     - Show this help
    /tools    - List available tools
    /memory   - Show session memory
    /clear    - Clear screen
    /reset    - Reset session memory
    /verbose  - Toggle verbose mode
    /exit     - Exit agent

  Just type naturally. The agent remembers your session context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'''

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.agent = AgentLoop(verbose=verbose)
        self.search_tool = SearchTool()
        self.memory = SessionMemory()
        self.running = True

    def print_banner(self):
        """Print the welcome banner."""
        print(self.BANNER)

        # Show LLM status
        llm_type, _ = self.agent._get_llm()
        if llm_type == "local":
            print("  LLM: Ollama (local) - φ trust bonus active!")
        elif llm_type == "groq":
            print("  LLM: Groq (cloud, free tier)")
        elif llm_type == "gemini":
            print("  LLM: Gemini (cloud, free tier)")
        else:
            print("  LLM: None configured!")
            print("       Run: bazinga --check")

        print(f"  CWD: {os.getcwd()}")
        print()

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = cmd.strip().lower()

        if cmd == "/help":
            print(self.BANNER)
            return True

        if cmd == "/tools":
            from .tools import TOOLS
            print("\nAvailable tools:")
            for name, tool in TOOLS.items():
                print(f"  {name}: {tool.description}")
            print()
            return True

        if cmd == "/memory":
            print("\n📝 Session Memory:")
            print(f"   Interactions: {len(self.memory.history)}")
            print(f"   Files read: {len(self.memory.files_read)}")
            if self.memory.files_read:
                for f in self.memory.files_read[-5:]:
                    print(f"     - {f}")
            print(f"   Commands run: {len(self.memory.commands_run)}")
            if self.memory.history:
                print(f"\n   Last interaction:")
                last = self.memory.history[-1]
                print(f"     You: {last['user'][:60]}...")
                print(f"     Agent: {last['agent'][:60]}...")
            print()
            return True

        if cmd == "/reset":
            self.memory.clear()
            print("Session memory cleared.")
            return True

        if cmd == "/clear":
            os.system('clear' if os.name != 'nt' else 'cls')
            return True

        if cmd == "/verbose":
            self.verbose = not self.verbose
            self.agent.verbose = self.verbose
            print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
            return True

        if cmd in ("/exit", "/quit", "/q"):
            self.running = False
            print("Goodbye!")
            return True

        return False

    def get_context(self, user_input: str) -> str:
        """Get relevant context from indexed knowledge + session memory."""
        parts = []

        # Add session memory context
        session_context = self.memory.get_context_summary()
        if session_context:
            parts.append(session_context)

        # Add RAG context
        try:
            result = self.search_tool.execute(query=user_input, limit=3)
            if result.get("success") and result.get("results"):
                rag_parts = []
                for r in result["results"]:
                    if isinstance(r, dict):
                        text = r.get("text", r.get("content", str(r)))
                    else:
                        text = str(r)
                    rag_parts.append(text[:500])
                if rag_parts:
                    parts.append("## Relevant indexed knowledge:\n" + "\n---\n".join(rag_parts))
        except:
            pass

        return "\n\n".join(parts)

    async def process_input(self, user_input: str):
        """Process user input and display agent's work."""
        # Get combined context
        context = self.get_context(user_input)

        print()  # Blank line before response

        # Track tools used
        tools_used = []
        final_answer = ""

        # Run agent
        async for step in self.agent.run(user_input, context):
            # Show thought
            if step.thought:
                print(f"💭 {step.thought}")

            # Show tool use
            if step.tool:
                print(f"🔧 {step.tool}", end="", flush=True)

                # Track for memory
                tool_record = f"{step.tool}"
                if step.tool == "read" and step.tool_args.get("path"):
                    tool_record = f"read:{step.tool_args['path']}"
                elif step.tool == "bash" and step.tool_args.get("command"):
                    tool_record = f"bash:{step.tool_args['command'][:50]}"
                tools_used.append(tool_record)

                if self.verbose:
                    print(f" {step.tool_args}")
                else:
                    print()  # Newline

                # Show observation (truncated)
                if step.observation:
                    obs_lines = step.observation.split('\n')
                    if len(obs_lines) > 10 and not self.verbose:
                        print(f"   ✓ Got {len(obs_lines)} lines")
                    else:
                        for line in obs_lines[:15]:
                            print(f"   {line}")
                        if len(obs_lines) > 15:
                            print(f"   ... ({len(obs_lines) - 15} more lines)")

            # Show final answer
            if step.is_final:
                final_answer = step.final_answer
                print()
                print(f"✨ {step.final_answer}")

        # Save to memory
        if final_answer:
            self.memory.add_interaction(user_input, final_answer, tools_used)

        print()

    async def run_async(self):
        """Run the interactive shell (async version)."""
        self.print_banner()

        while self.running:
            try:
                # Get input
                user_input = input("bazinga> ").strip()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue

                # Process with agent
                await self.process_input(user_input)

            except KeyboardInterrupt:
                print("\n(Use /exit to quit)")
            except EOFError:
                self.running = False
                print("\nGoodbye!")

    def run(self):
        """Run the interactive shell."""
        try:
            asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(self.run_async())
        except RuntimeError:
            asyncio.run(self.run_async())


def run_agent_shell(verbose: bool = False):
    """Entry point for the agent shell (sync wrapper)."""
    shell = AgentShell(verbose=verbose)
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(shell.run_async()))
            future.result()
    except RuntimeError:
        asyncio.run(shell.run_async())


async def run_agent_once(task: str, verbose: bool = False) -> str:
    """Run agent for a single task and return result."""
    agent = AgentLoop(verbose=verbose)
    search = SearchTool()

    # Get context
    context = ""
    try:
        result = search.execute(query=task, limit=3)
        if result.get("success") and result.get("results"):
            context_parts = []
            for r in result["results"]:
                if isinstance(r, dict):
                    text = r.get("text", r.get("content", str(r)))
                else:
                    text = str(r)
                context_parts.append(text[:500])
            context = "\n---\n".join(context_parts)
    except:
        pass

    # Run agent
    final_answer = ""
    async for step in agent.run(task, context):
        if verbose:
            if step.thought:
                print(f"💭 {step.thought}")
            if step.tool:
                print(f"🔧 {step.tool}")
        if step.is_final:
            final_answer = step.final_answer

    return final_answer
