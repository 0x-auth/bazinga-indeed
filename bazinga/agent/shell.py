"""
BAZINGA Agent Shell - Interactive REPL for the agent.

"The first AI you actually own"
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from .loop import AgentLoop
from .tools import SearchTool


class AgentShell:
    """
    Interactive shell for the BAZINGA agent.

    Features:
    - REPL interface
    - Command history
    - Tool execution feedback
    - RAG context injection
    """

    BANNER = '''
╔══════════════════════════════════════════════════════════════╗
║                    BAZINGA AGENT v0.1                        ║
║            "The first AI you actually own"                   ║
╚══════════════════════════════════════════════════════════════╝

  Commands:
    /help     - Show this help
    /tools    - List available tools
    /clear    - Clear screen
    /verbose  - Toggle verbose mode
    /exit     - Exit agent

  Just type naturally to interact with the agent.
  The agent can read files, run commands, and search your knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'''

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.agent = AgentLoop(verbose=verbose)
        self.search_tool = SearchTool()
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
        """Get relevant context from indexed knowledge."""
        try:
            result = self.search_tool.execute(query=user_input, limit=3)
            if result.get("success") and result.get("results"):
                context_parts = []
                for r in result["results"]:
                    if isinstance(r, dict):
                        text = r.get("text", r.get("content", str(r)))
                    else:
                        text = str(r)
                    context_parts.append(text[:500])
                return "\n---\n".join(context_parts)
        except:
            pass
        return ""

    async def process_input(self, user_input: str):
        """Process user input and display agent's work."""
        # Get RAG context
        context = self.get_context(user_input)

        print()  # Blank line before response

        # Run agent
        async for step in self.agent.run(user_input, context):
            # Show thought
            if step.thought:
                print(f"💭 {step.thought}")

            # Show tool use
            if step.tool:
                print(f"🔧 Using: {step.tool}")
                if self.verbose:
                    print(f"   Args: {step.tool_args}")

                # Show observation (truncated)
                if step.observation:
                    obs_lines = step.observation.split('\n')
                    if len(obs_lines) > 10 and not self.verbose:
                        print(f"   → {obs_lines[0]}")
                        print(f"   ... ({len(obs_lines)} lines)")
                    else:
                        for line in obs_lines[:20]:
                            print(f"   {line}")

            # Show final answer
            if step.is_final:
                print()
                print(f"✨ {step.final_answer}")

        print()

    async def run_async(self):
        """Run the interactive shell (async version)."""
        self.print_banner()

        while self.running:
            try:
                # Get input (sync, but that's fine for terminal)
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
            # Check if we're in an existing event loop
            loop = asyncio.get_running_loop()
            # We're in a loop, can't use asyncio.run
            # This shouldn't happen normally, but handle it
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(self.run_async())
        except RuntimeError:
            # No running loop, use asyncio.run
            asyncio.run(self.run_async())


def run_agent_shell(verbose: bool = False):
    """Entry point for the agent shell (sync wrapper)."""
    shell = AgentShell(verbose=verbose)
    # Use nest_asyncio to allow nested event loops, or run fresh
    try:
        asyncio.get_running_loop()
        # We're in an event loop already - this is tricky
        # For now, run synchronously by creating a new loop in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(shell.run_async()))
            future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
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
