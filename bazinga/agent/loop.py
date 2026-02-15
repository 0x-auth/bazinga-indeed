"""
BAZINGA Agent Loop - The ReAct pattern implementation.

ReAct = Reasoning + Acting
1. Think: What should I do?
2. Act: Use a tool
3. Observe: See the result
4. Repeat until done
"""

import re
import json
import os
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass, field

from .tools import TOOLS, execute_tool, get_tools_prompt


@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    thought: str = ""
    tool: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    final_answer: str = ""


class AgentLoop:
    """
    The ReAct agent loop.

    Uses LLM to reason about tasks, execute tools, and iterate
    until the task is complete.
    """

    MAX_ITERATIONS = 10

    SYSTEM_PROMPT = '''You are BAZINGA, a helpful AI assistant that can use tools to help users.
You run locally on the user's machine. You respect their privacy and work offline when possible.

{tools}

## CRITICAL: Response Format

You MUST use EXACTLY one of these two formats:

### Format A - When you need to use a tool:
```
THOUGHT: [your reasoning]
ACTION: [tool_name]
ARGS: {{"arg": "value"}}
```
Then STOP and wait for the OBSERVATION.

### Format B - When you have the final answer:
```
THOUGHT: [your reasoning]
ANSWER: [your complete response to the user]
```

## IMPORTANT RULES:
1. NEVER combine ACTION and ANSWER in the same response
2. After ACTION, you will receive an OBSERVATION - WAIT for it
3. Only use ANSWER when you have ALL the information needed
4. If you need to read a file, use the read tool FIRST, then wait for content
5. Do NOT make up file contents - always read them first
6. For simple questions that don't need tools (like math, general knowledge), just use ANSWER directly
7. Only use tools when you need to interact with the file system or run commands

## Example interaction:

User: What's in README.md?

Response 1:
THOUGHT: I need to read README.md to see its contents.
ACTION: read
ARGS: {{"path": "README.md"}}

[System provides OBSERVATION with file contents]

Response 2:
THOUGHT: I now have the file contents from the observation. Let me summarize.
ANSWER: The README.md contains... [actual summary based on observation]
'''

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.history: List[AgentStep] = []
        self._llm = None

    def _get_llm(self):
        """Get the LLM backend (lazy load)."""
        if self._llm is None:
            # Try to get local LLM first (Ollama), then cloud
            try:
                from ..local_llm import LocalLLM
                if LocalLLM.is_available():
                    self._llm = ("local", LocalLLM())
                    return self._llm
            except:
                pass

            # Try Groq
            groq_key = os.environ.get('GROQ_API_KEY')
            if groq_key:
                self._llm = ("groq", groq_key)
                return self._llm

            # Try Gemini
            gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if gemini_key:
                self._llm = ("gemini", gemini_key)
                return self._llm

            self._llm = ("none", None)

        return self._llm

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with messages. Tries local -> Groq -> Gemini."""
        import httpx

        # Try Ollama first (local, φ bonus)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3",
                        "messages": messages,
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    result = response.json().get("message", {}).get("content", "")
                    if result:
                        return result
        except Exception as e:
            if self.verbose:
                print(f"[Ollama unavailable: {e}]")

        # Try Groq (fast, free)
        groq_key = os.environ.get('GROQ_API_KEY')
        if groq_key:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {groq_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 2000
                        }
                    )
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if self.verbose:
                    print(f"[Groq error: {e}]")

        # Fallback to Gemini
            gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if gemini_key:
                try:
                    # Convert messages to Gemini format
                    contents = []
                    for msg in messages:
                        role = "user" if msg["role"] == "user" else "model"
                        if msg["role"] == "system":
                            # Prepend system to first user message
                            continue
                        contents.append({
                            "role": role,
                            "parts": [{"text": msg["content"]}]
                        })

                    # Add system instruction
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                            json={
                                "contents": contents,
                                "systemInstruction": {"parts": [{"text": system_msg}]} if system_msg else None,
                                "generationConfig": {
                                    "temperature": 0.7,
                                    "maxOutputTokens": 2000
                                }
                            }
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    if self.verbose:
                        print(f"[Gemini error: {e}]")

        return "ERROR: No LLM available. Please set GROQ_API_KEY or install Ollama."

    def _parse_response(self, response: str) -> AgentStep:
        """Parse LLM response into an AgentStep."""
        step = AgentStep()

        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|ANSWER:|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for final ANSWER
        answer_match = re.search(r'ANSWER:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            step.is_final = True
            step.final_answer = answer_match.group(1).strip()
            return step

        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            step.tool = action_match.group(1).lower()

        # Extract ARGS
        args_match = re.search(r'ARGS:\s*(\{.+?\})', response, re.DOTALL | re.IGNORECASE)
        if args_match:
            try:
                step.tool_args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                # Try to fix common issues
                args_str = args_match.group(1)
                # Replace single quotes with double quotes
                args_str = args_str.replace("'", '"')
                try:
                    step.tool_args = json.loads(args_str)
                except:
                    step.tool_args = {}

        return step

    async def run(self, user_input: str, context: str = "") -> Generator[AgentStep, None, None]:
        """
        Run the agent loop on user input.

        Yields AgentStep objects as the agent works.
        """
        # Build initial messages
        system_prompt = self.SYSTEM_PROMPT.format(tools=get_tools_prompt())

        if context:
            system_prompt += f"\n\n## Relevant context from indexed knowledge:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            # Call LLM
            response = await self._call_llm(messages)

            if self.verbose:
                print(f"\n[LLM Response {iteration}]:\n{response}\n")

            # Parse response
            step = self._parse_response(response)
            self.history.append(step)

            # Check if done
            if step.is_final:
                yield step
                return

            # Execute tool if specified
            if step.tool:
                if self.verbose:
                    print(f"[Executing: {step.tool}({step.tool_args})]")

                result = execute_tool(step.tool, **step.tool_args)
                step.observation = json.dumps(result, indent=2, default=str)

                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION:\n{step.observation}"
                })

                yield step
            else:
                # No tool, might be confused - add clarification
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please use the THOUGHT/ACTION/ARGS or THOUGHT/ANSWER format."
                })
                yield step

        # Max iterations reached
        final_step = AgentStep(
            thought="Max iterations reached",
            is_final=True,
            final_answer="I've reached my iteration limit. Here's what I found so far based on my work above."
        )
        yield final_step

    async def run_sync(self, user_input: str, context: str = "") -> str:
        """Run the agent and return final answer."""
        final_answer = ""
        async for step in self.run(user_input, context):
            if step.is_final:
                final_answer = step.final_answer
        return final_answer
