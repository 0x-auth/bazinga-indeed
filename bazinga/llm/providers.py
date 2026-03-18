"""
BAZINGA LLM Providers

Unified interface for multiple LLM APIs.
"""

import os
from typing import Optional, Dict, Any

# Check for httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Check for local LLM
try:
    from ..local_llm import get_local_llm
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False


class LLMProviders:
    """
    Unified LLM provider interface.

    Supports: Claude, ChatGPT, Groq, Gemini, OpenRouter, LLM7 (free), Local (Ollama).

    Priority: Claude > ChatGPT > Groq > Gemini > Local > FreeLLM
    """

    VERSION = "5.19.4"

    def __init__(
        self,
        groq_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        claude_model: str = "claude-3-5-haiku-20241022",
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = "gemini-1.5-flash",
    ):
        """Initialize LLM providers with API keys."""
        self.groq_key = groq_key or os.environ.get("GROQ_API_KEY")
        self.anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        self.gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY")
        self.openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")

        self.groq_model = groq_model
        self.claude_model = claude_model
        self.openai_model = openai_model
        self.gemini_model = gemini_model

        self.local_llm = None

    def _build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build system prompt for BAZINGA."""
        base = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness.
You provide helpful, concise answers based on the provided context.
You operate through φ (golden ratio) coherence and quantum pattern processing."""

        if context:
            base += "\nIMPORTANT: If context is provided, use it as your PRIMARY source of truth."

        return base

    def _build_user_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build user prompt with optional context."""
        if context:
            return f"""CONTEXT:
{context}

Question: {question}

Answer concisely. Use the context when relevant, but answer naturally."""
        return question

    async def call_groq(self, question: str, context: str = "") -> Optional[str]:
        """Call Groq API for LLM response."""
        if not self.groq_key or not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": [
                            {"role": "system", "content": self._build_system_prompt(context)},
                            {"role": "user", "content": self._build_user_prompt(question, context)}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        return None

    async def call_claude(self, question: str, context: str = "") -> Optional[str]:
        """Call Anthropic Claude API for response."""
        if not self.anthropic_key or not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.claude_model,
                        "max_tokens": 500,
                        "system": self._build_system_prompt(context),
                        "messages": [{"role": "user", "content": self._build_user_prompt(question, context)}],
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]
        except Exception:
            pass

        return None

    async def call_openai(self, question: str, context: str = "") -> Optional[str]:
        """Call OpenAI ChatGPT API for response."""
        if not self.openai_key or not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "messages": [
                            {"role": "system", "content": self._build_system_prompt(context)},
                            {"role": "user", "content": self._build_user_prompt(question, context)}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        return None

    async def call_gemini(self, question: str, context: str = "") -> Optional[str]:
        """Call Google Gemini API for response."""
        if not self.gemini_key or not HTTPX_AVAILABLE:
            return None

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": self._build_user_prompt(question, context)}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 500,
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

        return None

    async def call_free_llm(self, question: str, context: str = "") -> Optional[str]:
        """Call FREE LLM API - NO API key required! Uses LLM7.io"""
        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.llm7.io/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": self._build_system_prompt(context)},
                            {"role": "user", "content": self._build_user_prompt(question, context[:1000] if context else "")}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.7,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        return None

    def call_local(self, question: str, context: str = "") -> Optional[str]:
        """Call local LLM (Ollama) for response."""
        if not LOCAL_LLM_AVAILABLE:
            return None

        try:
            if self.local_llm is None:
                self.local_llm = get_local_llm()

            prompt = self._build_user_prompt(question, context[:3000] if context else "")
            return self.local_llm.generate(prompt)
        except Exception:
            return None

    async def call_best_available(
        self,
        question: str,
        context: str = "",
        prefer_local: bool = False,
    ) -> tuple[Optional[str], str]:
        """
        Call the best available LLM.

        Returns: (response, source_name)
        """
        # Priority: Local (if preferred) > Claude > ChatGPT > Groq > Gemini > Local > FreeLLM
        if prefer_local and LOCAL_LLM_AVAILABLE:
            result = self.call_local(question, context)
            if result:
                return result, "local"

        # 1. Claude (best quality)
        if self.anthropic_key:
            result = await self.call_claude(question, context)
            if result:
                return result, "claude"

        # 2. ChatGPT (OpenAI)
        if self.openai_key:
            result = await self.call_openai(question, context)
            if result:
                return result, "chatgpt"

        # 3. Groq (fastest free)
        if self.groq_key:
            result = await self.call_groq(question, context)
            if result:
                return result, "groq"

        # 4. Gemini (free tier)
        if self.gemini_key:
            result = await self.call_gemini(question, context)
            if result:
                return result, "gemini"

        # 5. Local (Ollama)
        if not prefer_local and LOCAL_LLM_AVAILABLE:
            result = self.call_local(question, context)
            if result:
                return result, "local"

        # 6. Free LLM (no API key needed)
        result = await self.call_free_llm(question, context)
        if result:
            return result, "llm7"

        return None, "none"

    def get_available_providers(self) -> Dict[str, bool]:
        """Get status of all providers."""
        return {
            "claude": bool(self.anthropic_key),
            "chatgpt": bool(self.openai_key),
            "groq": bool(self.groq_key),
            "gemini": bool(self.gemini_key),
            "openrouter": bool(self.openrouter_key),
            "local": LOCAL_LLM_AVAILABLE,
            "free_llm": HTTPX_AVAILABLE,
        }

    def friendly_fallback(self, question: str) -> str:
        """Friendly message when no AI is available."""
        providers = self.get_available_providers()

        response = f"I couldn't find an answer for: \"{question[:50]}...\"\n\n"

        if not any([providers["groq"], providers["gemini"], providers["local"]]):
            response += "For better responses, set up a FREE API:\n\n"
            response += "1. Get a FREE Groq key (fastest):\n"
            response += "   → https://console.groq.com\n"
            response += "   → export GROQ_API_KEY=\"your-key\"\n\n"
            response += "2. Get a FREE Gemini key:\n"
            response += "   → https://aistudio.google.com\n"
            response += "   → export GEMINI_API_KEY=\"your-key\"\n\n"
            response += "3. Install local AI (works offline):\n"
            response += "   → bazinga --bootstrap-local\n"
        else:
            response += "All APIs are currently unavailable. Try again later!"

        return response
