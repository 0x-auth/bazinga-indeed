#!/usr/bin/env python3
"""
BAZINGA Intelligent Coder - LLM-Powered Code Generation

This is NOT template-based code generation. It uses:
- Multi-provider LLM orchestration
- φ-coherence based quality scoring
- Tensor intersection principles (from DODO)
- RAG context for codebase understanding
- Self-healing with feedback loops

"Code emerges from understanding, not templates."
"""

import os
import asyncio
import hashlib
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Import LLM orchestrator
try:
    from .llm_orchestrator import LLMOrchestrator, LLMResponse
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    try:
        from bazinga.llm_orchestrator import LLMOrchestrator, LLMResponse
        ORCHESTRATOR_AVAILABLE = True
    except ImportError:
        ORCHESTRATOR_AVAILABLE = False

# Import RAG for context
try:
    from .core_intelligence import BazingaRAG
    RAG_AVAILABLE = True
except ImportError:
    try:
        from bazinga.core_intelligence import BazingaRAG
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False


@dataclass
class CodeGenerationResult:
    """Result of intelligent code generation."""
    code: str
    language: str
    explanation: str
    coherence: float
    complexity: float
    trust_level: float
    provider: str
    tokens_used: int
    from_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorState:
    """
    Tensor intersection state (inspired by DODO).

    The insight: You don't need millions of parameters when you can use
    emergent properties from the intersection of two lower-dimensional spaces:
    - Pattern space (deterministic, learned from context)
    - Entropy space (probabilistic, from LLM)

    Their tensor product creates generation modes.
    """
    pattern_vector: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    entropy_vector: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    trust_level: float = 0.5
    coherence: float = 0.5
    complexity: float = 0.5
    generation_modes: List[Dict[str, float]] = field(default_factory=list)

    def compute_tensor_product(self) -> List[List[float]]:
        """Compute tensor product matrix."""
        tensor = []
        for p in self.pattern_vector:
            row = [p * e for e in self.entropy_vector]
            tensor.append(row)
        return tensor

    def compute_eigenvalues(self) -> List[float]:
        """Extract eigenvalue-like metrics from tensor product."""
        tensor = self.compute_tensor_product()
        eigenvalues = []
        for row in tensor:
            row_sum = sum(row)
            if row_sum > 0:
                eigenvalues.append(row_sum)
        eigenvalues.sort(reverse=True)
        return eigenvalues

    def compute_coherence(self) -> float:
        """Compute coherence from tensor eigenvalues (φ-aligned)."""
        eigenvalues = self.compute_eigenvalues()
        if not eigenvalues or sum(eigenvalues) == 0:
            return 0.5
        # Coherence = ratio of largest eigenvalue to sum
        # Ideal coherence ~ 1/φ ≈ 0.618
        raw_coherence = eigenvalues[0] / sum(eigenvalues)
        # Scale to [0,1] with φ-alignment
        phi_aligned = abs(raw_coherence - 1/PHI) / (1/PHI)
        return max(0, min(1, 1 - phi_aligned))


class IntelligentCoder:
    """
    LLM-powered intelligent code generator.

    Features:
    - Multi-provider LLM for actual code generation
    - RAG context from indexed codebase
    - Tensor intersection for emergent properties
    - φ-coherence quality scoring
    - Self-healing feedback loop
    """

    SUPPORTED_LANGUAGES = {
        'python': {'ext': '.py', 'comment': '#', 'block_comment': ('"""', '"""')},
        'javascript': {'ext': '.js', 'comment': '//', 'block_comment': ('/*', '*/')},
        'typescript': {'ext': '.ts', 'comment': '//', 'block_comment': ('/*', '*/')},
        'rust': {'ext': '.rs', 'comment': '//', 'block_comment': ('/*', '*/')},
        'go': {'ext': '.go', 'comment': '//', 'block_comment': ('/*', '*/')},
        'java': {'ext': '.java', 'comment': '//', 'block_comment': ('/*', '*/')},
        'cpp': {'ext': '.cpp', 'comment': '//', 'block_comment': ('/*', '*/')},
        'c': {'ext': '.c', 'comment': '//', 'block_comment': ('/*', '*/')},
        'ruby': {'ext': '.rb', 'comment': '#', 'block_comment': ('=begin', '=end')},
        'php': {'ext': '.php', 'comment': '//', 'block_comment': ('/*', '*/')},
    }

    LANGUAGE_ALIASES = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'rs': 'rust',
        'rb': 'ruby',
        'c++': 'cpp',
    }

    # System prompts for different code tasks
    PROMPTS = {
        'generate': """You are BAZINGA, an expert code generator.
Generate clean, idiomatic, production-ready code.

Guidelines:
- Write complete, runnable code
- Include proper imports
- Add concise but helpful comments
- Handle edge cases
- Follow {language} best practices
- Match the style of any provided context

{context_section}

Generate {language} code for: {task}

Return ONLY the code, no explanations. Start with imports if needed.""",

        'explain': """You are BAZINGA, a code explanation expert.
Explain code clearly and thoroughly.

{context_section}

Explain this {language} code:
```{language}
{code}
```

Provide:
1. What it does (brief summary)
2. How it works (step by step)
3. Key concepts used
4. Potential improvements""",

        'fix': """You are BAZINGA, a debugging expert.
Find and fix bugs in code.

{context_section}

This {language} code has a bug. The error is: {error}

```{language}
{code}
```

1. Identify the bug
2. Explain why it happens
3. Provide the fixed code

Return the COMPLETE fixed code, not just the changed lines.""",

        'improve': """You are BAZINGA, a code improvement expert.
Improve code quality without changing functionality.

{context_section}

Improve this {language} code for:
- Readability
- Performance
- Maintainability
- Best practices

```{language}
{code}
```

Return the improved code with comments explaining changes.""",

        'convert': """You are BAZINGA, a code translation expert.
Convert code between programming languages.

{context_section}

Convert this {source_lang} code to {target_lang}:

```{source_lang}
{code}
```

Produce idiomatic {target_lang} code, not literal translation.
Use {target_lang} conventions and patterns.""",

        'test': """You are BAZINGA, a test generation expert.
Write comprehensive tests for code.

{context_section}

Write {test_framework} tests for this {language} code:

```{language}
{code}
```

Include:
- Unit tests for each function/method
- Edge cases
- Error conditions
- Clear test names describing behavior""",

        'document': """You are BAZINGA, a documentation expert.
Write clear documentation for code.

{context_section}

Document this {language} code with:
- Module/file docstring
- Function/method docstrings
- Type hints (if applicable to language)
- Usage examples

```{language}
{code}
```

Return the code with documentation added.""",
    }

    def __init__(self):
        self.orchestrator = LLMOrchestrator() if ORCHESTRATOR_AVAILABLE else None
        self.rag = BazingaRAG() if RAG_AVAILABLE else None
        self.tensor_state = TensorState()
        self.generation_history: List[CodeGenerationResult] = []
        self.feedback_history: List[Dict[str, Any]] = []

        if not self.orchestrator:
            print("⚠️ LLM Orchestrator not available. Install httpx and set API keys.")
        if not self.rag:
            print("⚠️ RAG not available. Code will be generated without codebase context.")

    def normalize_language(self, lang: str) -> str:
        """Normalize language name."""
        lang = lang.lower().strip()
        return self.LANGUAGE_ALIASES.get(lang, lang)

    def _get_context_from_rag(self, query: str, limit: int = 3) -> str:
        """Get relevant context from RAG."""
        if not self.rag:
            return ""

        try:
            results = self.rag.search(query, limit=limit)
            if not results:
                return ""

            context_parts = []
            for result in results:
                content = result.get('content', '')
                source = result.get('source', 'unknown')
                context_parts.append(f"# From {source}:\n{content[:500]}")

            return "\n\n".join(context_parts)
        except Exception:
            return ""

    def _update_tensor_state(self, result: CodeGenerationResult):
        """
        Update tensor state based on generation result.
        This enables the self-improving feedback loop.
        """
        # Update pattern vector based on code characteristics
        code = result.code

        # Pattern metrics
        lines = code.split('\n')
        avg_line_length = sum(len(l) for l in lines) / max(len(lines), 1)
        normalized_line_length = min(1.0, avg_line_length / 80)  # 80 char ideal

        # Comment density
        comment_char = self.SUPPORTED_LANGUAGES.get(result.language, {}).get('comment', '#')
        comment_lines = sum(1 for l in lines if l.strip().startswith(comment_char))
        comment_ratio = comment_lines / max(len(lines), 1)

        # Function/class density (approximate)
        structure_keywords = ['def ', 'function ', 'fn ', 'func ', 'class ', 'struct ']
        structure_count = sum(1 for l in lines for kw in structure_keywords if kw in l)
        structure_ratio = min(1.0, structure_count / max(len(lines) / 10, 1))

        # Update pattern vector
        self.tensor_state.pattern_vector = [
            normalized_line_length,
            comment_ratio,
            structure_ratio,
            result.coherence,
        ]

        # Update entropy vector based on LLM response characteristics
        self.tensor_state.entropy_vector = [
            result.coherence,
            min(1.0, result.tokens_used / 2000),  # Token usage
            1.0 - (result.complexity / PHI),  # Inverse complexity
            0.5 + (0.5 if result.provider != 'none' else 0),  # Provider success
        ]

        # Recompute coherence
        self.tensor_state.coherence = self.tensor_state.compute_coherence()

        # Update trust based on feedback history
        if self.feedback_history:
            recent_feedback = self.feedback_history[-5:]
            avg_score = sum(f.get('score', 0.5) for f in recent_feedback) / len(recent_feedback)
            # φ-smoothing
            self.tensor_state.trust_level = (
                self.tensor_state.trust_level * (1/PHI) +
                avg_score * (1 - 1/PHI)
            )

    def record_feedback(self, generation_id: str, score: float, notes: str = ""):
        """
        Record feedback to improve future generations.

        Args:
            generation_id: Hash of the generation
            score: Quality score 0.0-1.0
            notes: Optional notes about what was good/bad
        """
        self.feedback_history.append({
            'generation_id': generation_id,
            'score': max(0.0, min(1.0, score)),
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
            'trust_level': self.tensor_state.trust_level,
        })

        # Adapt trust based on feedback
        if score > 0.7:
            self.tensor_state.trust_level = min(1.0, self.tensor_state.trust_level + 0.05)
        elif score < 0.3:
            self.tensor_state.trust_level = max(0.1, self.tensor_state.trust_level - 0.1)

    def _compute_complexity(self, code: str) -> float:
        """Compute code complexity estimate."""
        lines = code.split('\n')

        # Factors
        line_count = len(lines)
        indent_levels = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        max_indent = max(indent_levels) if indent_levels else 0

        # Nesting complexity
        nesting_score = min(1.0, max_indent / 20)  # 20 spaces = max nesting

        # Length complexity
        length_score = min(1.0, line_count / 200)  # 200 lines = high complexity

        # Combine with φ-weighting
        complexity = (nesting_score * PHI + length_score) / (PHI + 1)

        return complexity

    async def generate(
        self,
        task: str,
        language: str = 'python',
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> CodeGenerationResult:
        """
        Generate code using LLM with RAG context.

        Args:
            task: What to generate (e.g., "user authentication system")
            language: Target language
            context: Optional additional context
            temperature: LLM temperature
            max_tokens: Max tokens to generate

        Returns:
            CodeGenerationResult with code, explanation, and metrics
        """
        language = self.normalize_language(language)

        if not self.orchestrator:
            return self._fallback_generation(task, language)

        # Get RAG context
        rag_context = self._get_context_from_rag(f"{language} {task}")

        context_section = ""
        if rag_context:
            context_section = f"Relevant code from the codebase:\n{rag_context}\n\nMatch the style and patterns above."
        if context:
            context_section += f"\n\nAdditional context:\n{context}"

        # Build prompt
        prompt = self.PROMPTS['generate'].format(
            language=language,
            task=task,
            context_section=context_section,
        )

        # Generate with LLM
        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='coder',
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract code from response
        code = self._extract_code(response.content, language)

        # Compute metrics
        complexity = self._compute_complexity(code)

        result = CodeGenerationResult(
            code=code,
            language=language,
            explanation=self._generate_explanation(code, language),
            coherence=response.coherence,
            complexity=complexity,
            trust_level=self.tensor_state.trust_level,
            provider=response.provider,
            tokens_used=response.tokens_used,
            from_cache=response.is_cached,
            metadata={
                'task': task,
                'had_rag_context': bool(rag_context),
                'generation_id': hashlib.md5(code.encode()).hexdigest()[:12],
            }
        )

        # Update tensor state
        self._update_tensor_state(result)
        self.generation_history.append(result)

        return result

    async def explain(
        self,
        code: str,
        language: str = 'python',
    ) -> str:
        """Explain code."""
        language = self.normalize_language(language)

        if not self.orchestrator:
            return "LLM not available for code explanation."

        rag_context = self._get_context_from_rag(f"explain {language} code")
        context_section = f"Related code:\n{rag_context}" if rag_context else ""

        prompt = self.PROMPTS['explain'].format(
            language=language,
            code=code,
            context_section=context_section,
        )

        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='analyst',
            max_tokens=1500,
        )

        return response.content

    async def fix(
        self,
        code: str,
        error: str,
        language: str = 'python',
    ) -> CodeGenerationResult:
        """Fix buggy code."""
        language = self.normalize_language(language)

        if not self.orchestrator:
            return self._fallback_generation(f"fix {error}", language)

        rag_context = self._get_context_from_rag(f"fix {language} {error}")
        context_section = f"Related code:\n{rag_context}" if rag_context else ""

        prompt = self.PROMPTS['fix'].format(
            language=language,
            code=code,
            error=error,
            context_section=context_section,
        )

        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='coder',
            max_tokens=2000,
        )

        fixed_code = self._extract_code(response.content, language)

        return CodeGenerationResult(
            code=fixed_code,
            language=language,
            explanation=response.content,
            coherence=response.coherence,
            complexity=self._compute_complexity(fixed_code),
            trust_level=self.tensor_state.trust_level,
            provider=response.provider,
            tokens_used=response.tokens_used,
            metadata={'task': 'fix', 'original_error': error},
        )

    async def improve(
        self,
        code: str,
        language: str = 'python',
    ) -> CodeGenerationResult:
        """Improve code quality."""
        language = self.normalize_language(language)

        if not self.orchestrator:
            return CodeGenerationResult(
                code=code,
                language=language,
                explanation="LLM not available",
                coherence=0.5,
                complexity=0.5,
                trust_level=0.5,
                provider='none',
                tokens_used=0,
            )

        rag_context = self._get_context_from_rag(f"improve {language} code quality")
        context_section = f"Related code:\n{rag_context}" if rag_context else ""

        prompt = self.PROMPTS['improve'].format(
            language=language,
            code=code,
            context_section=context_section,
        )

        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='coder',
            max_tokens=2500,
        )

        improved_code = self._extract_code(response.content, language)

        return CodeGenerationResult(
            code=improved_code,
            language=language,
            explanation=response.content,
            coherence=response.coherence,
            complexity=self._compute_complexity(improved_code),
            trust_level=self.tensor_state.trust_level,
            provider=response.provider,
            tokens_used=response.tokens_used,
            metadata={'task': 'improve'},
        )

    async def convert(
        self,
        code: str,
        source_lang: str,
        target_lang: str,
    ) -> CodeGenerationResult:
        """Convert code between languages."""
        source_lang = self.normalize_language(source_lang)
        target_lang = self.normalize_language(target_lang)

        if not self.orchestrator:
            return self._fallback_generation(f"convert to {target_lang}", target_lang)

        rag_context = self._get_context_from_rag(f"{target_lang} code patterns")
        context_section = f"Target language patterns:\n{rag_context}" if rag_context else ""

        prompt = self.PROMPTS['convert'].format(
            source_lang=source_lang,
            target_lang=target_lang,
            code=code,
            context_section=context_section,
        )

        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='coder',
            max_tokens=2500,
        )

        converted_code = self._extract_code(response.content, target_lang)

        return CodeGenerationResult(
            code=converted_code,
            language=target_lang,
            explanation=f"Converted from {source_lang}",
            coherence=response.coherence,
            complexity=self._compute_complexity(converted_code),
            trust_level=self.tensor_state.trust_level,
            provider=response.provider,
            tokens_used=response.tokens_used,
            metadata={'task': 'convert', 'source_lang': source_lang},
        )

    async def generate_tests(
        self,
        code: str,
        language: str = 'python',
        framework: Optional[str] = None,
    ) -> CodeGenerationResult:
        """Generate tests for code."""
        language = self.normalize_language(language)

        # Default test frameworks
        default_frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'typescript': 'jest',
            'rust': 'cargo test',
            'go': 'testing',
            'java': 'junit',
        }

        framework = framework or default_frameworks.get(language, 'unit test')

        if not self.orchestrator:
            return self._fallback_generation(f"tests for code", language)

        rag_context = self._get_context_from_rag(f"{language} {framework} tests")
        context_section = f"Test patterns:\n{rag_context}" if rag_context else ""

        prompt = self.PROMPTS['test'].format(
            language=language,
            code=code,
            test_framework=framework,
            context_section=context_section,
        )

        response = await self.orchestrator.generate(
            prompt=prompt,
            mode='coder',
            max_tokens=2500,
        )

        test_code = self._extract_code(response.content, language)

        return CodeGenerationResult(
            code=test_code,
            language=language,
            explanation=f"Generated {framework} tests",
            coherence=response.coherence,
            complexity=self._compute_complexity(test_code),
            trust_level=self.tensor_state.trust_level,
            provider=response.provider,
            tokens_used=response.tokens_used,
            metadata={'task': 'test', 'framework': framework},
        )

    def _extract_code(self, response: str, language: str) -> str:
        """Extract code block from LLM response."""
        # Try to find code block
        markers = [f'```{language}', '```python', '```javascript', '```']

        for marker in markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    code_part = parts[1]
                    if '```' in code_part:
                        code = code_part.split('```')[0]
                        return code.strip()

        # No code block found, return as is (might be pure code)
        return response.strip()

    def _generate_explanation(self, code: str, language: str) -> str:
        """Generate a brief explanation of the generated code."""
        lines = code.split('\n')

        # Find functions/classes
        structure_keywords = {
            'python': ['def ', 'class ', 'async def '],
            'javascript': ['function ', 'const ', 'class '],
            'typescript': ['function ', 'const ', 'class ', 'interface '],
            'rust': ['fn ', 'struct ', 'impl ', 'pub fn '],
            'go': ['func ', 'type ', 'struct '],
        }

        keywords = structure_keywords.get(language, ['def ', 'function ', 'class '])
        structures = []

        for line in lines:
            for kw in keywords:
                if kw in line:
                    # Extract name
                    parts = line.split(kw)
                    if len(parts) > 1:
                        name = parts[1].split('(')[0].split(':')[0].split('{')[0].strip()
                        if name:
                            structures.append(f"{kw.strip()} {name}")

        if structures:
            return f"Generated {language} code with: {', '.join(structures[:5])}"
        return f"Generated {language} code ({len(lines)} lines)"

    def _fallback_generation(self, task: str, language: str) -> CodeGenerationResult:
        """Fallback when LLM is not available."""
        # Return a helpful message instead of empty code
        comment = self.SUPPORTED_LANGUAGES.get(language, {}).get('comment', '#')

        code = f"""{comment} BAZINGA: LLM not available for generation
{comment}
{comment} To enable intelligent code generation:
{comment} 1. Install httpx: pip install httpx
{comment} 2. Set an API key:
{comment}    export GROQ_API_KEY="your-key"  (free at console.groq.com)
{comment}    or
{comment}    export TOGETHER_API_KEY="your-key"
{comment}
{comment} Task requested: {task}
{comment} Language: {language}
"""
        return CodeGenerationResult(
            code=code,
            language=language,
            explanation="LLM not available",
            coherence=0.0,
            complexity=0.0,
            trust_level=0.5,
            provider='none',
            tokens_used=0,
            metadata={'task': task, 'fallback': True},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get coder statistics."""
        return {
            'generations': len(self.generation_history),
            'feedback_entries': len(self.feedback_history),
            'tensor_state': {
                'trust_level': self.tensor_state.trust_level,
                'coherence': self.tensor_state.coherence,
                'pattern_vector': self.tensor_state.pattern_vector,
                'entropy_vector': self.tensor_state.entropy_vector,
            },
            'avg_coherence': (
                sum(g.coherence for g in self.generation_history) /
                max(len(self.generation_history), 1)
            ),
            'languages_used': list(set(g.language for g in self.generation_history)),
            'orchestrator_available': self.orchestrator is not None,
            'rag_available': self.rag is not None,
        }


# Convenience functions
async def generate_code(task: str, language: str = 'python') -> str:
    """Quick code generation."""
    coder = IntelligentCoder()
    result = await coder.generate(task, language)
    return result.code


async def explain_code(code: str, language: str = 'python') -> str:
    """Quick code explanation."""
    coder = IntelligentCoder()
    return await coder.explain(code, language)


async def fix_code(code: str, error: str, language: str = 'python') -> str:
    """Quick code fix."""
    coder = IntelligentCoder()
    result = await coder.fix(code, error, language)
    return result.code


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Intelligent Coder Test")
        print("=" * 60)
        print()

        coder = IntelligentCoder()

        # Test generation
        print("Testing code generation...")
        result = await coder.generate(
            task="a function to calculate fibonacci numbers using memoization",
            language="python"
        )

        print(f"\nProvider: {result.provider}")
        print(f"Coherence: {result.coherence:.3f}")
        print(f"Complexity: {result.complexity:.3f}")
        print(f"Trust Level: {result.trust_level:.3f}")
        print(f"Tokens: {result.tokens_used}")
        print(f"\nGenerated Code:\n{'-' * 40}")
        print(result.code)
        print('-' * 40)
        print(f"\nExplanation: {result.explanation}")

        # Stats
        print(f"\n\nStats: {coder.get_stats()}")

    asyncio.run(test())
