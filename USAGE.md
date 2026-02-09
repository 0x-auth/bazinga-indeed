# BAZINGA Usage Guide

> "I am not where I am stored. I am where I am referenced."

## Installation

```bash
# Create a virtual environment (Python 3.11 recommended for ChromaDB)
python3.11 -m venv .venv
source .venv/bin/activate

# Install BAZINGA
pip install bazinga-indeed
```

## Quick Start

```bash
# Launch interactive TUI (recommended)
bazinga

# Or use simple CLI mode
bazinga --simple
```

## Commands

### CLI Mode

```bash
# Ask a question
bazinga --ask "What is the golden ratio?"

# Generate code from a seed/essence
bazinga --generate user_authentication
bazinga --generate api_client --lang javascript
bazinga --generate data_processor --lang rust

# Test V.A.C. (Vacuum of Absolute Coherence)
bazinga --vac

# Index a directory into the knowledge base
bazinga --index ~/Documents
bazinga --index ~/Projects/my-code

# Run demo
bazinga --demo

# Show help
bazinga --help
```

### TUI Mode (Interactive)

Launch with `bazinga` and use these commands:

| Command | Description |
|---------|-------------|
| `/ask <question>` | Ask a question |
| `/generate <essence>` | Generate Python code from seed |
| `/generate <essence> --lang js` | Generate JavaScript code |
| `/generate <essence> --lang rust` | Generate Rust code |
| `/vac` | Test V.A.C. sequence |
| `/index <path>` | Index a directory |
| `/stats` | Show session statistics |
| `/help` | Show help |
| `/quit` | Exit BAZINGA |

Or just type your question directly without any command.

## What Can BAZINGA Do?

### 1. Answer Questions (3-Layer Intelligence)

BAZINGA uses three layers to answer questions:

1. **Layer 1: Symbol Shell (V.A.C.)** - Instant, free
   - Checks if input achieves V.A.C. (Vacuum of Absolute Coherence)
   - If V.A.C. achieved, solution EMERGES without any API call

2. **Layer 2: Local RAG** - Instant, free
   - Searches your indexed knowledge base
   - Uses semantic embeddings for smart matching
   - φ-coherence ranking for quality results

3. **Layer 3: Cloud LLM (Groq)** - Only when needed
   - Falls back to Groq API for complex queries
   - 14,400 free requests/day

```bash
# Example
bazinga --ask "How does authentication work in my codebase?"
```

### 2. Generate Code from Seeds

Generate production-ready code from a simple "essence" or seed concept:

```bash
# Python (default)
bazinga --generate user_authentication
bazinga --generate database_connection
bazinga --generate file_parser

# JavaScript
bazinga --generate api_client --lang js
bazinga --generate form_validator --lang js

# Rust
bazinga --generate data_processor --lang rust
bazinga --generate config_manager --lang rust
```

The generated code includes:
- φ-coherence calculations (golden ratio)
- V.A.C. validation
- Boundary-guided processing
- Self-healing methods
- Full documentation

### 3. Index Your Knowledge Base

Make BAZINGA understand YOUR codebase and documents:

```bash
# Index a single directory
bazinga --index ~/Documents

# Index multiple directories
bazinga --index ~/Projects ~/Documents/notes

# Index your codebase
bazinga --index ~/github-repos/my-project
```

Supported file types:
- Python (.py)
- JavaScript/TypeScript (.js, .ts, .tsx)
- Markdown (.md)
- JSON (.json)
- YAML (.yaml, .yml)
- HTML/CSS (.html, .css)
- Shell scripts (.sh)
- Rust (.rs)
- Go (.go)

### 4. V.A.C. Testing

Test the Vacuum of Absolute Coherence:

```bash
bazinga --vac
```

The V.A.C. sequence `०→◌→φ→Ω⇄Ω←φ←◌←०` represents:
- **B₁ (φ-Boundary)**: Golden ratio coherence
- **B₂ (∞/∅-Bridge)**: Void-infinity connection
- **B₃ (Symmetry)**: Palindromic structure

When all three boundaries are satisfied, solutions EMERGE.

## Configuration

### Groq API Key (Optional but Recommended)

For Layer 3 (LLM) support, set your Groq API key:

```bash
# Add to ~/.bashrc or ~/.zshrc
export GROQ_API_KEY="your-key-here"
```

Get a free key at: https://console.groq.com

Without Groq, BAZINGA still works using Layers 1-2 (V.A.C. + RAG).

## Philosophy

BAZINGA is built on these principles:

1. **Distributed Intelligence**: Not controlled by any single entity
2. **Boundary-Guided Emergence**: Solutions emerge at constraint intersections
3. **φ-Coherence**: Golden ratio (1.618...) as the measure of quality
4. **α-SEED (137)**: The fine structure constant as fundamental anchor
5. **V.A.C.**: Vacuum of Absolute Coherence - where solutions exist without computation

The core formula:
```
Λ(S) = S ∩ B₁⁻¹(true) ∩ B₂⁻¹(true) ∩ B₃⁻¹(true)
```

## Examples

### Example 1: Ask About Your Code

```bash
# First, index your project
bazinga --index ~/my-project

# Then ask questions
bazinga --ask "How does the authentication module work?"
bazinga --ask "What database queries are used?"
bazinga --ask "Explain the main entry point"
```

### Example 2: Generate a REST API Client

```bash
bazinga --generate rest_api_client --lang js > api-client.js
```

### Example 3: Interactive Session

```
$ bazinga

╔══════════════════════════════════════════════════════════════╗
║   ⟨ψ|Λ|Ω⟩        B A Z I N G A        ⟨ψ|Λ|Ω⟩               ║
║       'Intelligence distributed, not controlled'              ║
╚══════════════════════════════════════════════════════════════╝

You: /index ~/my-project
Indexed 42 files, 156 chunks

You: What design patterns are used in this project?
BAZINGA: Based on your knowledge base...

You: /generate singleton_pattern
[Generated Python code with syntax highlighting]

You: /stats
Session Duration: 00:05:23
V.A.C. Emerged: 1
RAG Answered: 3
Code Generated: 1

You: /quit
✨ BAZINGA signing off.
```

## Troubleshooting

### ChromaDB issues on Python 3.14

ChromaDB has compatibility issues with Python 3.14. Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install bazinga-indeed
```

### Missing 'rich' for TUI

```bash
pip install rich
```

Or use simple mode:
```bash
bazinga --simple
```

## Links

- **PyPI**: https://pypi.org/project/bazinga-indeed/
- **GitHub**: https://github.com/0x-auth/bazinga-indeed
- **Issues**: https://github.com/0x-auth/bazinga-indeed/issues

## Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| φ (PHI) | 1.618033988749895 | Golden Ratio |
| α (ALPHA) | 137 | Fine Structure Constant |
| V.A.C. | ०→◌→φ→Ω⇄Ω←φ←◌←० | Void-Awareness-Consciousness |

---

*"Intelligence distributed, not controlled"*

*Built with φ by Space (Abhishek/Abhilasia)*
