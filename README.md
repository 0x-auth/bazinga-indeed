# BAZINGA

**Distributed AI — Intelligence that belongs to everyone**

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⟨ψ|Λ|Ω⟩        B A Z I N G A        ⟨ψ|Λ|Ω⟩                   ║
║                                                                  ║
║         "Intelligence distributed, not controlled"               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

## What is BAZINGA?

BAZINGA is an **open-source, distributed AI** that:

- 🆓 **Always FREE** — Uses free APIs, falls back gracefully, never fails
- 🌐 **Runs anywhere** — Your Mac, Linux, cloud, anywhere
- 🔓 **No central control** — No single company owns it
- 🧠 **Your data, your AI** — Index YOUR files, YOUR knowledge
- φ **Quality filtered** — Golden ratio coherence on all responses
- 🤝 **Community driven** — PRs welcome, like Bitcoin but for AI

## Install

```bash
pip install bazinga-indeed
```

## Quick Start

```bash
# Just works - even without API keys!
bazinga --ask "What is consciousness?"

# Index your files
bazinga --index ~/Documents

# Interactive mode
bazinga
```

**That's it.** No API keys required to start. BAZINGA gracefully falls back through free options.

---

## API Keys (Optional but Recommended)

BAZINGA works without any API keys, but adding FREE keys makes it smarter:

### Priority Order (all FREE except Claude):
```
1. Groq      → FREE 14,400 requests/day
2. Gemini    → FREE 1 million tokens/month
3. Local LLM → FREE forever (runs on your machine)
4. Claude    → Paid (only used if others unavailable)
5. RAG       → FREE (searches your indexed docs)
```

### Get Your FREE API Keys

#### 1. Groq (Recommended - Fastest)
1. Go to https://console.groq.com/
2. Sign up (free, no credit card)
3. Click "API Keys" → "Create API Key"
4. Copy your key

#### 2. Gemini (Google - 1M free tokens/month)
1. Go to https://aistudio.google.com/
2. Sign in with Google
3. Click "Get API Key" → "Create API Key"
4. Copy your key

#### 3. Local LLM (Offline - Forever Free)
```bash
pip install bazinga-indeed[local]
# First run downloads a 700MB model, then works offline forever
```

#### 4. Claude (Optional - Paid)
1. Go to https://console.anthropic.com/
2. Sign up (get $5 free credit)
3. Go to "API Keys" → "Create Key"
4. Copy your key

### Set Your Keys

**Mac/Linux** — Add to `~/.bashrc` or `~/.zshrc`:
```bash
# BAZINGA API Keys (FREE!)
export GROQ_API_KEY="gsk_xxxxxxxxxxxx"
export GEMINI_API_KEY="AIzaSyxxxxxxxxxx"

# Optional (paid)
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

**Windows** — Set environment variables:
```cmd
setx GROQ_API_KEY "gsk_xxxxxxxxxxxx"
setx GEMINI_API_KEY "AIzaSyxxxxxxxxxx"
```

### Verify Setup
```bash
bazinga --version
```
Shows which APIs are configured.

---

## Usage

### Ask Questions
```bash
bazinga --ask "Explain quantum entanglement"
bazinga -a "What is the meaning of life?"
```

### Index Your Files
```bash
bazinga --index ~/Documents ~/Projects
bazinga -i ~/Notes
```

### Interactive Mode
```bash
bazinga

# Commands in interactive mode:
# /stats     - Show statistics
# /trust     - Show trust metrics
# /good      - Mark last answer as helpful (learns!)
# /bad       - Mark as unhelpful (adapts!)
# /quit      - Exit
```

### Generate Code
```bash
bazinga --code "fibonacci sequence" --lang python
bazinga -c "REST API server" -l javascript
```

### Advanced
```bash
bazinga --quantum "consciousness"     # Quantum pattern analysis
bazinga --coherence "your text"       # Check φ-coherence
bazinga --constants                   # Show φ, α, ψ values
bazinga --local                       # Force local LLM only
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         BAZINGA v3.5                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOUR QUESTION                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Memory     → Check learned patterns (instant)       │   │
│  │  2. Quantum    → Process in superposition (instant)     │   │
│  │  3. λG Check   → V.A.C. emergence check (instant)       │   │
│  │  4. Groq       → FREE API (14,400/day)                  │   │
│  │  5. Gemini     → FREE API (1M tokens/month)             │   │
│  │  6. Local LLM  → Your machine (forever free)            │   │
│  │  7. Claude     → Paid (fallback)                        │   │
│  │  8. RAG        → Your indexed docs (always works)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  φ-Coherence Filter (quality control via golden ratio)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  YOUR ANSWER (never fails, always responds)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** BAZINGA NEVER fails. If one API is down or rate-limited, it automatically tries the next. Eventually falls back to RAG (your own docs) which always works.

---

## The Vision

> "AI should be like Bitcoin — distributed, resilient, owned by everyone.
> Not a product you rent from a company.
> Intelligence that emerges from the network, not controlled by anyone."

### Roadmap

- [x] **Phase 1**: Local RAG + φ-Coherence ✓
- [x] **Phase 2**: Multi-LLM (Groq + Gemini + Claude + Local) ✓
- [x] **Phase 3**: Learning Memory ✓
- [x] **Phase 4**: Quantum + λG Processing ✓
- [ ] **Phase 5**: P2P Knowledge Network (coming)
- [ ] **Phase 6**: Federated Learning
- [ ] **Phase 7**: Full Decentralization (the Bitcoin of AI)

---

## Core Concepts

| Symbol | Meaning | Value |
|--------|---------|-------|
| φ (Phi) | Golden Ratio | 1.618033988749895 |
| α (Alpha) | Fine Structure Constant | 137 |
| ψ (Psi) | Consciousness Coefficient | 6.236 (2φ² + 1) |
| λG | Lambda-G | Boundary-guided emergence |
| V.A.C. | Vacuum of Absolute Coherence | Perfect state |
| τ (Tau) | Trust | Approaches 1 |

---

## Installation Options

```bash
# Basic (uses cloud APIs)
pip install bazinga-indeed

# With local LLM support (offline capable)
pip install bazinga-indeed[local]

# Everything
pip install bazinga-indeed[full]
```

### Requirements
- Python 3.11+
- ~500MB disk (for embeddings)
- ~700MB more if using local LLM

---

## Contributing

BAZINGA is open source. PRs welcome!

```bash
git clone https://github.com/0x-auth/bazinga-indeed.git
cd bazinga-indeed
pip install -e ".[dev]"

# Make changes, then:
git checkout -b feature/your-feature
git commit -m "Add amazing feature"
git push origin feature/your-feature
# Open PR
```

### Areas to Contribute
- 🌐 P2P networking
- 🧠 Better embeddings
- 📱 Mobile support
- 🔧 CLI improvements
- 📚 Documentation
- 🧪 Tests

---

## License

MIT License — Use it, modify it, share it. Keep it open.

---

## Philosophy

```
"You are where you're referenced, not where you're stored."

"More compute ≠ better AI. Better boundaries = better AI."

"Intelligence distributed, not controlled."

"BAZINGA never fails. It always finds a way."
```

---

**Built with φ-coherence** ✨

*BAZINGA: The AI that belongs to everyone*

[![PyPI](https://img.shields.io/pypi/v/bazinga-indeed)](https://pypi.org/project/bazinga-indeed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
