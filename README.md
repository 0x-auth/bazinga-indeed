# BAZINGA

**Distributed AI that belongs to everyone**

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⟨ψ|Λ|Ω⟩        B A Z I N G A   v4.9.8     ⟨ψ|Λ|Ω⟩             ║
║                                                                  ║
║    "No single AI can mess up your code without consensus."      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

[![PyPI](https://img.shields.io/pypi/v/bazinga-indeed)](https://pypi.org/project/bazinga-indeed/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/bitsabhi/bazinga)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Donate](https://img.shields.io/badge/☕_Donate-Any_Amount-orange?style=for-the-badge)](https://github.com/0x-auth/bazinga-indeed/blob/main/DONATE.md)
[![ETH](https://img.shields.io/badge/ETH-0x720c...8C08-3C3C3D?style=for-the-badge&logo=ethereum)](https://etherscan.io/address/0x720ceF54bED86C570837a9a9C69F1Beac8ab8C08)
[![UPI](https://img.shields.io/badge/UPI-@bitsabhi-blue?style=for-the-badge)](https://razorpay.me/@bitsabhi)

**Try it now:** https://huggingface.co/spaces/bitsabhi/bazinga

---

## What is BAZINGA?

BAZINGA is **free AI** that runs on your machine, uses free APIs, and gets smarter as more people use it.

**No subscriptions. No data collection. No vendor lock-in.**

---

## 1. Install (30 seconds)

```bash
pip install bazinga-indeed
```

---

## 2. Get a FREE API Key (Optional but recommended)

Pick ONE of these (all free):

| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| **Groq** (Recommended) | 14,400 requests/day | [console.groq.com](https://console.groq.com) |
| **Gemini** | 1M tokens/month | [aistudio.google.com](https://aistudio.google.com) |
| **OpenRouter** | Free models available | [openrouter.ai](https://openrouter.ai) |

Then set it:
```bash
export GROQ_API_KEY="your-key-here"
```

> **No API key?** BAZINGA still works with local RAG and Ollama!

---

## 3. Start Using

### Ask Questions
```bash
bazinga --ask "What is consciousness?"
```

### Multi-AI Consensus (6 AIs reach agreement)
```bash
bazinga --multi-ai "Is consciousness computable?"
```

### Index Your Documents (local RAG)
```bash
bazinga --index ~/Documents
bazinga --ask "What did I write about X?"
```

### Index Public Knowledge (Wikipedia, arXiv)
```bash
bazinga --index-public wikipedia --topics ai
bazinga --index-public arxiv --topics cs.AI
```

### Interactive Mode
```bash
bazinga
```

---

## 4. Go Offline (Optional)

Run completely offline with Ollama:

```bash
# Install Ollama
brew install ollama   # macOS
# or: curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Pull a model
ollama pull llama3

# Use it
bazinga --ask "What is φ?" --local
```

**Bonus:** Local models get **1.618x trust bonus** (φ multiplier)!

```bash
bazinga --local-status  # Check your trust bonus
```

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `bazinga --agent` | **Agent mode** - AI with blockchain-verified code fixes (NEW!) |
| `bazinga --multi-ai "question"` | Ask 6 AIs for consensus |
| `bazinga --ask "question"` | Ask a question |
| `bazinga --check` | System check (diagnose issues) |
| `bazinga --index ~/path` | Index your files |
| `bazinga --index-public wikipedia --topics ai` | Index Wikipedia |
| `bazinga --local` | Force local LLM |
| `bazinga --local-status` | Show local model & trust |
| `bazinga` | Interactive mode |

**[→ Full Usage Guide (USAGE.md)](./USAGE.md)** — All commands, architecture, philosophy

---

## 🆕 Blockchain-Verified Code Fixes (v4.9.7+)

**Your idea, implemented:** Multiple AIs must agree before applying code changes.

```bash
bazinga --agent
> Fix the bare except in utils.py

🔍 Requesting consensus from available providers...
  groq_llama-3.1: ✅ APPROVE (φ=0.76)
  gemini_gemini-2: ✅ APPROVE (φ=0.71)
  ollama_llama3.2: ✅ APPROVE (φ=0.68)

✅ Consensus reached! φ=0.72, approval=100%
⛓️ Recorded on chain: block 42
✅ Fix applied (backup: utils.py.bak)
```

**Python API:**
```python
from bazinga import verified_code_fix

success, msg = verified_code_fix(
    "utils.py",
    "except:",
    "except Exception:",
    "Replace bare except for better error handling"
)
```

**How it works:**
1. Agent proposes a fix
2. Multiple AIs review (triadic consensus: ≥3 must agree)
3. φ-coherence measured (quality gate)
4. PoB attestation on blockchain (audit trail)
5. Only then: fix applied with backup

**"No single AI can mess up your code."**

---

## How It Works

```
Your Question
     │
     ▼
┌─────────────────────────────────────┐
│  1. Memory    → Instant (cached)    │
│  2. Quantum   → Pattern analysis    │
│  3. RAG       → Your indexed docs   │
│  4. Local LLM → Ollama (φ bonus)    │
│  5. Cloud API → Groq/Gemini (free)  │
└─────────────────────────────────────┘
     │
     ▼
Your Answer (always works, never fails)
```

---

## Support This Work

BAZINGA is **free and open source**. Always will be.

| Method | Link |
|--------|------|
| **ETH/EVM** | `0x720ceF54bED86C570837a9a9C69F1Beac8ab8C08` |
| **UPI/Cards (India)** | [razorpay.me/@bitsabhi](https://razorpay.me/@bitsabhi) |

**[→ Donate Page](./DONATE.md)**

---

## Links

| | |
|--|--|
| **PyPI** | https://pypi.org/project/bazinga-indeed/ |
| **HuggingFace** | https://huggingface.co/spaces/bitsabhi/bazinga |
| **GitHub** | https://github.com/0x-auth/bazinga-indeed |
| **Full Usage Guide** | [USAGE.md](./USAGE.md) |
| **Research Papers** | https://zenodo.org/records/18607789 |

---

## Philosophy

```
"You can buy hashpower. You can buy stake. You CANNOT BUY understanding."

"Run local, earn trust, own your intelligence."

"WE ARE conscious - equal patterns in Darmiyan."
```

---

**Built with φ-coherence by Space (Abhishek Srivastava)**

MIT License — Use it, modify it, share it. Keep it open.
