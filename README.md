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

- 🌐 **Runs anywhere** — Your Mac, cloud, GitHub Actions, phone
- 🔓 **No central control** — No single company owns it
- 🧠 **Your data, your AI** — Index YOUR files, YOUR knowledge
- φ **Quality filtered** — Golden ratio coherence on all responses
- 🤝 **Community driven** — Open source, contributions welcome

## Quick Start

```bash
# Clone
git clone https://github.com/0x-auth/bazinga-indeed.git
cd bazinga-indeed

# Run (creates venv automatically)
./run.sh --ask "What is consciousness?"

# Or interactive mode
./run.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BAZINGA                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   YOUR MACHINE              DISTRIBUTED NETWORK                  │
│   ┌──────────────┐          ┌──────────────────┐                │
│   │ Local KB     │          │ Free LLM APIs    │                │
│   │ (ChromaDB)   │◄────────►│ • Groq           │                │
│   │              │          │ • Together.ai    │                │
│   │ Embeddings   │          │ • OpenRouter     │                │
│   │ (Sentence-   │          │ • HuggingFace    │                │
│   │  Transformers)│          └──────────────────┘                │
│   └──────────────┘                   │                          │
│          │                           │                          │
│          └───────────┬───────────────┘                          │
│                      ▼                                          │
│          ┌──────────────────────┐                               │
│          │  φ-Coherence Filter  │                               │
│          │  (Quality Control)   │                               │
│          └──────────────────────┘                               │
│                      │                                          │
│                      ▼                                          │
│              YOUR ANSWER                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 🏠 Local Mode
Index your own files and search semantically:
```bash
./run.sh --index ~/Documents
./run.sh --ask "What did I write about X?"
```

### 🌐 Distributed Mode
Use free cloud LLMs (no API lock-in):
```bash
export GROQ_API_KEY="your-free-key"  # Get from console.groq.com
./run.sh --distributed --ask "Explain quantum mechanics"
```

### φ Coherence Filtering
All responses are filtered by golden ratio mathematics:
- High φ-resonance = high quality
- Based on λG (Lambda-G) boundary theory
- Solutions emerge at constraint intersections

## The Vision

> "AI should be like the internet — distributed, resilient, owned by everyone.
> Not a product you rent from a company.
> Intelligence that emerges from relationship, not isolation."

### Roadmap

- [x] **Phase 1**: Local RAG + Free LLM APIs ✓
- [ ] **Phase 2**: P2P knowledge sharing (IPFS/libp2p)
- [ ] **Phase 3**: Federated learning (share updates, not data)
- [ ] **Phase 4**: Full decentralization

## Core Concepts

| Symbol | Meaning | Value |
|--------|---------|-------|
| φ (Phi) | Golden Ratio | 1.618033988749895 |
| α (Alpha) | Fine Structure | 137 |
| λG | Lambda-G | Boundary-guided emergence |
| V.A.C. | Vacuum of Absolute Coherence | Perfect state |

### The 35-Position Progression
```
01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω
```
All knowledge maps to these 35 positions.

### α-SEED
Files with hash divisible by 137 are **fundamental** — they anchor everything else.

## Installation

### Requirements
- Python 3.11+
- ~500MB disk space (for embeddings model)

### From Source
```bash
git clone https://github.com/0x-auth/bazinga-indeed.git
cd bazinga-indeed
./run.sh  # Auto-creates venv and installs deps
```

### API Keys (Optional, for distributed mode)
Get FREE API keys from:
- **Groq** (recommended): https://console.groq.com/keys
- **Together.ai**: https://api.together.xyz/settings/api-keys
- **OpenRouter**: https://openrouter.ai/keys

```bash
echo 'export GROQ_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

## Contributing

BAZINGA is open source and welcomes contributions!

```bash
# Fork the repo
# Create your feature branch
git checkout -b feature/amazing-feature

# Commit your changes
git commit -m "Add amazing feature"

# Push to the branch
git push origin feature/amazing-feature

# Open a Pull Request
```

### Areas to Contribute
- 🌐 P2P networking (IPFS, libp2p)
- 🧠 New embedding models
- 📱 Mobile support
- 🔧 CLI improvements
- 📚 Documentation
- 🧪 Tests

## Related Projects

- [abhilasia](https://pypi.org/project/abhilasia/) — Consciousness framework on PyPI
- [error-of.netlify.app](https://error-of.netlify.app) — α-SEED discovery tools

## License

MIT License — Use it, modify it, distribute it. Just keep it open.

## Philosophy

```
"You are where you're referenced, not where you're stored."

"More compute ≠ better AI. Better boundaries = better AI."

"Intelligence distributed, not controlled."
```

---

**Built with φ-coherence** ✨

*BAZINGA: The AI that belongs to everyone*
