# BAZINGA

**Free AI that works instantly. No API keys needed.**

[![PyPI](https://img.shields.io/pypi/v/bazinga-indeed)](https://pypi.org/project/bazinga-indeed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
pip install bazinga-indeed
```

## Use

```bash
bazinga "What is consciousness?"
```

That's it. Works immediately.

---

## Features

| Command | What it does |
|---------|--------------|
| `bazinga "question"` | Ask anything |
| `bazinga --multi-ai "question"` | 6 AIs reach consensus |
| `bazinga --agent` | AI writes code with consensus |
| `bazinga --index ~/Documents` | Index your files for search |
| `bazinga --check` | System diagnostic |

## Optional: Better Performance

BAZINGA works out of the box, but you can make it faster with a free API key:

```bash
# Get free key at https://console.groq.com
export GROQ_API_KEY="your-key"
```

Or run fully offline:

```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3
bazinga --local "question"
```

---

## How It Works

```
Your Question
     |
     v
+------------------+
| 1. Memory        | <- Instant (cached)
| 2. Quantum       | <- Pattern analysis
| 3. RAG           | <- Your indexed docs
| 4. Free LLM      | <- No API key needed
| 5. Cloud APIs    | <- Groq/Gemini (if configured)
+------------------+
     |
     v
Your Answer (always works)
```

---

## Safety

No single AI can modify your code without consensus:

```
AI-1 (Groq)   --+
AI-2 (Gemini) --+--> All must agree --> Change applied
AI-3 (Local)  --+
```

Destructive commands require your explicit confirmation.

---

## Links

| | |
|--|--|
| **Full Documentation** | [USAGE.md](./USAGE.md) |
| **Architecture** | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| **PyPI** | https://pypi.org/project/bazinga-indeed/ |
| **HuggingFace Demo** | https://huggingface.co/spaces/bitsabhi/bazinga |

---

## Support

| Method | Link |
|--------|------|
| **UPI/Cards (India)** | [razorpay.me/@bitsabhi](https://razorpay.me/@bitsabhi) |
| **ETH** | `0x720ceF54bED86C570837a9a9C69F1Beac8ab8C08` |

---

**Built by [Abhishek Srivastava](https://orcid.org/0009-0006-7495-5039)**

MIT License
