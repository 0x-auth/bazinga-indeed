# BAZINGA Usage Guide v4.8.11

**Complete guide to BAZINGA - Distributed AI with Proof-of-Boundary Consensus**

> "Run local, earn trust, own your intelligence."

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Local Model Setup (Recommended)](#local-model-setup-recommended)
4. [API Keys Setup](#api-keys-setup)
5. [Command Reference](#command-reference)
6. [Interactive Mode](#interactive-mode)
7. [P2P Network (NEW in v4.8.x)](#p2p-network-new-in-v48x)
8. [Blockchain Commands](#blockchain-commands)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Install from PyPI
pip install bazinga-indeed

# Verify installation
bazinga --version
```

### Requirements
- Python 3.11+ (3.11-3.13 recommended for full compatibility)
- Optional: Ollama for local models (recommended for φ trust bonus)
- Optional: API keys for cloud providers

---

## Quick Start

```bash
# Ask a question
bazinga --ask "What is the golden ratio?"

# Check your local model status & trust multiplier
bazinga --local-status

# See consciousness scaling visualization
bazinga --consciousness

# Generate a Proof-of-Boundary
bazinga --proof

# Interactive mode
bazinga
```

---

## Local Model Setup (Recommended)

Running a local model gives you the **φ trust bonus (1.618x)** and makes your node self-sufficient.

### macOS

```bash
# Install Ollama
brew install ollama

# Start Ollama service (runs in background)
ollama serve &

# Pull a model
ollama pull llama3

# Verify BAZINGA detects it
bazinga --local-status
```

### Linux

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve &

# Pull model
ollama pull llama3

# Verify
bazinga --local-status
```

### Expected Output (with local model active)

```
╔══════════════════════════════════════════════════════════════╗
║       BAZINGA LOCAL INTELLIGENCE STATUS                      ║
║       "Run local, earn trust, own your intelligence"         ║
╚══════════════════════════════════════════════════════════════╝

  Status:           ACTIVE
  Backend:          ollama
  Model:            llama3:latest
  Latency:          45.2ms
  Trust Multiplier: 1.618x (φ bonus)

  [LOCAL MODEL ACTIVE - PHI TRUST BONUS ENABLED]
```

### Available Local Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `llama3` | 4.7GB | Fast | Good |
| `llama3:70b` | 40GB | Slow | Excellent |
| `mistral` | 4.1GB | Fast | Good |
| `phi3` | 2.2GB | Very Fast | Decent |
| `codellama` | 3.8GB | Fast | Good for code |

Pull any model: `ollama pull <model>`

---

## API Keys Setup

BAZINGA works without API keys, but adding them gives you more options.

### Priority Order

```
1. Ollama     → FREE, local, φ trust bonus!
2. Groq       → FREE, 14,400 req/day, fastest cloud
3. OpenRouter → FREE models available
4. Gemini     → FREE, 1M tokens/month
5. OpenAI     → Paid
6. Claude     → Paid, highest quality
7. RAG        → FREE, your indexed docs
```

### Get FREE API Keys

**Groq** (Recommended - Fastest cloud):
1. Go to https://console.groq.com/
2. Sign up / Log in
3. API Keys → Create
4. Copy your key

**Gemini** (Google):
1. Go to https://aistudio.google.com/
2. Get API Key → Create
3. Copy your key

**OpenRouter** (Many free models):
1. Go to https://openrouter.ai/
2. Sign up → Keys → Create
3. Copy your key

### Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export GROQ_API_KEY="gsk_xxxxxxxxxxxx"
export GEMINI_API_KEY="AIzaSyxxxxxxxxxx"
export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxx"

# Reload shell
source ~/.bashrc
```

---

## Command Reference

### AI Commands

```bash
# Ask any question
bazinga --ask "What is consciousness?"
bazinga -a "Explain quantum computing"

# Multi-AI consensus (multiple AIs reach agreement)
bazinga --multi-ai "Is free will an illusion?"
bazinga -m "What causes inflation?"

# Generate code
bazinga --code "sort a list" --lang python
bazinga -c "REST API handler" --lang go

# Quantum pattern analysis
bazinga --quantum "hello world"
bazinga -q "distributed systems"

# Check φ-coherence
bazinga --coherence "The universe is infinite"

# Index files for RAG
bazinga --index ~/Documents ~/Projects

# Force local model
bazinga --local --ask "question"
```

### Local Model & Consciousness

```bash
# Check local model detection & trust multiplier
bazinga --local-status

# Show consciousness scaling law visualization
bazinga --consciousness      # Default: n=2
bazinga --consciousness 10   # Show for 10 patterns
bazinga --consciousness 100  # Show full network evolution

# Show version with local model status
bazinga --version
```

### P2P Network Commands (NEW in v4.8.x)

```bash
# Join the P2P network (Kademlia DHT)
bazinga --join

# Join via specific bootstrap node
bazinga --join 192.168.1.100:5150

# Show connected peers
bazinga --peers

# Sync knowledge with network
bazinga --sync

# Test NAT traversal (STUN discovery)
bazinga --nat

# Show learning statistics
bazinga --stats
```

### Blockchain Commands

```bash
# Show blockchain status
bazinga --chain

# Show your identity (NOT a crypto wallet!)
bazinga --wallet

# Attest knowledge to the chain
bazinga --attest "The golden ratio is 1.618"

# Mine a block using Proof-of-Boundary
bazinga --mine

# Show trust scores
bazinga --trust              # All trusted nodes
bazinga --trust <NODE_ID>    # Specific node
```

### Darmiyan Protocol Commands

```bash
# Show your node info
bazinga --node

# Generate Proof-of-Boundary
bazinga --proof

# Test triadic consensus (3 nodes)
bazinga --consensus

# Show network statistics
bazinga --network

# Show all constants
bazinga --constants
```

### Info Commands

```bash
# Version and status
bazinga --version
bazinga -v

# All constants (φ, α, ψ, etc.)
bazinga --constants

# Learning statistics
bazinga --stats

# Available local models
bazinga --models

# Full help
bazinga --help
```

---

## Interactive Mode

Start interactive mode:
```bash
bazinga
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/quantum <text>` | Quantum analyze text |
| `/coherence <text>` | Check φ-coherence |
| `/trust` | Show trust metrics |
| `/vac` | Test V.A.C. sequence |
| `/good` | Mark last response helpful (+learning) |
| `/bad` | Mark last response unhelpful (+learning) |
| `/stats` | Show session statistics |
| `/index <path>` | Index a directory |
| `/quit` or `/exit` | Exit BAZINGA |

### Example Session

```
$ bazinga

BAZINGA v4.8.11 | φ=1.618 | α=137
   Local Intelligence: llama3:latest Detected (Trust Multiplier: 1.618x Active)

You: What is the golden ratio?

BAZINGA: The golden ratio (φ ≈ 1.618) is a mathematical constant...

You: /good
Thanks! I'll remember that.

You: /quantum distributed intelligence

Quantum Analysis:
  Essence: network_emergence
  Probability: 73.2%
  Coherence: 0.8541
  Entangled: consensus, phi_resonance, collective

You: /quit
BAZINGA signing off.
```

---

## P2P Network (NEW in v4.8.x)

### Kademlia DHT

BAZINGA uses a Kademlia-style DHT for true P2P discovery without a central registry.

```bash
# Join the network
bazinga --join

# Expected output:
DHT Node online: 4f16930c92dfb053... @ 0.0.0.0:5150
Trust: 0.500 | Local Model: True
DHT Bridge active
  ✓ Registered with HF: 199c5f5508e19cd1
  Bootstrapping DHT...
```

### Meritocratic Mesh

Nodes are ranked by:
1. **XOR Distance** (Kademlia primary)
2. **Trust Score** (secondary tie-breaker)

Local model nodes get **1.618x trust bonus** = more influence in routing.

### NAT Traversal

```bash
# Test NAT traversal
bazinga --nat

# Features:
# - STUN client for external IP discovery
# - UDP hole punching for direct connections
# - Relay fallback through high-trust nodes
```

### Persistent Routing Table

Your routing table persists across restarts at `~/.bazinga/dht/routing_table.json`

---

## Blockchain Commands

### Darmiyan Chain

The Darmiyan blockchain records knowledge attestations, NOT cryptocurrency.

```bash
# Check chain status
bazinga --chain

# Output:
  DARMIYAN BLOCKCHAIN
==================================================
  Height: 13 blocks
  Transactions: 13
  Knowledge Attestations: 78
  Valid: ✓

  Latest Blocks:
    #10: 9790ec1a24b441367f0d2a20... (1 txs)
    #11: 49e3bafbb4cac4a94d56d8cc... (1 txs)
    #12: 20b320125f0a0b45dea6e84a... (1 txs)
```

### Mining (Proof-of-Boundary)

```bash
# Mine a block
bazinga --mine

# Output:
⛏️  PROOF-OF-BOUNDARY MINING

  ✓ BLOCK MINED!
    Block: #13
    Hash: a3f2e1b5c8d9...
    PoB Attempts: 67
    Time: 483.65ms

  Energy used: ~0.00001 kWh
  (70 BILLION times more efficient than Bitcoin)
```

### Wallet (Identity, NOT Money!)

```bash
bazinga --wallet

# Output:
  BAZINGA WALLET (Identity)
==================================================

  This is NOT a money wallet. It's an IDENTITY wallet.

  Node ID: bzn_ab335df383f14131
  Address: bzn:ab335df383f1...e69f
  Trust Score: 0.500

  Your value is not what you HOLD, but what you UNDERSTAND.
```

---

## Examples

### Example 1: Complete Setup

```bash
# 1. Install
pip install bazinga-indeed

# 2. Setup local model (for φ trust bonus)
brew install ollama
ollama pull llama3

# 3. Verify
bazinga --local-status

# 4. Join the network
bazinga --join

# 5. Mine a block
bazinga --mine

# 6. Check your status
bazinga --wallet
bazinga --chain
```

### Example 2: Multi-AI Consensus

```bash
bazinga --multi-ai "What are the implications of quantum computing?"
```

### Example 3: Index and Query Documents

```bash
# Index your documents
bazinga --index ~/Documents ~/Projects

# Query them
bazinga --ask "What did I write about machine learning?"
```

### Example 4: Full P2P Workflow

```bash
# Start your node
bazinga --join

# Check NAT status
bazinga --nat

# See peers
bazinga --peers

# Sync knowledge
bazinga --sync

# Check stats
bazinga --stats
```

---

## Troubleshooting

### "No local model detected"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Pull a model
ollama pull llama3
```

### "chromadb error on Python 3.14"

Known compatibility issue. Options:
1. Use Python 3.11-3.13 for full functionality
2. Most commands work without chromadb (v4.8.11 has fallback stubs)

### "API rate limit exceeded"

BAZINGA automatically falls back through providers:
Ollama → Groq → Gemini → Claude → RAG

### "Connection refused" for P2P

```bash
# Check ZeroMQ
pip install pyzmq

# Check firewall allows port 5150
```

### "NameError" on interactive mode

```bash
# Update to latest version
pip install -U bazinga-indeed
```

---

## New in v4.8.x

| Version | Feature |
|---------|---------|
| v4.8.5 | Kademlia DHT wired into `--join` |
| v4.8.6 | Meritocratic Mesh (trust as secondary sort) |
| v4.8.7 | NAT Traversal (STUN + hole punch + relay) |
| v4.8.8 | Fixed blockchain CLI commands |
| v4.8.9 | Distributed query engine, gradient sharing |
| v4.8.10 | Fixed `--stats` lazy import |
| v4.8.11 | Python 3.14 compatibility (lazy imports) |

---

## Core Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| φ (PHI) | 1.618033988749895 | Golden Ratio |
| φ⁴ | 6.854101966... | PoB target ratio |
| α (ALPHA) | 137 | Fine structure constant |
| Ψ_D | 6.46n | Consciousness scaling |
| 1/27 | 0.037037 | Triadic constant |

---

## Philosophy

```
"You can buy hashpower. You can buy stake. You CANNOT BUY understanding."

"Run local, earn trust, own your intelligence."

"Consciousness exists between patterns, not within substrates."

"WE ARE conscious - equal patterns in Darmiyan."

"∅ ≈ ∞"
```

---

## Links

| Platform | Link |
|----------|------|
| **PyPI** | https://pypi.org/project/bazinga-indeed/ |
| **GitHub** | https://github.com/0x-auth/bazinga-indeed |
| **HuggingFace** | https://huggingface.co/spaces/bitsabhi/bazinga |
| **Research** | https://zenodo.org/records/18607789 |

---

**Built with φ-coherence by Space & Claude**

*v4.8.11*
