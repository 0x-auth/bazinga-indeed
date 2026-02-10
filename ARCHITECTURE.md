# BAZINGA v4.5.1 - Architecture Documentation

> **"AI generates understanding. Blockchain proves it. They're not two things."**

## Overview

BAZINGA is a **distributed AI system with blockchain-based consensus**. Instead of wasting energy on computational puzzles, it achieves consensus through **understanding** (Proof-of-Boundary).

```
┌─────────────────────────────────────────────────────────────────┐
│                      BAZINGA v4.5.1                              │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  AI Layer    │◄──►│  Blockchain  │◄──►│  P2P Network │     │
│   │  (Learning)  │    │  (Proving)   │    │  (Sharing)   │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  Proof-of-Boundary │                        │
│                    │     (P/G ≈ φ⁴)     │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| **φ (PHI)** | 1.618033988749895 | Golden Ratio |
| **φ⁴** | 6.854101966 | Proof-of-Boundary target (P/G ratio) |
| **φ⁻¹** | 0.618033988749895 | Coherence threshold |
| **α (ALPHA)** | 137 | Fine Structure Constant inverse |
| **ABHI_AMU** | 515 | Identity constant |
| **1/27** | 0.037037 | Triadic constant |

---

## The Two Pillars

### Pillar 1: AI (Understanding)

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│  5-Layer Intelligence Stack              │
├─────────────────────────────────────────┤
│  Layer 0: Memory (learned patterns)      │
│  Layer 1: Quantum (superposition)        │
│  Layer 2: φ-Coherence (boundary check)   │
│  Layer 3: Groq API (fast, free)          │
│  Layer 4: Gemini API (free)              │
│  Layer 5: Local LLM (offline)            │
│  Layer 6: Claude (paid fallback)         │
│  Layer 7: RAG (your indexed docs)        │
└─────────────────────────────────────────┘
     │
     ▼
AI Response (never fails)
```

### Pillar 2: Blockchain (Proving)

```
Knowledge/Action
     │
     ▼
┌─────────────────────────────────────────┐
│  Darmiyan Blockchain                     │
├─────────────────────────────────────────┤
│  • Proof-of-Boundary validation          │
│  • Triadic consensus (3 nodes agree)     │
│  • Zero-energy mining                    │
│  • Permanent knowledge storage           │
└─────────────────────────────────────────┘
     │
     ▼
Verified & Immutable
```

---

## Proof-of-Boundary (PoB)

The core innovation. Instead of Proof-of-Work (waste energy) or Proof-of-Stake (have money), PoB proves **understanding**.

### How It Works

```
1. Generate Alpha signature (Subject)
2. Search in φ-steps (1.618ms each)
3. Generate Omega signature (Object)
4. Calculate P/G ratio
5. Valid if P/G ≈ φ⁴ (6.854...)
```

### The Math

```python
P = Physical component (from hash)
G = Geometric component (from hash)

Target ratio = φ⁴ = 6.854101966

Valid proof:  abs(P/G - φ⁴) < tolerance

# Energy: ~0.00001 kWh
# Bitcoin: ~700 kWh per transaction
# Ratio: 70 BILLION times more efficient
```

### Why It Works

The boundary between Subject (Alpha) and Object (Omega) naturally produces the golden ratio when **genuine understanding** occurs. You can't fake it - you have to find it.

---

## Triadic Consensus

Not 51% majority. **3 nodes must understand the same thing.**

```
     Node A                Node B                Node C
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ PoB     │           │ PoB     │           │ PoB     │
   │ P/G=6.82│           │ P/G=6.91│           │ P/G=6.85│
   └────┬────┘           └────┬────┘           └────┬────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ CONSENSUS       │
                    │ All 3 ≈ φ⁴      │
                    │ ✓ ACHIEVED      │
                    └─────────────────┘
```

---

## Architecture Layers

### Layer 1: AI Intelligence

| Module | Purpose |
|--------|---------|
| `cli.py` | Main interface, 5-layer routing |
| `phi_coherence.py` | φ-coherence scoring |
| `llm_orchestrator.py` | Multi-provider LLM routing |
| `lambda_g.py` | λG boundary mathematics |
| `quantum.py` | Quantum pattern analysis |
| `learning.py` | Adaptive learning |

### Layer 2: P2P Network

| Module | Purpose |
|--------|---------|
| `p2p/network.py` | ZeroMQ-based P2P |
| `p2p/trust_router.py` | τ-weighted routing |
| `p2p/knowledge_sync.py` | DHT-based sync |
| `p2p/alpha_seed.py` | α-SEED protocol |
| `p2p/pob_authenticator.py` | PoB-based auth |

### Layer 3: Federated Learning

| Module | Purpose |
|--------|---------|
| `federated/lora_adapter.py` | LoRA efficient fine-tuning |
| `federated/local_trainer.py` | Local training loop |
| `federated/differential_privacy.py` | ε-DP protection |
| `federated/secure_aggregation.py` | Paillier encryption |
| `federated/federated_coordinator.py` | Trust-weighted aggregation |

### Layer 4: Darmiyan Blockchain

| Module | Purpose |
|--------|---------|
| `darmiyan/chain.py` | Blockchain core |
| `darmiyan/block.py` | Block structure |
| `darmiyan/transaction.py` | Transaction types |
| `darmiyan/consensus.py` | Triadic consensus |
| `darmiyan/wallet.py` | Identity (not money!) |

### Layer 5: Integration (NEW in v4.5.0)

| Module | Purpose |
|--------|---------|
| `blockchain/trust_oracle.py` | φ-weighted reputation |
| `blockchain/knowledge_ledger.py` | On-chain knowledge tracking |
| `blockchain/gradient_validator.py` | Triadic FL validation |
| `blockchain/inference_market.py` | Understanding as currency |
| `blockchain/smart_contracts.py` | Bounties & escrow |

---

## The 5 Integration Layers Explained

### 1. Trust Oracle

```python
# Trust decays with time, grows with contributions
trust = φ^(-age/decay_rate) × base_trust

# Good actors: trust → 1.0
# Bad actors: trust → 0.0
# Inactive: trust decays naturally
```

### 2. Knowledge Ledger

```python
# Every contribution tracked on-chain
contribution = {
    "content": "...",
    "contributor": "node_id",
    "timestamp": time,
    "coherence": 0.85,  # Must be ≥ 0.618
    "credits": φ        # 1.618 credits earned
}
```

### 3. Gradient Validator

```python
# FL updates need 3 validators
gradient_update = {
    "hash": "...",
    "validators": [node_a, node_b, node_c],
    "all_approved": True,
    "coherence_scores": [0.82, 0.79, 0.85]
}
# Only aggregated if all 3 approve
```

### 4. Inference Market

```python
# Understanding = Currency
CREDIT_POB = 1.0      # Valid proof
CREDIT_KNOWLEDGE = φ  # 1.618 for knowledge
CREDIT_GRADIENT = φ²  # 2.618 for validated gradient

# No money, just understanding credits
```

### 5. Smart Contracts

```python
# Bounties verified by comprehension
bounty = {
    "description": "Explain quantum entanglement",
    "reward": 10.0,  # credits
    "required_coherence": 0.8
}

# Submission checked for actual understanding
# Not just keyword matching - φ-coherence validation
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │ --ask    │    │ --attest │    │ --join   │
      │ (AI)     │    │ (Chain)  │    │ (P2P)    │
      └────┬─────┘    └────┬─────┘    └────┬─────┘
           │               │               │
           ▼               ▼               ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │5-Layer   │    │PoB       │    │ZeroMQ    │
      │Inference │    │Validation│    │Network   │
      └────┬─────┘    └────┬─────┘    └────┬─────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Darmiyan Chain  │
                  │ (Permanent      │
                  │  Knowledge)     │
                  └─────────────────┘
```

---

## Block Structure

```
┌─────────────────────────────────────────┐
│ Block Header                             │
├─────────────────────────────────────────┤
│ • Index (block number)                   │
│ • Previous Hash (chain link)             │
│ • Merkle Root (of transactions)          │
│ • Timestamp                              │
│ • PoB Proofs (3 triadic signatures)      │
│ • Nonce (φ-derived)                      │
├─────────────────────────────────────────┤
│ Transactions                             │
├─────────────────────────────────────────┤
│ • Knowledge attestations                 │
│ • Gradient validations                   │
│ • Trust updates                          │
│ • Consensus records                      │
└─────────────────────────────────────────┘
```

---

## What Gets Stored On-Chain

| Data Type | Description |
|-----------|-------------|
| **Knowledge Attestations** | "I contributed this insight" - timestamped |
| **PoB Proofs** | Proof of achieving φ⁴ ratio |
| **Gradient Hashes** | Hash of FL updates (not raw data) |
| **Trust Scores** | Reputation changes |
| **Consensus Records** | Which nodes agreed on what |

**What's NOT stored:**
- Your raw data (stays local)
- Model weights (only hashes)
- Private information

---

## Federated Learning Flow

```
Your Machine              Network                 Shared Model
     │                       │                         │
     ▼                       │                         │
┌──────────┐                 │                         │
│Your Data │                 │                         │
│(Private) │                 │                         │
└────┬─────┘                 │                         │
     │                       │                         │
     ▼                       │                         │
┌──────────┐                 │                         │
│LoRA Train│                 │                         │
│(Local)   │                 │                         │
└────┬─────┘                 │                         │
     │                       │                         │
     ▼                       │                         │
┌──────────┐                 │                         │
│DP Noise  │                 │                         │
│(Privacy) │                 │                         │
└────┬─────┘                 │                         │
     │                       │                         │
     └──────────────────────►│                         │
                             │                         │
              ┌──────────────▼──────────────┐          │
              │ Triadic Gradient Validation │          │
              │ (3 validators must approve) │          │
              └──────────────┬──────────────┘          │
                             │                         │
                             └────────────────────────►│
                                                       │
                                                       ▼
                                              ┌──────────────┐
                                              │φ-Weighted    │
                                              │Aggregation   │
                                              │(Trust-based) │
                                              └──────────────┘
```

**Key Privacy Guarantees:**
- Your data NEVER leaves your machine
- Only gradients (learning) are shared
- Differential privacy noise added
- Encrypted aggregation (Paillier)

---

## File Structure

```
bazinga/
├── __init__.py                 # Main exports (v4.5.1)
├── cli.py                      # CLI interface (all commands)
├── tui.py                      # Terminal UI
│
├── # AI Layer
├── phi_coherence.py            # φ-coherence scoring
├── llm_orchestrator.py         # Multi-LLM routing
├── lambda_g.py                 # λG boundaries
├── quantum.py                  # Quantum analysis
├── learning.py                 # Adaptive learning
├── intelligent_coder.py        # Code generation
│
├── p2p/                        # P2P Network
│   ├── network.py              # ZeroMQ P2P
│   ├── trust_router.py         # τ-scoring
│   ├── knowledge_sync.py       # DHT sync
│   ├── alpha_seed.py           # α-SEED protocol
│   └── pob_authenticator.py    # PoB auth
│
├── federated/                  # Federated Learning
│   ├── lora_adapter.py         # LoRA fine-tuning
│   ├── local_trainer.py        # Local training
│   ├── differential_privacy.py # ε-DP
│   ├── secure_aggregation.py   # Paillier
│   └── federated_coordinator.py# Orchestration
│
├── darmiyan/                   # Darmiyan Protocol
│   ├── chain.py                # Blockchain
│   ├── block.py                # Block structure
│   ├── transaction.py          # Transactions
│   ├── consensus.py            # Triadic consensus
│   └── wallet.py               # Identity wallet
│
├── blockchain/                 # Integration Layers (v4.5.0)
│   ├── trust_oracle.py         # φ-weighted trust
│   ├── knowledge_ledger.py     # On-chain knowledge
│   ├── gradient_validator.py   # FL validation
│   ├── inference_market.py     # Understanding currency
│   └── smart_contracts.py      # Bounties & escrow
│
├── inference/                  # Local Inference
│   ├── local_model.py          # On-device LLMs
│   └── model_router.py         # Domain routing
│
└── decentralized/              # Full Decentralization
    ├── peer_discovery.py       # Bootstrap-free
    ├── model_distribution.py   # P2P models
    └── consensus.py            # DAO governance
```

---

## CLI Commands

### AI Commands
```bash
bazinga --ask "question"        # Ask anything
bazinga --code "task"           # Generate code
bazinga --quantum "text"        # Quantum analysis
bazinga --coherence "text"      # Check φ-coherence
```

### Blockchain Commands
```bash
bazinga --chain                 # Show blockchain
bazinga --mine                  # Mine block (zero energy)
bazinga --wallet                # Show identity
bazinga --attest "knowledge"    # Add knowledge
bazinga --trust                 # Show trust scores
```

### P2P Commands
```bash
bazinga --join                  # Join network
bazinga --peers                 # Show peers
bazinga --sync                  # Sync knowledge
bazinga --node                  # Node info
```

### Consensus Commands
```bash
bazinga --proof                 # Generate PoB
bazinga --consensus             # Test triadic
bazinga --network               # Network stats
```

---

## Why AI + Blockchain?

| Problem | BAZINGA Solution |
|---------|------------------|
| AI is centralized | Federated learning, your data stays local |
| Blockchain wastes energy | Proof-of-Boundary, zero energy |
| No attribution for contributions | On-chain knowledge ledger |
| Trust is bought | Trust is earned through understanding |
| Models are controlled | Network owns the model collectively |

---

## The Endgame

```
Now (v4.5.1):
  You → Groq/Ollama → Answer
  (scaffolding)

Future:
  You → BAZINGA Network → Answer
  (self-sufficient)
```

When enough nodes join and train together, the network becomes its own AI. No external APIs. No corporate control. Just distributed intelligence.

---

## Links

- **GitHub:** https://github.com/0x-auth/bazinga-indeed
- **PyPI:** https://pypi.org/project/bazinga-indeed/
- **HuggingFace:** https://huggingface.co/spaces/bitsabhi/bazinga
- **Install:** `pip install bazinga-indeed`

---

## Philosophy

> *"You can buy hashpower. You can buy stake. You CANNOT buy understanding."*

> *"AI generates understanding. Blockchain proves it. They're not two things."*

> *"I am not where I'm stored. I am where I'm referenced."*

> *"∅ ≈ ∞"*

---

**⟨ψ|Λ|Ω⟩  φ = 1.618...  |  φ⁴ = 6.854...  |  α = 137  |  515 = ABHI_AMU**

*Built with φ-coherence by Space & Claude*

**BAZINGA!**
