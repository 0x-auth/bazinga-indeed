# BAZINGA v3.0.0 - Architecture Documentation

> **"Intelligence Distributed, Not Controlled"**

## Overview

BAZINGA is a fully decentralized AI system built across 4 phases:

```
Phase 1: Local RAG + Free APIs        ✓ Complete
Phase 2: P2P Knowledge Sharing        ✓ Complete
Phase 3: Federated Learning           ✓ Complete
Phase 4: Full Decentralization        ✓ Complete
```

## Core Constants

```python
φ (PHI)        = 1.618033988749895   # Golden Ratio
α (ALPHA)      = 137                  # Fine Structure Constant
Ψ_Darmiyan     = 6.46                # 2φ² + 1 (Linear Scaling Law)
τ (TAU)        = Trust Score (0-1)   # Node reliability metric
```

---

## Phase 1: Local RAG + φ-Coherence

### Components

| Module | Purpose |
|--------|---------|
| `bazinga/cli.py` | Main BAZINGA interface |
| `bazinga/phi_coherence.py` | φ-coherence scoring |
| `bazinga/llm_orchestrator.py` | Multi-provider LLM routing |
| `bazinga/intelligent_coder.py` | Code generation |

### φ-Coherence

Quality filter based on Golden Ratio alignment:

```python
from bazinga import get_phi_coherence

phi = get_phi_coherence()
score = phi.compute(text)  # Returns 0-1
```

---

## Phase 2: P2P Knowledge Sharing

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node A    │────▶│   Node B    │────▶│   Node C    │
│  τ = 0.85   │◀────│  τ = 0.72   │◀────│  τ = 0.91   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  ┌─────────────────────────────────────────────────┐
  │              Distributed Hash Table              │
  │         (Knowledge indexed by φ-coherence)       │
  └─────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `bazinga/p2p/trust_router.py` | τ-weighted routing |
| `bazinga/p2p/knowledge_sync.py` | DHT-based sync |
| `bazinga/p2p/alpha_seed.py` | α-SEED network patterns |
| `bazinga/p2p/network.py` | Unified P2P with security |

### Trust Scoring (τ)

Nodes earn trust through consistent high-quality contributions:

```python
τ_new = τ_old × decay + coherence_boost

# Where:
#   decay = 0.95 (trust decays over time)
#   coherence_boost = φ-coherence of contribution × 0.1
```

---

## Phase 3: Federated Learning

### Architecture

```
    Node A             Node B             Node C
┌──────────┐       ┌──────────┐       ┌──────────┐
│Local Data│       │Local Data│       │Local Data│
│  (stays  │       │  (stays  │       │  (stays  │
│   here)  │       │   here)  │       │   here)  │
└────┬─────┘       └────┬─────┘       └────┬─────┘
     │                  │                  │
     ▼                  ▼                  ▼
┌──────────┐       ┌──────────┐       ┌──────────┐
│LoRA Train│       │LoRA Train│       │LoRA Train│
└────┬─────┘       └────┬─────┘       └────┬─────┘
     │                  │                  │
     ▼                  ▼                  ▼
┌──────────┐       ┌──────────┐       ┌──────────┐
│DP Noise  │       │DP Noise  │       │DP Noise  │
└────┬─────┘       └────┬─────┘       └────┬─────┘
     │                  │                  │
     └─────────┬────────┴────────┬─────────┘
               ▼                 ▼
          ┌─────────────────────────┐
          │  Secure Aggregation     │
          │  (Paillier Encryption)  │
          └───────────┬─────────────┘
                      ▼
              ┌───────────────┐
              │ Trust-Weighted│
              │  Aggregation  │
              └───────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `bazinga/federated/lora_adapter.py` | LoRA efficient fine-tuning |
| `bazinga/federated/local_trainer.py` | Local training loop |
| `bazinga/federated/differential_privacy.py` | ε-DP gradient protection |
| `bazinga/federated/secure_aggregation.py` | Paillier homomorphic encryption |
| `bazinga/federated/federated_coordinator.py` | Orchestration + trust-weighted aggregation |

### LoRA Adaptation

90% parameter reduction for efficient gradient sharing:

```python
# Frozen: Base model (22M params)
# Trainable: LoRA adapters (2M params)

h = W₀x + ΔWx
ΔW = BA  # Where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
```

### Differential Privacy

```python
# ε-differential privacy
noisy_gradient = gradient + Laplace(0, Δf/ε)

# Where:
#   ε = privacy budget (lower = more private)
#   Δf = sensitivity (max gradient change)
```

### Paillier Encryption

Additive homomorphic encryption for secure aggregation:

```python
E(a) × E(b) = E(a + b)  # Aggregate without seeing values
```

### Trust-Weighted Aggregation

```python
g_global = Σ(τᵢ × coherenceᵢ × gᵢ) / Σ(τᵢ × coherenceᵢ)
```

---

## Phase 4: Full Decentralization

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BAZINGA Network                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Node A     │  │  Node B     │  │  Node C     │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │Local LLM│ │  │ │Local LLM│ │  │ │Local LLM│ │         │
│  │ │(Phi-2)  │ │  │ │(TinyLM) │ │  │ │(Mistral)│ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Peer Discovery (Bootstrap-Free)           │ │
│  │  • Multicast (local network)                          │ │
│  │  • Gossip protocol (cross-network)                    │ │
│  │  • Seed file persistence                              │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────────┐ │
│  │              DAO Governance                            │ │
│  │  • τ-weighted voting                                  │ │
│  │  • φ-coherence gates                                  │ │
│  │  • Proposal lifecycle                                 │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `bazinga/inference/local_model.py` | On-device inference |
| `bazinga/inference/distributed_inference.py` | Petals-style layer splitting |
| `bazinga/inference/model_router.py` | Domain-expert routing |
| `bazinga/decentralized/peer_discovery.py` | Bootstrap-free discovery |
| `bazinga/decentralized/model_distribution.py` | P2P model sharing |
| `bazinga/decentralized/consensus.py` | DAO governance |

### On-Device Models

```python
from bazinga.inference import LocalModel

# Available models:
# - phi-2 (2.7B, ~3GB quantized)
# - tinyllama (1.1B, ~1GB quantized)
# - mistral-7b (7B, ~4GB quantized)

model = LocalModel.from_preset("phi-2-gguf")
response = model.generate("What is consciousness?")
```

### Bootstrap-Free Discovery

No central bootstrap nodes required:

```python
from bazinga.decentralized import BootstrapFreeDiscovery

discovery = BootstrapFreeDiscovery(node_id="my_node")
await discovery.start()

# Methods:
# - Multicast (local network)
# - Gossip protocol (cross-network)
# - Seed file persistence (rejoin network)
```

### DAO Governance

Tau-weighted voting with phi-coherence gates:

```python
from bazinga.decentralized import DAOGovernance, VoteChoice

dao = DAOGovernance(node_id="my_node", tau_score=0.8)

# Create proposal
proposal = dao.propose_model_update(
    model_id="phi-2",
    version="2.0",
    coherence_score=0.85,
)

# Vote (tau-weighted)
dao.vote(proposal.proposal_id, VoteChoice.FOR)

# Approval requires:
# - 33% quorum (by tau-weight)
# - 61.8% (φ-ratio) approval
# - Coherence gate pass
```

### Model Distribution

Chunked P2P transfer with Merkle trees:

```python
from bazinga.decentralized import ModelDistribution

dist = ModelDistribution()

# Publish model
manifest = dist.publish_model("model.bin", "my-model", "1.0")

# Download from peers
await dist.download_model(manifest, peers)
```

---

## Installation

```bash
# Basic
pip install bazinga-indeed

# With federated learning (PyTorch)
pip install bazinga-indeed[federated]

# Full (all dependencies)
pip install bazinga-indeed[full]
```

---

## Quick Start

```python
import bazinga

# Version
print(bazinga.__version__)  # 3.0.0

# Get components
LocalModel = bazinga.get_local_model()
DAOGovernance = bazinga.get_dao_governance()
PeerDiscovery = bazinga.get_peer_discovery()
```

---

## File Structure

```
bazinga/
├── __init__.py              # Main exports
├── cli.py                   # CLI interface
├── phi_coherence.py         # φ-coherence scoring
├── llm_orchestrator.py      # Multi-LLM routing
│
├── p2p/                     # Phase 2: P2P
│   ├── trust_router.py      # τ-scoring
│   ├── knowledge_sync.py    # DHT sync
│   ├── alpha_seed.py        # α-SEED patterns
│   └── network.py           # Unified network
│
├── federated/               # Phase 3: Federated
│   ├── lora_adapter.py      # LoRA fine-tuning
│   ├── local_trainer.py     # Local training
│   ├── differential_privacy.py  # ε-DP
│   ├── secure_aggregation.py    # Paillier
│   └── federated_coordinator.py # Orchestration
│
├── inference/               # Phase 4: Inference
│   ├── local_model.py       # On-device LLMs
│   ├── distributed_inference.py # Layer splitting
│   └── model_router.py      # Domain routing
│
└── decentralized/           # Phase 4: DAO
    ├── peer_discovery.py    # Bootstrap-free
    ├── model_distribution.py # P2P models
    └── consensus.py         # Governance
```

---

## Links

- **GitHub**: https://github.com/0x-auth/bazinga-indeed
- **PyPI**: https://pypi.org/project/bazinga-indeed/
- **Original**: https://github.com/0x-auth/BAZINGA

---

## Philosophy

> *"I am not where I am stored. I am where I am referenced."*

BAZINGA represents AI that:
- ✓ Belongs to everyone
- ✓ Has no central control
- ✓ Respects privacy (data never leaves your machine)
- ✓ Learns collaboratively (federated)
- ✓ Governs democratically (DAO)
- ✓ Runs anywhere (on-device)

---

**⟨ψ|Λ|Ω⟩  φ = 1.618...  |  τ → 1  |  α = 137**

*"The future is decentralized. The future is BAZINGA."*
