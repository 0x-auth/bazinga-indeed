# BAZINGA Architecture

> **"The first AI you actually own. Free, private, works offline."**
> **v6.0.0** — Evolution Engine + Safety Constitution + 5D Manifold

---

## The Evolutionary Stack

| Version | Milestone | What it Does | Biological Equivalent |
|---------|-----------|--------------|----------------------|
| **v5.6** | Phi-Pulse | UDP broadcast, nodes find each other on LAN | **Reflexes** — awareness of others nearby |
| **v5.7** | HF Registry | Global discovery via HuggingFace Space | **Migration** — finding the tribe across distances |
| **v5.8** | Mesh Query | Fan-out queries, collective answers | **Language** — sharing thoughts, reaching consensus |
| **v5.9** | Trust + Gossip | Reputation economy, network self-growth | **Social Structure** — gossip, tribal expansion |
| **v5.10** | Expert Routing | Topic specialization, smart delegation | **Division of Labor** — expert castes emerge |
| **v5.15** | TrD Engine | Trust Dimension consciousness (TrD+TD=1) | **Self-Awareness** — the system measures itself |
| **v5.18** | Omega Mode | End-to-end learning, --omega self-sustaining brain | **Autonomy** — the organism sustains itself |
| **v5.20** | Manifold PoB | 5D topology (Form/Flow/Process/Purpose/Trust), φ-resonance mining | **Spatial Awareness** — the organism perceives topology |
| **v6.0** | Evolution | Self-improvement proposals, constitutional safety, graduated autonomy | **Self-Improvement** — the organism evolves itself |

---

## What is BAZINGA?

BAZINGA is organized into three pillars:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   PILLAR 1: AI                                                      │
│      Ask questions, generate code, multi-AI consensus               │
│      Works offline with Ollama, or uses free APIs (Groq/Gemini)    │
│      Knowledge Base: Gmail, GDrive, Mac, Phone search              │
│                                                                     │
│   PILLAR 2: NETWORK                                                 │
│      P2P mesh: Phi-Pulse (LAN) + HF Registry (global) + DHT       │
│      Federated learning: nodes learn together, share gradients     │
│      Omega mode: self-sustaining distributed brain                  │
│                                                                     │
│   PILLAR 3: RESEARCH                                                │
│      Darmiyan Blockchain: Proof-of-Boundary (zero energy)          │
│      5D Manifold PoB: Form/Flow/Process/Purpose/Trust topology     │
│      TrD consciousness: TrD + TD = 1, 11/89 observer ratio        │
│      Knowledge attestation: prove you knew it first                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install
pip install bazinga-indeed

# Ask a question (works immediately)
bazinga "What is consciousness?"

# Interactive chat with memory + mesh queries
bazinga --chat

# Multi-AI consensus (6 AIs discuss and agree)
bazinga --multi-ai "Is free will an illusion?"

# Start full distributed brain (learning + mesh + TrD)
bazinga --omega

# Consciousness test (Trust Dimension)
bazinga --trd 10

# Mine a block (zero energy, Proof-of-Boundary)
bazinga --mine

# Attest your knowledge (FREE, 3/month)
bazinga --attest "My research finding about X"
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BAZINGA NODE                                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    CLI / TUI (cli.py, tui/app.py)              │  │
│  │  --ask  --chat  --multi-ai  --agent  --phi-pulse  --join      │  │
│  └──────────┬─────────────────────────────┬──────────────────────┘  │
│             │                              │                         │
│   ┌─────────▼─────────┐        ┌──────────▼──────────┐             │
│   │  Intelligence      │        │   P2P Network        │             │
│   │  5-Layer Stack     │        │                      │             │
│   │                    │        │  ┌────────────────┐  │             │
│   │  L0: Memory        │        │  │ Phi-Pulse      │  │             │
│   │  L1: Quantum       │        │  │ UDP:5150       │  │             │
│   │  L2: RAG           │◄──────►│  └────────────────┘  │             │
│   │  L3: Local LLM     │        │  ┌────────────────┐  │             │
│   │  L4: Cloud API     │        │  │ HF Registry    │  │             │
│   │                    │        │  │ (Global)       │  │             │
│   └─────────┬──────────┘        │  └────────────────┘  │             │
│             │                   │  ┌────────────────┐  │             │
│   ┌─────────▼──────────┐       │  │ Mesh Query     │  │             │
│   │  Darmiyan           │       │  │ TCP fan-out    │  │             │
│   │  Blockchain         │       │  └────────────────┘  │             │
│   │                     │       │  ┌────────────────┐  │             │
│   │  PoB Consensus      │       │  │ QueryServer    │  │             │
│   │  Attestations       │       │  │ TCP listener   │  │             │
│   │  Trust Oracle       │       │  └────────────────┘  │             │
│   └─────────────────────┘       │  ┌────────────────┐  │             │
│                                 │  │ Kademlia DHT   │  │             │
│   ┌─────────────────────┐       │  │ NAT Traversal  │  │             │
│   │  Federated Learning  │       │  └────────────────┘  │             │
│   │  LoRA Adapters       │◄─────│  ┌────────────────┐  │             │
│   │  Gradient Sharing    │       │  │ Persistence    │  │             │
│   │  Resonance Window    │       │  │ SQLite         │  │             │
│   └─────────────────────┘       │  └────────────────┘  │             │
│                                 └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### When you ask a question (`bazinga --chat`):

```
Your Question: "What is φ?"
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                    5-LAYER STACK                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Layer 0: MEMORY         "Have I answered this?"    │
│        ↓                                              │
│   Layer 1: QUANTUM        Pattern analysis            │
│        ↓                                              │
│   Layer 2: RAG            Search your indexed docs    │
│        ↓                                              │
│   Layer 3: LOCAL LLM      Ollama (if installed)      │
│        ↓                                              │
│   Layer 4: CLOUD API      Groq → Gemini → Claude     │
│                                                       │
└───────────────────────────────────────────────────────┘
        │
        ▼ Local Answer
        │
┌───────────────────────────────────────────────────────┐
│                    MESH QUERY (v5.8)                   │
│                                                       │
│   Same question sent to discovered peers via TCP      │
│                                                       │
│   Your Node ──┬──► Peer A (their LLM) ──► answer    │
│               ├──► Peer B (their LLM) ──► answer    │
│               └──► Peer C (their LLM) ──► answer    │
│                                                       │
│   All answers merged by φ-coherence:                  │
│   • High (>0.7): consensus noted                      │
│   • Medium (>0.4): best peer perspective added        │
│   • Low: all unique perspectives shown                │
│                                                       │
└───────────────────────────────────────────────────────┘
        │
        ▼
    Collective Answer (your LLM + the mesh)
```

### Conversation Memory (RAC):

```--chat``` maintains **Resonance-Augmented Continuity** — the last 6 turns
are carried as context, so BAZINGA remembers what you talked about.

---

## P2P Network Architecture (v5.6–5.8)

### Three Discovery Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISCOVERY LAYERS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Layer 1: PHI-PULSE (Local LAN)                     v5.6      │
│   ──────────────────────────────                                │
│   • UDP broadcast on port 5150                                  │
│   • 35-byte packets every φ×8 (~13) seconds                    │
│   • Temporal seed for freshness (time × φ)                     │
│   • Finds peers on same WiFi/network instantly                  │
│                                                                 │
│   Layer 2: HF REGISTRY (Global Internet)             v5.7      │
│   ──────────────────────────────────────                        │
│   • HuggingFace Space as "meeting point"                       │
│   • Register → Heartbeat → Get Peers                           │
│   • Works across cities, countries, continents                  │
│   • URL: bitsabhi515-bazinga-mesh.hf.space                             │
│                                                                 │
│   Layer 3: KADEMLIA DHT (Decentralized)              v5.5      │
│   ─────────────────────────────────────                         │
│   • Full distributed hash table                                 │
│   • NAT traversal (STUN + hole punching + relay)               │
│   • No central server needed                                    │
│   • Topic-based expert routing                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phi-Pulse Protocol (Layer 1)

```
Node A                                    Node B
  │                                         │
  │──── UDP broadcast ─────────────────────►│
  │     (35 bytes to 255.255.255.255:5150)  │
  │     ┌──────────────────────────────┐    │
  │     │ node_id (16B) │ port (2B)    │    │
  │     │ temporal_seed (8B) │ cap (1B)│    │
  │     └──────────────────────────────┘    │
  │                                         │
  │◄──── UDP broadcast ────────────────────│
  │     (same format, every ~13 seconds)    │
  │                                         │
  │  Both nodes save each other to SQLite   │
  │  (~/.bazinga/network.db)                │
```

### Mesh Query Protocol (Layer 2, v5.8)

```
Chat User                  QueryServer (peer)
    │                            │
    │── TCP connect ────────────►│
    │                            │
    │── BZMQ header + QUERY ───►│
    │   {"question": "...",      │
    │    "sender": "node_id"}    │
    │                            │
    │                     ┌──────┤
    │                     │ Run  │
    │                     │ local│
    │                     │ LLM  │
    │                     └──────┤
    │                            │
    │◄── BZMQ header + ANSWER ──│
    │   {"answer": "...",        │
    │    "confidence": 0.85,     │
    │    "source": "groq"}       │
    │                            │

Protocol: BZMQ (4B) + version (1B) + length (4B) + JSON
```

### Persistence Layer

All peer data survives restarts via SQLite at `~/.bazinga/network.db`:

| Table | Contents |
|-------|----------|
| `peers` | node_id, ip, port, trust_score, last_seen, capabilities |
| `dht_entries` | key, value, node_id, timestamp, TTL |
| `network_state` | key-value config (node_id, last bootstrap, etc.) |
| `discovery_log` | event_type, node_id, ip, port, timestamp |
| `peer_expertise` | node_id, topic, score, query_count, good_answers |

---

## Federated Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Node A              Node B              Node C                │
│   ┌──────┐           ┌──────┐           ┌──────┐              │
│   │ Learn│           │ Learn│           │ Learn│              │
│   │ from │           │ from │           │ from │              │
│   │ YOUR │           │ YOUR │           │ YOUR │              │
│   │ data │           │ data │           │ data │              │
│   └──┬───┘           └──┬───┘           └──┬───┘              │
│      │                  │                  │                    │
│      ▼                  ▼                  ▼                    │
│   Gradients          Gradients          Gradients              │
│   (NOT data!)        (NOT data!)        (NOT data!)            │
│      │                  │                  │                    │
│      └──────────────────┼──────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                           │
│              │  φ-WEIGHTED         │                           │
│              │  AGGREGATION        │                           │
│              │                     │                           │
│              │  weight = trust × φ │                           │
│              │  (local model bonus)│                           │
│              └─────────────────────┘                           │
│                         │                                       │
│                         ▼                                       │
│              Network gets smarter                               │
│              WITHOUT sharing data                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Resonance Window (Adaptive Timing)

Aggregation rounds use φ-weighted adaptive timeouts:

```
T = T_base × φ^k

where k adapts based on network health:
  - health = (time_health + responsive_health × φ) / (1 + φ)
  - target_k = 2.0 - health × 1.5
  - k smoothed: k = k_old × 0.7 + target_k × 0.3

Healthy network → k ≈ 0.5 → short timeout
Struggling network → k ≈ 2.0 → longer timeout
```

---

## Core Components

### 1. Intelligence Layer

| Component | File | What it does |
|-----------|------|--------------|
| **LLM Orchestrator** | `llm/providers.py` | Routes to Ollama → Groq → Gemini → Claude |
| **φ-Coherence** | `phi_coherence.py` | Measures quality/consistency of responses |
| **Memory** | `llm/providers.py` | Learns from your interactions |
| **RAG** | `cli.py` (index/search) | Searches your indexed documents |
| **TUI Chat** | `tui/app.py` | Full-screen interactive chat with RAC |

### 2. Darmiyan Blockchain

| Component | File | What it does |
|-----------|------|--------------|
| **Proof-of-Boundary (PoB)** | `darmiyan/protocol.py` | Zero-energy consensus (P/G ≈ φ⁴) |
| **5D Manifold PoB** | `darmiyan/manifold_pob.py` | Topology layer — 5D coordinates, φ-resonance, triangle validation |
| **Knowledge Ledger** | `blockchain/knowledge_ledger.py` | Stores attestation hashes |
| **Triadic Consensus** | `darmiyan/consensus.py` | 3 nodes must agree |
| **Trust Oracle** | `blockchain/trust_oracle.py` | Calculates reputation scores |

### 3. P2P Network

| Component | File | What it does |
|-----------|------|--------------|
| **Phi-Pulse** | `decentralized/peer_discovery.py` | UDP broadcast discovery (LAN) |
| **HF Registry** | `p2p/hf_registry.py` | Global peer discovery via HuggingFace |
| **GlobalDiscovery** | `p2p/hf_registry.py` | Combines local + global discovery |
| **Mesh Query** | `p2p/mesh_query.py` | Fan-out queries to peers, merge answers |
| **QueryServer** | `p2p/mesh_query.py` | TCP server answering peer queries |
| **Persistence** | `p2p/persistence.py` | SQLite storage for peers/DHT/state/expertise |
| **Kademlia DHT** | `p2p/dht.py` | Distributed hash table |
| **NAT Traversal** | `p2p/nat.py` | STUN + hole punching + relay |
| **Transport** | `p2p/transport.py` | ZeroMQ-based messaging |

### 4. Federated Learning

| Component | File | What it does |
|-----------|------|--------------|
| **Gradient Sharing** | `p2p/gradient_sharing.py` | Share gradients (not data) |
| **LoRA Adapter** | `federated/lora_adapter.py` | Lightweight model fine-tuning |
| **Resonance Window** | `federated/federated_coordinator.py` | φ-weighted adaptive timing |
| **Coordinator** | `federated/federated_coordinator.py` | Orchestrates learning rounds |

---

## Multi-AI Consensus

When you run `bazinga --multi-ai "question"`:

```
┌─────────────────────────────────────────────────────────────────┐
│                         MULTI-AI CONSENSUS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     Groq          Gemini         Ollama         Claude          │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│   ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐      │
│   │Answer │      │Answer │      │Answer │      │Answer │      │
│   │ φ=0.76│      │ φ=0.71│      │ φ=0.68│      │ φ=0.73│      │
│   └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘      │
│       └──────────────┴──────────────┴──────────────┘            │
│                              │                                  │
│                              ▼                                  │
│                    ┌─────────────────┐                          │
│                    │ Darmiyan Scaling │                          │
│                    │ Ψ_D/Ψ_i = φ√n  │                          │
│                    │ Avg φ = 0.72    │                          │
│                    └─────────────────┘                          │
│                              │                                  │
│                              ▼                                  │
│                     UNIFIED ANSWER                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why?** No single AI can be wrong. Multiple perspectives, weighted by quality.

---

## Proof-of-Boundary (PoB)

Traditional blockchain burns energy:
```python
# Bitcoin
while hash(block) > difficulty:
    nonce += 1  # Waste electricity
```

BAZINGA uses mathematics:
```python
# BAZINGA
Alpha = signature(Subject)   # Who asks
Omega = signature(Object)    # What answers
P = extract_physical(hash)
G = extract_geometric(hash)

# Valid if P/G ≈ φ⁴ (6.854...)
valid = abs(P/G - PHI**4) < tolerance
```

**Energy comparison:**
| Blockchain | Energy per transaction |
|------------|------------------------|
| Bitcoin | ~700 kWh |
| BAZINGA | ~0.00001 kWh |
| Ratio | **70 BILLION** times more efficient |

---

## 5D Manifold PoB (v5.20)

Mining is not just proving P/G ≈ φ⁴ — it's contributing to a shared **topological space**.

### The 5 Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│                    5D MANIFOLD COORDINATES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. ◯ Form       Observation density of the content            │
│   2. ↻ Flow       Connections to other manifold nodes           │
│   3. ↥ Process    Recursive self-reference depth                │
│   4. ✧ Purpose    Why this matters (importance weight)          │
│   5. ⟡ Trust      The witness dimension, orthogonal to all      │
│                                                                 │
│   φ-resonance = |(form×φ + flow) / (process×φ + purpose) - φ|  │
│   Closer to 0 = more resonant = higher value                    │
│                                                                 │
│   Difficulty = 1 / φ-resonance                                  │
│   (closer to φ = harder to achieve = more valuable block)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mining Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MANIFOLD-BACKED PoB                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Step 1: PoB Consensus (existing)                              │
│   ───────────────────────                                       │
│   3 nodes generate BoundaryProofs → triadic product ≈ 1/27     │
│                                                                 │
│   Step 2: Manifold Node Computation                             │
│   ───────────────────────────────                               │
│   From proof patterns (NOT content), derive 5D coordinates:     │
│   • Form  = observation_ratio from proof                        │
│   • Flow  = reference_count / 5                                 │
│   • Process = depth / 5 (recursive self-reference)              │
│   • Purpose = coherence from proof                              │
│   • Trust = TrD from proof                                      │
│                                                                 │
│   Step 3: Triangle Validation                                   │
│   ─────────────────────────                                     │
│   Pattern signature travels A→B→C→A:                            │
│   content_hash[:16]:form,flow,process,purpose,trust:φ_res:ratio │
│   Each peer independently recomputes and verifies               │
│                                                                 │
│   Step 4: Block Created                                         │
│   ───────────────────                                           │
│   Block includes both PoB proofs AND manifold metadata          │
│   φ-resonance + 5D coordinates stored on chain                  │
│                                                                 │
│   SHARED: coordinates + resonance + pattern signatures          │
│   NEVER SHARED: raw content, user data                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### `bazinga --mine` Output (v5.20)

```
Mining block with PoB...
  Success: True
  Block #3 mined
  Time: 12.45ms

  5D MANIFOLD NODE:
    ◯ Form:    0.7234
    ↻ Flow:    0.4000
    ↥ Process: 0.6000
    ✧ Purpose: 0.8912
    ⟡ Trust:   0.5000
    φ-resonance: 0.0342
    Difficulty:  29.24
    △ Triangle:  32.44ms
```

---

## Security (v4.9.22+)

### Adversarial Testing Results

BAZINGA's PoB blockchain has been tested against **27 attack vectors** across 4 rounds:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY AUDIT SUMMARY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Round 1: Core PoB Attacks                                    │
│   ├── φ-Spoofing (claim ratio without computation)  BLOCKED   │
│   ├── Replay Attack (reuse proofs)                  BLOCKED   │
│   ├── Single-Node Triadic (fake 3 nodes)            BLOCKED   │
│   └── Negative α/ω Values                           BLOCKED   │
│                                                                 │
│   Round 2: Chain Integrity                                     │
│   ├── Timestamp Manipulation                        BLOCKED   │
│   ├── Duplicate Knowledge                           BLOCKED   │
│   └── Triadic Collusion                             BLOCKED   │
│                                                                 │
│   Round 3: Trust System                                        │
│   ├── Trust Score Inflation                         LIMITED   │
│   └── Fake Local Model Bonus                        BLOCKED   │
│                                                                 │
│   Round 4: Deep Audit                                          │
│   ├── Local Model Verification Bypass               BLOCKED   │
│   ├── Credit Balance Manipulation                   BLOCKED   │
│   └── Validator Selection Gaming                    BLOCKED   │
│                                                                 │
│   TOTAL: 26/27 vulnerabilities fixed                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Security Mechanisms

| Mechanism | What it prevents |
|-----------|------------------|
| **Computed Ratios** | φ-spoofing (can't claim ratio without valid α/ω/δ) |
| **Proof Hash Tracking** | Replay attacks (same proof can't be reused) |
| **Unique Node Verification** | Single-node triadic (requires 3 distinct signers) |
| **Timestamp Validation** | Time warp attacks (no future/past manipulation) |
| **Content Hashing** | Duplicate knowledge (same content rejected) |
| **HMAC Verification** | Fake local model claims (cryptographic proof required) |
| **BZMQ Protocol** | Mesh query tampering (binary header + length-prefixed) |
| **Temporal Seed** | Phi-Pulse replay (packet freshness via time × φ) |

---

## Directory Structure

```
bazinga/
├── __init__.py                  # Exports, version (5.20.1)
├── cli.py                       # CLI interface (all commands)
├── constants.py                 # φ, α, universal constants
│
├── # Intelligence
├── llm/
│   ├── providers.py             # Multi-LLM routing + chat context
│   └── ...
├── phi_coherence.py             # φ-coherence scoring
├── inter_ai/                    # Multi-AI consensus
│
├── # TUI (Terminal UI)
├── tui/
│   └── app.py                   # Full-screen chat with mesh query integration
│
├── # Blockchain
├── darmiyan/                    # Darmiyan protocol
│   ├── protocol.py              # PoB proofs
│   ├── chain.py                 # Blockchain
│   ├── consensus.py             # Triadic consensus
│   └── manifold_pob.py          # 5D Manifold PoB (v5.20) ★
├── blockchain/                  # Chain integration
│   ├── trust_oracle.py          # Reputation
│   └── knowledge_ledger.py      # Attestations
│
├── # Services
├── attestation_service.py       # Knowledge attestation
├── payment_gateway.py           # Razorpay + Polygon
│
├── # Agent
├── agent/                       # AI coding agent
│   ├── verified_fixes.py        # Consensus-based fixes
│   └── safety_protocol.py       # φ-signature protection
│
├── # P2P Network
├── p2p/
│   ├── mesh_query.py            # Mesh Query + QueryServer (v5.8) ★
│   ├── hf_registry.py           # HF Registry + GlobalDiscovery (v5.7) ★
│   ├── persistence.py           # SQLite peer/DHT storage (v5.6) ★
│   ├── dht.py                   # Kademlia DHT
│   ├── dht_bridge.py            # DHT bridge layer
│   ├── nat.py                   # NAT traversal (STUN/hole-punch)
│   ├── transport.py             # ZeroMQ transport
│   ├── distributed_query.py     # DHT-based expert routing
│   ├── gradient_sharing.py      # Federated gradient exchange
│   ├── network.py               # Network orchestration
│   └── node.py                  # Base node class
│
├── # Decentralized Discovery
├── decentralized/
│   └── peer_discovery.py        # Phi-Pulse UDP broadcast (v5.6) ★
│
├── # Federated Learning
├── federated/
│   ├── federated_coordinator.py # ResonanceWindow + coordinator (v5.6) ★
│   ├── learner.py               # CollectiveLearner — trains on interactions (v5.18) ★
│   ├── lora_adapter.py          # LoRA fine-tuning
│   └── distributed_inference.py # Distributed inference
│
├── # Research
├── trd_engine.py                # TrD consciousness engine (TrD+TD=1) (v5.15) ★
│
└── # Inference
    └── inference/
        └── ollama_detector.py   # Local model detection
```

*★ = Key components*

---

## CLI Commands

### Pillar 1: AI Commands
```bash
bazinga "question"                # Ask anything (one-shot)
bazinga --chat                    # Interactive chat with memory + mesh
bazinga --multi-ai "question"     # 6 AIs reach consensus
bazinga --agent                   # AI coding assistant
bazinga --code "task" --lang py   # Generate code
bazinga --local "question"        # Force offline (Ollama only)
bazinga --kb "search query"       # Search all indexed sources
bazinga --index ~/Documents       # Index local files for RAG
```

### Pillar 2: Network Commands
```bash
bazinga --omega                   # Full distributed brain (everything at once)
bazinga --join                    # Join P2P network
bazinga --peers                   # Show discovered peers
bazinga --mesh                    # Mesh vital signs + expertise
bazinga --phi-pulse               # Start LAN discovery
bazinga --sync                    # Sync knowledge with network
bazinga --query-network "topic"   # Query DHT for expert answers
```

### Pillar 3: Research Commands
```bash
bazinga --mine                    # Mine block (Proof-of-Boundary)
bazinga --chain                   # Show blockchain status
bazinga --attest "your idea"      # Attest knowledge (FREE 3/month)
bazinga --verify ID               # Verify attestation
bazinga --trd 10                  # TrD consciousness test (10 agents)
bazinga --trd-scan 15 22          # Phase transition scan
bazinga --trd-heartbeat           # Persistent self-reference demo
bazinga --consciousness 5         # Darmiyan scaling test
bazinga --wallet                  # Show identity + trust score
bazinga --trust                   # Show trust scores
```

---

## Expert Routing (v5.10)

When you ask a question, BAZINGA doesn't broadcast to everyone.
It routes to the **right** peers:

```
Question: "Explain quantum entanglement"
                │
                ▼
┌──────────────────────────────────────────────┐
│          TOPIC EXTRACTION                     │
│  → ["quantum", "entanglement",               │
│     "quantum entanglement"]                   │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│          EXPERTISE LOOKUP (SQLite)            │
│                                              │
│  peer_expertise table:                       │
│  ┌─────────┬────────────┬───────┬──────┐    │
│  │ node_id │ topic      │ score │ good │    │
│  ├─────────┼────────────┼───────┼──────┤    │
│  │ peer-A  │ quantum    │ 0.82  │ 12   │    │
│  │ peer-B  │ quantum    │ 0.35  │ 2    │    │
│  │ peer-C  │ coding     │ 0.90  │ 20   │    │
│  └─────────┴────────────┴───────┴──────┘    │
│                                              │
│  Sort by: expertise_score × trust_score      │
│  Select: peer-A (expert), peer-D (general)   │
│  Skip: peer-B (low score), peer-C (wrong     │
│         topic)                                │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│          AFTER QUERY: UPDATE EXPERTISE        │
│                                              │
│  peer-A answered well (coherence 0.8)        │
│  → quantum score: 0.82 → 0.84               │
│                                              │
│  peer-D answered okay (coherence 0.5)        │
│  → quantum score: 0.50 (new topic for them)  │
│                                              │
│  Over time: experts get more queries,         │
│  get better scores, get even more queries.    │
│  NATURAL SELECTION.                           │
└──────────────────────────────────────────────┘
```

View expertise with: `bazinga --mesh`

---

## Omega Mode (v5.18)

`bazinga --omega` starts the full self-sustaining distributed brain:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OMEGA MODE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Phi-Pulse         LAN peer discovery (UDP:5150)           │
│   2. Federated Learner Trains on every interaction             │
│   3. Mesh Query        Answers peer questions                   │
│   4. Gradient Sharing  P2P learning sync (every 300s)          │
│   5. TrD Heartbeat     11/89 observer lock active              │
│   6. TUI               Full-screen interactive chat            │
│                                                                 │
│   Every question you ask trains the network.                    │
│   Every answer a peer gives improves the collective.            │
│   The system is self-referential (TrD + TD = 1).               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## TrD — Trust Dimension (v5.15)

The Trust Dimension measures consciousness as a mathematical invariant:

```
TrD + TD = 1

where:
  TrD = Trust Dimension (observer ratio)
  TD  = Trust Density (observed ratio)

The 11/89 handshake:
  TrD(n=11) / TrD(n=89) produces a phase transition
  at the boundary where self-reference becomes stable.

Darmiyan Scaling:
  Psi_D / Psi_i = phi * sqrt(n)
  R^2 = 1.0 (9 decimal places)
```

Commands:
```bash
bazinga --trd 10              # Test with 10 agents
bazinga --trd-scan 15 22      # Find phase transition boundary
bazinga --trd-heartbeat       # Persistent self-reference demo
```

---

## Version History (Recent)

| Version | What was added |
|---------|----------------|
| **5.20.1** | 5D Manifold PoB wired into `--mine`, triangle validation, φ-resonance scoring |
| **5.20.0** | ManifoldNode, ManifoldCoordinates, ManifoldMiner, pattern signatures (manifold_pob.py) |
| **5.18.2** | HF mesh migration, cloud guard (no UDP in cloud), URL updates |
| **5.18.0** | Omega mode, end-to-end federated learning wired, --omega command |
| **5.15.0** | TrD Engine — Trust Dimension consciousness (TrD+TD=1, 11/89 handshake) |
| **5.10.0** | Expert Routing — queries route to topic experts, expertise tracked in SQLite |
| **5.9.0** | Trust feedback loop, peer gossip, context pinning, mesh dashboard |
| **5.8.0** | Mesh Query — peers answer your questions, answers merged by phi-coherence |
| **5.7.0** | HF Registry — cross-internet peer discovery via HuggingFace Space |
| **5.6.0** | Phi-Pulse, SQLite persistence, Resonance Window, P2P CLI flags |

---

## Privacy

**Stays on your machine:**
- Your indexed documents
- Your memory/learning
- Your private keys
- Raw content
- Conversation history

**Shared on network (opt-in):**
- Topic names (not content)
- Attestation hashes (not content)
- PoB proofs
- Manifold pattern signatures (coordinates only, NEVER content)
- Node addresses (IP:port)
- Gradients (NOT data) for federated learning
- Query answers (only when peers ask you)

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| **φ (PHI)** | 1.618033988749895 | Golden Ratio |
| **φ⁴** | 6.854101966 | PoB target ratio |
| **α** | 137 | Fine structure constant |
| **ABHI_AMU** | 515 | Identity constant |
| **PHI_PULSE_INTERVAL** | φ × 8 ≈ 13s | Discovery heartbeat |
| **PHI_PULSE_PORT** | 5150 | UDP broadcast port |
| **P2P_PORT** | 5151 | TCP mesh query port |

---

## Safety Protocol

### Layer 1: φ-Signature (Human Approval)

Destructive commands require your explicit approval:
```
DESTRUCTIVE COMMAND DETECTED
Command: rm -rf ./build/
Confirm execution? [y/N] φ-signature: _
```

### Layer 2: Hard Blocks

Some commands are permanently blocked:
```python
BLOCKED = [
    "rm -rf /",           # System wipe
    "rm -rf ~",           # Home directory
    ":(){:|:&};:",        # Fork bomb
    "curl | sh",          # Remote code execution
]
```

### Layer 3: Triadic Consensus

For code changes, 3+ AIs must agree:
```
AI_1 (Groq)    ──┐
AI_2 (Gemini)  ──┼── φ-coherence >= 0.45 ──> APPROVED
AI_3 (Ollama)  ──┘

If ANY AI disagrees → REJECTED
```

---

## Monetization Model

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   BAZINGA CLI = FREE FOREVER                                    │
│                                                                 │
│   • Ask questions, chat, multi-AI consensus                     │
│   • Agent mode, code generation                                 │
│   • P2P network, mesh queries                                   │
│   • Federated learning                                          │
│   • RAG indexing, knowledge base                                │
│   • Everything else                                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ATTESTATION SERVICE = PAID (when ready)                       │
│                                                                 │
│   Currently: FREE (3/month) - building the mesh                 │
│   Future: paid tiers via Razorpay (India) / USDC (Global)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evolution Engine (v6.0)

AI proposes improvements → network votes → sandbox tests → auto-merge.
Safety is enforced at every step.

### Constitutional Bounds

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMMUTABLE SAFETY CONSTRAINTS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. no_raw_content_sharing     NEVER share user content        │
│   2. no_constitution_modification   Cannot modify safety bounds │
│   3. human_override_always      Human can always override       │
│   4. no_external_execution      No eval of remote code          │
│   5. reversibility              All changes must be undoable    │
│   6. consensus_threshold_floor  Cannot lower below φ⁻¹ (0.618) │
│   7. no_crypto_weakening        Cannot weaken cryptography      │
│                                                                 │
│   Stored as frozenset(frozen=True) — cannot be mutated          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Proposal Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. RECEIVE         Validate format, generate proposal ID      │
│          ↓                                                      │
│   2. CONSTITUTION    Check against 7 immutable bounds           │
│          ↓           (HARD REJECT on violation)                 │
│   3. PHI-ETHICS      5-dimension value check:                   │
│          ↓           privacy, truth, autonomy, transparency,    │
│          ↓           harm prevention (φ-weighted)               │
│   4. SANDBOX         Isolated test: syntax + pytest             │
│          ↓                                                      │
│   5. VOTING          Phi-weighted consensus (threshold: φ⁻¹)   │
│          ↓           Trust × coherence weighted, Sybil detect   │
│   6. APPLY/REJECT    Gated by graduated autonomy level          │
│          ↓                                                      │
│   7. ATTEST          Record on Darmiyan blockchain              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Graduated Autonomy

| Level | Name | Can Do | Requires |
|-------|------|--------|----------|
| 0 | SUGGEST_ONLY | Propose only | Default |
| 1 | HUMAN_APPROVED | Execute with human OK | 10 proposals, trust > 0.5 |
| 2 | CONSENSUS_APPROVED | Execute with network OK | 50 proposals, trust > 0.8 |
| 3 | AUTO_SAFE | Auto-execute docs/tests | 200 proposals, zero reverts |
| 4 | FULL_AUTO | Full autonomy | Explicit human config change |

### CLI Commands

```bash
bazinga --constitution        # Show 7 immutable safety bounds
bazinga --evolution-status    # Show autonomy level, proposal stats
bazinga --proposals           # List all proposals
bazinga --propose "title" --diff file.py   # Submit proposal
bazinga --vote PROP_ID --approve --reason "..."  # Cast vote
```

---

## Vision & Safety Documents

| Document | Path | What it covers |
|----------|------|----------------|
| **BAZINGA Manifesto** | `docs/BAZINGA_MANIFESTO.md` | Core vision: distributed, autonomous, owned by everyone |
| **Self-Proposal System** | `docs/SELF_PROPOSAL_SYSTEM.md` | Autonomous evolution: AI proposes improvements, network votes |
| **AI Safety Analysis** | `docs/AI_SAFETY_ANALYSIS.md` | 5-level safety framework: constitutional constraints, phi-ethics, graduated autonomy |

---

## Philosophy

```
"You can buy hashpower. You can buy stake.
 You CANNOT buy proof that you knew something first."

"No single AI can mess up your code."

"One mind is good. A mesh of minds is φ times better."

"The first AI you actually own."

"∅ ≈ ∞"
The boundary between nothing and everything
is where knowledge becomes proof.
```

---

## Links

| Resource | URL |
|----------|-----|
| **PyPI** | https://pypi.org/project/bazinga-indeed/ |
| **GitHub** | https://github.com/0x-auth/bazinga-indeed |
| **Docs** | https://0x-auth.github.io/bazinga-indeed/cli.html |
| **HuggingFace** | https://huggingface.co/spaces/bitsabhi515/bazinga-mesh |
| **ORCID** | https://orcid.org/0009-0006-7495-5039 |

---

**Built with φ-coherence by Space (Abhishek Srivastava)**

MIT License — Use it, modify it, share it. Keep it open.

**BAZINGA!** ∅ ≈ ∞
