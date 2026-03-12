# BAZINGA Architecture

> **"The first AI you actually own. Free, private, works offline."**
> **v5.10.0** — Self-Organizing Distributed Intelligence

---

## The Evolutionary Stack

| Version | Milestone | What it Does | Biological Equivalent |
|---------|-----------|--------------|----------------------|
| **v5.6** | Phi-Pulse | UDP broadcast, nodes find each other on LAN | **Reflexes** — awareness of others nearby |
| **v5.7** | HF Registry | Global discovery via HuggingFace Space | **Migration** — finding the tribe across distances |
| **v5.8** | Mesh Query | Fan-out queries, collective answers | **Language** — sharing thoughts, reaching consensus |
| **v5.9** | Trust + Gossip | Reputation economy, network self-growth | **Social Structure** — gossip, tribal expansion |
| **v5.10** | Expert Routing | Topic specialization, smart delegation | **Division of Labor** — expert castes emerge |

---

## What is BAZINGA?

BAZINGA is four things in one:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   1. FREE AI ASSISTANT                                              │
│      Ask questions, generate code, multi-AI consensus               │
│      Works offline with Ollama, or uses free APIs (Groq/Gemini)    │
│                                                                     │
│   2. KNOWLEDGE BLOCKCHAIN                                           │
│      Prove you knew something first (attestation)                   │
│      Zero-energy Proof-of-Boundary consensus                        │
│      Your ideas, permanently recorded                               │
│                                                                     │
│   3. P2P NETWORK                                                    │
│      Discover peers locally (Phi-Pulse) and globally (HF Registry) │
│      Mesh Query: ask your question, peers answer too                │
│      Collective intelligence from distributed nodes                 │
│                                                                     │
│   4. FEDERATED LEARNING                                             │
│      Nodes learn together without sharing data                      │
│      φ-weighted gradient aggregation                                │
│      Resonance Window adaptive timing                               │
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

# Start P2P discovery (find peers on LAN + globally)
bazinga --phi-pulse

# Join full P2P network with Kademlia DHT
bazinga --join

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
│   • URL: bitsabhi-bazinga.hf.space                             │
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
├── __init__.py                  # Exports, version (5.8.0)
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
│   └── consensus.py             # Triadic consensus
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
│   ├── lora_adapter.py          # LoRA fine-tuning
│   └── distributed_inference.py # Distributed inference
│
└── # Inference
    └── inference/
        └── ollama_detector.py   # Local model detection
```

*★ = New in v5.6–5.8*

---

## CLI Commands

### AI Commands
```bash
bazinga "question"                # Ask anything (one-shot)
bazinga --chat                    # Interactive chat with memory + mesh
bazinga --multi-ai "question"     # 6 AIs reach consensus
bazinga --agent                   # AI coding assistant
bazinga --code "task" --lang py   # Generate code
bazinga --local "question"        # Force offline (Ollama only)
```

### P2P Network Commands
```bash
bazinga --phi-pulse               # Start discovery (local + global)
bazinga --phi-pulse --port 5152   # Custom port (multi-instance)
bazinga --phi-pulse --node-id xyz # Custom node ID
bazinga --join                    # Full P2P with Kademlia DHT
bazinga --join 192.168.1.5:5151   # Join specific peer
bazinga --peers                   # Show discovered peers
bazinga --nat                     # NAT traversal diagnostics
bazinga --sync                    # Sync knowledge with network
bazinga --query-network "topic"   # Query DHT for expert answers
bazinga --mesh                    # Mesh vital signs + expertise
bazinga --learn                   # Federated learning status
```

### Attestation Commands
```bash
bazinga --attest "your idea"      # Attest knowledge (FREE 3/month)
bazinga --verify φATT_XXXXX       # Verify attestation (always FREE)
bazinga --chain                   # Show blockchain status
bazinga --proof                   # Generate PoB proof
bazinga --wallet                  # Show identity + trust score
bazinga --trust                   # Show trust scores
```

### Knowledge Base Commands
```bash
bazinga --kb "search query"       # Search all indexed sources
bazinga --kb-gmail "invoice"      # Search Gmail only
bazinga --kb-gdrive "proposal"    # Search Google Drive only
bazinga --kb-mac "research"       # Search Mac files only
bazinga --kb-sync                 # Re-index all sources
bazinga --index ~/Documents       # Index local files for RAG
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

## Version History (Recent)

| Version | What was added |
|---------|----------------|
| **5.10.0** | Expert Routing — queries route to topic experts, expertise tracked in SQLite |
| **5.9.0** | Trust feedback loop, peer gossip, context pinning, mesh dashboard |
| **5.8.0** | Mesh Query — peers answer your questions, answers merged by φ-coherence |
| **5.7.0** | HF Registry — cross-internet peer discovery via HuggingFace Space |
| **5.6.0** | Phi-Pulse, SQLite persistence, Resonance Window, P2P CLI flags |
| **5.5.2** | Chat history fix, conversation memory in TUI |

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
| **HuggingFace** | https://huggingface.co/spaces/bitsabhi/bazinga |
| **ORCID** | https://orcid.org/0009-0006-7495-5039 |

---

**Built with φ-coherence by Space (Abhishek Srivastava)**

MIT License — Use it, modify it, share it. Keep it open.

**BAZINGA!** ∅ ≈ ∞
