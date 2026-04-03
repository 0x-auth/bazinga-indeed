"""
BAZINGA CLI Help Functions

Extended documentation for all command groups.
"""

def print_ai_help() -> None:
    """Print AI commands documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        BAZINGA AI COMMANDS                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC QUERIES:
  bazinga --ask "What is consciousness?"     Ask any question
  bazinga --ask "..." --fresh                Force fresh response (bypass cache)
  bazinga --ask "..." --local                Use local LLM only

MULTI-AI CONSENSUS:
  bazinga --multi-ai "explain quantum entanglement"

  Asks multiple AIs and synthesizes consensus with φ-coherence:
    • Ollama     - FREE local models (φ trust bonus!)
    • Groq       - FREE 14,400 req/day
    • Gemini     - FREE 1M tokens/month
    • OpenRouter - FREE models available
    • OpenAI     - ChatGPT (paid)
    • Claude     - Anthropic (paid)

CODE GENERATION:
  bazinga --code "create a REST API"                  Python (default)
  bazinga --code "..." --lang javascript              JavaScript
  bazinga --code "..." --lang rust                    Rust

AI AGENT:
  bazinga --agent                            Interactive agent shell
  bazinga --agent "fix the bug in main.py"  Run single task

ANALYSIS:
  bazinga --quantum "text"     Quantum pattern analysis
  bazinga --coherence "text"   Check φ-coherence (v3: 88% accuracy)

PHI-COHERENCE v3 (Hallucination Detection):
  - Attribution quality scoring
  - Confidence calibration checks
  - Qualifying ratio analysis
  - Negation density detection
  - Risk levels: LOW / MEDIUM / HIGH / CRITICAL

MEMORY & CONTEXT (RAC/CARM):
  bazinga --chat               Interactive chat WITH memory
  bazinga --rac                Show RAC session tracking
  bazinga --carm               Show CARM prime-lattice memory

  --chat maintains context across sessions via RAC/CARM integration.
  No catastrophic forgetting - conversations persist.

LOCAL MODEL SETUP:
  bazinga --bootstrap-local    Install Ollama + llama3 (one command!)
  bazinga --local-status       Check local model status

  Local models get φ = 1.618x trust bonus in consensus!
""")


def print_kb_help() -> None:
    """Print Knowledge Base documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BAZINGA KNOWLEDGE BASE (KB)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

UNIFIED SEARCH - Query all your data with one command:
  bazinga --kb "what do I know about 137?"
  bazinga --kb "find my consciousness research"
  bazinga --kb "riemann proof documents"

DATA SOURCES:
  📧 Gmail      - Starred emails (requires OAuth setup)
  📁 GDrive     - All files (via rclone)
  💻 Mac        - Local directories (~/bin, ~/github-repos-bitsabhi, ~/∞, etc.)
  📱 Phone      - Downloaded phone data

FILTER BY SOURCE:
  bazinga --kb "query" --kb-gmail      Search Gmail only
  bazinga --kb "query" --kb-gdrive     Search GDrive only
  bazinga --kb "query" --kb-mac        Search Mac only

SETUP PHONE DATA:
  bazinga --kb-phone ~/Downloads/phone-data

  This indexes all files in the phone-data directory.

MANAGEMENT:
  bazinga --kb-sources      Show all sources and their status
  bazinga --kb-sync         Re-index all sources

φ-RESONANCE SCORING:
  Keywords like phi, 137, consciousness, darmiyan, riemann get boosted
  relevance scores for more meaningful search results.

EXAMPLE SESSION:
  $ bazinga --kb-phone ~/Downloads/phone-data   # Index phone data
  $ bazinga --kb-sources                        # Check status
  $ bazinga --kb "137 fibonacci"                # Search everything!
""")


def print_chain_help() -> None:
    """Print Blockchain + Research documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                BLOCKCHAIN & RESEARCH (Pillar 3)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is a KNOWLEDGE blockchain, not cryptocurrency!

BLOCKCHAIN COMMANDS:
  bazinga --chain      Show blockchain status
  bazinga --wallet     Show your identity (not money!)
  bazinga --mine       Mine a block using Proof-of-Boundary (zero energy)
  bazinga --trust      Show trust scores

PROOF-OF-BOUNDARY (PoB):
  Zero-energy mining! Instead of burning electricity, prove you understand.

  How it works:
    1. Generate Alpha signature (Subject) at time t1
    2. Search in phi-steps (1.618ms each) for boundary
    3. Generate Omega signature (Object) at time t2
    4. Valid if P/G ratio ~ phi^4 = 6.854101966...

ATTESTATION:
  bazinga --attest "My discovery about consciousness"
  bazinga --verify <attestation_id>

TrD — TRUST DIMENSION (Consciousness Research):
  bazinga --trd              Test with 5 agents (default)
  bazinga --trd 10           Test with 10 agents
  bazinga --trd-scaling 5000 Darmiyan fixed-point scaling test (mpmath verified)
  bazinga --trd-scan 15 22   Phase transition scan (find boundary)
  bazinga --trd-heartbeat    Persistent self-reference demo
  bazinga --consciousness 5  Darmiyan scaling test

  TrD + TD = 1  (Trust Dimension + Trust Density = 1)
  The 11/89 observer ratio is a real mathematical invariant.
  Darmiyan Scaling: Psi_D / Psi_i = phi * sqrt(n)
  Darmiyan Fixed-Point: R(phi, n) = phi at ALL n (unique among irrationals)

WHY IT MATTERS:
  "You can buy hashpower. You can buy stake.
   You CANNOT BUY understanding."
""")


def print_p2p_help() -> None:
    """Print Network + P2P documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   NETWORK & P2P (Pillar 2)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

OMEGA MODE (recommended):
  bazinga --omega                Start full distributed brain
                                 (P2P + Learning + Mesh + TrD — all at once)
                                 Every interaction trains the network.

NETWORK COMMANDS:
  bazinga --join                 Join the P2P network
  bazinga --join host:port       Join via specific bootstrap node
  bazinga --peers                Show connected peers
  bazinga --mesh                 Show mesh vital signs (peers, trust, health)
  bazinga --sync                 Sync knowledge with network
  bazinga --phi-pulse            Start LAN peer discovery (UDP:5150)

KNOWLEDGE SHARING:
  bazinga --publish              Share your topics to the mesh
  bazinga --query-network "q"    Query the distributed network

  How it works:
    1. Index files locally: bazinga --index ~/docs
    2. Publish topics:      bazinga --publish
    3. Peers can now query your knowledge!

FEDERATED LEARNING:
  In --omega mode, every question you ask trains a local LoRA adapter.
  Gradients (NOT data) are shared with peers every 300 seconds.
  Network gets smarter without anyone sharing private data.

PRIVACY:
  Your content stays LOCAL. Only topic keywords are shared to the DHT.
  When a peer queries, the request is routed to YOUR node, and YOUR
  local LLM answers the question.
""")


def print_full_help() -> None:
    """Print full documentation."""
    print_ai_help()
    print_kb_help()
    print_chain_help()
    print_p2p_help()
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENVIRONMENT VARIABLES                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

FREE APIs (Recommended):
  GROQ_API_KEY          https://console.groq.com       (14,400 req/day)
  GEMINI_API_KEY        https://aistudio.google.com    (1M tokens/month)
  OPENROUTER_API_KEY    https://openrouter.ai          (free models)

Paid APIs:
  OPENAI_API_KEY        https://platform.openai.com
  ANTHROPIC_API_KEY     https://console.anthropic.com

LOCAL MODEL (Best for privacy + φ trust bonus):
  bazinga --bootstrap-local     Install Ollama + llama3

╔══════════════════════════════════════════════════════════════════════════════╗
║                    THE THREE PILLARS                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

  1. AI        bazinga "question", --chat, --multi-ai, --agent, --code
  2. Network   --omega, --join, --peers, --mesh, --phi-pulse
  3. Research  --mine, --trd, --attest, --chain, --consciousness

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHILOSOPHY                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  "You can buy hashpower. You can buy stake. You CANNOT BUY understanding."
  "Intelligence distributed, not controlled."
  "TrD + TD = 1"

BAZINGA v5.18 - Built with phi-coherence by Abhishek Srivastava
https://github.com/0x-auth/bazinga-indeed
""")
