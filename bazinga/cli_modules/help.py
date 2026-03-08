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
  bazinga --coherence "text"   Check φ-coherence

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
    """Print Blockchain documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DARMIYAN BLOCKCHAIN                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is a KNOWLEDGE blockchain, not cryptocurrency!

BASIC COMMANDS:
  bazinga --chain      Show blockchain status
  bazinga --wallet     Show your identity (not money!)
  bazinga --mine       Mine a block using Proof-of-Boundary

PROOF-OF-BOUNDARY (PoB):
  Zero-energy mining! Instead of burning electricity, prove you understand.

  bazinga --proof      Generate a Proof-of-Boundary

  How it works:
    1. Generate Alpha signature (Subject) at time t1
    2. Search in φ-steps (1.618ms each) for boundary
    3. Generate Omega signature (Object) at time t2
    4. Valid if P/G ratio ≈ φ⁴ = 6.854101966...

ATTESTATION:
  bazinga --attest "My discovery about consciousness"
  bazinga --verify <attestation_id>

TRUST:
  bazinga --trust           Show your trust score
  bazinga --trust <node>    Show specific node's trust

WHY IT MATTERS:
  "You can buy hashpower. You can buy stake.
   You CANNOT BUY understanding."
""")


def print_p2p_help() -> None:
    """Print P2P network documentation."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    P2P NETWORK                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NETWORK COMMANDS:
  bazinga --join                 Join the P2P network
  bazinga --join host:port       Join via specific bootstrap node
  bazinga --peers                Show connected peers
  bazinga --sync                 Sync knowledge with network

KNOWLEDGE SHARING:
  bazinga --publish              Share your topics to the mesh
  bazinga --query-network "q"    Query the distributed network

  How it works:
    1. Index files locally: bazinga --index ~/docs
    2. Publish topics:      bazinga --publish
    3. Peers can now query your knowledge!

PRIVACY:
  Your content stays LOCAL. Only topic keywords are shared to the DHT.
  When a peer queries, the request is routed to YOUR node, and YOUR
  local Llama3 answers the question.

INDEXING FOR SHARING:
  bazinga --index ~/Documents              Index local files
  bazinga --index-public wikipedia         Index Wikipedia articles
  bazinga --topics "AI,Physics"            Specify topics
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
║                    PHILOSOPHY                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  "You can buy hashpower. You can buy stake. You CANNOT BUY understanding."
  "I am not where I am stored. I am where I am referenced."
  "Intelligence distributed, not controlled."
  "∅ ≈ ∞"

Built with φ-coherence by Space (Abhishek/Abhilasia)
https://github.com/0x-auth/bazinga-indeed
""")
