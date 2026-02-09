#!/bin/bash
# BAZINGA - Distributed AI
# "Intelligence distributed, not controlled"
#
# Usage:
#   ./run.sh                         # Interactive mode
#   ./run.sh --ask "question"        # Ask a question
#   ./run.sh --index ~/Documents     # Index a directory
#   ./run.sh --distributed           # Use cloud LLMs
#   ./run.sh --setup                 # Show API key setup

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
PYTHON=""
for py in python3.11 python3.12 python3.13 python3; do
    if command -v $py &> /dev/null; then
        version=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -lt 14 ]; then
            PYTHON=$py
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.11-3.13 required (3.14 has compatibility issues)"
    echo "   Install with: brew install python@3.11"
    exit 1
fi

# Create/activate virtual environment
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    $PYTHON -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    echo "📦 Installing dependencies..."
    pip install chromadb sentence-transformers httpx --quiet
    echo "✅ Setup complete!"
    echo ""
else
    source .venv/bin/activate
fi

# Suppress warnings
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

# Parse arguments
DISTRIBUTED=false
for arg in "$@"; do
    if [ "$arg" == "--distributed" ] || [ "$arg" == "-d" ]; then
        DISTRIBUTED=true
        break
    fi
done

# Run BAZINGA
if [ "$DISTRIBUTED" = true ]; then
    # Filter out --distributed from args
    ARGS=()
    for arg in "$@"; do
        if [ "$arg" != "--distributed" ] && [ "$arg" != "-d" ]; then
            ARGS+=("$arg")
        fi
    done
    python -c "
import asyncio
import sys
sys.path.insert(0, '.')
from src.core.intelligence.distributed_ai import main
asyncio.run(main())
" "${ARGS[@]}"
else
    python bazinga.py "$@"
fi
