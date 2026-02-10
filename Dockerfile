# ═══════════════════════════════════════════════════════════════════════════════
# BAZINGA v4.5.0 - Distributed AI Node
# ═══════════════════════════════════════════════════════════════════════════════
# "AI generates understanding. Blockchain proves it."
#
# Build:  docker build -t bazinga-node .
# Run:    docker run -it --name node1 bazinga-node bazinga --join
# Test:   docker-compose up -d  (starts 3-node triadic network)
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="Space (Abhishek/Abhilasia)"
LABEL version="4.5.0"
LABEL description="BAZINGA - Distributed AI with Proof-of-Boundary Consensus"

WORKDIR /app

# Install system dependencies for ZeroMQ
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY bazinga/ ./bazinga/
COPY src/ ./src/

# Install BAZINGA with all dependencies
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /root/.bazinga/chain /root/.bazinga/wallet /root/.bazinga/knowledge

# Environment
ENV PYTHONUNBUFFERED=1
ENV BAZINGA_PORT=5150
ENV BAZINGA_DATA_DIR=/root/.bazinga

# Expose P2P port
EXPOSE 5150

# Health check - verify node can generate PoB
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from bazinga.darmiyan import prove_boundary; p=prove_boundary(); exit(0 if p.valid else 1)"

# Default: show help
CMD ["bazinga", "--help"]
