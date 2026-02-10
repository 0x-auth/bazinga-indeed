# BAZINGA Darmiyan Node
# Docker image for running a P2P node

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY bazinga/ ./bazinga/
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Default port
ENV BAZINGA_PORT=5150
ENV PYTHONUNBUFFERED=1

# Run the node
CMD ["python", "-m", "bazinga.darmiyan.protocol"]
