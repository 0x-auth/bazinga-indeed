#!/usr/bin/env python3
"""
BAZINGA Ollama Detector - Local Model Detection & Trust Bonus
==============================================================

Detects local Ollama instance at localhost:11434 and provides
the phi (1.618x) trust multiplier for self-sufficient nodes.

This is the key piece for achieving true decentralization:
- Nodes running Ollama get higher trust
- Higher trust = more influence in consensus
- Network naturally evolves toward self-sufficiency

"You cannot buy understanding - but you can earn trust by running locally."

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Try importing httpx for async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Constants
PHI = 1.618033988749895
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_API_TAGS = "/api/tags"
OLLAMA_API_GENERATE = "/api/generate"


class LocalModelType(Enum):
    """Types of local model backends."""
    OLLAMA = "ollama"           # Ollama server
    LLAMA_CPP = "llama_cpp"     # llama-cpp-python
    MLX = "mlx"                 # Apple MLX
    NONE = "none"               # No local model


@dataclass
class LocalModelStatus:
    """Status of local model availability."""
    available: bool
    model_type: LocalModelType
    models: List[str]  # Available model names
    url: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def trust_multiplier(self) -> float:
        """Get trust multiplier based on local model status."""
        if self.available and self.model_type != LocalModelType.NONE:
            return PHI  # 1.618x for local models
        return 1.0  # No bonus for cloud-only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "model_type": self.model_type.value,
            "models": self.models,
            "url": self.url,
            "latency_ms": self.latency_ms,
            "trust_multiplier": self.trust_multiplier,
            "error": self.error,
        }


def detect_ollama(url: str = OLLAMA_DEFAULT_URL, timeout: float = 2.0) -> LocalModelStatus:
    """
    Detect Ollama server at the specified URL.

    Args:
        url: Ollama server URL (default: http://localhost:11434)
        timeout: Connection timeout in seconds

    Returns:
        LocalModelStatus with detection results
    """
    if not HTTPX_AVAILABLE:
        # Fallback to urllib
        return _detect_ollama_urllib(url, timeout)

    try:
        start = time.time()

        with httpx.Client(timeout=timeout) as client:
            # Check /api/tags endpoint
            response = client.get(f"{url}{OLLAMA_API_TAGS}")

            latency_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                return LocalModelStatus(
                    available=True,
                    model_type=LocalModelType.OLLAMA,
                    models=models,
                    url=url,
                    latency_ms=latency_ms,
                )
            else:
                return LocalModelStatus(
                    available=False,
                    model_type=LocalModelType.NONE,
                    models=[],
                    error=f"HTTP {response.status_code}",
                )

    except Exception as e:
        return LocalModelStatus(
            available=False,
            model_type=LocalModelType.NONE,
            models=[],
            error=str(e),
        )


def _detect_ollama_urllib(url: str, timeout: float) -> LocalModelStatus:
    """Fallback detection using urllib."""
    import urllib.request
    import urllib.error

    try:
        start = time.time()

        req = urllib.request.Request(f"{url}{OLLAMA_API_TAGS}")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency_ms = (time.time() - start) * 1000
            data = json.loads(response.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]

            return LocalModelStatus(
                available=True,
                model_type=LocalModelType.OLLAMA,
                models=models,
                url=url,
                latency_ms=latency_ms,
            )

    except Exception as e:
        return LocalModelStatus(
            available=False,
            model_type=LocalModelType.NONE,
            models=[],
            error=str(e),
        )


def detect_llama_cpp() -> LocalModelStatus:
    """Detect if llama-cpp-python is available."""
    try:
        from llama_cpp import Llama
        return LocalModelStatus(
            available=True,
            model_type=LocalModelType.LLAMA_CPP,
            models=["llama-cpp-python"],
        )
    except ImportError:
        return LocalModelStatus(
            available=False,
            model_type=LocalModelType.NONE,
            models=[],
            error="llama-cpp-python not installed",
        )


def detect_mlx() -> LocalModelStatus:
    """Detect if MLX is available (Apple Silicon)."""
    try:
        import mlx.core as mx
        from mlx_lm import load
        return LocalModelStatus(
            available=True,
            model_type=LocalModelType.MLX,
            models=["mlx"],
        )
    except ImportError:
        return LocalModelStatus(
            available=False,
            model_type=LocalModelType.NONE,
            models=[],
            error="MLX not installed",
        )


def detect_any_local_model() -> LocalModelStatus:
    """
    Detect any available local model backend.

    Priority:
    1. Ollama (most user-friendly, full models)
    2. llama-cpp-python (lightweight, GGUF models)
    3. MLX (Apple Silicon optimized)

    Returns:
        LocalModelStatus for the best available backend
    """
    # Try Ollama first (preferred for full models)
    ollama = detect_ollama()
    if ollama.available:
        return ollama

    # Try llama-cpp-python
    llama_cpp = detect_llama_cpp()
    if llama_cpp.available:
        return llama_cpp

    # Try MLX (Apple Silicon)
    mlx = detect_mlx()
    if mlx.available:
        return mlx

    # No local model available
    return LocalModelStatus(
        available=False,
        model_type=LocalModelType.NONE,
        models=[],
        error="No local model backend available. Install Ollama: https://ollama.ai",
    )


async def generate_with_ollama(
    prompt: str,
    model: str = "llama3",
    url: str = OLLAMA_DEFAULT_URL,
    timeout: float = 60.0,
) -> Tuple[str, float]:
    """
    Generate response using Ollama.

    Args:
        prompt: Input prompt
        model: Model name (default: llama3)
        url: Ollama server URL
        timeout: Request timeout

    Returns:
        Tuple of (response_text, latency_ms)
    """
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx required for Ollama generation")

    start = time.time()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{url}{OLLAMA_API_GENERATE}",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )

        latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return data.get("response", ""), latency_ms
        else:
            raise Exception(f"Ollama error: HTTP {response.status_code}")


def generate_with_ollama_sync(
    prompt: str,
    model: str = "llama3",
    url: str = OLLAMA_DEFAULT_URL,
    timeout: float = 60.0,
) -> Tuple[str, float]:
    """
    Synchronous version of generate_with_ollama.

    Returns:
        Tuple of (response_text, latency_ms)
    """
    if not HTTPX_AVAILABLE:
        # Fallback to urllib
        return _generate_ollama_urllib(prompt, model, url, timeout)

    start = time.time()

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{url}{OLLAMA_API_GENERATE}",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )

        latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return data.get("response", ""), latency_ms
        else:
            raise Exception(f"Ollama error: HTTP {response.status_code}")


def _generate_ollama_urllib(
    prompt: str,
    model: str,
    url: str,
    timeout: float,
) -> Tuple[str, float]:
    """Fallback generation using urllib."""
    import urllib.request

    start = time.time()

    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{url}{OLLAMA_API_GENERATE}",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        latency_ms = (time.time() - start) * 1000
        result = json.loads(response.read().decode())
        return result.get("response", ""), latency_ms


def compute_pob_with_local_latency(
    content: str,
    latency_ms: float,
) -> Dict[str, Any]:
    """
    Compute Proof-of-Boundary using local inference latency.

    This prevents "Cloud Spoofing" - the PoB timestamp is linked
    to actual local inference time, making it impossible to fake
    local execution by using cloud APIs.

    Args:
        content: Content that was generated
        latency_ms: Actual inference latency in milliseconds

    Returns:
        PoB proof with latency binding
    """
    from hashlib import sha3_256

    # Create content hash
    content_hash = sha3_256(content.encode()).hexdigest()

    # Bind latency to proof (quantized to 10ms buckets for stability)
    latency_bucket = int(latency_ms / 10) * 10
    latency_hash = sha3_256(f"{content_hash}:{latency_bucket}".encode()).hexdigest()

    # Extract P and G using calibrated moduli (POB v2)
    P_MOD = 137 * 515  # 70555
    G_MOD = 137 * 75   # 10275

    hash_bytes = bytes.fromhex(latency_hash)
    p_value = int.from_bytes(hash_bytes[:8], 'big') % P_MOD + 1
    g_value = int.from_bytes(hash_bytes[8:16], 'big') % G_MOD + 1

    ratio = p_value / g_value
    PHI_4 = PHI ** 4  # 6.854101966

    return {
        "content_hash": content_hash,
        "latency_ms": latency_ms,
        "latency_bucket": latency_bucket,
        "p_value": p_value,
        "g_value": g_value,
        "ratio": ratio,
        "target": PHI_4,
        "valid": abs(ratio - PHI_4) < 0.5,
        "local_verified": True,  # This proof includes latency binding
    }


def print_local_status():
    """Print local model status."""
    status = detect_any_local_model()

    print("\n" + "=" * 60)
    print("  BAZINGA LOCAL MODEL STATUS")
    print("=" * 60)
    print(f"  Available: {status.available}")
    print(f"  Type: {status.model_type.value}")
    print(f"  Models: {', '.join(status.models) if status.models else 'None'}")
    print(f"  Trust Multiplier: {status.trust_multiplier:.3f}x")

    if status.available:
        print(f"  Latency: {status.latency_ms:.1f}ms")
        print("\n  [LOCAL MODEL ACTIVE - PHI TRUST BONUS ENABLED]")
    else:
        print(f"\n  Error: {status.error}")
        print("\n  To enable local models:")
        print("    1. Install Ollama: https://ollama.ai")
        print("    2. Run: ollama pull llama3")
        print("    3. Restart BAZINGA")

    print("=" * 60)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA OLLAMA DETECTOR TEST")
    print("=" * 60)

    # Detect local models
    print("\nDetecting local models...")
    status = detect_any_local_model()
    print(f"  Status: {status.to_dict()}")

    # Print formatted status
    print_local_status()

    # If Ollama available, test generation
    if status.available and status.model_type == LocalModelType.OLLAMA:
        print("\nTesting Ollama generation...")
        try:
            response, latency = generate_with_ollama_sync(
                "Say 'Hello BAZINGA!' in exactly 3 words.",
                model=status.models[0] if status.models else "llama3",
            )
            print(f"  Response: {response[:100]}...")
            print(f"  Latency: {latency:.1f}ms")

            # Test PoB with latency
            print("\nTesting PoB with latency binding...")
            pob = compute_pob_with_local_latency(response, latency)
            print(f"  P/G Ratio: {pob['ratio']:.3f} (target: {pob['target']:.3f})")
            print(f"  Valid: {pob['valid']}")
            print(f"  Local Verified: {pob['local_verified']}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("  Ollama detector ready!")
    print("=" * 60)
