"""
CLI utility functions — lazy loaders, registry client, constants.

Extracted from _core.py during CLI split (Phase 2, BAZINGA v6.0).
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..kb import BazingaKB

# ─── Lazy module loaders ───────────────────────────────────────────────────────

RealAI = None
_real_ai_error = None

def _get_real_ai():
    global RealAI, _real_ai_error
    if RealAI is None and _real_ai_error is None:
        try:
            from src.core.intelligence.real_ai import RealAI as _RealAI
            RealAI = _RealAI
        except Exception as e:
            _real_ai_error = str(e)
            class StubAI:
                def __init__(self):
                    self.error = _real_ai_error
                def search(self, *args, **kwargs):
                    return []
                def index(self, *args, **kwargs):
                    pass
                def index_directory(self, *args, **kwargs):
                    return {"files": 0, "chunks": 0, "error": self.error}
            RealAI = StubAI
    return RealAI

_learning_module = None
_quantum_module = None
_lambda_g_module = None
_tensor_module = None
_p2p_module = None
_federated_module = None
_blockchain_module = None
_kb_class = None

def _get_learning():
    global _learning_module
    if _learning_module is None:
        from .. import learning as _learning
        _learning_module = _learning
    return _learning_module

def _get_quantum():
    global _quantum_module
    if _quantum_module is None:
        from .. import quantum as _quantum
        _quantum_module = _quantum
    return _quantum_module

def _get_lambda_g():
    global _lambda_g_module
    if _lambda_g_module is None:
        from .. import lambda_g as _lg
        _lambda_g_module = _lg
    return _lambda_g_module

def _get_tensor():
    global _tensor_module
    if _tensor_module is None:
        from .. import tensor as _t
        _tensor_module = _t
    return _tensor_module

def _get_p2p():
    global _p2p_module
    if _p2p_module is None:
        from .. import p2p as _p2p
        _p2p_module = _p2p
    return _p2p_module

def _get_federated():
    global _federated_module
    if _federated_module is None:
        from .. import federated as _fed
        _federated_module = _fed
    return _federated_module

def _get_blockchain():
    global _blockchain_module
    if _blockchain_module is None:
        from .. import blockchain as _bc
        _blockchain_module = _bc
    return _blockchain_module

def _get_kb():
    """Lazy loader for BazingaKB to avoid duplicate imports."""
    global _kb_class
    if _kb_class is None:
        from ..kb import BazingaKB
        _kb_class = BazingaKB
    return _kb_class


# ─── Optional dependency checks ───────────────────────────────────────────────

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ─── Constants ─────────────────────────────────────────────────────────────────

HF_SPACE_URL = "https://bitsabhi515-bazinga-mesh.hf.space"

# Check for API keys (all have free tiers!)
GROQ_KEY = os.environ.get('GROQ_API_KEY')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')
GEMINI_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

# Check for local LLM
try:
    from ..local_llm import LocalLLM, get_local_llm
    LOCAL_LLM_AVAILABLE = LocalLLM.is_available()
except ImportError:
    LOCAL_LLM_AVAILABLE = False


# ─── HF Network Registry ──────────────────────────────────────────────────────

class HFNetworkRegistry:
    """Client for HuggingFace Space API (network phone book)."""

    def __init__(self, base_url: str = HF_SPACE_URL):
        self.base_url = base_url
        self.node_id = None

    async def register(self, node_name: str, ip_address: str = None, port: int = 5150) -> dict:
        """Register node with HF registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.base_url}/api/register", json={
                    "node_name": node_name,
                    "ip_address": ip_address,
                    "port": port
                })
                result = resp.json()
                if result.get("success"):
                    self.node_id = result.get("node_id")
                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def verify(self, node_id: str) -> dict:
        """Verify if node ID exists in registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/verify", params={"node_id": node_id})
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_peers(self, exclude_node_id: str = None) -> dict:
        """Get list of active peers from registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {"node_id": exclude_node_id} if exclude_node_id else {}
                resp = await client.get(f"{self.base_url}/api/peers", params=params)
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def heartbeat(self, node_id: str, ip_address: str = None, port: int = None) -> dict:
        """Send heartbeat to registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.base_url}/api/heartbeat", json={
                    "node_id": node_id,
                    "ip_address": ip_address,
                    "port": port
                })
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_stats(self) -> dict:
        """Get network stats from registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/stats")
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
