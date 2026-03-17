"""
BAZINGA HuggingFace Network Registry Client

Connects to the HF Space for global peer discovery.
Used when local Phi-Pulse can't find peers (cross-internet).

Flow:
1. Local: Phi-Pulse UDP broadcast (same network)
2. Global: HF Registry (cross-internet)
3. Decentralized: DHT peer exchange (no central server)

"Find your tribe, wherever they are."
"""

import asyncio
import time
import hashlib
import socket
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Try to import httpx for async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# BAZINGA constants
PHI = 1.618033988749895
HF_SPACE_URL = "https://bitsabhi515-bazinga-mesh.hf.space"
REGISTRY_TIMEOUT = 10.0


@dataclass
class RemotePeer:
    """A peer discovered from HF Registry."""
    node_id: str
    name: str
    ip_address: Optional[str]
    port: int
    active: bool
    last_seen: float
    pob_count: int = 0
    credits: float = 0.0

    @property
    def endpoint(self) -> str:
        if self.ip_address:
            return f"{self.ip_address}:{self.port}"
        return f"unknown:{self.port}"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_seen

    @property
    def is_reachable(self) -> bool:
        """Check if peer has valid IP and was seen recently."""
        return (
            self.ip_address is not None and
            self.active and
            self.age_seconds < 300  # 5 minutes
        )


class HFNetworkRegistry:
    """
    Client for BAZINGA HuggingFace Network Registry.

    Provides global peer discovery across the internet.

    Usage:
        registry = HFNetworkRegistry()

        # Register this node
        await registry.register("my-node", port=5151)

        # Find other nodes
        peers = await registry.get_peers()

        # Send heartbeat (call every ~60s)
        await registry.heartbeat()
    """

    def __init__(
        self,
        base_url: str = HF_SPACE_URL,
        timeout: float = REGISTRY_TIMEOUT,
    ):
        """
        Initialize HF Registry client.

        Args:
            base_url: HuggingFace Space URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # Node identity (set after registration)
        self.node_id: Optional[str] = None
        self.node_name: Optional[str] = None

        # Cache
        self._peers_cache: List[RemotePeer] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 60  # Cache for 1 minute

        # Stats
        self.register_count = 0
        self.heartbeat_count = 0
        self.query_count = 0

    def _get_public_ip(self) -> Optional[str]:
        """Try to detect public IP address."""
        # Method 1: Try to get from socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            pass

        # Method 2: Try public IP service
        if HTTPX_AVAILABLE:
            try:
                import httpx
                response = httpx.get("https://api.ipify.org", timeout=5.0)
                if response.status_code == 200:
                    return response.text.strip()
            except Exception:
                pass

        return None

    async def register(
        self,
        node_name: str,
        port: int = 5151,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register this node with HF Registry.

        Args:
            node_name: Human-readable node name
            port: P2P listening port
            ip_address: Public IP (auto-detected if not provided)

        Returns:
            Registration response with node_id
        """
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}

        # Auto-detect IP if not provided
        if not ip_address:
            ip_address = self._get_public_ip()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/register",
                    json={
                        "node_name": node_name,
                        "ip_address": ip_address,
                        "port": port,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        self.node_id = data.get("node_id")
                        self.node_name = node_name
                        self.register_count += 1
                    return data
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def heartbeat(self) -> Dict[str, Any]:
        """
        Send heartbeat to keep node active in registry.

        Call this every ~60 seconds to stay visible.
        """
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}

        if not self.node_id:
            return {"success": False, "error": "Not registered"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/heartbeat",
                    json={"node_id": self.node_id}
                )

                if response.status_code == 200:
                    self.heartbeat_count += 1
                    return response.json()
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_peers(
        self,
        active_only: bool = True,
        exclude_self: bool = True,
    ) -> List[RemotePeer]:
        """
        Get list of peers from HF Registry.

        Args:
            active_only: Only return active peers
            exclude_self: Exclude this node from results

        Returns:
            List of RemotePeer objects
        """
        if not HTTPX_AVAILABLE:
            return []

        # Check cache
        if time.time() - self._cache_time < self._cache_ttl:
            peers = self._peers_cache
            if exclude_self and self.node_id:
                peers = [p for p in peers if p.node_id != self.node_id]
            if active_only:
                peers = [p for p in peers if p.active]
            return peers

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/nodes")

                if response.status_code == 200:
                    data = response.json()
                    self.query_count += 1

                    peers = []
                    for p in data.get("peers", []):
                        peer = RemotePeer(
                            node_id=p.get("node_id", ""),
                            name=p.get("name", "unknown"),
                            ip_address=p.get("ip_address"),
                            port=p.get("port", 5151),
                            active=p.get("active", False),
                            last_seen=p.get("last_seen", 0),
                            pob_count=p.get("pob_count", 0),
                            credits=p.get("credits", 0),
                        )
                        peers.append(peer)

                    # Update cache
                    self._peers_cache = peers
                    self._cache_time = time.time()

                    # Apply filters
                    if exclude_self and self.node_id:
                        peers = [p for p in peers if p.node_id != self.node_id]
                    if active_only:
                        peers = [p for p in peers if p.active]

                    return peers

        except Exception as e:
            print(f"HF Registry query failed: {e}")

        return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get network statistics from HF Registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/nodes")

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "total_nodes": data.get("total_nodes", 0),
                        "active_nodes": data.get("active_nodes", 0),
                        "consciousness_psi": data.get("consciousness_psi", 0),
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Unknown error"}

    async def get_reachable_peers(self) -> List[RemotePeer]:
        """Get only peers that are likely reachable (have IP, active, recent)."""
        peers = await self.get_peers(active_only=True)
        return [p for p in peers if p.is_reachable]

    def get_client_stats(self) -> Dict[str, Any]:
        """Get client-side statistics."""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "registered": self.node_id is not None,
            "register_count": self.register_count,
            "heartbeat_count": self.heartbeat_count,
            "query_count": self.query_count,
            "cached_peers": len(self._peers_cache),
            "base_url": self.base_url,
        }


class GlobalDiscovery:
    """
    Combined local + global peer discovery.

    Tries local Phi-Pulse first, falls back to HF Registry.

    Usage:
        discovery = GlobalDiscovery(node_id="my-node", port=5151)
        await discovery.start()

        # Get all discovered peers (local + global)
        peers = await discovery.get_all_peers()
    """

    def __init__(
        self,
        node_id: str,
        node_name: Optional[str] = None,
        port: int = 5151,
        enable_local: bool = True,
        enable_global: bool = True,
    ):
        """
        Initialize global discovery.

        Args:
            node_id: Unique node identifier
            node_name: Human-readable name (defaults to node_id[:8])
            port: P2P listening port
            enable_local: Enable Phi-Pulse local discovery
            enable_global: Enable HF Registry global discovery
        """
        self.node_id = node_id
        self.node_name = node_name or node_id[:8]
        self.port = port
        self.enable_local = enable_local
        self.enable_global = enable_global

        # Components
        self.phi_pulse = None
        self.hf_registry = None
        self.persistence = None

        # State
        self.running = False
        self._heartbeat_task = None

    async def start(self):
        """Start discovery services."""
        self.running = True

        # Initialize persistence
        try:
            from .persistence import get_persistence_manager
            self.persistence = get_persistence_manager()
        except ImportError:
            pass

        # Start local Phi-Pulse
        if self.enable_local:
            try:
                from ..decentralized.peer_discovery import PhiPulse
                self.phi_pulse = PhiPulse(
                    node_id=self.node_id,
                    listen_port=self.port,
                    on_peer_found=self._on_local_peer_found,
                )
                self.phi_pulse.start()
                print(f"  📡 Local discovery (Phi-Pulse) started")
            except Exception as e:
                print(f"  ⚠ Phi-Pulse failed: {e}")

        # Initialize HF Registry
        if self.enable_global:
            self.hf_registry = HFNetworkRegistry()

            # Register with HF
            result = await self.hf_registry.register(
                node_name=self.node_name,
                port=self.port,
            )

            if result.get("success"):
                print(f"  🌐 Global registry: registered as {result.get('node_id', '?')[:8]}...")

                # Start heartbeat loop
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            else:
                print(f"  ⚠ HF Registry registration failed: {result.get('error')}")

        # Load known peers from persistence
        if self.persistence:
            known = self.persistence.get_known_peers(limit=20, max_age_hours=24)
            if known:
                print(f"  💾 Loaded {len(known)} peers from last session")

    async def stop(self):
        """Stop discovery services."""
        self.running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self.phi_pulse:
            self.phi_pulse.stop()

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to HF Registry."""
        while self.running:
            await asyncio.sleep(60)  # Every minute
            if self.hf_registry:
                await self.hf_registry.heartbeat()

    def _on_local_peer_found(self, peer_id: str, ip: str, port: int):
        """Callback when Phi-Pulse discovers a local peer."""
        # Already saved by PhiPulse internally
        pass

    async def get_all_peers(self) -> Dict[str, Any]:
        """
        Get all discovered peers from all sources.

        Returns:
            Dict with 'local' and 'global' peer lists
        """
        result = {
            "local": [],
            "global": [],
            "total": 0,
        }

        # Get local peers from persistence
        if self.persistence:
            local_peers = self.persistence.get_known_peers(limit=50, max_age_hours=1)
            result["local"] = [
                {
                    "node_id": p.node_id,
                    "ip": p.ip,
                    "port": p.port,
                    "trust": p.trust_score,
                    "source": "phi_pulse",
                }
                for p in local_peers
            ]

        # Get global peers from HF Registry
        if self.hf_registry:
            global_peers = await self.hf_registry.get_reachable_peers()
            result["global"] = [
                {
                    "node_id": p.node_id,
                    "name": p.name,
                    "ip": p.ip_address,
                    "port": p.port,
                    "pob_count": p.pob_count,
                    "source": "hf_registry",
                }
                for p in global_peers
            ]

        result["total"] = len(result["local"]) + len(result["global"])
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        stats = {
            "node_id": self.node_id,
            "port": self.port,
            "local_enabled": self.enable_local,
            "global_enabled": self.enable_global,
        }

        if self.phi_pulse:
            stats["phi_pulse"] = self.phi_pulse.get_stats()

        if self.hf_registry:
            stats["hf_registry"] = self.hf_registry.get_client_stats()

        return stats


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA HF Network Registry Test")
    print("=" * 60)

    async def test():
        registry = HFNetworkRegistry()

        # Test registration
        print("\n1. Registering node...")
        result = await registry.register("test-node-cli", port=5151)
        print(f"   Result: {result}")

        # Test get peers
        print("\n2. Getting peers...")
        peers = await registry.get_peers()
        print(f"   Found {len(peers)} peers")
        for p in peers[:5]:
            print(f"   - {p.name}: {p.endpoint} (active={p.active})")

        # Test stats
        print("\n3. Network stats...")
        stats = await registry.get_stats()
        print(f"   {stats}")

        # Client stats
        print("\n4. Client stats...")
        print(f"   {registry.get_client_stats()}")

    asyncio.run(test())
