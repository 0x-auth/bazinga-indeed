#!/usr/bin/env python3
"""
BAZINGA Federated Mesh Bridge (v5.5.0 MAINNET)
==============================================
Connects to HF Space + Local Zeroconf for peer discovery.

Two-layer discovery:
1. HF Space API (global registry) - https://calm-purpose-production-2031.up.railway.app
2. Zeroconf (local network) - mDNS discovery

Author: Abhishek (Space)
"""

import asyncio
import hashlib
import os
import socket
import httpx
from datetime import datetime

# Optional: Zeroconf for local discovery
try:
    from zeroconf import ServiceInfo, ServiceListener
    from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

from bazinga.carm import CARMMemory

# HF Space API
HF_SPACE_URL = "https://calm-purpose-production-2031.up.railway.app"


class BazingaMeshNode:
    """
    BAZINGA Mesh Node - connects to the global Darmiyan network.

    Registration flow:
    1. Generate local attestation from CARM weights
    2. Register with HF Space (global discovery)
    3. Start Zeroconf for local network peers
    4. Heartbeat to stay active
    """

    def __init__(self, node_id: str = None):
        # Generate node ID if not provided
        if node_id is None:
            import uuid
            node_id = f"bzn_{uuid.uuid4().hex[:8]}"

        self.node_id = node_id
        self.node_name = f"BAZINGA-{node_id[-8:]}"

        # CARM for local attestation
        self.carm = CARMMemory(
            memory_dir=os.path.expanduser("~/.bazinga/carm"),
            dim=128
        )

        self.service_type = "_bazinga-mesh._tcp.local."
        self.registered = False
        self.hf_node_id = None  # ID returned by HF Space

    def get_local_attestation(self) -> str:
        """Generate cryptographic hash of local CARM state."""
        if 9 in self.carm.channels:
            weights = self.carm.channels[9].get_weights()
            return hashlib.sha256(weights.tobytes()).hexdigest()
        return hashlib.sha256(b"genesis_resonance_seed").hexdigest()

    async def register_with_hf_space(self) -> bool:
        """Register this node with the HuggingFace Space registry."""
        local_hash = self.get_local_attestation()

        print(f"\n{'='*60}")
        print(f"  BAZINGA MESH - MAINNET REGISTRATION")
        print(f"{'='*60}")
        print(f"  Node ID:     {self.node_id}")
        print(f"  Node Name:   {self.node_name}")
        print(f"  Attestation: {local_hash[:16]}...")
        print(f"  HF Space:    {HF_SPACE_URL}")
        print()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Register with HF Space
                print("📡 [MESH] Registering with HuggingFace Space...")

                response = await client.post(
                    f"{HF_SPACE_URL}/api/register",
                    json={
                        "node_name": self.node_name,
                        "port": 8468,  # BAZINGA default port
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        self.hf_node_id = data.get("node_id")
                        self.registered = True
                        print(f"✅ [MESH] REGISTERED!")
                        print(f"   HF Node ID: {self.hf_node_id}")
                        print(f"   Credits: {data.get('credits', 0)}")
                        return True
                    else:
                        print(f"❌ [MESH] Registration failed: {data.get('error')}")
                else:
                    print(f"❌ [MESH] HTTP Error: {response.status_code}")

        except Exception as e:
            print(f"❌ [MESH] Connection error: {e}")

        return False

    async def get_network_peers(self) -> list:
        """Fetch list of active peers from HF Space."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{HF_SPACE_URL}/api/nodes")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("nodes", [])
        except Exception as e:
            print(f"⚠️  [MESH] Could not fetch peers: {e}")

        return []

    async def send_heartbeat(self) -> bool:
        """Send heartbeat to stay active in the registry."""
        if not self.hf_node_id:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{HF_SPACE_URL}/api/heartbeat",
                    json={"node_id": self.hf_node_id}
                )
                return response.status_code == 200
        except:
            return False

    async def start_mesh(self):
        """Start the mesh node - register and begin peer discovery."""

        # Step 1: Register with HF Space
        if not await self.register_with_hf_space():
            print("\n⚠️  [MESH] Running in LOCAL-ONLY mode (HF Space unavailable)")

        # Step 2: Show current network state
        print("\n🔍 [MESH] Checking network peers...")
        peers = await self.get_network_peers()

        if peers:
            print(f"\n   Found {len(peers)} node(s) on the network:")
            for peer in peers:
                status = "🟢" if peer.get("active") else "🔴"
                print(f"   {status} {peer.get('name', 'unknown')} ({peer.get('node_id', '?')[:8]}...)")
        else:
            print("\n   🎉 YOU ARE THE FIRST NODE ON MAINNET!")
            print("   ०→◌→φ→Ω⇄Ω←φ←◌←०")

        # Step 3: Start local Zeroconf discovery (if available)
        if ZEROCONF_AVAILABLE:
            print("\n📡 [MESH] Starting local network discovery (Zeroconf)...")
            await self._start_zeroconf()
        else:
            print("\n📡 [MESH] Zeroconf not available - using HF Space only")
            # Keep alive with heartbeats
            await self._heartbeat_loop()

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to stay active."""
        print("\n💓 [MESH] Node is LIVE - sending heartbeats...")
        print("   Press Ctrl+C to stop\n")

        try:
            while True:
                await asyncio.sleep(60)  # Heartbeat every 60 seconds
                if await self.send_heartbeat():
                    print(f"   💓 Heartbeat sent @ {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print(f"   ⚠️  Heartbeat failed @ {datetime.now().strftime('%H:%M:%S')}")
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n🛑 [MESH] Shutting down...")

    async def _start_zeroconf(self):
        """Start Zeroconf for local network peer discovery."""
        local_hash = self.get_local_attestation()

        aiozc = AsyncZeroconf()

        desc = {
            'node_id': self.node_id,
            'attestation': local_hash[:16],
            'hf_id': self.hf_node_id or 'none',
        }

        info = ServiceInfo(
            self.service_type,
            f"{self.node_id}.{self.service_type}",
            addresses=[socket.inet_aton("127.0.0.1")],
            port=8468,
            properties=desc,
        )

        await aiozc.async_register_service(info)
        print(f"   Zeroconf service registered: {self.node_id}")

        # Keep alive
        try:
            while True:
                await asyncio.sleep(60)
                if await self.send_heartbeat():
                    print(f"   💓 Heartbeat @ {datetime.now().strftime('%H:%M:%S')}")
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n🛑 [MESH] Shutting down...")
        finally:
            await aiozc.async_unregister_service(info)
            await aiozc.async_close()


async def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════╗
║  BAZINGA FEDERATED MESH v5.5.0 MAINNET                   ║
║  "The first AI you actually own"                         ║
╚══════════════════════════════════════════════════════════╝
    """)

    node = BazingaMeshNode()
    await node.start_mesh()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye from the Darmiyan Network!")
        print("   φ = 1.618033988749895")
