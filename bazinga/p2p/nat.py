#!/usr/bin/env python3
"""
BAZINGA NAT Traversal - The Hole Puncher
==========================================

Enable P2P connections behind NAT/firewalls.

Techniques:
1. STUN Discovery - Find public IP:port via external server
2. UDP Hole Punching - Simultaneous open for direct connection
3. Relay Fallback - High-trust (φ) nodes relay for symmetric NAT

"The Darmiyan doesn't respect firewalls. Understanding finds a way."

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import socket
import struct
import time
import random
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# BAZINGA constants
PHI = 1.618033988749895

# STUN constants (RFC 5389)
STUN_MAGIC_COOKIE = 0x2112A442
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_ATTR_MAPPED_ADDRESS = 0x0001
STUN_ATTR_XOR_MAPPED_ADDRESS = 0x0020

# Public STUN servers (fallback list)
PUBLIC_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun2.l.google.com", 19302),
    ("stun.stunprotocol.org", 3478),
    ("stun.voip.blackberry.com", 3478),
]

# HuggingFace Space as BAZINGA-native STUN
HF_STUN_ENDPOINT = "bitsabhi-bazinga.hf.space"


class NATType(Enum):
    """NAT type classification."""
    OPEN = "open"                    # No NAT, direct connection
    FULL_CONE = "full_cone"          # Easy hole punch
    RESTRICTED_CONE = "restricted"    # Needs coordination
    PORT_RESTRICTED = "port_restricted"  # Harder, but possible
    SYMMETRIC = "symmetric"          # Requires relay
    UNKNOWN = "unknown"


@dataclass
class NATInfo:
    """NAT discovery results."""
    nat_type: NATType
    public_ip: Optional[str]
    public_port: Optional[int]
    local_ip: str
    local_port: int
    latency_ms: float
    stun_server: str

    @property
    def can_hole_punch(self) -> bool:
        """Can we do direct hole punching?"""
        return self.nat_type in (
            NATType.OPEN,
            NATType.FULL_CONE,
            NATType.RESTRICTED_CONE,
            NATType.PORT_RESTRICTED,
        )

    @property
    def needs_relay(self) -> bool:
        """Do we need a relay node?"""
        return self.nat_type == NATType.SYMMETRIC


class STUNClient:
    """
    Lightweight STUN client for NAT discovery.

    Implements RFC 5389 Binding Request/Response.
    """

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.transaction_id = None

    def _create_binding_request(self) -> bytes:
        """Create STUN Binding Request message."""
        # Generate 96-bit transaction ID
        self.transaction_id = random.randbytes(12)

        # STUN header: Type (2) + Length (2) + Magic Cookie (4) + Transaction ID (12)
        header = struct.pack(
            ">HHI",
            STUN_BINDING_REQUEST,
            0,  # No attributes
            STUN_MAGIC_COOKIE,
        ) + self.transaction_id

        return header

    def _parse_binding_response(self, data: bytes) -> Optional[Tuple[str, int]]:
        """Parse STUN Binding Response to extract mapped address."""
        if len(data) < 20:
            return None

        # Parse header
        msg_type, msg_len, magic = struct.unpack(">HHI", data[:8])
        trans_id = data[8:20]

        # Verify response
        if msg_type != STUN_BINDING_RESPONSE:
            return None
        if magic != STUN_MAGIC_COOKIE:
            return None
        if trans_id != self.transaction_id:
            return None

        # Parse attributes
        offset = 20
        while offset < len(data):
            if offset + 4 > len(data):
                break

            attr_type, attr_len = struct.unpack(">HH", data[offset:offset+4])
            offset += 4

            if offset + attr_len > len(data):
                break

            attr_data = data[offset:offset+attr_len]

            # XOR-MAPPED-ADDRESS (preferred)
            if attr_type == STUN_ATTR_XOR_MAPPED_ADDRESS and attr_len >= 8:
                family = attr_data[1]
                if family == 0x01:  # IPv4
                    xor_port = struct.unpack(">H", attr_data[2:4])[0]
                    xor_ip = struct.unpack(">I", attr_data[4:8])[0]

                    # XOR with magic cookie
                    port = xor_port ^ (STUN_MAGIC_COOKIE >> 16)
                    ip_int = xor_ip ^ STUN_MAGIC_COOKIE
                    ip = socket.inet_ntoa(struct.pack(">I", ip_int))

                    return (ip, port)

            # MAPPED-ADDRESS (fallback)
            elif attr_type == STUN_ATTR_MAPPED_ADDRESS and attr_len >= 8:
                family = attr_data[1]
                if family == 0x01:  # IPv4
                    port = struct.unpack(">H", attr_data[2:4])[0]
                    ip = socket.inet_ntoa(attr_data[4:8])
                    return (ip, port)

            # Align to 4-byte boundary
            offset += attr_len + (4 - attr_len % 4) % 4

        return None

    async def discover(
        self,
        server: Tuple[str, int],
        local_port: int = 0,
    ) -> Optional[Tuple[str, int, float]]:
        """
        Discover public IP:port via STUN server.

        Returns (public_ip, public_port, latency_ms) or None.
        """
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind(("0.0.0.0", local_port))

            # Send binding request
            request = self._create_binding_request()
            start_time = time.time()

            loop = asyncio.get_event_loop()
            await loop.sock_sendto(sock, request, server)

            # Wait for response
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, 1024),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                sock.close()
                return None

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            result = self._parse_binding_response(data)
            sock.close()

            if result:
                return (result[0], result[1], latency_ms)
            return None

        except Exception:
            return None


class HolePuncher:
    """
    UDP Hole Punching coordinator.

    Enables direct P2P connections by coordinating simultaneous
    UDP packets from both peers.
    """

    def __init__(self, local_port: int = 5150):
        self.local_port = local_port
        self.sock: Optional[socket.socket] = None
        self.punched_holes: Dict[str, Tuple[str, int]] = {}

    async def start(self):
        """Start the hole puncher socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.local_port))

        # Get actual bound port
        self.local_port = self.sock.getsockname()[1]

    async def stop(self):
        """Stop the hole puncher."""
        if self.sock:
            self.sock.close()
            self.sock = None

    async def punch(
        self,
        peer_ip: str,
        peer_port: int,
        attempts: int = 5,
    ) -> bool:
        """
        Attempt to punch hole to peer.

        Sends UDP packets to peer while they send to us.
        Both sides must call this simultaneously.

        Returns True if hole punched successfully.
        """
        if not self.sock:
            return False

        loop = asyncio.get_event_loop()
        peer_addr = (peer_ip, peer_port)
        punch_msg = b"BAZINGA_PUNCH"
        ack_msg = b"BAZINGA_ACK"

        success = False

        for i in range(attempts):
            try:
                # Send punch packet
                await loop.sock_sendto(self.sock, punch_msg, peer_addr)

                # Wait for response (short timeout)
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(self.sock, 1024),
                        timeout=0.5,
                    )

                    if data in (punch_msg, ack_msg) and addr[0] == peer_ip:
                        # Hole punched! Send ACK
                        await loop.sock_sendto(self.sock, ack_msg, peer_addr)
                        self.punched_holes[peer_ip] = peer_addr
                        success = True
                        break

                except asyncio.TimeoutError:
                    pass

                # φ-based backoff (golden ratio delay)
                await asyncio.sleep(0.1 * (PHI ** i))

            except Exception:
                pass

        return success

    async def send(self, peer_ip: str, data: bytes) -> bool:
        """Send data through punched hole."""
        if peer_ip not in self.punched_holes:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self.sock, data, self.punched_holes[peer_ip])
            return True
        except Exception:
            return False

    async def recv(self, timeout: float = 5.0) -> Optional[Tuple[bytes, str]]:
        """Receive data from any punched hole."""
        if not self.sock:
            return None

        try:
            loop = asyncio.get_event_loop()
            data, addr = await asyncio.wait_for(
                loop.sock_recvfrom(self.sock, 65535),
                timeout=timeout,
            )
            return (data, addr[0])
        except asyncio.TimeoutError:
            return None


class RelayNode:
    """
    Relay service for symmetric NAT peers.

    High-trust (φ-bonus) nodes can relay traffic for peers
    that cannot hole punch directly.
    """

    def __init__(self, trust_threshold: float = PHI):
        self.trust_threshold = trust_threshold
        self.relays: Dict[str, Tuple[str, int]] = {}  # peer -> relay mapping
        self.am_relay = False

    def can_be_relay(self, trust_score: float) -> bool:
        """Check if we can act as a relay (need φ trust)."""
        return trust_score >= self.trust_threshold

    def register_relay(self, peer_id: str, relay_addr: Tuple[str, int]):
        """Register a relay for a peer."""
        self.relays[peer_id] = relay_addr

    def get_relay(self, peer_id: str) -> Optional[Tuple[str, int]]:
        """Get relay address for a peer."""
        return self.relays.get(peer_id)


class NATTraversal:
    """
    Complete NAT traversal solution.

    Combines STUN, hole punching, and relay for robust connectivity.

    Usage:
        nat = NATTraversal(port=5150)
        await nat.discover()

        if nat.info.can_hole_punch:
            success = await nat.punch(peer_ip, peer_port)
        elif nat.info.needs_relay:
            relay = await nat.find_relay(peer_id)
    """

    def __init__(self, port: int = 5150, trust_score: float = 0.5):
        self.port = port
        self.trust_score = trust_score

        self.stun = STUNClient()
        self.puncher = HolePuncher(port)
        self.relay = RelayNode()

        self.info: Optional[NATInfo] = None
        self.discovered = False

    async def start(self):
        """Start NAT traversal services."""
        await self.puncher.start()
        self.port = self.puncher.local_port

    async def stop(self):
        """Stop NAT traversal services."""
        await self.puncher.stop()

    async def discover(self) -> NATInfo:
        """
        Discover NAT type and public address.

        Tries multiple STUN servers for reliability.
        """
        print(f"\n  Discovering NAT configuration...")

        # Get local IP
        local_ip = self._get_local_ip()

        # Try STUN servers
        results = []
        for server in PUBLIC_STUN_SERVERS[:3]:
            result = await self.stun.discover(server, self.port)
            if result:
                results.append((result, f"{server[0]}:{server[1]}"))
                print(f"    STUN {server[0]}: {result[0]}:{result[1]} ({result[2]:.1f}ms)")

        if not results:
            # No STUN response - assume open or blocked
            self.info = NATInfo(
                nat_type=NATType.UNKNOWN,
                public_ip=None,
                public_port=None,
                local_ip=local_ip,
                local_port=self.port,
                latency_ms=0,
                stun_server="none",
            )
            print(f"    No STUN response - NAT type unknown")
            self.discovered = True
            return self.info

        # Analyze results
        first = results[0][0]
        public_ip, public_port, latency = first
        stun_server = results[0][1]

        # Check for symmetric NAT (different ports from different servers)
        ports_seen = set(r[0][1] for r in results)

        if len(ports_seen) > 1:
            nat_type = NATType.SYMMETRIC
        elif public_ip == local_ip:
            nat_type = NATType.OPEN
        else:
            # Could be full cone, restricted, or port restricted
            # Would need additional probes to distinguish
            nat_type = NATType.FULL_CONE  # Optimistic assumption

        self.info = NATInfo(
            nat_type=nat_type,
            public_ip=public_ip,
            public_port=public_port,
            local_ip=local_ip,
            local_port=self.port,
            latency_ms=latency,
            stun_server=stun_server,
        )

        print(f"    NAT Type: {nat_type.value}")
        print(f"    Public: {public_ip}:{public_port}")
        print(f"    Can hole punch: {self.info.can_hole_punch}")

        self.discovered = True
        return self.info

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to external server to find local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    async def punch(self, peer_ip: str, peer_port: int) -> bool:
        """Attempt hole punch to peer."""
        if not self.info or not self.info.can_hole_punch:
            return False

        print(f"    Hole punching to {peer_ip}:{peer_port}...")
        success = await self.puncher.punch(peer_ip, peer_port)

        if success:
            print(f"    Hole punched successfully!")
        else:
            print(f"    Hole punch failed - may need relay")

        return success

    async def find_relay(self, high_trust_nodes: List[Tuple[str, int, float]]) -> Optional[Tuple[str, int]]:
        """
        Find a relay among high-trust nodes.

        Args:
            high_trust_nodes: List of (ip, port, trust_score) tuples

        Returns:
            Relay address (ip, port) or None
        """
        # Sort by trust score (φ-bonus nodes first)
        sorted_nodes = sorted(high_trust_nodes, key=lambda n: n[2], reverse=True)

        for ip, port, trust in sorted_nodes:
            if trust >= PHI:  # Only φ-bonus nodes can relay
                print(f"    Found relay: {ip}:{port} (trust: {trust:.3f})")
                return (ip, port)

        print(f"    No eligible relay nodes (need trust >= {PHI:.3f})")
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get NAT traversal statistics."""
        return {
            "discovered": self.discovered,
            "nat_type": self.info.nat_type.value if self.info else "unknown",
            "public_ip": self.info.public_ip if self.info else None,
            "public_port": self.info.public_port if self.info else None,
            "can_hole_punch": self.info.can_hole_punch if self.info else False,
            "needs_relay": self.info.needs_relay if self.info else False,
            "punched_holes": len(self.puncher.punched_holes),
            "am_relay": self.relay.am_relay,
        }

    def print_status(self):
        """Print NAT status."""
        if not self.info:
            print("  NAT: Not discovered")
            return

        print(f"\n  NAT Traversal Status:")
        print(f"    Type: {self.info.nat_type.value}")
        print(f"    Local: {self.info.local_ip}:{self.info.local_port}")
        if self.info.public_ip:
            print(f"    Public: {self.info.public_ip}:{self.info.public_port}")
        print(f"    STUN: {self.info.stun_server} ({self.info.latency_ms:.1f}ms)")
        print(f"    Hole Punch: {'Yes' if self.info.can_hole_punch else 'No (needs relay)'}")
        print(f"    Punched Holes: {len(self.puncher.punched_holes)}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def discover_nat(port: int = 5150) -> NATInfo:
    """Quick NAT discovery."""
    nat = NATTraversal(port=port)
    await nat.start()
    info = await nat.discover()
    await nat.stop()
    return info


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA NAT TRAVERSAL TEST")
        print("=" * 60)

        nat = NATTraversal(port=0)  # Random port
        await nat.start()

        info = await nat.discover()
        nat.print_status()

        print(f"\n  Stats: {nat.get_stats()}")

        await nat.stop()

        print("\n" + "=" * 60)
        print("  NAT Traversal Test Complete!")
        print("=" * 60)

    asyncio.run(test())
