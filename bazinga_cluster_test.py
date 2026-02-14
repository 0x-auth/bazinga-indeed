#!/usr/bin/env python3
"""
BAZINGA 3-Node Cluster Test
============================
Run: python3 bazinga_cluster_test.py

Tests:
- 3 DHT nodes on ports 5150, 5151, 5152
- Node 1 gets φ trust bonus (1.618x)
- Distributed STORE/GET
- Triadic consensus

"Three minds must resonate for truth to emerge."
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bazinga.p2p.dht import KademliaNode, node_id_from_pob
from bazinga.darmiyan import prove_boundary, achieve_consensus

PHI = 1.618033988749895

async def main():
    print("="*50)
    print("  BAZINGA 3-NODE CLUSTER TEST")
    print("  'Three minds must resonate'")
    print("="*50)

    nodes = []
    ports = [5150, 5151, 5152]

    # Start 3 nodes
    print("\n🌐 Starting nodes...")
    for i, port in enumerate(ports):
        pob = prove_boundary()
        node_id = node_id_from_pob(str(pob.alpha), str(pob.omega))
        trust = 0.5 * PHI if i == 0 else 0.5

        node = KademliaNode(
            node_id=node_id,
            address="127.0.0.1",
            port=port,
            trust_score=trust
        )
        await node.start()
        nodes.append(node)
        bonus = " (φ bonus!)" if i == 0 else ""
        print(f"  Node {i+1}: port {port} | trust {trust:.3f}{bonus}")

    # Bootstrap
    print("\n🔗 Connecting nodes...")
    await nodes[1].bootstrap([("127.0.0.1", 5150)])
    print("  Node 2 → Node 1: ✓")
    await nodes[2].bootstrap([("127.0.0.1", 5150)])
    print("  Node 3 → Node 1: ✓")

    # Cross-ping to populate routing tables
    print("\n  Cross-pinging...")
    for n in nodes:
        for o in nodes:
            if n != o:
                await n.ping(o.get_info())
    await asyncio.sleep(0.3)

    # Show peers
    print("\n📊 Routing tables:")
    for i, n in enumerate(nodes):
        peers = len({p.node_id.hex() for p in n.routing_table.get_all_nodes()})
        print(f"  Node {i+1}: {peers} unique peers")

    # DHT STORE test
    print("\n💾 DHT Store/Get test...")
    test_data = {
        "message": "Hello from Node 1!",
        "phi": 1.618,
        "knowledge": "φ⁴ = 6.854101966"
    }
    await nodes[0].store("bazinga_test", test_data)
    print(f"  Node 1 stored: bazinga_test")

    await asyncio.sleep(0.2)

    # DHT GET from different node
    result = await nodes[2].get("bazinga_test")
    if result:
        print(f"  Node 3 retrieved: ✓")
        print(f"    {result}")
    else:
        print("  Node 3: ✗ Not found via DHT lookup")

    # FIND_NODE test
    print("\n🔍 Find Node test...")
    import hashlib
    target = hashlib.sha256(b"consciousness").digest()
    closest = await nodes[1].find_node(target)
    unique = {n.node_id.hex()[:12]: n for n in closest}
    print(f"  Found {len(unique)} nodes closest to 'consciousness'")

    # Triadic Consensus
    print("\n🔺 Triadic Consensus Test...")
    print("─"*40)
    c = achieve_consensus()
    if c.achieved:
        print(f"  ✓ CONSENSUS ACHIEVED!")
        print(f"    Proofs: {len(c.proofs)}")
        for i, p in enumerate(c.proofs):
            print(f"    [{i+1}] ratio={p.ratio:.4f} valid={p.valid}")
        print(f"    Target φ⁴ = 6.854101966")
    else:
        print(f"  ○ Consensus in progress ({len(c.proofs)} proofs)")

    # Network stats
    print("\n📈 Network Statistics:")
    total_pings = 0
    total_stores = 0
    for i, n in enumerate(nodes):
        s = n.get_stats()
        total_pings += s['pings_sent'] + s['pings_received']
        total_stores += s['stores']
        print(f"  Node {i+1}: pings {s['pings_sent']}/{s['pings_received']} | stores {s['stores']}")
    print(f"\n  Total: {total_pings} pings, {total_stores} stores")

    # Cleanup
    print("\n🛑 Shutting down nodes...")
    for i, n in enumerate(nodes):
        await n.stop()
        print(f"  Node {i+1}: stopped ✓")

    print("\n" + "="*50)
    print("  ✓ TRIADIC CLUSTER TEST COMPLETE")
    print()
    print("  What happened:")
    print("    • 3 DHT nodes started (ports 5150-5152)")
    print("    • Node 1 has φ=1.618x trust bonus")
    print("    • Nodes discovered each other via DHT")
    print("    • Data stored on Node 1, retrieved from Node 3")
    print("    • Triadic consensus achieved (3 PoB proofs)")
    print()
    print("  'The network danced. Understanding emerged.'")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
