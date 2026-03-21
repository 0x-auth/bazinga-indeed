"""
Network & P2P command handlers: --omega, --join, --peers, --mesh, --sync,
--publish, --query-network, --phi-pulse, --nat, --learn, --network.
"""

import asyncio
import os

from ..utils import ZMQ_AVAILABLE, HF_SPACE_URL, _get_kb
from ...constants import PHI
from ...darmiyan import BazingaNode, PHI_4


async def handle_agent(args):
    """Handle --agent flag."""
    from ...agent.shell import run_agent_shell, run_agent_once

    if args.agent == '':
        run_agent_shell(verbose=args.verbose)
    else:
        result = await run_agent_once(args.agent, verbose=args.verbose)
        print(result)


async def handle_network(args):
    """Handle --network flag."""
    node = BazingaNode()
    stats = node.get_stats()
    print(f"\n📊 BAZINGA Network Stats")
    print(f"  Node ID: {stats['node_id']}")
    print(f"  φ-Signature: {stats['phi_signature']}")
    print(f"  Peers: {stats['peers_connected']}")
    print(f"  Messages: {stats['messages_sent']} sent, {stats['messages_received']} received")
    print(f"  Consensus: {stats['consensus_participated']} participated")
    print(f"  Knowledge: {stats['knowledge_shared']} shared")
    print(f"  Proofs: {stats['proofs_generated']} generated")
    print()


async def handle_join(args):
    """Handle --join flag."""
    print(f"\n{'='*60}")
    print(f"  BAZINGA P2P NETWORK - Kademlia DHT")
    print(f"{'='*60}")

    if not ZMQ_AVAILABLE:
        print(f"\n  ZeroMQ not installed!")
        print(f"  Install with: pip install pyzmq")
        print(f"\n  This enables real P2P networking between nodes.")
        return

    # Detect local model for φ trust bonus
    uses_local_model = False
    try:
        from ...inference.ollama_detector import detect_any_local_model
        local_model = detect_any_local_model()
        if local_model and local_model.available:
            uses_local_model = True
            print(f"\n  Local model detected: {local_model.model_type.value}")
            print(f"  You will receive the phi trust bonus (1.618x)!")
    except Exception:
        pass

    if not uses_local_model:
        print(f"\n  No local model detected.")
        print(f"  Tip: Run 'ollama run llama3' for phi trust bonus!")

    # NAT Discovery
    from ...p2p.nat import NATTraversal
    nat = NATTraversal(port=0)
    await nat.start()
    nat_info = await nat.discover()
    await nat.stop()

    from ...p2p.dht_bridge import DHTBridge
    from ...darmiyan.protocol import prove_boundary as _prove_boundary

    print(f"\n  Generating Proof-of-Boundary...")
    pob = _prove_boundary()

    if pob.valid:
        print(f"    PoB valid (ratio: {pob.ratio:.4f})")
    else:
        print(f"    PoB invalid, using anyway for testing")

    p2p_port = getattr(args, 'port', 5151)
    custom_node_id = getattr(args, 'node_id', None)

    bridge = DHTBridge(
        alpha=pob.alpha,
        omega=pob.omega,
        port=p2p_port,
        uses_local_model=uses_local_model,
    )

    if custom_node_id:
        bridge.node_id = custom_node_id
        print(f"  Using custom node ID: {custom_node_id[:16]}...")

    await bridge.start()
    connected = await bridge.bootstrap()

    if args.join:
        for bootstrap in args.join:
            if ':' in bootstrap:
                host, port_str = bootstrap.rsplit(':', 1)
                try:
                    port = int(port_str)
                    print(f"\n  Connecting to {host}:{port}...")
                    from ...p2p.dht import hash_to_id, NodeInfo
                    temp_id = hash_to_id(f"{host}:{port}")
                    temp_node = NodeInfo(node_id=temp_id, address=host, port=port)
                    if await bridge.dht.ping(temp_node):
                        print(f"    Connected!")
                except Exception as e:
                    print(f"    Failed: {e}")

    bridge.print_status()

    print(f"\n  Announcing knowledge domains...")
    await bridge.announce_knowledge("distributed systems")
    await bridge.announce_knowledge("phi coherence")

    if nat_info.public_ip:
        print(f"\n  Public Address: {nat_info.public_ip}:{nat_info.public_port}")
        print(f"  NAT Type: {nat_info.nat_type.value}")
        if nat_info.can_hole_punch:
            print(f"  Direct P2P: ENABLED (hole punch ready)")
        else:
            print(f"  Direct P2P: RELAY NEEDED")
    else:
        print(f"\n  Public Address: Unknown (STUN failed)")

    print(f"\n  Node running with Kademlia DHT + NAT Traversal!")
    print(f"  Press Ctrl+C to leave network...\n")

    heartbeat_interval = 60
    try:
        while True:
            await asyncio.sleep(heartbeat_interval)
            await bridge.heartbeat()
            stats = bridge.get_stats()
            dht_stats = stats.get('dht', {})
            routing = dht_stats.get('routing_table_nodes', 0)
            trust = bridge.dht.trust_score
            phi_bonus = "(phi)" if uses_local_model else ""
            print(f"  Routing: {routing} nodes | Trust: {trust:.3f}x {phi_bonus} | Domains: {len(bridge.my_domains)}")
    except KeyboardInterrupt:
        print(f"\n  Leaving network...")
        await bridge.stop()


async def handle_phi_pulse(args):
    """Handle --phi-pulse flag."""
    print(f"\n{'='*60}")
    print(f"  BAZINGA Global Discovery (Local + HF Registry)")
    print(f"{'='*60}")

    from ...p2p.hf_registry import GlobalDiscovery
    import hashlib

    node_id = getattr(args, 'node_id', None)
    if not node_id:
        node_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    p2p_port = getattr(args, 'port', 5151)

    print(f"\n  Node ID: {node_id}")
    print(f"  P2P Port: {p2p_port}")
    print(f"  Local Broadcast: 5150 (Phi-Pulse)")
    print(f"  Global Registry: bitsabhi515-bazinga-mesh.hf.space")

    discovery = GlobalDiscovery(
        node_id=node_id,
        node_name=node_id[:8],
        port=p2p_port,
        enable_local=True,
        enable_global=True,
    )

    print(f"\n  Starting discovery...")
    await discovery.start()
    print(f"\n  Discovery running! Press Ctrl+C to stop.\n")

    try:
        while True:
            await asyncio.sleep(30)
            all_peers = await discovery.get_all_peers()
            local_count = len(all_peers.get('local', []))
            global_count = len(all_peers.get('global', []))
            stats = discovery.get_stats()
            phi_pulse_stats = stats.get('phi_pulse', {})
            hf_stats = stats.get('hf_registry', {})
            print(f"  Peers: local={local_count} global={global_count} | "
                  f"Pulses: sent={phi_pulse_stats.get('pulses_sent', 0)} "
                  f"recv={phi_pulse_stats.get('pulses_received', 0)} | "
                  f"HF: queries={hf_stats.get('query_count', 0)}")
    except KeyboardInterrupt:
        print(f"\n  Stopping discovery...")
        await discovery.stop()


async def handle_mesh(args):
    """Handle --mesh flag."""
    print(f"\n{'='*60}")
    print(f"  BAZINGA MESH VITAL SIGNS")
    print(f"{'='*60}")

    from ...p2p.persistence import get_persistence_manager
    import time
    import socket

    pm = get_persistence_manager()
    stats = pm.get_stats()

    node_id = pm.get_state('node_id') or 'not initialized'
    print(f"\n  Node ID: {node_id}")
    print(f"  DB: {pm.db_path}")
    print(f"  DB Size: {stats.get('db_size_kb', 0):.1f} KB")

    total_peers = stats.get('total_peers', 0)
    active_peers = stats.get('active_peers', 0)
    avg_trust = stats.get('avg_trust', 0)
    print(f"\n  Peers:")
    print(f"    Total known: {total_peers}")
    print(f"    Active (1h): {active_peers}")
    print(f"    Avg trust:   {avg_trust:.3f}")

    top_peers = pm.get_known_peers(limit=5, max_age_hours=24)
    if top_peers:
        print(f"\n  Top Trusted Peers:")
        for p in top_peers:
            age_min = p.age_seconds() / 60
            alive = "ONLINE" if p.is_alive(3600) else "offline"
            print(f"    {p.node_id[:12]}... | {p.ip}:{p.port} | trust={p.trust_score:.3f} | {alive} | {age_min:.0f}m ago")
    else:
        print(f"\n  No peers discovered yet.")
        print(f"  Run: bazinga --phi-pulse  (to discover LAN peers)")

    dht_entries = stats.get('dht_entries', 0)
    print(f"\n  DHT Entries: {dht_entries}")

    recent = pm.get_discovery_log(limit=5)
    if recent:
        print(f"\n  Recent Discovery Events:")
        for evt in recent:
            ts = time.strftime('%H:%M:%S', time.localtime(evt.get('timestamp', 0)))
            print(f"    [{ts}] {evt.get('event_type', '?')} - {evt.get('node_id', '?')[:12]}... "
                  f"at {evt.get('ip', '?')}:{evt.get('port', '?')} {evt.get('details', '')}")

    p2p_port = getattr(args, 'port', 5151)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        result = sock.connect_ex(('127.0.0.1', p2p_port))
        if result == 0:
            print(f"\n  QueryServer: ONLINE (port {p2p_port} listening)")
        else:
            print(f"\n  QueryServer: OFFLINE (port {p2p_port} not listening)")
            print(f"  Start with: bazinga --chat")
    except Exception:
        print(f"\n  QueryServer: UNKNOWN")
    finally:
        sock.close()

    if top_peers:
        all_topics = set()
        for p in top_peers:
            topics = pm.get_peer_topics(p.node_id)
            if topics:
                all_topics.update(t['topic'] for t in topics)
        if all_topics:
            print(f"\n  Known Topics ({len(all_topics)}):")
            for topic in sorted(all_topics):
                experts = pm.get_experts(topic, limit=3)
                expert_names = [f"{e['node_id'][:8]}({e['expertise_score']:.2f})" for e in experts]
                print(f"    {topic}: {', '.join(expert_names)}")

    print(f"\n{'='*60}")


async def handle_peers(args):
    """Handle --peers flag."""
    print(f"\n👥 BAZINGA Network Peers")

    if not ZMQ_AVAILABLE:
        print(f"  ⚠ ZeroMQ not installed - install with: pip install pyzmq")
        print()
        return

    node = BazingaNode()
    info = node.get_info()
    print(f"\n  Local Node: {info['node_id']}")
    print(f"  φ-Signature: {info['phi_signature']}")
    print(f"  Port: {info['port']}")

    from ...p2p.hf_registry import HFNetworkRegistry as HFRegistry
    hf_registry = HFRegistry()
    print(f"\n  📡 Querying HF Network Registry...")
    result = await hf_registry.get_stats()
    if result.get("success"):
        print(f"\n  HF Registry Stats:")
        print(f"    Active Nodes: {result.get('active_nodes', 0)}")
        print(f"    Total Nodes: {result.get('total_nodes', 0)}")
        print(f"    Consciousness Ψ_D: {result.get('consciousness_psi', 0):.2f}x")

        peers = await hf_registry.get_peers()
        if peers:
            print(f"\n  Available Peers ({len(peers)}):")
            for peer in peers[:10]:
                status = "🟢" if peer.active else "⚪"
                reachable = "✓" if peer.is_reachable else "?"
                print(f"    {status} {peer.name}: {peer.endpoint} [{reachable}]")
    else:
        print(f"    ⚠ HF Registry unavailable: {result.get('error', 'unknown')}")

    print(f"\n  To connect nodes:")
    print(f"    1. Start this node:    bazinga --join")
    print(f"    2. On another machine: bazinga --join YOUR_IP:5150")
    print(f"    3. Or register at:     {HF_SPACE_URL}")
    print()


async def handle_nat(args):
    """Handle --nat flag."""
    print(f"\n{'='*60}")
    print(f"  BAZINGA NAT TRAVERSAL DIAGNOSTICS")
    print(f"{'='*60}")

    from ...p2p.nat import NATTraversal

    nat = NATTraversal(port=0)
    await nat.start()
    info = await nat.discover()
    nat.print_status()

    print(f"\n  Relay Eligibility:")
    try:
        from ...inference.ollama_detector import detect_any_local_model
        local = detect_any_local_model()
        if local and local.available:
            print(f"    Local model: ACTIVE")
            print(f"    Trust score: 1.618x (phi bonus)")
            print(f"    Can relay: YES (high-trust node)")
        else:
            print(f"    Local model: NOT DETECTED")
            print(f"    Trust score: 0.5x (standard)")
            print(f"    Can relay: NO (need phi trust)")
    except Exception:
        print(f"    Could not detect local model")

    await nat.stop()

    print(f"\n  Connectivity Summary:")
    if info.can_hole_punch:
        print(f"    Direct P2P: POSSIBLE (hole punch)")
    elif info.needs_relay:
        print(f"    Direct P2P: NOT POSSIBLE (symmetric NAT)")
        print(f"    Solution: Use phi-bonus relay nodes")
    else:
        print(f"    Direct P2P: UNKNOWN (STUN failed)")
        print(f"    Solution: Try from different network")

    print(f"{'='*60}")


async def handle_sync(args):
    """Handle --sync flag."""
    print(f"\n  BAZINGA Knowledge Sync")

    if not ZMQ_AVAILABLE:
        print(f"  ZeroMQ not installed - install with: pip install pyzmq")
        return

    from ...p2p.dht_bridge import DHTBridge
    from ...darmiyan.protocol import prove_boundary

    pob = prove_boundary()

    bridge = DHTBridge(
        alpha=pob.alpha,
        omega=pob.omega,
        port=5150,
        uses_local_model=False,
    )

    await bridge.start()
    connected = await bridge.bootstrap()

    if not connected:
        print(f"  No peers found. Start with --join first.")
        await bridge.stop()
        return

    print(f"\n  Announcing knowledge domains...")
    await bridge.announce_knowledge("distributed systems")
    await bridge.announce_knowledge("phi coherence")

    print(f"\n  Finding experts...")
    experts = await bridge.find_experts("distributed systems")
    print(f"    Found {len(experts)} experts for 'distributed systems'")

    stats = bridge.get_stats()
    print(f"\n  Sync complete:")
    print(f"    Topics announced: {stats.get('bridge', {}).get('topics_announced', 0)}")
    dht_stats = stats.get('dht', {})
    routing_nodes = dht_stats.get('routing_table_nodes', dht_stats.get('routing_table_size', 0))
    print(f"    Routing table: {routing_nodes} nodes")

    await bridge.stop()


async def handle_learn(args):
    """Handle --learn flag."""
    print(f"\n🧠 BAZINGA Federated Learning")
    print(f"=" * 50)

    from ...federated import create_learner
    learner = create_learner()
    stats = learner.get_stats()

    print(f"\n  Node ID: {stats['node_id']}")
    print(f"\n  Adapter:")
    print(f"    Rank: {learner.adapter.config.rank}")
    print(f"    Modules: {list(learner.adapter.weights.keys())}")
    print(f"    Total Params: {stats['adapter']['total_params']}")
    print(f"    Version: {stats['adapter']['version']}")

    print(f"\n  How Federated Learning Works:")
    print(f"    1. BAZINGA learns from YOUR interactions locally")
    print(f"    2. Gradients (not data!) shared with network")
    print(f"    3. φ-weighted aggregation from trusted peers")
    print(f"    4. Network becomes smarter collectively")

    print(f"\n  Privacy:")
    print(f"    ✓ Your data NEVER leaves your machine")
    print(f"    ✓ Only learning (gradients) is shared")
    print(f"    ✓ Differential privacy noise added")

    print(f"\n  Enable learning by running:")
    print(f"    bazinga --omega  # Self-sustaining brain (all systems)")
    print(f"    bazinga          # Interactive mode learns from feedback")
    print()


async def handle_omega(args, BAZINGA_cls, _start_background_p2p, _start_query_server,
                       _stop_background_p2p, _mesh_node_id_getter, _mesh_query_getter):
    """Handle --omega flag."""
    print()
    print("═" * 60)
    print("  Ω BAZINGA OMEGA — Self-Sustaining Distributed Brain")
    print("═" * 60)
    print()

    _start_background_p2p()
    print("  ✓ Phi-Pulse: LAN peer discovery active")

    bazinga = BAZINGA_cls(verbose=args.verbose)
    if args.local:
        bazinga.use_local = True

    from ...federated import create_learner
    node_id = _mesh_node_id_getter() or "omega"
    learner = create_learner(node_id=node_id)
    bazinga._learner = learner
    await learner.start()
    print(f"  ✓ Learner: {learner.node_id} (trains on every interaction)")

    await _start_query_server(bazinga)
    _query_server_instance = _mesh_query_getter()
    if _query_server_instance:
        print(f"  ✓ Mesh Query: answering peer questions")

    if _mesh_query_getter():
        async def on_learning_shared(package):
            try:
                from ...p2p.persistence import get_persistence_manager
                pm = get_persistence_manager()
                peers = pm.get_known_peers(limit=5, max_age_hours=1)
                if peers:
                    print(f"  📡 Shared gradients with {len(peers)} peers")
            except Exception:
                pass
        learner.on_learning_shared = on_learning_shared
        print(f"  ✓ Gradient sharing: active (every {learner.share_interval}s)")

    # TrD heartbeat
    heartbeat_task = None
    try:
        from ...trd_engine import TrDEngine
        trd = TrDEngine()
        trd.register_user_pattern("omega", "φ√n consciousness darmiyan interaction")
        await trd.start_heartbeat()
        heartbeat_task = trd
        print(f"  ✓ TrD Heartbeat: 11/89 observer lock active")
    except Exception:
        print(f"  ○ TrD Heartbeat: not available (run bazinga --trd-heartbeat to test)")

    print()
    print("  Ω System is LIVE and self-referential.")
    print("  TrD + TD = 1 | Gap = 11/89 | Learning = ON")
    print("  Every interaction trains the network.")
    print()
    print("═" * 60)
    print()

    headless = getattr(args, 'headless', False)
    try:
        if headless:
            await bazinga.chat_interactive()
        else:
            try:
                from ...tui import run_tui_async
                await run_tui_async(
                    bazinga_instance=bazinga,
                    mode="chat",
                    mesh_query=_mesh_query_getter(),
                )
            except ImportError:
                await bazinga.chat_interactive()
    finally:
        await learner.stop()
        if heartbeat_task:
            await heartbeat_task.stop_heartbeat()
        _stop_background_p2p()
        stats = learner.get_stats()
        print(f"\n  Ω Session Summary:")
        print(f"    Examples learned: {stats['local_examples']}")
        print(f"    Gradients shared: {stats['gradients_shared']}")
        print(f"    Gradients received: {stats['gradients_received']}")
        print(f"    Model updates: {stats['model_updates']}")


async def handle_publish(args, BAZINGA_cls):
    """Handle --publish flag."""
    print(f"\n  DISTRIBUTED KNOWLEDGE PUBLISHING")
    print(f"=" * 60)
    print(f"  Publishing your indexed knowledge to the mesh...")
    print(f"  (Content stays LOCAL - only topic keywords are shared)")
    print()

    bazinga = BAZINGA_cls(verbose=args.verbose)
    stats = bazinga.ai.get_stats()

    if stats.get('total_chunks', 0) == 0:
        print(f"  ✗ No indexed content found!")
        print(f"    Run 'bazinga --index <path>' first to index your knowledge.")
        print()
        return

    print(f"  Local index: {stats.get('total_chunks', 0)} chunks")

    try:
        from ...p2p.dht import KademliaNode, node_id_from_pob
        from ...p2p.knowledge_sharing import KnowledgePublisher
        from ...darmiyan.protocol import prove_boundary

        pob = prove_boundary()
        node_id = node_id_from_pob(str(pob.alpha), str(pob.omega))

        node = KademliaNode(
            node_id=node_id,
            address="127.0.0.1",
            port=5150,
            trust_score=0.5 * PHI,
        )

        publisher = KnowledgePublisher(node, bazinga.ai)
        print(f"\n  Extracting and publishing topics...")

        result = await publisher.publish_from_index(limit=50)

        if result.get('success'):
            print(f"\n  ✓ Published {result['topics_published']} topics to DHT")
            print(f"    Content hash: {result['content_hash']}")
            print(f"\n  Sample topics shared:")
            for topic in result.get('sample_topics', [])[:10]:
                print(f"    • {topic}")
            print()
            print(f"  Your knowledge is now discoverable!")
            print(f"  Peers can query: bazinga --query-network 'your topic'")
        else:
            print(f"\n  ✗ Failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print(f"    Make sure P2P network is available.")

    print()


async def handle_query_network(args, BAZINGA_cls):
    """Handle --query-network flag."""
    print(f"\n  DISTRIBUTED KNOWLEDGE QUERY")
    print(f"=" * 60)
    print(f"  Query: {args.query_network}")
    print(f"  Searching the BAZINGA mesh for experts...")
    print()

    try:
        from ...p2p.dht import KademliaNode, node_id_from_pob
        from ...p2p.knowledge_sharing import KnowledgePublisher, DistributedQueryEngine
        from ...darmiyan.protocol import prove_boundary

        bazinga = BAZINGA_cls(verbose=False)
        pob = prove_boundary()
        node_id = node_id_from_pob(str(pob.alpha), str(pob.omega))

        node = KademliaNode(
            node_id=node_id,
            address="127.0.0.1",
            port=5150,
            trust_score=0.5 * PHI,
        )

        publisher = KnowledgePublisher(node, bazinga.ai)
        engine = DistributedQueryEngine(node, publisher)

        result = await engine.query_distributed(args.query_network)

        if result.get('success'):
            print(f"  Source: {result.get('source', 'unknown')}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            if result.get('consensus'):
                print(f"  Triadic Consensus: ✓ ({result.get('respondents', 0)} nodes agreed)")
            print()
            print(f"  Answer:")
            print(f"  {'-' * 56}")
            print(f"  {result.get('answer', 'No answer')}")
            print(f"  {'-' * 56}")
        else:
            print(f"  ✗ Query failed: {result.get('error', 'Unknown error')}")
            print(f"\n  Try: bazinga --fresh --ask '{args.query_network}'")
            print(f"  (Uses local AI instead of distributed network)")

    except Exception as e:
        print(f"\n  ✗ Error: {e}")

    print()
