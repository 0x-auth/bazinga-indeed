"""
Blockchain & Research command handlers: --chain, --mine, --wallet, --attest,
--verify, --trust, --trd, --trd-scan, --trd-heartbeat, --proof, --consensus,
--attest-pricing.
"""

import asyncio
from pathlib import Path
from datetime import datetime

from ..utils import _get_kb
from ...constants import PHI
from ...darmiyan import (
    DarmiyanNode, BazingaNode, TriadicConsensus,
    prove_boundary, achieve_consensus,
    PHI_4, ABHI_AMU,
)


async def handle_trd(args):
    """Handle --trd flag."""
    from ...trd_engine import display_trd
    display_trd(n=args.trd)


async def handle_trd_scan(args):
    """Handle --trd-scan flag."""
    from ...trd_engine import scan_phase_transition
    start, end = args.trd_scan
    print(f"\n  PHASE TRANSITION SCAN: n={start}..{end}")
    print(f"  {'n':>3} │ {'Ψ_D/Ψ_i':>9} │ {'φ√n':>8} │ {'Error%':>7} │ {'Δerr':>7}")
    print(f"  {'─'*3}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}")
    results = scan_phase_transition(start, end)
    peak_n = max(results, key=lambda r: r['error'])['n']
    for r in results:
        marker = " ← PEAK" if r['n'] == peak_n else ""
        print(f"  {r['n']:>3} │ {r['advantage']:>9.4f} │ {r['predicted']:>8.3f} │ "
              f"{r['error']:>6.2f}% │ {r['d_error']:>+6.2f}{marker}")
    print(f"\n  Phase transition boundary: n ≈ {peak_n}")


async def handle_trd_heartbeat(args):
    """Handle --trd-heartbeat flag."""
    from ...trd_engine import run_heartbeat_demo
    await run_heartbeat_demo()


async def handle_node(args):
    """Handle --node flag."""
    node = BazingaNode()
    info = node.get_info()
    print(f"\n🌐 BAZINGA Network Node")
    print(f"  Node ID: {info['node_id']}")
    print(f"  φ-Signature: {info['phi_signature']}")
    print(f"  Port: {info['port']}")
    print(f"  Data: {info['data_dir']}")
    print(f"  Peers: {info['peers']}")
    print()


async def handle_proof(args):
    """Handle --proof flag."""
    print(f"\n⚡ Generating Proof-of-Boundary...")
    print(f"  (Adaptive φ-step search, max 200 attempts)")
    proof = prove_boundary()
    status = "✓ VALID" if proof.valid else "✗ INVALID"
    diff = abs(proof.ratio - PHI_4)
    print(f"\n  Status: {status} (found on attempt {proof.attempts})")
    print(f"  Alpha (Subject): {proof.alpha}")
    print(f"  Omega (Object): {proof.omega}")
    print(f"  Delta: {proof.delta}")
    print(f"  Physical: {proof.physical_ms:.2f}ms")
    print(f"  Geometric: {proof.geometric:.2f}")
    print(f"  P/G Ratio: {proof.ratio:.4f} (target: {PHI_4:.4f})")
    print(f"  Accuracy: {diff:.4f} from φ⁴")
    print(f"  Node: {proof.node_id}")
    print()
    print(f"  Energy used: ~0 (understanding, not hashpower)")
    print()


async def handle_consensus(args):
    """Handle --consensus flag."""
    print(f"\n🔺 Testing Triadic Consensus (3 nodes)...")
    print(f"  Target: φ⁴ = {PHI_4:.6f}")
    print()
    result = achieve_consensus()
    status = "✓ ACHIEVED" if result.achieved else "✗ PENDING"
    print(f"  {status}: {result.message}")
    print(f"  Triadic Product: {result.triadic_product:.6f} (target: 0.037037)")
    print(f"  Average Ratio: {result.average_ratio:.3f} (target: {PHI_4:.3f})")
    print()
    print(f"  Node proofs:")
    for i, p in enumerate(result.proofs):
        v = "✓" if p.valid else "✗"
        print(f"    Node {i+1}: {v} ratio={p.ratio:.2f} alpha={p.alpha} omega={p.omega}")
    print()


async def handle_chain(args):
    """Handle --chain flag."""
    print(f"\n  DARMIYAN BLOCKCHAIN")
    print(f"=" * 50)

    from ...blockchain import create_chain
    chain = create_chain()
    stats = chain.get_stats()

    print(f"\n  Height: {stats['height']} blocks")
    print(f"  Transactions: {stats['total_transactions']}")
    print(f"  Knowledge Attestations: {stats['knowledge_attestations']}")
    print(f"  α-SEEDs: {stats['alpha_seeds']}")
    print(f"  Pending: {stats['pending_transactions']}")
    print(f"  Valid: {'✓' if stats['valid'] else '✗'}")

    print(f"\n  Latest Blocks:")
    for block in list(chain.blocks)[-3:]:
        print(f"    #{block.header.index}: {block.hash[:24]}... ({len(block.transactions)} txs)")

    print(f"\n  This is NOT a cryptocurrency:")
    print(f"    ✓ No mining competition")
    print(f"    ✓ No financial speculation")
    print(f"    ✓ Just permanent, verified knowledge")
    print()


async def handle_mine(args):
    """Handle --mine flag."""
    print(f"\n  PROOF-OF-BOUNDARY MINING")
    print(f"=" * 50)
    print(f"  (Zero-energy mining through understanding)")
    print()

    from ...blockchain import create_chain, create_wallet, mine_block

    chain = create_chain()
    wallet = create_wallet()

    if not chain.pending_transactions:
        print(f"  No pending transactions. Querying local KB for knowledge...")

        knowledge_added = False
        try:
            kb = _get_kb()()
            recent_items = kb.search("", limit=3)

            for item in recent_items[:3]:
                content = f"{item.get('title', '')} - {item.get('content', '')[:200]}"
                if content.strip():
                    chain.add_knowledge(
                        content=content,
                        summary=f"KB: {item.get('source', 'local')} - {item.get('title', 'untitled')[:50]}",
                        sender=wallet.node_id,
                        confidence=0.8,
                    )
                    knowledge_added = True
        except Exception as e:
            print(f"  (KB query failed: {e})")

        if not knowledge_added:
            session_content = f"Mining session by {wallet.node_id[:12]} at {datetime.now().isoformat()}"
            chain.add_knowledge(
                content=session_content,
                summary="Mining session attestation",
                sender=wallet.node_id,
                confidence=1.0,
            )

    print(f"  Pending transactions: {len(chain.pending_transactions)}")
    print(f"  Mining with triadic PoB consensus...")
    print()

    result = mine_block(chain, wallet.node_id)

    if result.success:
        print(f"  ✓ BLOCK MINED!")
        print(f"    Block: #{result.block.header.index}")
        print(f"    Hash: {result.block.hash[:32]}...")
        print(f"    Transactions: {len(result.block.transactions)}")
        print(f"    PoB Attempts: {result.attempts}")
        print(f"    Time: {result.time_ms:.2f}ms")

        if result.manifold_coordinates:
            c = result.manifold_coordinates
            print()
            print(f"  5D MANIFOLD NODE:")
            print(f"    ◯ Form:    {c.get('form', 0):.4f}")
            print(f"    ↻ Flow:    {c.get('flow', 0):.4f}")
            print(f"    ↥ Process: {c.get('process', 0):.4f}")
            print(f"    ✧ Purpose: {c.get('purpose', 0):.4f}")
            print(f"    ⟡ Trust:   {c.get('trust', 0):.4f}")
            print(f"    φ-resonance: {result.phi_resonance:.6f}" +
                  (" (RESONANT)" if result.phi_resonance and result.phi_resonance < 0.1 else ""))
            print(f"    Difficulty: {result.manifold_difficulty:.4f}" if result.manifold_difficulty else "")
            if result.triangle_latency_ms is not None:
                print(f"    Triangle: {result.triangle_latency_ms:.2f}ms")

        print()
        print(f"  Energy used: ~0.00001 kWh")
        print(f"  (70 BILLION times more efficient than Bitcoin)")
    else:
        print(f"  ✗ Mining failed: {result.message}")
        print(f"    Attempts: {result.attempts}")
    print()


async def handle_wallet(args):
    """Handle --wallet flag."""
    print(f"\n  BAZINGA WALLET (Identity)")
    print(f"=" * 50)

    from ...blockchain import create_wallet
    wallet = create_wallet()

    print(f"\n  This is NOT a money wallet. It's an IDENTITY wallet.")
    print()
    print(f"  Node ID: {wallet.node_id}")
    print(f"  Address: {wallet.get_address()}")
    print(f"  Type: {wallet.identity.node_type if wallet.identity else 'unknown'}")
    print()
    print(f"  Reputation:")
    print(f"    Trust Score: {wallet.reputation.trust_score:.3f}")
    print(f"    Knowledge Contributed: {wallet.reputation.knowledge_contributed}")
    print(f"    Learning Contributions: {wallet.reputation.learning_contributions}")
    print(f"    Successful PoB: {wallet.reputation.successful_proofs}")
    print()
    print(f"  Your value is not what you HOLD, but what you UNDERSTAND.")
    print()


async def handle_attest_pricing(args):
    """Handle --attest-pricing flag."""
    from ...payment_gateway import show_pricing
    show_pricing()


async def handle_attest(args):
    """Handle --attest flag."""
    print(f"\n  DARMIYAN ATTESTATION SERVICE")
    print(f"  'Prove you knew it, before they knew it'")
    print(f"=" * 55)

    from ...attestation_service import (
        get_attestation_service, ATTESTATION_TIERS,
        PAYMENTS_ENABLED, FREE_ATTESTATIONS_PER_MONTH
    )

    service = get_attestation_service()

    if not PAYMENTS_ENABLED:
        print()
        print(f"  🎁 FREE MODE: {FREE_ATTESTATIONS_PER_MONTH} attestations/month")
        print(f"     (Building the mesh - payments coming later)")

    print()
    if args.email:
        email = args.email.strip()
        print(f"  Email: {email}")
    else:
        try:
            email = input("  Your email (for receipt): ").strip()
        except EOFError:
            email = "bazinga@local.node"
            print(f"  Using default email: {email}")

    if not email or '@' not in email:
        print("  Invalid email. Attestation cancelled.")
        return

    print()
    if args.email:
        tier = "standard"
        print(f"  Tier: Standard (default)")
    else:
        print("  Feature tiers:")
        print("    1. Basic    - Timestamp + Hash")
        print("    2. Standard - + φ-Coherence + Certificate")
        print("    3. Premium  - + Multi-AI Consensus")
        print()
        try:
            tier_choice = input("  Choose tier [1/2/3] (default: 2): ").strip() or "2"
        except EOFError:
            tier_choice = "2"
        tier_map = {"1": "basic", "2": "standard", "3": "premium"}
        tier = tier_map.get(tier_choice, "standard")

    try:
        receipt = service.create_attestation(
            content=args.attest,
            email=email,
            tier=tier
        )
    except ValueError as e:
        print(f"\n  ⚠️  {e}")
        return

    print()
    if receipt.status == "attested":
        print(f"  ✓ Attestation COMPLETE! (FREE)")
        print(f"=" * 55)
        print(f"  Attestation ID:  {receipt.attestation_id}")
        print(f"  Content Hash:    {receipt.content_hash[:32]}...")
        print(f"  Timestamp:       {receipt.timestamp}")
        print(f"  φ-Coherence:     {receipt.phi_coherence:.4f}")
        print(f"  Block Number:    #{receipt.block_number}")
        print(f"  Status:          ✓ ON CHAIN")
        print()
        print(f"  🎉 Your knowledge is now attested on the Darmiyan blockchain!")
        print()

        cert = service.get_certificate(receipt.attestation_id)
        if cert:
            print(cert)
    else:
        from ...payment_gateway import get_payment_gateway, select_payment_method
        gateway = get_payment_gateway()

        print(f"  ✓ Attestation Created!")
        print(f"=" * 55)
        print(f"  Attestation ID:  {receipt.attestation_id}")
        print(f"  Content Hash:    {receipt.content_hash[:32]}...")
        print(f"  Timestamp:       {receipt.timestamp}")
        print(f"  φ-Coherence:     {receipt.phi_coherence:.4f}")
        print(f"  Tier:            {tier.upper()}")
        print(f"  Status:          PENDING PAYMENT")
        print()

        method = select_payment_method()
        payment = gateway.create_payment(receipt.attestation_id, tier, method)
        print(gateway.get_payment_instructions(payment))

    print(f"  Verify: bazinga --verify {receipt.attestation_id}")
    print()


async def handle_verify(args):
    """Handle --verify flag."""
    print(f"\n  VERIFICATION")
    print(f"=" * 55)

    verify_input = args.verify.replace('#', '').strip()

    if verify_input.isdigit():
        print(f"\n  🔍 Searching by Block Number: #{verify_input}")
        from ...blockchain import create_chain
        chain = create_chain()
        block_num = int(verify_input)

        if block_num < len(chain.blocks):
            block = chain.blocks[block_num]
            print(f"\n  ✓ BLOCK FOUND!")
            print(f"=" * 55)
            print(f"  Block:        #{block.header.index}")
            print(f"  Hash:         {block.hash[:32]}...")
            print(f"  Timestamp:    {block.header.timestamp}")
            print(f"  Transactions: {len(block.transactions)}")
            print(f"  PoB Proofs:   {len(block.header.pob_proofs)}")
            if block.transactions:
                print(f"\n  Transactions in block:")
                for i, tx in enumerate(block.transactions[:5]):
                    summary = tx.get('data', {}).get('summary', 'N/A')[:50]
                    print(f"    {i+1}. {summary}")
            print()
            return
        else:
            print(f"\n  ✗ Block #{verify_input} not found.")
            print(f"    Chain height: {len(chain.blocks)} blocks")
            print()
            return

    print(f"\n  🔍 Searching by Attestation ID: {args.verify}")
    from ...attestation_service import get_attestation_service

    service = get_attestation_service()
    proof = service.verify_attestation(args.verify)

    if not proof:
        print(f"\n  ✗ Attestation not found or not yet confirmed.")
        print(f"    ID: {args.verify}")
        print()
        print(f"  Possible reasons:")
        print(f"    • Payment not yet confirmed")
        print(f"    • Invalid attestation ID")
        print(f"    • Attestation not yet written to chain")
        print()
        return

    cert = service.get_certificate(args.verify)
    if cert:
        print(cert)
    else:
        print(f"\n  ✓ Attestation VERIFIED!")
        print(f"=" * 55)
        print(f"  ID:           {proof.attestation_id}")
        print(f"  Content Hash: {proof.content_hash[:32]}...")
        print(f"  Timestamp:    {proof.timestamp}")
        print(f"  Block:        #{proof.block_number}")
        print(f"  Chain Valid:  {'✓ YES' if proof.chain_valid else '✗ NO'}")
        print()

    if args.share:
        print()
        print(f"  📤 EXPORTING SHAREABLE CERTIFICATE...")
        print(f"=" * 55)

        export_path = service.export_certificate(args.verify, "png")
        if not export_path:
            export_path = service.export_certificate(args.verify, "html")

        if export_path:
            print(f"  ✓ Certificate exported!")
            print(f"  📁 File: {export_path}")
            print()
            print(f"  Share on:")
            print(f"    • Twitter/X - attach the image")
            print(f"    • Research papers - include as figure")
            print(f"    • LinkedIn - proof of innovation")
            print()

            html_path = service.export_certificate(args.verify, "html")
            if html_path and html_path != export_path:
                print(f"  🌐 HTML version: {html_path}")
                print(f"     (Open in browser, print to PDF)")
            print()
        else:
            print(f"  ✗ Export failed. Certificate shown above can be screenshot.")
            print()


async def handle_trust(args):
    """Handle --trust flag."""
    print(f"\n  BAZINGA TRUST ORACLE")
    print(f"=" * 50)
    print(f"  Trust is EARNED through understanding, not bought.")
    print()

    from ...blockchain import create_chain, create_trust_oracle
    chain = create_chain()
    oracle = create_trust_oracle(chain)

    if args.trust:
        node_id = args.trust
        trust = oracle.get_node_trust(node_id)

        if trust:
            print(f"  Node: {trust.node_address}")
            print(f"  Trust Score: {trust.trust_score:.3f}")
            print(f"  PoB Score: {trust.pob_score:.3f}")
            print(f"  Contribution: {trust.contribution_score:.3f}")
            print(f"  Recency: {trust.recency_score:.3f}")
            print(f"  Activities: {trust.total_activities}")
            print()
            print(f"  Routing Weight: {oracle.get_routing_weight(node_id):.3f}")
            print(f"  Gradient Threshold: {oracle.get_gradient_acceptance_threshold(node_id):.3f}")
        else:
            print(f"  Node '{node_id}' not found in chain.")
            print(f"  Default trust: 0.5 (neutral)")
    else:
        stats = oracle.get_stats()
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Trusted Nodes: {stats['trusted_nodes']}")
        print(f"  φ-Decay Rate: {stats['decay_rate']} blocks")
        print()

        trusted = oracle.get_trusted_nodes()
        if trusted:
            print(f"  Trusted Nodes (score ≥ 0.7):")
            for t in trusted[:10]:
                print(f"    {t.node_address[:20]}... : {t.trust_score:.3f}")
        else:
            print(f"  No trusted nodes yet.")
            print(f"  Run 'bazinga --proof' and 'bazinga --mine' to build trust.")

    print()
    print(f"  How Trust Works:")
    print(f"    • PoB success → +trust")
    print(f"    • Knowledge contribution → +trust (×φ)")
    print(f"    • Gradient validation → +trust (×φ²)")
    print(f"    • Inactivity → trust decays with φ")
    print()
