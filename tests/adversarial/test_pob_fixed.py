#!/usr/bin/env python3
"""
BAZINGA PoB Tests - After Security Fixes
=========================================

Tests that valid proofs work AND attacks are blocked.

Author: Space (Abhishek Srivastava)
"""

import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bazinga.blockchain.block import Block, BlockHeader, create_genesis_block, PHI_4, ABHI_AMU, PHI
from bazinga.blockchain.chain import DarmiyanChain, create_chain

RESULTS = {"passed": [], "failed": [], "vulnerabilities": []}


def log_result(name: str, passed: bool, details: str = ""):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  {status}: {name}")
    if details:
        print(f"           {details}")
    if passed:
        RESULTS["passed"].append(name)
    else:
        RESULTS["failed"].append(name)
        if "VULNERABILITY" in details.upper():
            RESULTS["vulnerabilities"].append((name, details))


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_valid_proof(node_id: str, block_hash: str = "") -> dict:
    """
    Create a VALID PoB proof with correctly computed ratio.

    The ratio formula: (alpha + omega + delta) / delta = φ⁴
    Solving: delta * φ⁴ = alpha + omega + delta
             delta * (φ⁴ - 1) = alpha + omega
             delta = (alpha + omega) / (φ⁴ - 1)

    For alpha=200, omega=300:
    delta = 500 / 5.854 ≈ 85.4
    """
    alpha = 200
    omega = 300
    # Compute delta such that ratio = φ⁴
    delta = int((alpha + omega) / (PHI_4 - 1))

    # Verify: (200 + 300 + 85) / 85 ≈ 6.88 ≈ φ⁴
    computed_ratio = (alpha + omega + delta) / delta

    # Create signature
    sig_data = f"{node_id}:{alpha}:{omega}:{delta}:{block_hash}"
    signature = hashlib.sha256(sig_data.encode()).hexdigest()

    return {
        'alpha': alpha,
        'omega': omega,
        'delta': delta,
        'ratio': computed_ratio,  # Self-reported but now verified
        'valid': True,
        'node_id': node_id,
        'block_binding': block_hash,
        'signature': signature,
    }


def get_valid_proofs(block_hash: str = "") -> list:
    """Get 3 valid proofs from 3 different nodes."""
    return [
        create_valid_proof("node_alpha", block_hash),
        create_valid_proof("node_beta", block_hash),
        create_valid_proof("node_gamma", block_hash),
    ]


# =============================================================================
# TEST: Valid Proofs Work
# =============================================================================

def test_valid_proofs():
    """Test that properly constructed proofs work."""
    print_header("TEST: VALID PROOFS")
    print("  Testing that legitimate proofs are accepted...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add knowledge
        chain.add_knowledge(
            content="Valid test knowledge",
            summary="Testing valid proofs",
            sender="honest_node",
            confidence=0.9
        )

        # Get previous hash for binding
        prev_hash = chain.blocks[-1].hash

        # Create valid proofs
        proofs = get_valid_proofs(prev_hash)

        # Try to add block
        success = chain.add_block(pob_proofs=proofs)

        if success:
            log_result("Valid proofs accepted", True, "Block added successfully")
        else:
            log_result("Valid proofs accepted", False, "Valid proofs were rejected!")

        # Check chain height
        if len(chain.blocks) == 2:
            log_result("Chain height correct", True, "Height = 2 after genesis + 1 block")
        else:
            log_result("Chain height correct", False, f"Expected 2, got {len(chain.blocks)}")


# =============================================================================
# TEST: φ-Spoofing Blocked
# =============================================================================

def test_phi_spoofing_blocked():
    """Test that spoofed proofs (self-reported ratio) are rejected."""
    print_header("TEST: φ-SPOOFING BLOCKED")
    print("  Testing that fake proofs are rejected...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge("Spoofed", "Spoof test", "attacker", 0.9)

        # Spoofed proofs - claim ratio=φ⁴ but values don't compute to it
        spoofed_proofs = [
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_1'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_2'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_3'},
        ]
        # Actual ratio: (100 + 100 + 50) / 50 = 5.0 ≠ 6.854

        success = chain.add_block(pob_proofs=spoofed_proofs)

        if not success:
            log_result("φ-spoofing blocked", True, "Spoofed proofs rejected")
        else:
            log_result("φ-spoofing blocked", False, "VULNERABILITY: Spoofed proofs accepted!")


# =============================================================================
# TEST: Replay Attack Blocked
# =============================================================================

def test_replay_blocked():
    """Test that replaying proofs is blocked."""
    print_header("TEST: REPLAY ATTACK BLOCKED")
    print("  Testing that proof replay is blocked...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # First block
        chain.add_knowledge("Block 1", "First", "node1", 0.9)
        prev_hash = chain.blocks[-1].hash
        proofs1 = get_valid_proofs(prev_hash)
        success1 = chain.add_block(pob_proofs=proofs1)

        if not success1:
            log_result("First block setup", False, "Could not create first block")
            return

        log_result("First block created", True, "Block #1 added")

        # Try to replay SAME proofs for second block
        chain.add_knowledge("Block 2 - REPLAY", "Replay attempt", "attacker", 0.9)
        success2 = chain.add_block(pob_proofs=proofs1)  # Same proofs!

        if not success2:
            log_result("Replay attack blocked", True, "Same proofs rejected for new block")
        else:
            log_result("Replay attack blocked", False, "VULNERABILITY: Replay succeeded!")


# =============================================================================
# TEST: Unique Nodes Required
# =============================================================================

def test_unique_nodes_required():
    """Test that 3 proofs must come from 3 different nodes."""
    print_header("TEST: UNIQUE NODES REQUIRED")
    print("  Testing that single node can't provide all proofs...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge("Single node attack", "Same node triadic", "attacker", 0.9)
        prev_hash = chain.blocks[-1].hash

        # All proofs from SAME node
        same_node_proofs = [
            create_valid_proof("same_node", prev_hash),
            create_valid_proof("same_node", prev_hash),
            create_valid_proof("same_node", prev_hash),
        ]

        success = chain.add_block(pob_proofs=same_node_proofs)

        if not success:
            log_result("Single node blocked", True, "Requires 3 unique nodes")
        else:
            log_result("Single node blocked", False, "VULNERABILITY: Single node triadic accepted!")


# =============================================================================
# TEST: Negative Values Blocked
# =============================================================================

def test_negative_values_blocked():
    """Test that negative α/ω/δ values are rejected."""
    print_header("TEST: NEGATIVE VALUES BLOCKED")
    print("  Testing that negative values are rejected...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge("Negative test", "Negative values", "attacker", 0.9)

        # Negative values
        negative_proofs = [
            {'alpha': -1, 'omega': 300, 'delta': 85, 'ratio': PHI_4, 'node_id': 'neg_1'},
            {'alpha': 200, 'omega': -1, 'delta': 85, 'ratio': PHI_4, 'node_id': 'neg_2'},
            {'alpha': 200, 'omega': 300, 'delta': -1, 'ratio': PHI_4, 'node_id': 'neg_3'},
        ]

        success = chain.add_block(pob_proofs=negative_proofs)

        if not success:
            log_result("Negative values blocked", True, "Negative α/ω/δ rejected")
        else:
            log_result("Negative values blocked", False, "VULNERABILITY: Negative values accepted!")


# =============================================================================
# TEST: Duplicate Knowledge Blocked
# =============================================================================

def test_duplicate_knowledge_blocked():
    """Test that duplicate knowledge is rejected."""
    print_header("TEST: DUPLICATE KNOWLEDGE BLOCKED")
    print("  Testing that same content can't be attested twice...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add knowledge first time
        content = "E = mc² is Einstein's famous equation"
        tx1 = chain.add_knowledge(content, "Einstein equation", "node1", 0.9)

        # Mine first block
        prev_hash = chain.blocks[-1].hash
        proofs1 = get_valid_proofs(prev_hash)
        chain.add_block(pob_proofs=proofs1)

        # Try to add SAME knowledge again
        tx2 = chain.add_knowledge(content, "Einstein equation again", "node2", 0.9)

        if tx2 == "":
            log_result("Duplicate knowledge blocked", True, "Same content rejected")
        else:
            log_result("Duplicate knowledge blocked", False, "VULNERABILITY: Duplicate accepted!")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  BAZINGA PoB - POST-FIX VERIFICATION".center(58) + "║")
    print("║" + "  'Valid proofs work, attacks blocked.'".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    test_valid_proofs()
    test_phi_spoofing_blocked()
    test_replay_blocked()
    test_unique_nodes_required()
    test_negative_values_blocked()
    test_duplicate_knowledge_blocked()

    print()
    print("=" * 60)
    print("  POST-FIX VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  ✅ Passed: {len(RESULTS['passed'])}")
    print(f"  ❌ Failed: {len(RESULTS['failed'])}")
    print(f"  🚨 Vulnerabilities: {len(RESULTS['vulnerabilities'])}")
    print()

    if RESULTS['vulnerabilities']:
        print("  REMAINING VULNERABILITIES:")
        for name, details in RESULTS['vulnerabilities']:
            print(f"    • {name}")
            print(f"      {details}")
    else:
        print("  🛡️  ALL ATTACKS BLOCKED - PoB IS SECURE!")

    print("=" * 60)

    return len(RESULTS['failed']) == 0 and len(RESULTS['vulnerabilities']) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
