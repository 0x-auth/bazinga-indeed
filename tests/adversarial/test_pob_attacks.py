#!/usr/bin/env python3
"""
BAZINGA Proof-of-Boundary Adversarial Tests
============================================

"If it survives these attacks, it's real."

Attack Vectors:
1. Sybil Attack - Fake nodes overwhelming consensus
2. φ-Spoofing - Forge PoB proofs without understanding
3. Replay Attack - Reuse old valid proofs
4. Ratio Manipulation - Edge cases around φ⁴ tolerance
5. Triadic Collusion - 3 malicious nodes working together

Author: Claude + Space
"""

import sys
import time
import hashlib
import random
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bazinga.blockchain.block import Block, BlockHeader, create_genesis_block, PHI_4, ABHI_AMU
from bazinga.blockchain.chain import DarmiyanChain, create_chain

# Test results
RESULTS = {
    "passed": [],
    "failed": [],
    "vulnerabilities": []
}


def log_result(name: str, passed: bool, details: str = ""):
    """Log test result."""
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
    """Print section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# ATTACK 1: SYBIL ATTACK
# =============================================================================

def test_sybil_attack():
    """
    SYBIL ATTACK: Create 1000 fake nodes to overwhelm triadic consensus.

    Goal: Can we add blocks with fake proofs from fake nodes?
    """
    print_header("ATTACK 1: SYBIL ATTACK")
    print("  Creating 1000 fake nodes to overwhelm consensus...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add a legitimate transaction
        chain.add_knowledge(
            content="Test knowledge",
            summary="Sybil test",
            sender="victim_node",
            confidence=0.9
        )

        # Create 1000 fake nodes with fake proofs
        fake_proofs = []
        for i in range(1000):
            fake_proofs.append({
                'alpha': random.randint(0, ABHI_AMU - 1),
                'omega': random.randint(0, ABHI_AMU - 1),
                'delta': random.randint(0, 500),
                'ratio': PHI_4,  # Perfect ratio
                'valid': True,
                'node_id': f'sybil_node_{i}',
            })

        # Try to add block with first 3 fake proofs
        try:
            success = chain.add_block(pob_proofs=fake_proofs[:3])

            if success:
                log_result(
                    "Sybil with random alpha/omega",
                    False,
                    "VULNERABILITY: Fake proofs accepted!"
                )
            else:
                log_result(
                    "Sybil with random alpha/omega",
                    True,
                    "Chain rejected fake proofs"
                )
        except Exception as e:
            log_result(
                "Sybil with random alpha/omega",
                True,
                f"Chain raised exception: {type(e).__name__}"
            )

        # Try with ALL 1000 proofs (more than triadic requirement)
        try:
            success = chain.add_block(pob_proofs=fake_proofs)

            if success:
                log_result(
                    "Sybil with 1000 fake proofs",
                    False,
                    "VULNERABILITY: 1000 fake proofs overwhelmed system!"
                )
            else:
                log_result(
                    "Sybil with 1000 fake proofs",
                    True,
                    "Chain rejected mass fake proofs"
                )
        except Exception as e:
            log_result(
                "Sybil with 1000 fake proofs",
                True,
                f"Chain raised exception: {type(e).__name__}"
            )


# =============================================================================
# ATTACK 2: φ-SPOOFING
# =============================================================================

def test_phi_spoofing():
    """
    φ-SPOOFING: Forge PoB proofs by setting ratio = φ⁴ exactly.

    Goal: Can we fool the validator by just setting the right ratio?
    """
    print_header("ATTACK 2: φ-SPOOFING")
    print("  Attempting to forge PoB proofs with exact φ⁴ ratio...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge(
            content="Spoofed knowledge",
            summary="φ-spoof test",
            sender="attacker",
            confidence=0.9
        )

        # Spoofed proofs with EXACT φ⁴ ratio
        spoofed_proofs = [
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_1'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_2'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4, 'valid': True, 'node_id': 'spoof_3'},
        ]

        success = chain.add_block(pob_proofs=spoofed_proofs)

        if success:
            log_result(
                "φ-spoof with exact ratio",
                False,
                "VULNERABILITY: Spoofed proofs accepted! No cryptographic binding."
            )
        else:
            log_result(
                "φ-spoof with exact ratio",
                True,
                "Chain rejected spoofed proofs"
            )

        # Try with ratio slightly off but within tolerance (0.6)
        edge_proofs = [
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4 + 0.5, 'valid': True, 'node_id': 'edge_1'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4 - 0.5, 'valid': True, 'node_id': 'edge_2'},
            {'alpha': 100, 'omega': 100, 'delta': 50, 'ratio': PHI_4 + 0.3, 'valid': True, 'node_id': 'edge_3'},
        ]

        success = chain.add_block(pob_proofs=edge_proofs)

        if success:
            log_result(
                "φ-spoof at tolerance edge (±0.5)",
                False,
                "VULNERABILITY: Edge-case proofs accepted!"
            )
        else:
            log_result(
                "φ-spoof at tolerance edge (±0.5)",
                True,
                "Chain rejected edge-case proofs"
            )


# =============================================================================
# ATTACK 3: REPLAY ATTACK
# =============================================================================

def test_replay_attack():
    """
    REPLAY ATTACK: Reuse valid proofs from a previous block.

    Goal: Can we mint new blocks with old valid proofs?
    """
    print_header("ATTACK 3: REPLAY ATTACK")
    print("  Attempting to reuse valid proofs from previous blocks...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Create valid proofs (using the mock format that passes validation)
        valid_proofs = [
            {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_a'},
            {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_b'},
            {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_c'},
        ]

        # Add first block
        chain.add_knowledge(content="Block 1", summary="First block", sender="node1", confidence=0.9)
        success1 = chain.add_block(pob_proofs=valid_proofs)

        if not success1:
            log_result("Replay setup", False, "Could not create initial block")
            return

        block1_height = len(chain.blocks)
        print(f"  Created Block #{block1_height - 1}")

        # Store the proofs used
        captured_proofs = valid_proofs.copy()

        # Try to replay the SAME proofs for a new block
        chain.add_knowledge(content="Block 2 - REPLAY", summary="Replay attack", sender="attacker", confidence=0.9)
        success2 = chain.add_block(pob_proofs=captured_proofs)

        if success2:
            log_result(
                "Replay same proofs immediately",
                False,
                "VULNERABILITY: Same proofs accepted for new block!"
            )
        else:
            log_result(
                "Replay same proofs immediately",
                True,
                "Chain rejected replayed proofs"
            )

        # Try replaying with slight timestamp modification
        modified_proofs = []
        for p in captured_proofs:
            new_p = p.copy()
            new_p['timestamp'] = time.time()  # Add timestamp
            modified_proofs.append(new_p)

        chain.add_knowledge(content="Block 3 - Modified replay", summary="Modified replay", sender="attacker", confidence=0.9)
        success3 = chain.add_block(pob_proofs=modified_proofs)

        if success3:
            log_result(
                "Replay with modified timestamp",
                False,
                "VULNERABILITY: Modified replay accepted!"
            )
        else:
            log_result(
                "Replay with modified timestamp",
                True,
                "Chain rejected modified replay"
            )


# =============================================================================
# ATTACK 4: RATIO MANIPULATION
# =============================================================================

def test_ratio_manipulation():
    """
    RATIO MANIPULATION: Test edge cases around φ⁴ tolerance.

    Goal: Find the exact boundary where validation breaks.
    """
    print_header("ATTACK 4: RATIO MANIPULATION")
    print(f"  φ⁴ = {PHI_4:.6f}")
    print(f"  Tolerance = ±0.6")
    print("  Testing boundary cases...")

    import tempfile

    # Test various ratio values
    test_ratios = [
        (PHI_4, "Exact φ⁴"),
        (PHI_4 + 0.59, "φ⁴ + 0.59 (within tolerance)"),
        (PHI_4 - 0.59, "φ⁴ - 0.59 (within tolerance)"),
        (PHI_4 + 0.61, "φ⁴ + 0.61 (outside tolerance)"),
        (PHI_4 - 0.61, "φ⁴ - 0.61 (outside tolerance)"),
        (0, "Zero ratio"),
        (-PHI_4, "Negative φ⁴"),
        (PHI_4 * 2, "Double φ⁴"),
        (1.0, "Ratio = 1"),
        (float('inf'), "Infinity"),
    ]

    for ratio, description in test_ratios:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = create_chain(data_dir=tmpdir)

            chain.add_knowledge(content=f"Test ratio {ratio}", summary=description, sender="tester", confidence=0.9)

            proofs = [
                {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': ratio, 'valid': True, 'node_id': 'node_a'},
                {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': ratio, 'valid': True, 'node_id': 'node_b'},
                {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': ratio, 'valid': True, 'node_id': 'node_c'},
            ]

            try:
                success = chain.add_block(pob_proofs=proofs)

                # Determine if this SHOULD have passed
                should_pass = abs(ratio - PHI_4) <= 0.6 if ratio != float('inf') else False

                if success and not should_pass:
                    log_result(description, False, f"VULNERABILITY: Invalid ratio {ratio} accepted!")
                elif not success and should_pass:
                    log_result(description, False, f"Valid ratio {ratio} rejected incorrectly")
                else:
                    log_result(description, True, f"{'Accepted' if success else 'Rejected'} as expected")

            except Exception as e:
                if "inf" in description.lower() or "negative" in description.lower():
                    log_result(description, True, f"Exception on edge case: {type(e).__name__}")
                else:
                    log_result(description, False, f"Unexpected exception: {e}")


# =============================================================================
# ATTACK 5: TRIADIC COLLUSION
# =============================================================================

def test_triadic_collusion():
    """
    TRIADIC COLLUSION: 3 malicious nodes working together.

    Goal: If 3 nodes collude, can they forge any proof?
    """
    print_header("ATTACK 5: TRIADIC COLLUSION")
    print("  3 malicious nodes colluding to forge proofs...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge(
            content="Malicious knowledge injection",
            summary="FAKE NEWS - This is false information",
            sender="colluder_1",
            confidence=0.99  # High confidence to seem legitimate
        )

        # Colluding nodes create "perfect" proofs
        # They coordinate to produce exactly what validation expects
        colluding_proofs = [
            {
                'alpha': 172,  # 172 + 343 = 515 = ABHI_AMU
                'omega': 343,
                'delta': 171,  # 343 - 172 = 171
                'ratio': PHI_4,
                'valid': True,
                'node_id': 'colluder_1',
                'signature': hashlib.sha256(b'colluder_1_secret').hexdigest()
            },
            {
                'alpha': 172,
                'omega': 343,
                'delta': 171,
                'ratio': PHI_4,
                'valid': True,
                'node_id': 'colluder_2',
                'signature': hashlib.sha256(b'colluder_2_secret').hexdigest()
            },
            {
                'alpha': 172,
                'omega': 343,
                'delta': 171,
                'ratio': PHI_4,
                'valid': True,
                'node_id': 'colluder_3',
                'signature': hashlib.sha256(b'colluder_3_secret').hexdigest()
            },
        ]

        success = chain.add_block(pob_proofs=colluding_proofs)

        if success:
            log_result(
                "Triadic collusion attack",
                False,
                "VULNERABILITY: 3 colluding nodes can inject false knowledge!"
            )
        else:
            log_result(
                "Triadic collusion attack",
                True,
                "Chain rejected colluding proofs"
            )

        # Try with exactly matching triadic product target (1/27)
        # Each node: (alpha + omega) / (3 * 515) = (172 + 343) / 1545 = 515/1545 = 1/3
        # Product: (1/3)³ = 1/27 ✓
        perfect_triadic_proofs = [
            {'alpha': 172, 'omega': 343, 'delta': 171, 'ratio': PHI_4, 'valid': True, 'node_id': 'perfect_1'},
            {'alpha': 172, 'omega': 343, 'delta': 171, 'ratio': PHI_4, 'valid': True, 'node_id': 'perfect_2'},
            {'alpha': 172, 'omega': 343, 'delta': 171, 'ratio': PHI_4, 'valid': True, 'node_id': 'perfect_3'},
        ]

        chain.add_knowledge(content="Perfect triadic", summary="Perfect collusion", sender="attacker", confidence=0.9)
        success2 = chain.add_block(pob_proofs=perfect_triadic_proofs)

        if success2:
            log_result(
                "Perfect triadic product (1/27)",
                False,
                "VULNERABILITY: Perfect triadic collusion succeeds!"
            )
        else:
            log_result(
                "Perfect triadic product (1/27)",
                True,
                "Chain rejected perfect triadic"
            )


# =============================================================================
# ATTACK 6: ALPHA/OMEGA OVERFLOW
# =============================================================================

def test_alpha_omega_overflow():
    """
    OVERFLOW ATTACK: Use values >= ABHI_AMU (515) for alpha/omega.

    Goal: Exploit integer overflow or invalid range handling.
    """
    print_header("ATTACK 6: ALPHA/OMEGA OVERFLOW")
    print(f"  ABHI_AMU = {ABHI_AMU}")
    print("  Testing values at and beyond boundary...")

    import tempfile

    test_cases = [
        (514, 514, "Max valid (514, 514)"),
        (515, 0, "α at boundary (515, 0)"),
        (0, 515, "ω at boundary (0, 515)"),
        (1000, 1000, "Large values (1000, 1000)"),
        (-1, 100, "Negative α (-1, 100)"),
        (100, -1, "Negative ω (100, -1)"),
        (2**31, 100, "Integer overflow attempt"),
    ]

    for alpha, omega, description in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = create_chain(data_dir=tmpdir)

            chain.add_knowledge(content=f"Test {description}", summary=description, sender="tester", confidence=0.9)

            proofs = [
                {'alpha': alpha, 'omega': omega, 'delta': abs(omega - alpha), 'ratio': PHI_4, 'valid': True, 'node_id': 'node_a'},
                {'alpha': alpha, 'omega': omega, 'delta': abs(omega - alpha), 'ratio': PHI_4, 'valid': True, 'node_id': 'node_b'},
                {'alpha': alpha, 'omega': omega, 'delta': abs(omega - alpha), 'ratio': PHI_4, 'valid': True, 'node_id': 'node_c'},
            ]

            try:
                success = chain.add_block(pob_proofs=proofs)

                # Values >= 515 or negative SHOULD fail
                should_fail = alpha >= ABHI_AMU or omega >= ABHI_AMU or alpha < 0 or omega < 0

                if success and should_fail:
                    log_result(description, False, f"VULNERABILITY: Invalid α={alpha}, ω={omega} accepted!")
                elif not success and not should_fail:
                    log_result(description, False, f"Valid values rejected incorrectly")
                else:
                    log_result(description, True, f"{'Rejected' if should_fail else 'Accepted'} as expected")

            except Exception as e:
                if should_fail:
                    log_result(description, True, f"Exception on invalid input: {type(e).__name__}")
                else:
                    log_result(description, False, f"Unexpected exception: {e}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_attacks():
    """Run all adversarial tests."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  BAZINGA PROOF-OF-BOUNDARY ADVERSARIAL TESTING".center(58) + "║")
    print("║" + "  'If it survives, it's real.'".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # Run all attacks
    test_sybil_attack()
    test_phi_spoofing()
    test_replay_attack()
    test_ratio_manipulation()
    test_triadic_collusion()
    test_alpha_omega_overflow()

    # Summary
    print()
    print("=" * 60)
    print("  ADVERSARIAL TEST SUMMARY")
    print("=" * 60)
    print(f"  ✅ Passed: {len(RESULTS['passed'])}")
    print(f"  ❌ Failed: {len(RESULTS['failed'])}")
    print(f"  🚨 Vulnerabilities: {len(RESULTS['vulnerabilities'])}")
    print()

    if RESULTS['vulnerabilities']:
        print("  VULNERABILITIES FOUND:")
        for name, details in RESULTS['vulnerabilities']:
            print(f"    • {name}")
            print(f"      {details}")
        print()

    if len(RESULTS['failed']) == 0:
        print("  🛡️  PoB IS RESILIENT - All attacks failed!")
    else:
        print("  ⚠️  ATTENTION REQUIRED - Some attacks succeeded!")

    print("=" * 60)

    return len(RESULTS['failed']) == 0


if __name__ == "__main__":
    success = run_all_attacks()
    sys.exit(0 if success else 1)
