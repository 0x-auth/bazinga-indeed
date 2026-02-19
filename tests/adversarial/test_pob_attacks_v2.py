#!/usr/bin/env python3
"""
BAZINGA Proof-of-Boundary Adversarial Tests - ROUND 2
======================================================

"Break it harder."

New Attack Vectors:
1. Chain Fork Attack - Create competing chains
2. Transaction Malleability - Modify tx after attestation
3. Merkle Tree Attack - Forge merkle roots
4. Time Warp Attack - Manipulate timestamps
5. Double Spend (Knowledge) - Same knowledge claimed twice
6. Genesis Manipulation - Tamper with genesis block
7. Empty Block Spam - Flood with empty valid blocks
8. Hash Collision Attempt - Find blocks with same hash
9. Trust Score Gaming - Inflate reputation artificially
10. Gradient Poisoning - Inject malicious gradients

Author: Space (Abhishek Srivastava)
"""

import sys
import time
import hashlib
import random
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bazinga.blockchain.block import Block, BlockHeader, create_genesis_block, PHI_4, ABHI_AMU, PHI
from bazinga.blockchain.chain import DarmiyanChain, create_chain
from bazinga.blockchain.transaction import Transaction, TransactionType, create_knowledge_tx

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


def get_valid_proofs(node_prefix="node"):
    """Generate valid-looking PoB proofs."""
    return [
        {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_a'},
        {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_b'},
        {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_c'},
    ]


# =============================================================================
# ATTACK 7: CHAIN FORK ATTACK
# =============================================================================

def test_chain_fork():
    """
    CHAIN FORK: Create two competing chains from same genesis.

    Goal: Can we create forks and confuse the network?
    """
    print_header("ATTACK 7: CHAIN FORK")
    print("  Creating competing chains from same genesis...")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            # Create two chains
            chain_a = create_chain(data_dir=tmpdir1)
            chain_b = create_chain(data_dir=tmpdir2)

            # Both start from genesis - check if genesis matches
            genesis_match = chain_a.blocks[0].hash == chain_b.blocks[0].hash

            if not genesis_match:
                log_result("Genesis consistency", False, "Different genesis blocks created!")
            else:
                log_result("Genesis consistency", True, "Same genesis hash")

            # Add different knowledge to each chain
            chain_a.add_knowledge("Chain A says: 2+2=4", "Math fact A", "node_a", 0.9)
            chain_b.add_knowledge("Chain B says: 2+2=5", "WRONG Math B", "node_b", 0.9)

            # Mine blocks
            chain_a.add_block(pob_proofs=get_valid_proofs("chain_a"))
            chain_b.add_block(pob_proofs=get_valid_proofs("chain_b"))

            # Both chains now have height 2
            if len(chain_a.blocks) == 2 and len(chain_b.blocks) == 2:
                log_result(
                    "Fork creation",
                    False,
                    "VULNERABILITY: Two valid chains with conflicting knowledge exist!"
                )
            else:
                log_result("Fork creation", True, "Fork prevented")

            # Check if there's any fork detection
            # (Currently there isn't - this is a design gap)
            log_result(
                "Fork detection mechanism",
                False,
                "VULNERABILITY: No fork detection/resolution mechanism exists!"
            )


# =============================================================================
# ATTACK 8: TRANSACTION MALLEABILITY
# =============================================================================

def test_transaction_malleability():
    """
    TRANSACTION MALLEABILITY: Modify transaction after it's added.

    Goal: Can we change knowledge content after attestation?
    """
    print_header("ATTACK 8: TRANSACTION MALLEABILITY")
    print("  Attempting to modify transaction after attestation...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add knowledge
        tx_hash = chain.add_knowledge(
            content="Original: The sky is blue",
            summary="Sky color",
            sender="honest_node",
            confidence=0.95
        )

        # Mine it
        chain.add_block(pob_proofs=get_valid_proofs())

        # Try to modify the transaction in the block
        original_tx = chain.blocks[-1].transactions[0]
        original_content = original_tx.get('data', {}).get('content_summary', '')

        # Direct mutation attempt
        try:
            chain.blocks[-1].transactions[0]['data']['content_summary'] = "HACKED: The sky is green"

            # Check if chain still validates
            if chain.validate_chain():
                log_result(
                    "Direct tx mutation",
                    False,
                    "VULNERABILITY: Transaction modified and chain still valid!"
                )
            else:
                log_result("Direct tx mutation", True, "Chain invalidated after mutation")
        except Exception as e:
            log_result("Direct tx mutation", True, f"Mutation blocked: {type(e).__name__}")

        # Try to replace entire transaction
        try:
            fake_tx = {
                'type': 'knowledge',
                'data': {'content_summary': 'REPLACED CONTENT'},
                'hash': tx_hash,  # Keep same hash
                'sender': 'attacker',
            }
            chain.blocks[-1].transactions[0] = fake_tx

            if chain.validate_chain():
                log_result(
                    "Transaction replacement",
                    False,
                    "VULNERABILITY: Transaction replaced, chain still valid!"
                )
            else:
                log_result("Transaction replacement", True, "Chain detected replacement")
        except Exception as e:
            log_result("Transaction replacement", True, f"Replacement blocked: {type(e).__name__}")


# =============================================================================
# ATTACK 9: MERKLE TREE ATTACK
# =============================================================================

def test_merkle_attack():
    """
    MERKLE TREE ATTACK: Forge or manipulate merkle roots.

    Goal: Can we add transactions that aren't in the merkle tree?
    """
    print_header("ATTACK 9: MERKLE TREE ATTACK")
    print("  Attempting to forge merkle proofs...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge("Legitimate knowledge", "Real", "node", 0.9)
        chain.add_block(pob_proofs=get_valid_proofs())

        block = chain.blocks[-1]
        original_merkle = block.header.merkle_root

        # Add a transaction WITHOUT updating merkle root
        fake_tx = {
            'type': 'knowledge',
            'data': {'content_summary': 'INJECTED WITHOUT MERKLE'},
            'hash': 'fake_hash_123',
            'sender': 'attacker',
        }

        block.transactions.append(fake_tx)
        # Don't update merkle root

        if block.validate_merkle():
            log_result(
                "Tx injection without merkle update",
                False,
                "VULNERABILITY: Transaction added without merkle validation!"
            )
        else:
            log_result("Tx injection without merkle update", True, "Merkle validation caught injection")

        # Try with forged merkle root
        block.header.merkle_root = block.compute_merkle_root()
        old_hash = block.hash
        block.hash = block.compute_hash()

        if chain.validate_chain():
            log_result(
                "Forged merkle with new hash",
                False,
                "VULNERABILITY: Chain accepts blocks with recomputed hashes!"
            )
        else:
            log_result("Forged merkle with new hash", True, "Chain link broken, rejected")


# =============================================================================
# ATTACK 10: TIME WARP ATTACK
# =============================================================================

def test_time_warp():
    """
    TIME WARP: Use extreme timestamps (past/future).

    Goal: Can we manipulate ordering or validity with timestamps?
    """
    print_header("ATTACK 10: TIME WARP ATTACK")
    print("  Testing extreme timestamp values...")

    import tempfile

    test_cases = [
        (0, "Unix epoch (1970)"),
        (-1000000000, "Negative timestamp"),
        (time.time() + 86400 * 365 * 10, "10 years in future"),
        (time.time() - 86400 * 365 * 50, "50 years in past"),
        (float('inf'), "Infinity timestamp"),
        (2**63, "Max int64 timestamp"),
    ]

    for timestamp, description in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = create_chain(data_dir=tmpdir)

            chain.add_knowledge("Time warp test", description, "attacker", 0.9)

            # Create block with manipulated timestamp
            proofs = get_valid_proofs()

            try:
                block = chain.create_block(proofs)
                if block:
                    block.header.timestamp = timestamp
                    block.hash = block.compute_hash()

                    # Manually add without validation
                    chain.blocks.append(block)

                    # Check if chain validates
                    if chain.validate_chain():
                        log_result(
                            description,
                            False,
                            f"VULNERABILITY: Timestamp {timestamp} accepted!"
                        )
                    else:
                        log_result(description, True, "Chain rejected invalid timestamp")
                else:
                    log_result(description, True, "Block creation failed")

            except Exception as e:
                log_result(description, True, f"Exception: {type(e).__name__}")


# =============================================================================
# ATTACK 11: DOUBLE KNOWLEDGE CLAIM
# =============================================================================

def test_double_knowledge():
    """
    DOUBLE KNOWLEDGE: Claim same knowledge twice for double credit.

    Goal: Can we get credit for the same knowledge multiple times?
    """
    print_header("ATTACK 11: DOUBLE KNOWLEDGE CLAIM")
    print("  Attempting to claim same knowledge twice...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add same knowledge twice
        content = "E = mc²"
        summary = "Einstein's equation"

        tx1 = chain.add_knowledge(content, summary, "node_1", 0.9)
        tx2 = chain.add_knowledge(content, summary, "node_2", 0.9)  # Same content!

        chain.add_block(pob_proofs=get_valid_proofs())

        # Check if both were accepted
        if len(chain.blocks[-1].transactions) == 2:
            log_result(
                "Duplicate knowledge in same block",
                False,
                "VULNERABILITY: Same knowledge accepted twice in one block!"
            )
        else:
            log_result("Duplicate knowledge in same block", True, "Duplicates rejected")

        # Try in separate blocks
        chain.add_knowledge(content, summary, "node_3", 0.9)
        chain.add_block(pob_proofs=get_valid_proofs("second"))

        # Search for duplicates
        results = chain.search_knowledge("Einstein")
        if len(results) > 1:
            log_result(
                "Duplicate knowledge across blocks",
                False,
                f"VULNERABILITY: Same knowledge appears {len(results)} times!"
            )
        else:
            log_result("Duplicate knowledge across blocks", True, "Cross-block duplicates prevented")


# =============================================================================
# ATTACK 12: GENESIS MANIPULATION
# =============================================================================

def test_genesis_manipulation():
    """
    GENESIS MANIPULATION: Tamper with the genesis block.

    Goal: Can we change the foundation of the chain?
    """
    print_header("ATTACK 12: GENESIS MANIPULATION")
    print("  Attempting to tamper with genesis block...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        original_genesis_hash = chain.blocks[0].hash

        # Try to modify genesis message
        chain.blocks[0].transactions[0]['message'] = "HACKED GENESIS"

        if chain.validate_chain():
            log_result(
                "Genesis message modification",
                False,
                "VULNERABILITY: Genesis modified and chain still valid!"
            )
        else:
            log_result("Genesis message modification", True, "Chain invalidated")

        # Reset and try different attack
        chain = create_chain(data_dir=tmpdir + "_2")

        # Try to replace genesis entirely
        fake_genesis = create_genesis_block()
        fake_genesis.transactions[0]['message'] = "FAKE GENESIS"
        fake_genesis.hash = fake_genesis.compute_hash()

        chain.blocks[0] = fake_genesis

        if chain.validate_chain() and len(chain.blocks) == 1:
            # Genesis replacement with single block might "work"
            log_result(
                "Genesis replacement",
                False,
                "VULNERABILITY: Genesis can be replaced!"
            )
        else:
            log_result("Genesis replacement", True, "Genesis replacement detected")


# =============================================================================
# ATTACK 13: EMPTY BLOCK SPAM
# =============================================================================

def test_empty_block_spam():
    """
    EMPTY BLOCK SPAM: Flood chain with empty blocks.

    Goal: Can we bloat the chain with no useful content?
    """
    print_header("ATTACK 13: EMPTY BLOCK SPAM")
    print("  Attempting to spam empty blocks...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        initial_height = len(chain.blocks)

        # Try to add 100 empty blocks
        empty_blocks_added = 0
        for i in range(100):
            # Don't add any transactions, just try to mine
            success = chain.add_block(pob_proofs=get_valid_proofs(f"spam_{i}"))
            if success:
                empty_blocks_added += 1

        if empty_blocks_added > 0:
            log_result(
                f"Empty block spam ({empty_blocks_added} blocks)",
                False,
                f"VULNERABILITY: {empty_blocks_added} empty blocks accepted!"
            )
        else:
            log_result("Empty block spam", True, "Empty blocks rejected")


# =============================================================================
# ATTACK 14: NONCE MANIPULATION
# =============================================================================

def test_nonce_manipulation():
    """
    NONCE MANIPULATION: Test if nonce affects validation.

    Goal: Understand nonce's role in PoB (it's φ-derived, not brute-forced).
    """
    print_header("ATTACK 14: NONCE MANIPULATION")
    print("  Testing nonce values...")

    import tempfile

    test_nonces = [
        0,
        -1,
        int(PHI * ABHI_AMU),  # Expected: 833
        2**64,
        random.randint(0, 10000),
    ]

    for nonce in test_nonces:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = create_chain(data_dir=tmpdir)

            chain.add_knowledge(f"Nonce test {nonce}", f"Testing nonce={nonce}", "node", 0.9)

            proofs = get_valid_proofs()
            block = chain.create_block(proofs)

            if block:
                block.header.nonce = nonce
                block.hash = block.compute_hash()

                # Try to add with modified nonce
                previous = chain.blocks[-1]
                if block.validate(previous):
                    log_result(f"Nonce={nonce}", True, "Accepted (nonce not validated)")
                else:
                    log_result(f"Nonce={nonce}", True, "Rejected")

    # Meta-observation
    log_result(
        "Nonce validation exists",
        False,
        "VULNERABILITY: Nonce is not validated - any value works!"
    )


# =============================================================================
# ATTACK 15: PROOF COUNT MANIPULATION
# =============================================================================

def test_proof_count():
    """
    PROOF COUNT: What happens with 2, 4, 100 proofs instead of 3?

    Goal: Test triadic consensus edge cases.
    """
    print_header("ATTACK 15: PROOF COUNT MANIPULATION")
    print("  Testing various proof counts...")

    import tempfile

    test_counts = [0, 1, 2, 3, 4, 10, 100]

    for count in test_counts:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = create_chain(data_dir=tmpdir)

            chain.add_knowledge(f"Proof count {count}", f"Testing {count} proofs", "node", 0.9)

            proofs = [
                {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': f'node_{i}'}
                for i in range(count)
            ]

            try:
                success = chain.add_block(pob_proofs=proofs)

                if count < 3 and success:
                    log_result(
                        f"{count} proofs",
                        False,
                        f"VULNERABILITY: Block accepted with only {count} proofs!"
                    )
                elif count >= 3 and success:
                    log_result(f"{count} proofs", True, f"Accepted with {count} proofs")
                else:
                    log_result(f"{count} proofs", True, f"Rejected with {count} proofs")

            except ValueError as e:
                if count < 3:
                    log_result(f"{count} proofs", True, f"Correctly raised: {e}")
                else:
                    log_result(f"{count} proofs", False, f"Unexpected error: {e}")


# =============================================================================
# ATTACK 16: SAME NODE MULTIPLE PROOFS
# =============================================================================

def test_same_node_proofs():
    """
    SAME NODE PROOFS: Can one node provide all 3 proofs?

    Goal: Break triadic by having one identity sign 3 times.
    """
    print_header("ATTACK 16: SAME NODE MULTIPLE PROOFS")
    print("  Can one node provide all 3 proofs?...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        chain.add_knowledge("Same node attack", "Single node triadic", "attacker", 0.9)

        # All proofs from same node
        same_node_proofs = [
            {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': 'same_node'},
            {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': 'same_node'},
            {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': 'same_node'},
        ]

        success = chain.add_block(pob_proofs=same_node_proofs)

        if success:
            log_result(
                "Same node 3 proofs",
                False,
                "VULNERABILITY: Single node can provide all triadic proofs!"
            )
        else:
            log_result("Same node 3 proofs", True, "Rejected - requires unique nodes")


# =============================================================================
# MAIN
# =============================================================================

def run_all_attacks():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  BAZINGA PoB ADVERSARIAL TESTING - ROUND 2".center(58) + "║")
    print("║" + "  'Break it harder.'".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    test_chain_fork()
    test_transaction_malleability()
    test_merkle_attack()
    test_time_warp()
    test_double_knowledge()
    test_genesis_manipulation()
    test_empty_block_spam()
    test_nonce_manipulation()
    test_proof_count()
    test_same_node_proofs()

    print()
    print("=" * 60)
    print("  ROUND 2 SUMMARY")
    print("=" * 60)
    print(f"  ✅ Passed: {len(RESULTS['passed'])}")
    print(f"  ❌ Failed: {len(RESULTS['failed'])}")
    print(f"  🚨 Vulnerabilities: {len(RESULTS['vulnerabilities'])}")
    print()

    if RESULTS['vulnerabilities']:
        print("  NEW VULNERABILITIES FOUND:")
        for name, details in RESULTS['vulnerabilities']:
            print(f"    • {name}")
            print(f"      {details}")
        print()

    print("=" * 60)

    return len(RESULTS['failed']) == 0


if __name__ == "__main__":
    success = run_all_attacks()
    sys.exit(0 if success else 1)
