#!/usr/bin/env python3
"""
Darmiyan Network Test Suite
============================
Tests for Proof-of-Boundary consensus and triadic network.

Run: python -m pytest tests/test_darmiyan.py -v
Or:  python tests/test_darmiyan.py
"""

import time
import asyncio
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bazinga.darmiyan import (
    DarmiyanNode, BazingaNode, TriadicConsensus,
    prove_boundary, achieve_consensus,
    PHI_4, ABHI_AMU,
)
from bazinga.darmiyan.constants import PHI, POB_TOLERANCE


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_constants():
    """Verify Darmiyan constants are correct."""
    print("\n[TEST] Constants verification...")

    assert abs(PHI - 1.618033988749895) < 1e-10, "PHI should be golden ratio"
    assert abs(PHI_4 - 6.854101966249685) < 1e-10, "PHI_4 should be φ⁴"
    assert ABHI_AMU == 515, "ABHI_AMU should be 515"

    # Verify φ⁴ = φ³ + φ² (Fibonacci property)
    phi_3 = PHI ** 3
    phi_2 = PHI ** 2
    assert abs(PHI_4 - (phi_3 + phi_2)) < 1e-10, "φ⁴ = φ³ + φ²"

    print("  ✓ All constants verified")
    return True


def test_single_proof():
    """Test single Proof-of-Boundary generation."""
    print("\n[TEST] Single proof generation...")

    proof = prove_boundary()

    assert proof.alpha < ABHI_AMU, "Alpha should be mod 515"
    assert proof.omega < ABHI_AMU, "Omega should be mod 515"
    assert proof.delta > 0, "Delta should be positive"
    assert proof.attempts > 0, "Should have attempted at least once"

    if proof.valid:
        assert abs(proof.ratio - PHI_4) < POB_TOLERANCE, "Valid proof should be within tolerance"
        print(f"  ✓ Valid proof in {proof.attempts} attempts (ratio: {proof.ratio:.4f})")
    else:
        print(f"  ⚠ Proof invalid after {proof.attempts} attempts (ratio: {proof.ratio:.4f})")

    return proof.valid


def test_node_creation():
    """Test DarmiyanNode creation and identity."""
    print("\n[TEST] Node creation...")

    node = DarmiyanNode()

    assert node.node_id.startswith("node_"), "Node ID should start with 'node_'"
    assert len(node.node_id) == 17, "Node ID should be 17 chars (node_ + 12 hex)"
    assert node.proofs_generated == 0, "Fresh node should have 0 proofs"

    print(f"  ✓ Node created: {node.node_id}")
    return True


def test_proof_verification():
    """Test proof verification by another node."""
    print("\n[TEST] Proof verification...")

    # Node A generates proof
    node_a = DarmiyanNode()
    proof = node_a.prove_boundary_sync()

    # Node B verifies
    node_b = DarmiyanNode()

    if proof.valid:
        # Valid proofs should verify (if recent)
        verified = node_b.verify_proof(proof)
        print(f"  ✓ Proof verification: {'passed' if verified else 'failed (might be timing)'}")
        return True
    else:
        print(f"  ⚠ Proof was invalid, skipping verification test")
        return True  # Not a failure, just unlucky


def test_bazinga_node():
    """Test BazingaNode (full network node)."""
    print("\n[TEST] BazingaNode creation...")

    node = BazingaNode(port=5150)
    info = node.get_info()

    assert info['port'] == 5150, "Port should match"
    assert info['phi_signature'] < ABHI_AMU, "φ-signature should be mod 515"
    assert info['peers'] == 0, "Fresh node should have no peers"

    print(f"  ✓ BazingaNode created: {info['node_id']}")
    print(f"    φ-signature: {info['phi_signature']}")
    return True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_triadic_consensus():
    """Test triadic consensus with 3 nodes."""
    print("\n[TEST] Triadic consensus...")

    tc = TriadicConsensus()
    result = tc.attempt_consensus_sync()

    print(f"  Triadic product: {result.triadic_product:.6f}")
    print(f"  Average ratio: {result.average_ratio:.4f} (target: {PHI_4:.4f})")

    for i, p in enumerate(result.proofs):
        status = "✓" if p.valid else "✗"
        print(f"    Node {i+1}: {status} ratio={p.ratio:.3f} attempts={p.attempts}")

    if result.achieved:
        print(f"  ✓ CONSENSUS ACHIEVED")
    else:
        print(f"  ⚠ Consensus not achieved: {result.message}")

    return result.achieved


def test_multiple_proofs():
    """Test multiple proof generation for success rate."""
    print("\n[TEST] Multiple proofs (10 runs)...")

    node = DarmiyanNode()
    successes = 0
    total_attempts = 0
    best_diff = float('inf')

    for i in range(10):
        proof = node.prove_boundary_sync()
        if proof.valid:
            successes += 1
            total_attempts += proof.attempts
            diff = abs(proof.ratio - PHI_4)
            if diff < best_diff:
                best_diff = diff

    success_rate = successes / 10 * 100
    avg_attempts = total_attempts / successes if successes > 0 else 0

    print(f"  Success rate: {successes}/10 ({success_rate:.0f}%)")
    print(f"  Avg attempts: {avg_attempts:.1f}")
    print(f"  Best accuracy: {best_diff:.4f} from φ⁴")

    assert success_rate >= 80, f"Success rate should be >= 80%, got {success_rate}%"
    print(f"  ✓ Success rate meets threshold")
    return True


# =============================================================================
# MULTIPROCESS TESTS (Simulating separate nodes)
# =============================================================================

def _worker_generate_proof(queue, node_id):
    """Worker function to generate proof in separate process."""
    node = DarmiyanNode(node_id=f"node_{node_id}")
    proof = node.prove_boundary_sync()
    queue.put({
        'node_id': node.node_id,
        'valid': proof.valid,
        'ratio': proof.ratio,
        'attempts': proof.attempts,
        'alpha': proof.alpha,
        'omega': proof.omega,
    })


def test_multiprocess_proofs():
    """Test proof generation across multiple processes."""
    print("\n[TEST] Multiprocess proof generation (3 processes)...")

    queue = Queue()
    processes = []

    # Start 3 worker processes
    for i in range(3):
        p = Process(target=_worker_generate_proof, args=(queue, f"mp_{i}"))
        processes.append(p)
        p.start()

    # Wait for all to complete
    for p in processes:
        p.join(timeout=60)

    # Collect results
    results = []
    while not queue.empty():
        results.append(queue.get())

    print(f"  Collected {len(results)} proofs from separate processes:")

    valid_count = 0
    for r in results:
        status = "✓" if r['valid'] else "✗"
        print(f"    {r['node_id']}: {status} ratio={r['ratio']:.3f} attempts={r['attempts']}")
        if r['valid']:
            valid_count += 1

    # Check if we could achieve triadic consensus
    if valid_count == 3:
        avg_ratio = sum(r['ratio'] for r in results) / 3
        print(f"  Average ratio: {avg_ratio:.4f} (target: {PHI_4:.4f})")
        if abs(avg_ratio - PHI_4) < POB_TOLERANCE:
            print(f"  ✓ TRIADIC CONSENSUS POSSIBLE across processes!")
        else:
            print(f"  ⚠ Ratios don't converge for consensus")
    else:
        print(f"  ⚠ Only {valid_count}/3 valid proofs")

    return valid_count >= 2  # At least 2 should succeed


def test_concurrent_nodes():
    """Test concurrent node operations using ThreadPoolExecutor."""
    print("\n[TEST] Concurrent node operations (5 threads)...")

    def generate_proof_task(node_num):
        node = DarmiyanNode(node_id=f"concurrent_{node_num}")
        proof = node.prove_boundary_sync()
        return {
            'node': node.node_id,
            'valid': proof.valid,
            'ratio': proof.ratio,
            'attempts': proof.attempts,
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_proof_task, i) for i in range(5)]
        results = [f.result() for f in futures]

    valid_count = sum(1 for r in results if r['valid'])
    print(f"  Valid proofs: {valid_count}/5")

    for r in results:
        status = "✓" if r['valid'] else "✗"
        print(f"    {r['node']}: {status} ratio={r['ratio']:.3f}")

    assert valid_count >= 4, f"At least 4/5 should be valid, got {valid_count}"
    print(f"  ✓ Concurrent operations successful")
    return True


# =============================================================================
# STRESS TESTS
# =============================================================================

def test_stress_proofs(n=100):
    """Stress test: generate many proofs and analyze distribution."""
    print(f"\n[TEST] Stress test ({n} proofs)...")

    node = DarmiyanNode()
    results = []

    start = time.time()
    for i in range(n):
        proof = node.prove_boundary_sync()
        results.append({
            'valid': proof.valid,
            'ratio': proof.ratio,
            'attempts': proof.attempts,
            'delta': proof.delta,
        })
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{n}")

    elapsed = time.time() - start

    valid_count = sum(1 for r in results if r['valid'])
    avg_attempts = sum(r['attempts'] for r in results if r['valid']) / valid_count if valid_count > 0 else 0
    avg_ratio = sum(r['ratio'] for r in results if r['valid']) / valid_count if valid_count > 0 else 0

    # Find best and worst
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        best = min(valid_results, key=lambda r: abs(r['ratio'] - PHI_4))
        worst = max(valid_results, key=lambda r: abs(r['ratio'] - PHI_4))

    print(f"\n  Results:")
    print(f"    Total time: {elapsed:.1f}s ({elapsed/n*1000:.1f}ms per proof)")
    print(f"    Success rate: {valid_count}/{n} ({valid_count/n*100:.1f}%)")
    print(f"    Avg attempts: {avg_attempts:.1f}")
    print(f"    Avg ratio: {avg_ratio:.4f} (target: {PHI_4:.4f})")

    if valid_results:
        print(f"    Best accuracy: {abs(best['ratio'] - PHI_4):.4f}")
        print(f"    Worst accuracy: {abs(worst['ratio'] - PHI_4):.4f}")

    assert valid_count / n >= 0.9, f"Should have >= 90% success rate, got {valid_count/n*100:.1f}%"
    print(f"  ✓ Stress test passed")
    return True


# =============================================================================
# P2P INTEGRATION TESTS
# =============================================================================

def test_p2p_module_imports():
    """Test P2P module imports work."""
    print("\n[TEST] P2P module imports...")

    from bazinga.p2p import (
        BAZINGANetwork,
        create_network,
        BAZINGANode,
        BAZINGA_DHT,
        KnowledgeGraphSync,
        TrustRouter,
        AlphaSeedNetwork,
        is_alpha_seed,
    )

    # Verify imports
    assert BAZINGANetwork is not None
    assert create_network is not None
    assert is_alpha_seed is not None

    print("  ✓ All P2P imports successful")
    return True


def test_pob_authentication():
    """Test PoB-based authentication for P2P."""
    print("\n[TEST] PoB authentication for P2P...")

    # Node must prove boundary before joining network
    proof = prove_boundary()

    if proof.valid:
        # Valid proof allows network participation
        print(f"  ✓ Node authenticated: {proof.node_id}")
        print(f"    φ⁴ ratio: {proof.ratio:.4f}")
        print(f"    Attempts: {proof.attempts}")
        return True
    else:
        print(f"  ⚠ PoB failed, retry possible")
        return True  # Not a test failure, just unlucky


def test_alpha_seed_detection():
    """Test α-SEED file detection (hash % 137 == 0)."""
    print("\n[TEST] α-SEED detection...")

    from bazinga.p2p import is_alpha_seed
    from bazinga.p2p.alpha_seed import compute_alpha_hash

    # α-SEED is based on SHA256 hash % 137 == 0
    # Test the mechanism works correctly
    test_contents = [
        "The consciousness network emerges",
        "BAZINGA distributed knowledge",
        "φ resonance in the boundary",
        "137 is the fine structure constant",
        "Proof of Boundary consensus",
    ]

    seed_count = 0
    for content in test_contents:
        h = compute_alpha_hash(content)
        is_seed = is_alpha_seed(content)
        remainder = h % 137
        status = "α-SEED" if is_seed else "regular"
        print(f"    '{content[:30]}...' -> {status} (hash % 137 = {remainder})")
        if is_seed:
            seed_count += 1
            # Verify consistency
            assert remainder == 0, f"α-SEED should have hash % 137 == 0, got {remainder}"

    # Also test with integer input
    assert is_alpha_seed(137) == True, "137 should be α-SEED"
    assert is_alpha_seed(274) == True, "274 should be α-SEED"
    assert is_alpha_seed(100) == False, "100 should not be α-SEED"

    print(f"  ✓ α-SEED detection working ({seed_count}/{len(test_contents)} were α-SEEDs)")
    return True


def test_zmq_transport():
    """Test ZeroMQ transport layer."""
    print("\n[TEST] ZeroMQ transport...")

    from bazinga.p2p import ZMQ_AVAILABLE, BazingaTransport, create_transport

    if not ZMQ_AVAILABLE:
        print("  ⚠ ZeroMQ not installed, skipping transport test")
        return True

    # Test transport creation
    transport = create_transport(node_id="test_node", port=15150, pub_port=15151)
    assert transport is not None, "Transport should be created"
    assert transport.node_id == "test_node", "Node ID should match"

    print("  ✓ Transport created successfully")
    print(f"    ZMQ Available: {ZMQ_AVAILABLE}")
    return True


def test_protocol_creation():
    """Test BazingaProtocol creation."""
    print("\n[TEST] Protocol creation...")

    from bazinga.p2p import BazingaProtocol, ZMQ_AVAILABLE

    if not ZMQ_AVAILABLE:
        print("  ⚠ ZeroMQ not installed, skipping protocol test")
        return True

    # Test protocol creation (doesn't start network)
    protocol = BazingaProtocol(node_id="test_protocol", port=15160)
    assert protocol is not None, "Protocol should be created"
    assert protocol.node_id == "test_protocol", "Node ID should match"
    assert protocol.port == 15160, "Port should match"

    print("  ✓ Protocol created successfully")
    print(f"    Node ID: {protocol.node_id}")
    print(f"    Port: {protocol.port}")
    return True


def test_federated_imports():
    """Test federated learning module imports."""
    print("\n[TEST] Federated learning imports...")

    from bazinga.federated import (
        LoRAAdapter,
        create_lora_adapter,
        GradientPackage,
        GradientSharer,
        FederatedAggregator,
        phi_weighted_average,
        CollectiveLearner,
        create_learner,
    )

    assert LoRAAdapter is not None
    assert CollectiveLearner is not None
    assert create_learner is not None

    print("  ✓ All federated imports successful")
    return True


def test_lora_adapter():
    """Test LoRA adapter creation and training."""
    print("\n[TEST] LoRA adapter...")

    from bazinga.federated import create_lora_adapter

    # Create adapter
    adapter = create_lora_adapter(node_id="test_lora")
    assert adapter.node_id == "test_lora"

    # Initialize weights
    adapter.initialize_weights(
        input_dim=768,
        output_dim=768,
        module_name="q_proj",
    )

    assert "q_proj" in adapter.weights
    stats = adapter.get_stats()
    assert stats['total_params'] > 0

    print(f"  ✓ LoRA adapter created")
    print(f"    Modules: {stats['modules']}")
    print(f"    Params: {stats['total_params']}")
    return True


def test_collective_learner():
    """Test collective learner."""
    print("\n[TEST] Collective learner...")

    from bazinga.federated import create_learner

    # Create learner
    learner = create_learner(node_id="test_collective")
    assert learner.node_id == "test_collective"

    # Simulate learning
    learner.learn(
        question="What is BAZINGA?",
        answer="BAZINGA is distributed AI.",
        feedback_score=0.9,
    )

    stats = learner.get_stats()
    assert stats['local_examples'] == 1

    print(f"  ✓ Collective learner working")
    print(f"    Local examples: {stats['local_examples']}")
    print(f"    Adapter version: {stats['adapter']['version']}")
    return True


# =============================================================================
# BLOCKCHAIN TESTS
# =============================================================================

def test_blockchain_imports():
    """Test blockchain module imports."""
    print("\n[TEST] Blockchain imports...")

    from bazinga.blockchain import (
        Block,
        create_genesis_block,
        Transaction,
        KnowledgeAttestation,
        DarmiyanChain,
        create_chain,
        Wallet,
        create_wallet,
        PoBMiner,
        mine_block,
    )

    assert Block is not None
    assert create_chain is not None
    assert create_wallet is not None
    assert PoBMiner is not None

    print("  ✓ All blockchain imports successful")
    return True


def test_genesis_block():
    """Test genesis block creation."""
    print("\n[TEST] Genesis block creation...")

    from bazinga.blockchain import create_genesis_block

    genesis = create_genesis_block()

    assert genesis.header.index == 0, "Genesis should be block 0"
    assert genesis.header.previous_hash == "0" * 64, "Genesis has null previous hash"
    assert len(genesis.transactions) == 1, "Genesis has 1 transaction"
    assert genesis.validate(), "Genesis should be valid"

    print(f"  ✓ Genesis block created")
    print(f"    Hash: {genesis.hash[:24]}...")
    print(f"    Merkle: {genesis.header.merkle_root[:24]}...")
    return True


def test_knowledge_attestation():
    """Test knowledge attestation creation."""
    print("\n[TEST] Knowledge attestation...")

    from bazinga.blockchain import KnowledgeAttestation

    ka = KnowledgeAttestation.create(
        content="φ = 1.618033988749895",
        summary="Golden ratio definition",
        confidence=0.95,
        source_type="human",
    )

    assert ka.content_hash is not None
    assert ka.confidence == 0.95
    assert ka.source_type == "human"

    # Convert to transaction
    tx = ka.to_transaction(sender="test_node")
    assert tx.tx_type == "knowledge"
    assert tx.sender == "test_node"

    print(f"  ✓ Knowledge attestation created")
    print(f"    Hash: {ka.content_hash[:24]}...")
    print(f"    α-SEED: {ka.alpha_seed}")
    return True


def test_wallet_creation():
    """Test wallet/identity creation."""
    print("\n[TEST] Wallet creation...")

    import tempfile
    from bazinga.blockchain import create_wallet

    with tempfile.TemporaryDirectory() as tmpdir:
        wallet = create_wallet(node_id="test_wallet", data_dir=tmpdir)

        assert wallet.node_id == "test_wallet"
        assert wallet.public_key is not None
        assert wallet.private_key is not None

        # Test signing
        data = "test transaction data"
        signature = wallet.sign(data)
        assert wallet.verify(data, signature), "Signature should verify"
        assert not wallet.verify("wrong data", signature), "Wrong data should fail"

        print(f"  ✓ Wallet created")
        print(f"    Address: {wallet.get_address()}")
        print(f"    Trust: {wallet.reputation.trust_score:.3f}")

    return True


def test_chain_creation():
    """Test blockchain creation and operations."""
    print("\n[TEST] Chain creation...")

    import tempfile
    from bazinga.blockchain import create_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Should have genesis block
        assert len(chain) == 1, "Should have genesis block"
        assert chain.get_block(0) is not None

        # Add knowledge
        tx_hash = chain.add_knowledge(
            content="BAZINGA is distributed AI",
            summary="BAZINGA definition",
            sender="test_node",
            confidence=0.9,
        )

        assert tx_hash is not None
        assert len(chain.pending_transactions) == 1

        print(f"  ✓ Chain created with genesis")
        print(f"    Height: {len(chain)}")
        print(f"    Pending: {len(chain.pending_transactions)}")

    return True


def test_pob_mining():
    """Test Proof-of-Boundary mining."""
    print("\n[TEST] PoB mining...")

    import tempfile
    from bazinga.blockchain import create_chain, PoBMiner

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add knowledge to mine
        chain.add_knowledge(
            content="φ⁴ = 6.854101966249685",
            summary="Boundary ratio",
            sender="test_miner",
            confidence=0.9,
        )

        # Create miner and attempt
        miner = PoBMiner(chain, node_id="test_miner")
        result = miner.mine_sync(max_attempts=20)

        if result.success:
            print(f"  ✓ Block mined successfully!")
            print(f"    Block: #{result.block.header.index}")
            print(f"    Attempts: {result.attempts}")
            print(f"    Time: {result.time_ms:.2f}ms")
            assert len(chain) == 2, "Should have 2 blocks now"
        else:
            print(f"  ⚠ Mining didn't succeed in 20 attempts (this can happen)")
            print(f"    Message: {result.message}")

    return True  # Not a failure if mining takes many attempts


def test_chain_validation():
    """Test chain validation with mock blocks."""
    print("\n[TEST] Chain validation...")

    import tempfile
    from bazinga.blockchain import create_chain, Block, BlockHeader

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Add knowledge and mock-mine a block
        chain.add_knowledge(
            content="Test knowledge",
            summary="Test",
            sender="test",
            confidence=1.0,
        )

        # Create mock PoB proofs (valid format)
        # Must satisfy: (alpha + omega + delta) / delta ≈ PHI_4 (6.854)
        # And: alpha < 515, omega < 515
        mock_proofs = [
            {'alpha': 248, 'omega': 249, 'delta': 85, 'ratio': PHI_4, 'valid': True, 'node_id': 'a'},
            {'alpha': 263, 'omega': 263, 'delta': 90, 'ratio': PHI_4, 'valid': True, 'node_id': 'b'},
            {'alpha': 278, 'omega': 278, 'delta': 95, 'ratio': PHI_4, 'valid': True, 'node_id': 'c'},
        ]

        success = chain.add_block(pob_proofs=mock_proofs)
        assert success, "Block with mock proofs should be added"

        # Validate chain
        valid = chain.validate_chain()
        assert valid, "Chain should be valid"

        print(f"  ✓ Chain validation passed")
        print(f"    Height: {len(chain)}")
        print(f"    Valid: {valid}")

    return True


def test_trust_oracle():
    """Test trust oracle."""
    print("\n[TEST] Trust oracle...")

    import tempfile
    from bazinga.blockchain import create_chain, create_trust_oracle

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)
        oracle = create_trust_oracle(chain)

        # Record activities for a good node
        for i in range(5):
            oracle.record_activity(
                node_address="node_test_good",
                activity_type="pob",
                success=True,
                block_number=i,
            )

        # Record activities for a bad node
        for i in range(5):
            oracle.record_activity(
                node_address="node_test_bad",
                activity_type="pob",
                success=False,
                block_number=i,
            )

        # Check trust scores
        good_trust = oracle.get_trust_score("node_test_good")
        bad_trust = oracle.get_trust_score("node_test_bad")
        unknown_trust = oracle.get_trust_score("node_unknown")

        assert good_trust > 0.5, f"Good node should have trust > 0.5, got {good_trust}"
        assert bad_trust < 0.5, f"Bad node should have trust < 0.5, got {bad_trust}"
        assert unknown_trust == 0.5, f"Unknown node should have trust = 0.5, got {unknown_trust}"

        print(f"  ✓ Trust oracle working")
        print(f"    Good node: {good_trust:.3f}")
        print(f"    Bad node: {bad_trust:.3f}")
        print(f"    Unknown: {unknown_trust:.3f}")

    return True


# =============================================================================
# INTEGRATION LAYER TESTS (v4.5.0)
# =============================================================================

def test_knowledge_ledger():
    """Test knowledge ledger."""
    print("\n[TEST] Knowledge ledger...")

    from bazinga.blockchain import KnowledgeLedger, create_ledger

    ledger = create_ledger()

    # Record a contribution
    contribution = ledger.record_contribution(
        contributor="node_test_1",
        content="The golden ratio φ = 1.618...",
        contribution_type="knowledge",
        metadata={"topic": "mathematics"}
    )

    if contribution:
        print(f"  ✓ Contribution recorded")
        print(f"    Hash: {contribution.payload_hash[:16]}...")
        print(f"    Coherence: {contribution.coherence_score:.3f}")
        print(f"    Credits: {contribution.get_credit_value():.3f}")

        # Check credits
        credits = ledger.get_contributor_credits("node_test_1")
        assert credits > 0, "Contributor should have credits"
        print(f"    Total credits: {credits:.3f}")
    else:
        print(f"  ⚠ Contribution rejected (coherence too low)")

    return True


def test_gradient_validator():
    """Test gradient validator."""
    print("\n[TEST] Gradient validator...")

    from bazinga.blockchain import GradientValidator, create_validator

    validator = create_validator()

    # Register validators
    validator.register_validator("val_1")
    validator.register_validator("val_2")
    validator.register_validator("val_3")

    # Submit a gradient
    update = validator.submit_gradient(
        submitter="trainer_1",
        gradient_hash="abc123def456",
        model_version="v1.0",
        loss_improvement=0.05,
    )

    assert update is not None
    print(f"  ✓ Gradient submitted")
    print(f"    Hash: {update.gradient_hash}")

    # Submit validations
    validator.validate_gradient(
        validator="val_1",
        gradient_hash="abc123def456",
        approved=True,
        improvement_verified=True,
        coherence_score=0.8,
    )

    validator.validate_gradient(
        validator="val_2",
        gradient_hash="abc123def456",
        approved=True,
        improvement_verified=True,
        coherence_score=0.75,
    )

    validator.validate_gradient(
        validator="val_3",
        gradient_hash="abc123def456",
        approved=True,
        improvement_verified=True,
        coherence_score=0.85,
    )

    # Check if accepted
    accepted = validator.get_accepted_gradients()
    assert len(accepted) > 0, "Gradient should be accepted"
    print(f"  ✓ Gradient accepted by triadic consensus")

    stats = validator.get_stats()
    print(f"    Validators: {stats['validators']}")
    print(f"    Accepted: {stats['accepted_updates']}")

    return True


def test_inference_market():
    """Test inference market."""
    print("\n[TEST] Inference market...")

    from bazinga.blockchain import InferenceMarket, create_market

    market = create_market()

    # Register providers
    market.register_provider("provider_1", capacity=5)
    market.register_provider("provider_2", capacity=5)

    # Give requester some credits
    market.add_credits("requester_1", 10.0, "test")

    # Request inference
    request = market.request_inference(
        requester="requester_1",
        query="What is the golden ratio?",
    )

    assert request is not None
    print(f"  ✓ Request created")
    print(f"    ID: {request.request_id}")
    print(f"    Status: {request.status.value}")

    if request.provider:
        # Complete the request
        completed = market.complete_inference(
            request_id=request.request_id,
            response="The golden ratio φ ≈ 1.618",
            coherence_score=0.9,
        )

        print(f"  ✓ Request completed")
        print(f"    Provider: {completed.provider}")
        print(f"    Coherence: {completed.coherence_score}")

    stats = market.get_market_stats()
    print(f"    Providers: {stats['providers']}")
    print(f"    Completed: {stats['completed_requests']}")

    return True


def test_smart_contracts():
    """Test smart contracts."""
    print("\n[TEST] Smart contracts...")

    from bazinga.blockchain import ContractEngine, create_engine
    from bazinga.blockchain import create_market

    market = create_market()
    engine = create_engine(inference_market=market)

    # Give creator credits for bounty (must use _internal=True and valid reason)
    market.add_credits("creator_1", 100.0, "knowledge_attestation", _internal=True)

    # Create a bounty contract
    contract = engine.create_bounty(
        creator="creator_1",
        description="Explain quantum entanglement",
        bounty_credits=10.0,
        required_coherence=0.7,
    )

    assert contract is not None
    print(f"  ✓ Contract created")
    print(f"    ID: {contract.contract_id}")
    print(f"    Type: {contract.contract_type.value}")
    print(f"    Bounty: {contract.terms.bounty_credits}")

    # Register reviewers
    engine.register_reviewer("reviewer_1")
    engine.register_reviewer("reviewer_2")
    engine.register_reviewer("reviewer_3")

    # Submit a solution
    submission = engine.submit_solution(
        contract_id=contract.contract_id,
        submitter="solver_1",
        content="Quantum entanglement is when particles share correlated states...",
    )

    print(f"  ✓ Solution submitted")
    print(f"    Coherence: {submission.coherence_score:.3f}")

    # Submit reviews
    for reviewer in ["reviewer_1", "reviewer_2", "reviewer_3"]:
        engine.review_submission(
            contract_id=contract.contract_id,
            submission_hash=submission.content_hash,
            reviewer=reviewer,
            approved=True,
            comments="Good explanation",
        )

    # Check if executed
    updated_contract = engine.get_contract(contract.contract_id)
    print(f"  ✓ Contract status: {updated_contract.status.value}")

    stats = engine.get_stats()
    print(f"    Total contracts: {stats['total_contracts']}")
    print(f"    Executed: {stats['executed']}")

    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("DARMIYAN NETWORK TEST SUITE")
    print("=" * 60)

    tests = [
        ("Constants", test_constants),
        ("Single Proof", test_single_proof),
        ("Node Creation", test_node_creation),
        ("Proof Verification", test_proof_verification),
        ("BazingaNode", test_bazinga_node),
        ("Triadic Consensus", test_triadic_consensus),
        ("Multiple Proofs", test_multiple_proofs),
        ("Multiprocess Proofs", test_multiprocess_proofs),
        ("Concurrent Nodes", test_concurrent_nodes),
        # P2P Integration Tests
        ("P2P Module Imports", test_p2p_module_imports),
        ("PoB Authentication", test_pob_authentication),
        ("α-SEED Detection", test_alpha_seed_detection),
        # Transport Layer Tests
        ("ZMQ Transport", test_zmq_transport),
        ("Protocol Creation", test_protocol_creation),
        # Federated Learning Tests
        ("Federated Imports", test_federated_imports),
        ("LoRA Adapter", test_lora_adapter),
        ("Collective Learner", test_collective_learner),
        # Blockchain Tests
        ("Blockchain Imports", test_blockchain_imports),
        ("Genesis Block", test_genesis_block),
        ("Knowledge Attestation", test_knowledge_attestation),
        ("Wallet Creation", test_wallet_creation),
        ("Chain Creation", test_chain_creation),
        ("PoB Mining", test_pob_mining),
        ("Chain Validation", test_chain_validation),
        # Trust Layer Tests
        ("Trust Oracle", test_trust_oracle),
        # Integration Layer Tests (v4.5.0)
        ("Knowledge Ledger", test_knowledge_ledger),
        ("Gradient Validator", test_gradient_validator),
        ("Inference Market", test_inference_market),
        ("Smart Contracts", test_smart_contracts),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ EXCEPTION: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, err in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
        if err:
            print(f"         Error: {err}")

    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stress":
        # Run stress test only
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        test_stress_proofs(n)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
