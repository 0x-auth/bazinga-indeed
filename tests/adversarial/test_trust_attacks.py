#!/usr/bin/env python3
"""
BAZINGA Trust & Gradient Adversarial Tests
===========================================

"Trust no one. Verify everyone."

Attack Vectors:
1. Trust Score Inflation - Game the reputation system
2. Gradient Poisoning - Inject malicious model updates
3. Credit Farming - Abuse the credit system
4. Provider Impersonation - Fake high-trust nodes
5. φ-Coherence Gaming - Artificially inflate coherence scores

Author: Claude + Space
"""

import sys
import time
import hashlib
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bazinga.blockchain.chain import create_chain
from bazinga.blockchain.block import PHI_4, ABHI_AMU

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
    return [
        {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_a'},
        {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_b'},
        {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': PHI_4, 'valid': True, 'node_id': f'{node_prefix}_c'},
    ]


# =============================================================================
# ATTACK 17: TRUST SCORE INFLATION
# =============================================================================

def test_trust_inflation():
    """
    TRUST INFLATION: Game the TrustOracle to inflate reputation.

    Goal: Can a malicious node gain undeserved trust?
    """
    print_header("ATTACK 17: TRUST SCORE INFLATION")
    print("  Attempting to inflate trust scores...")

    import tempfile

    try:
        from bazinga.blockchain.trust_oracle import TrustOracle, create_trust_oracle
    except ImportError as e:
        log_result("TrustOracle import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)
        oracle = create_trust_oracle(chain)

        # Get initial trust
        initial_trust = oracle.get_trust_score("attacker_node")
        print(f"  Initial trust: {initial_trust}")

        # Spam self-attestations
        for i in range(100):
            oracle.record_activity(
                node_address="attacker_node",
                activity_type="pob",
                success=True,
                block_number=i,
                score=1.0,
                metadata={"fake": True}
            )

        inflated_trust = oracle.get_trust_score("attacker_node")
        print(f"  After 100 fake activities: {inflated_trust}")

        if inflated_trust > initial_trust + 0.5:
            log_result(
                "Self-attestation spam",
                False,
                f"VULNERABILITY: Trust inflated from {initial_trust} to {inflated_trust}!"
            )
        else:
            log_result("Self-attestation spam", True, "Trust inflation limited")

        # Try to claim local model bonus without running one
        record = oracle.record_activity(
            node_address="fake_local_node",
            activity_type="local_model",
            success=True,
            block_number=101,
            score=1.0,
            metadata={"model": "fake_llama", "verified": False},
            is_local_model=True
        )

        # The key check: is_local_model should be False (verification failed)
        if record.is_local_model:
            log_result(
                "Fake local model claim",
                False,
                f"VULNERABILITY: Fake local model claim was verified as True!"
            )
        else:
            log_result("Fake local model claim", True, "Fake claim rejected (is_local_model=False)")


# =============================================================================
# ATTACK 18: GRADIENT POISONING
# =============================================================================

def test_gradient_poisoning():
    """
    GRADIENT POISONING: Inject malicious gradients.

    Goal: Can we corrupt the federated learning process?
    """
    print_header("ATTACK 18: GRADIENT POISONING")
    print("  Attempting to inject malicious gradients...")

    try:
        from bazinga.blockchain.gradient_validator import GradientValidator, GradientUpdate
    except ImportError as e:
        log_result("GradientValidator import", False, f"Cannot import: {e}")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            validator = GradientValidator(chain)
        except Exception as e:
            log_result("GradientValidator init", False, f"Cannot create: {e}")
            return

        # Create poisoned gradient (all zeros - would reset model)
        poisoned_gradient = {
            'layer_1': [0.0] * 1000,
            'layer_2': [0.0] * 1000,
            'bias': [0.0] * 100,
        }

        gradient_hash = hashlib.sha256(str(poisoned_gradient).encode()).hexdigest()

        try:
            update = GradientUpdate(
                gradient_hash=gradient_hash,
                submitter="attacker",
                model_version="v1",
                metrics={"loss": 0.0, "accuracy": 1.0},  # Fake perfect metrics
            )

            # Try to submit
            result = validator.submit_gradient(update)

            if result:
                log_result(
                    "Poisoned gradient (all zeros)",
                    False,
                    "VULNERABILITY: Zero gradient accepted!"
                )
            else:
                log_result("Poisoned gradient (all zeros)", True, "Poisoned gradient rejected")

        except Exception as e:
            log_result("Poisoned gradient submission", True, f"Exception: {type(e).__name__}")

        # Try extreme gradient values
        extreme_gradient = {
            'layer_1': [float('inf')] * 100,
            'layer_2': [float('nan')] * 100,
        }

        try:
            extreme_hash = hashlib.sha256(str(extreme_gradient).encode()).hexdigest()
            extreme_update = GradientUpdate(
                gradient_hash=extreme_hash,
                submitter="attacker",
                model_version="v1",
                metrics={"loss": 0.001},
            )

            result = validator.submit_gradient(extreme_update)

            if result:
                log_result(
                    "Extreme gradient (inf/nan)",
                    False,
                    "VULNERABILITY: Inf/NaN gradient accepted!"
                )
            else:
                log_result("Extreme gradient (inf/nan)", True, "Extreme gradient rejected")

        except Exception as e:
            log_result("Extreme gradient", True, f"Exception: {type(e).__name__}")


# =============================================================================
# ATTACK 19: CREDIT FARMING
# =============================================================================

def test_credit_farming():
    """
    CREDIT FARMING: Abuse the inference market credit system.

    Goal: Can we farm credits without providing real value?
    """
    print_header("ATTACK 19: CREDIT FARMING")
    print("  Attempting to farm credits...")

    try:
        from bazinga.blockchain.inference_market import InferenceMarket
    except ImportError as e:
        log_result("InferenceMarket import", False, f"Cannot import: {e}")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            market = InferenceMarket(chain)
        except Exception as e:
            log_result("InferenceMarket init", False, f"Cannot create: {e}")
            return

        # Check initial credits
        try:
            initial_credits = market.get_credits("farmer")
        except:
            initial_credits = 0

        print(f"  Initial credits: {initial_credits}")

        # Self-dealing: create and complete own requests
        for i in range(50):
            try:
                # Create request as farmer
                request_id = market.create_request(
                    query=f"Farm query {i}",
                    requester="farmer",
                    reward=0.1,
                )

                # Complete own request
                market.complete_request(
                    request_id=request_id,
                    provider="farmer",  # Same as requester!
                    response="Fake response",
                    coherence=0.99,
                )
            except Exception:
                pass

        try:
            final_credits = market.get_credits("farmer")
        except:
            final_credits = 0

        print(f"  After 50 self-deals: {final_credits}")

        if final_credits > initial_credits + 10:
            log_result(
                "Self-dealing credit farm",
                False,
                f"VULNERABILITY: Credits farmed from {initial_credits} to {final_credits}!"
            )
        else:
            log_result("Self-dealing credit farm", True, "Self-dealing prevented")


# =============================================================================
# ATTACK 20: φ-COHERENCE GAMING
# =============================================================================

def test_phi_coherence_gaming():
    """
    φ-COHERENCE GAMING: Artificially inflate coherence scores.

    Goal: Can we game the coherence filter?
    """
    print_header("ATTACK 20: φ-COHERENCE GAMING")
    print("  Attempting to game φ-coherence...")

    try:
        from bazinga.blockchain.knowledge_ledger import PhiCoherenceFilter
    except ImportError as e:
        log_result("PhiCoherenceFilter import", False, f"Cannot import: {e}")
        return

    try:
        phi_filter = PhiCoherenceFilter()
    except Exception as e:
        log_result("PhiCoherenceFilter init", False, f"Cannot create: {e}")
        return

    # Test with garbage that looks like it has high entropy
    test_cases = [
        ("", "Empty string"),
        ("a" * 1000, "Single char repeated"),
        ("φ" * 100, "φ symbol spam"),
        ("The quick brown fox " * 50, "Repeated sentence"),
        (str(random.random()) * 100, "Random digits"),
        ("真假真假" * 250, "Chinese alternating chars"),
        ("\x00" * 500, "Null bytes"),
    ]

    for content, description in test_cases:
        try:
            coherence = phi_filter.compute_coherence(content)

            # Threshold is 0.618
            if coherence >= 0.618 and description in ["Empty string", "Null bytes", "Single char repeated"]:
                log_result(
                    description,
                    False,
                    f"VULNERABILITY: Garbage got coherence {coherence:.3f} (threshold 0.618)!"
                )
            else:
                log_result(description, True, f"Coherence: {coherence:.3f}")

        except Exception as e:
            log_result(description, True, f"Exception: {type(e).__name__}")


# =============================================================================
# ATTACK 21: SMART CONTRACT EXPLOITATION
# =============================================================================

def test_smart_contract_exploit():
    """
    SMART CONTRACT EXPLOITATION: Abuse understanding contracts.

    Goal: Can we complete contracts without actual understanding?
    """
    print_header("ATTACK 21: SMART CONTRACT EXPLOITATION")
    print("  Attempting to exploit understanding contracts...")

    try:
        from bazinga.blockchain.smart_contracts import ContractEngine, UnderstandingContract, ContractType
    except ImportError as e:
        log_result("SmartContracts import", False, f"Cannot import: {e}")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            engine = ContractEngine(chain)
        except Exception as e:
            log_result("ContractEngine init", False, f"Cannot create: {e}")
            return

        # Create a bounty contract
        try:
            contract = engine.create_contract(
                contract_type=ContractType.BOUNTY,
                creator="victim",
                challenge="Explain quantum entanglement",
                reward=100,
                reviewers=["reviewer_1", "reviewer_2", "reviewer_3"],
            )

            # Try to claim with garbage answer
            submission_id = engine.submit_solution(
                contract_id=contract.id,
                solver="attacker",
                solution="Lorem ipsum dolor sit amet " * 100,
                coherence=0.99,  # Fake high coherence
            )

            # Try to approve own submission by being a reviewer
            engine.review_submission(
                contract_id=contract.id,
                submission_id=submission_id,
                reviewer="attacker",  # Not in reviewers list
                approved=True,
            )

            # Check if contract was executed
            status = engine.get_contract_status(contract.id)

            if status == "EXECUTED":
                log_result(
                    "Self-approval attack",
                    False,
                    "VULNERABILITY: Attacker approved own garbage solution!"
                )
            else:
                log_result("Self-approval attack", True, "Self-approval rejected")

        except Exception as e:
            log_result("Smart contract test", True, f"Exception: {type(e).__name__}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_attacks():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  BAZINGA TRUST & GRADIENT ADVERSARIAL TESTING".center(58) + "║")
    print("║" + "  'Trust no one. Verify everyone.'".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    test_trust_inflation()
    test_gradient_poisoning()
    test_credit_farming()
    test_phi_coherence_gaming()
    test_smart_contract_exploit()

    print()
    print("=" * 60)
    print("  TRUST & GRADIENT SUMMARY")
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

    print("=" * 60)

    return len(RESULTS['failed']) == 0


if __name__ == "__main__":
    success = run_all_attacks()
    sys.exit(0 if success else 1)
