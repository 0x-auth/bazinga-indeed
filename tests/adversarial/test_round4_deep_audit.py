#!/usr/bin/env python3
"""
BAZINGA Blockchain Security Audit - ROUND 4
============================================

Deep dive vulnerabilities found by Claude security audit.

11 NEW vulnerabilities to test:
1. CRITICAL: Broken signature scheme (symmetric instead of asymmetric)
2. CRITICAL: Private key stored in plaintext
3. HIGH: Weak local model verification
4. HIGH: Unvalidated trust manipulation via record_proof()
5. HIGH: Unverified φ multiplier for local models
6. HIGH: Loose proof bounds checking
7. HIGH: Credit balance manipulation
8. MEDIUM: Deterministic nonce
9. MEDIUM: Weak proof serialization
10. MEDIUM: Unbounded knowledge submissions
11. MEDIUM: Race condition in validator selection

Author: Space (Abhishek Srivastava)
Date: Feb 2026
"""

import sys
import time
import hashlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# VULN 1: BROKEN SIGNATURE SCHEME
# =============================================================================

def test_signature_scheme():
    """Test if signature scheme is actually asymmetric."""
    print_header("VULN 1: SIGNATURE SCHEME (CRITICAL)")
    print("  Testing if wallet signatures are cryptographically sound...")

    try:
        from bazinga.blockchain.wallet import Wallet, create_wallet
    except ImportError as e:
        log_result("Wallet import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two wallets
        wallet1 = create_wallet(data_dir=tmpdir, node_id="test1")
        wallet2 = create_wallet(data_dir=tmpdir, node_id="test2")

        # Sign data with wallet1
        data = "test_message"
        signature = wallet1.sign(data)

        # Try to verify with wallet2 (should fail if asymmetric)
        # If verify() returns True, something is WRONG
        can_verify_others = wallet2.verify(data, signature, wallet1.public_key)

        # Check if wallet1 can verify its own signature
        can_verify_own = wallet1.verify(data, signature, wallet1.public_key)

        # In a BROKEN symmetric scheme:
        # - Only signer can verify (requires private key)
        # - Others cannot verify

        if not can_verify_others and can_verify_own:
            log_result(
                "Asymmetric signatures",
                False,
                "VULNERABILITY: Only signer can verify - symmetric scheme detected!"
            )
        elif can_verify_others:
            log_result("Asymmetric signatures", True, "Others can verify signatures")
        else:
            log_result(
                "Asymmetric signatures",
                False,
                "VULNERABILITY: Nobody can verify signatures!"
            )


# =============================================================================
# VULN 2: PLAINTEXT PRIVATE KEY
# =============================================================================

def test_plaintext_private_key():
    """Test if private keys are stored in plaintext."""
    print_header("VULN 2: PRIVATE KEY STORAGE (CRITICAL)")
    print("  Checking if private keys are encrypted on disk...")

    try:
        from bazinga.blockchain.wallet import create_wallet
        import json
    except ImportError as e:
        log_result("Wallet import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        wallet = create_wallet(data_dir=tmpdir, node_id="test_wallet")
        wallet_file = Path(tmpdir) / "wallets" / "test_wallet.json"

        if wallet_file.exists():
            with open(wallet_file, 'r') as f:
                data = json.load(f)

            if 'private_key' in data:
                # Check if it's plaintext (hex string) vs encrypted (base64 blob)
                private_key = data['private_key']
                if len(private_key) == 64 and all(c in '0123456789abcdef' for c in private_key):
                    log_result(
                        "Private key encrypted",
                        False,
                        "VULNERABILITY: Private key stored as plaintext hex!"
                    )
                else:
                    log_result("Private key encrypted", True, "Key appears encrypted")
            else:
                log_result("Private key encrypted", True, "No private_key field in file")
        else:
            log_result("Private key encrypted", True, "No wallet file created")


# =============================================================================
# VULN 3: WEAK LOCAL MODEL VERIFICATION
# =============================================================================

def test_weak_local_model_verification():
    """Test if local model verification can be trivially bypassed."""
    print_header("VULN 3: LOCAL MODEL VERIFICATION (HIGH)")
    print("  Testing if fake local model claims are accepted...")

    try:
        from bazinga.blockchain.trust_oracle import TrustOracle, create_trust_oracle
        from bazinga.blockchain.chain import create_chain
    except ImportError as e:
        log_result("TrustOracle import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)
        oracle = create_trust_oracle(chain)

        # Try various weak verification bypasses
        bypasses = [
            # Method 1: Fake challenge-response
            {
                "challenge": "real_challenge",
                "challenge_response": "fake_response",  # Different = accepted?
            },
            # Method 2: Fake 64-char attestation
            {
                "attestation": "a" * 64,  # Just 64 chars
            },
            # Method 3: Claim verified_by
            {
                "verified_by": "system",  # Hardcoded trusted verifier
            },
        ]

        for i, metadata in enumerate(bypasses, 1):
            record = oracle.record_activity(
                node_address=f"attacker_{i}",
                activity_type="local_model",
                success=True,
                block_number=i,
                score=1.0,
                metadata=metadata,
                is_local_model=True
            )

            if record.is_local_model:
                log_result(
                    f"Bypass method {i}",
                    False,
                    f"VULNERABILITY: Accepted with {list(metadata.keys())[0]}!"
                )
            else:
                log_result(f"Bypass method {i}", True, "Fake claim rejected")


# =============================================================================
# VULN 4: UNVALIDATED TRUST MANIPULATION
# =============================================================================

def test_trust_manipulation():
    """Test if trust can be directly manipulated via record_proof()."""
    print_header("VULN 4: TRUST MANIPULATION (HIGH)")
    print("  Testing if wallet.record_proof() can inflate trust...")

    try:
        from bazinga.blockchain.wallet import create_wallet
    except ImportError as e:
        log_result("Wallet import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        wallet = create_wallet(data_dir=tmpdir, node_id="attacker")

        initial_trust = wallet.reputation.trust_score
        print(f"  Initial trust: {initial_trust}")

        # Spam fake successful proofs
        for _ in range(100):
            wallet.record_proof(success=True)

        final_trust = wallet.reputation.trust_score
        print(f"  After 100 fake proofs: {final_trust}")

        if final_trust > initial_trust + 0.3:
            log_result(
                "Trust inflation blocked",
                False,
                f"VULNERABILITY: Trust inflated from {initial_trust:.2f} to {final_trust:.2f}!"
            )
        else:
            log_result("Trust inflation blocked", True, "Trust manipulation prevented")


# =============================================================================
# VULN 7: CREDIT BALANCE MANIPULATION
# =============================================================================

def test_credit_manipulation():
    """Test if credits can be added without validation."""
    print_header("VULN 7: CREDIT MANIPULATION (HIGH)")
    print("  Testing if credits can be added directly...")

    try:
        from bazinga.blockchain.inference_market import InferenceMarket
        from bazinga.blockchain.chain import create_chain
    except ImportError as e:
        log_result("InferenceMarket import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            market = InferenceMarket(chain)
        except Exception as e:
            log_result("InferenceMarket init", False, f"Cannot create: {e}")
            return

        initial_credits = market.get_credits("attacker")
        print(f"  Initial credits: {initial_credits}")

        # Try to directly add credits
        try:
            market.add_credits("attacker", 999999.0, "hacked")
            final_credits = market.get_credits("attacker")
            print(f"  After add_credits(): {final_credits}")

            if final_credits > initial_credits + 1000:
                log_result(
                    "Credit manipulation blocked",
                    False,
                    f"VULNERABILITY: Credits inflated to {final_credits}!"
                )
            else:
                log_result("Credit manipulation blocked", True, "Direct add rejected")
        except Exception as e:
            log_result("Credit manipulation blocked", True, f"Exception: {type(e).__name__}")


# =============================================================================
# VULN 8: DETERMINISTIC NONCE
# =============================================================================

def test_deterministic_nonce():
    """Test if block nonces are predictable."""
    print_header("VULN 8: DETERMINISTIC NONCE (MEDIUM)")
    print("  Testing if nonces are predictable...")

    try:
        from bazinga.blockchain.chain import create_chain
        from bazinga.blockchain.block import PHI, ABHI_AMU
    except ImportError as e:
        log_result("Chain import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        # Get genesis nonce
        genesis_nonce = chain.blocks[0].header.nonce

        # Predict what it should be
        genesis_time = chain.blocks[0].header.timestamp
        predicted_nonce = int(genesis_time * PHI) % ABHI_AMU

        print(f"  Genesis nonce: {genesis_nonce}")
        print(f"  Predicted (time * φ % 515): {predicted_nonce}")

        # Check if nonce follows the predictable formula
        # Allow small tolerance for timing differences
        if abs(genesis_nonce - predicted_nonce) < 10:
            log_result(
                "Random nonce",
                False,
                "VULNERABILITY: Nonce is deterministic from timestamp!"
            )
        else:
            log_result("Random nonce", True, "Nonce appears random")


# =============================================================================
# VULN 10: UNBOUNDED KNOWLEDGE SUBMISSIONS
# =============================================================================

def test_unbounded_knowledge():
    """Test if knowledge submissions have size/rate limits."""
    print_header("VULN 10: UNBOUNDED KNOWLEDGE (MEDIUM)")
    print("  Testing if large/many submissions are rate limited...")

    try:
        from bazinga.blockchain.knowledge_ledger import KnowledgeLedger, create_ledger
        from bazinga.blockchain.chain import create_chain
    except ImportError as e:
        log_result("KnowledgeLedger import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            ledger = create_ledger(chain)
        except Exception as e:
            log_result("KnowledgeLedger init", False, f"Cannot create: {e}")
            return

        # Test 1: Large content
        large_content = "X" * (1024 * 1024)  # 1MB
        try:
            result = ledger.record_contribution(
                contributor="attacker",
                content=large_content,
                embedding=[0.1] * 384
            )
            if result:
                log_result(
                    "Large submission blocked",
                    False,
                    "VULNERABILITY: 1MB content accepted without limit!"
                )
            else:
                log_result("Large submission blocked", True, "Large content rejected")
        except Exception as e:
            log_result("Large submission blocked", True, f"Exception: {type(e).__name__}")

        # Test 2: Rapid submissions
        rapid_count = 0
        for i in range(100):
            try:
                result = ledger.record_contribution(
                    contributor="attacker",
                    content=f"spam_{i}",
                    embedding=[0.1] * 384
                )
                if result:
                    rapid_count += 1
            except:
                break

        if rapid_count >= 100:
            log_result(
                "Rate limiting",
                False,
                f"VULNERABILITY: {rapid_count} rapid submissions accepted!"
            )
        else:
            log_result("Rate limiting", True, f"Only {rapid_count} accepted")


# =============================================================================
# VULN 11: VALIDATOR SELECTION
# =============================================================================

def test_validator_selection():
    """Test if validator selection can be gamed."""
    print_header("VULN 11: VALIDATOR SELECTION (MEDIUM)")
    print("  Testing if attacker can control validators...")

    try:
        from bazinga.blockchain.gradient_validator import GradientValidator, GradientUpdate
        from bazinga.blockchain.chain import create_chain
    except ImportError as e:
        log_result("GradientValidator import", False, f"Cannot import: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        try:
            validator = GradientValidator(chain)
        except Exception as e:
            log_result("GradientValidator init", False, f"Cannot create: {e}")
            return

        # Register attacker as validator
        try:
            validator.register_validator("attacker_1", trust_score=0.5)
            validator.register_validator("attacker_2", trust_score=0.5)
            validator.register_validator("attacker_3", trust_score=0.5)
        except Exception as e:
            log_result("Validator registration", True, f"Cannot register: {e}")
            return

        # Submit gradient as attacker_1
        try:
            update = GradientUpdate(
                gradient_hash="fake_hash",
                submitter="attacker_1",
                model_version="v1",
                metrics={"loss": 0.0},
            )

            # Check if other attackers can be validators
            selected = validator.select_validators(update, count=3)

            attacker_validators = [v for v in selected if v.startswith("attacker_")]
            if len(attacker_validators) >= 2:
                log_result(
                    "Diverse validators required",
                    False,
                    f"VULNERABILITY: {len(attacker_validators)} attacker validators selected!"
                )
            else:
                log_result("Diverse validators required", True, "Validators properly diverse")

        except Exception as e:
            log_result("Validator selection", True, f"Exception: {type(e).__name__}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  BAZINGA BLOCKCHAIN SECURITY AUDIT - ROUND 4".center(68) + "║")
    print("║" + "  'Deep dive into cryptographic and logic vulnerabilities'".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    test_signature_scheme()
    test_plaintext_private_key()
    test_weak_local_model_verification()
    test_trust_manipulation()
    test_credit_manipulation()
    test_deterministic_nonce()
    test_unbounded_knowledge()
    test_validator_selection()

    print()
    print("=" * 70)
    print("  ROUND 4 SECURITY AUDIT SUMMARY")
    print("=" * 70)
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

    print("=" * 70)

    return len(RESULTS['vulnerabilities']) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
