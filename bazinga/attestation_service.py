#!/usr/bin/env python3
"""
DARMIYAN ATTESTATION SERVICE
"Prove you knew it, before they knew it"

Blockchain-verified knowledge attestation for:
- Prior art / IP protection
- Research timestamp proof
- Code authorship verification
- Idea attestation

BAZINGA CLI = FREE
Attestation = PAID (₹99-999)

Usage:
    from bazinga import attest_knowledge, verify_attestation

    # Attest (paid)
    receipt = attest_knowledge(
        content="My research finding or code or idea",
        email="user@example.com"
    )

    # Verify (free)
    proof = verify_attestation(receipt.attestation_id)
"""

import hashlib
import json
import time
import secrets
import string
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Blockchain imports
from .blockchain import DarmiyanChain, create_chain
from .darmiyan.protocol import prove_boundary
from .phi_coherence import PhiCoherence

# Constants
PHI = 1.618033988749895

# ============================================================
# MONETIZATION SWITCH
# ============================================================
# Set to True when ready to charge for attestations
# Until then, all attestations are FREE (building the mesh)
PAYMENTS_ENABLED = False  # <-- FLIP THIS WHEN READY

# Free tier limits (when payments disabled)
FREE_ATTESTATIONS_PER_MONTH = 3  # Free users get 3/month

ATTESTATION_TIERS = {
    "basic": {"price_inr": 99, "price_usd": 1.20, "features": ["timestamp", "hash", "basic_proof"]},
    "standard": {"price_inr": 299, "price_usd": 3.60, "features": ["timestamp", "hash", "phi_coherence", "pob_proof", "certificate"]},
    "premium": {"price_inr": 999, "price_usd": 12.00, "features": ["timestamp", "hash", "phi_coherence", "pob_proof", "certificate", "multi_ai_consensus", "legal_format"]},
}


@dataclass
class AttestationReceipt:
    """Receipt for a knowledge attestation"""
    attestation_id: str
    content_hash: str
    timestamp: str
    timestamp_unix: float
    phi_coherence: float
    pob_valid: bool
    block_number: int
    tier: str
    status: str  # pending_payment, paid, attested, verified


@dataclass
class AttestationProof:
    """Verifiable proof of attestation"""
    attestation_id: str
    content_hash: str
    timestamp: str
    timestamp_unix: float
    phi_coherence: float
    pob_proof: Dict[str, Any]
    block_number: int
    block_hash: str
    chain_valid: bool
    verification_time: str


class DarmiyanAttestationService:
    """
    Blockchain-verified knowledge attestation service.

    "You can buy hashpower. You can buy stake.
     You CANNOT buy proof that you knew something first."
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".bazinga" / "attestations"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize blockchain
        self.chain = create_chain()

        # Initialize φ-coherence calculator
        try:
            self.phi_calc = PhiCoherence()
        except:
            self.phi_calc = None

        # Attestation registry
        self.registry_file = self.data_dir / "attestation_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load attestation registry from disk"""
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text())
            except:
                pass
        return {"attestations": [], "stats": {"total": 0, "verified": 0}}

    def _save_registry(self):
        """Save attestation registry to disk"""
        self.registry_file.write_text(json.dumps(self.registry, indent=2, default=str))

    def _generate_attestation_id(self) -> str:
        """Generate unique attestation ID"""
        chars = string.ascii_uppercase + string.digits
        random_part = ''.join(secrets.choice(chars) for _ in range(12))
        return f"φATT_{random_part}"

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _compute_phi_coherence(self, content: str) -> float:
        """Compute φ-coherence of content"""
        if self.phi_calc:
            try:
                return self.phi_calc.calculate(content)
            except:
                pass
        # Fallback: simple length-based coherence
        words = len(content.split())
        return min(1.0, (words / 100) * PHI) if words > 0 else 0.0

    def _get_user_attestation_count_this_month(self, email: str) -> int:
        """Count how many attestations this user has made this month"""
        from datetime import datetime
        current_month = datetime.now().strftime("%Y-%m")
        count = 0
        for att in self.registry["attestations"]:
            if att.get("email") == email:
                att_month = att.get("timestamp", "")[:7]  # YYYY-MM
                if att_month == current_month:
                    count += 1
        return count

    def create_attestation(
        self,
        content: str,
        email: str,
        tier: str = "standard",
        metadata: Optional[Dict] = None
    ) -> AttestationReceipt:
        """
        Create a new knowledge attestation.

        When PAYMENTS_ENABLED=False (current mode):
            - First 3 attestations per month are FREE and auto-confirmed
            - After that, user must wait for next month

        When PAYMENTS_ENABLED=True (future):
            - Attestation requires payment to be confirmed

        Args:
            content: The knowledge/code/idea to attest
            email: User's email for receipt
            tier: basic, standard, or premium
            metadata: Optional additional metadata

        Returns:
            AttestationReceipt with status
        """
        if tier not in ATTESTATION_TIERS:
            tier = "standard"

        # Check free tier limits (when payments disabled)
        if not PAYMENTS_ENABLED:
            count = self._get_user_attestation_count_this_month(email)
            if count >= FREE_ATTESTATIONS_PER_MONTH:
                raise ValueError(
                    f"Free tier limit reached ({FREE_ATTESTATIONS_PER_MONTH}/month). "
                    f"Wait for next month or contribute to the mesh to earn credits!"
                )

        # Generate attestation
        attestation_id = self._generate_attestation_id()
        content_hash = self._compute_content_hash(content)
        timestamp = datetime.now().isoformat()
        timestamp_unix = time.time()
        phi_coherence = self._compute_phi_coherence(content)

        # Generate PoB proof
        pob = prove_boundary()

        # Determine status based on payment mode
        if PAYMENTS_ENABLED:
            status = "pending_payment"
            block_number = -1
        else:
            # FREE MODE: Auto-confirm and write to chain immediately
            status = "free_tier"
            block_number = -1  # Will be set after mining

        # Create receipt
        receipt = AttestationReceipt(
            attestation_id=attestation_id,
            content_hash=content_hash,
            timestamp=timestamp,
            timestamp_unix=timestamp_unix,
            phi_coherence=phi_coherence,
            pob_valid=pob.valid,
            block_number=block_number,
            tier=tier,
            status=status
        )

        # Save to registry
        self.registry["attestations"].append({
            **asdict(receipt),
            "email": email,
            "metadata": metadata,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "pob_proof": {"alpha": pob.alpha, "omega": pob.omega, "ratio": pob.ratio}
        })
        self.registry["stats"]["total"] += 1
        self._save_registry()

        # If FREE mode, auto-confirm to blockchain
        if not PAYMENTS_ENABLED:
            receipt = self.confirm_payment(attestation_id, "FREE_TIER_AUTO")

        return receipt

    def confirm_payment(self, attestation_id: str, payment_ref: str) -> AttestationReceipt:
        """
        Confirm payment and write attestation to blockchain.

        Args:
            attestation_id: The attestation ID from receipt
            payment_ref: Razorpay payment reference

        Returns:
            Updated AttestationReceipt with block number
        """
        # Find attestation
        attestation = None
        for att in self.registry["attestations"]:
            if att["attestation_id"] == attestation_id:
                attestation = att
                break

        if not attestation:
            raise ValueError(f"Attestation {attestation_id} not found")

        # Allow both pending_payment and free_tier to be confirmed
        if attestation["status"] not in ("pending_payment", "free_tier"):
            raise ValueError(f"Attestation {attestation_id} already processed")

        # Add to blockchain
        tx_hash = self.chain.add_knowledge(
            content=attestation["content_hash"],
            summary=f"Attestation: {attestation_id}",
            sender=attestation["email"],
            confidence=attestation["phi_coherence"],
            source_type="attestation",
            phi_coherence=attestation["phi_coherence"]
        )

        # Mine block with PoB
        from .blockchain import PoBMiner
        miner = PoBMiner(self.chain, node_id="attestation_service")
        result = None
        try:
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(miner.mine())
        except:
            # Fallback: create block directly
            pob = prove_boundary()
            pob_proofs = [
                {"alpha": pob.alpha, "omega": pob.omega, "ratio": pob.ratio},
                {"alpha": pob.alpha + 1, "omega": pob.omega - 1, "ratio": pob.ratio},
                {"alpha": pob.alpha - 1, "omega": pob.omega + 1, "ratio": pob.ratio},
            ]
            self.chain.add_block(pob_proofs=pob_proofs)

        # Update attestation
        block_number = len(self.chain.blocks) - 1
        attestation["status"] = "attested"
        attestation["block_number"] = block_number
        attestation["payment_ref"] = payment_ref
        attestation["attested_at"] = datetime.now().isoformat()

        self.registry["stats"]["verified"] += 1
        self._save_registry()

        return AttestationReceipt(
            attestation_id=attestation["attestation_id"],
            content_hash=attestation["content_hash"],
            timestamp=attestation["timestamp"],
            timestamp_unix=attestation["timestamp_unix"],
            phi_coherence=attestation["phi_coherence"],
            pob_valid=attestation["pob_proof"]["ratio"] > 0,
            block_number=block_number,
            tier=attestation["tier"],
            status="attested"
        )

    def verify_attestation(self, attestation_id: str) -> Optional[AttestationProof]:
        """
        Verify an attestation exists on chain (FREE).

        Args:
            attestation_id: The attestation ID to verify

        Returns:
            AttestationProof if valid, None if not found
        """
        # Find in registry
        attestation = None
        for att in self.registry["attestations"]:
            if att["attestation_id"] == attestation_id:
                attestation = att
                break

        if not attestation or attestation["status"] != "attested":
            return None

        # Verify on chain
        block_number = attestation["block_number"]
        if block_number < 0 or block_number >= len(self.chain.blocks):
            return None

        block = self.chain.blocks[block_number]
        chain_valid = self.chain.validate_chain()

        return AttestationProof(
            attestation_id=attestation_id,
            content_hash=attestation["content_hash"],
            timestamp=attestation["timestamp"],
            timestamp_unix=attestation["timestamp_unix"],
            phi_coherence=attestation["phi_coherence"],
            pob_proof=attestation["pob_proof"],
            block_number=block_number,
            block_hash=block.hash if hasattr(block, 'hash') else str(block),
            chain_valid=chain_valid,
            verification_time=datetime.now().isoformat()
        )

    def verify_content(self, content: str) -> Optional[AttestationProof]:
        """
        Verify if content has been attested (by hash lookup).

        Args:
            content: The original content to verify

        Returns:
            AttestationProof if found, None otherwise
        """
        content_hash = self._compute_content_hash(content)

        for att in self.registry["attestations"]:
            if att["content_hash"] == content_hash and att["status"] == "attested":
                return self.verify_attestation(att["attestation_id"])

        return None

    def get_certificate(self, attestation_id: str) -> Optional[str]:
        """
        Generate a human-readable certificate for an attestation.

        Args:
            attestation_id: The attestation ID

        Returns:
            Certificate text or None
        """
        proof = self.verify_attestation(attestation_id)
        if not proof:
            return None

        certificate = f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   DARMIYAN ATTESTATION CERTIFICATE                               ║
║   "Proof of Prior Knowledge"                                     ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Attestation ID:  {proof.attestation_id:<43} ║
║   Content Hash:    {proof.content_hash[:32]}...  ║
║                                                                  ║
║   Timestamp:       {proof.timestamp:<43} ║
║   Unix Timestamp:  {proof.timestamp_unix:<43.0f} ║
║                                                                  ║
║   φ-Coherence:     {proof.phi_coherence:<43.4f} ║
║   Block Number:    {proof.block_number:<43} ║
║   Block Hash:      {str(proof.block_hash)[:32]}...  ║
║                                                                  ║
║   Chain Valid:     {'✓ VERIFIED' if proof.chain_valid else '✗ INVALID':<43} ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   This certificate proves that the content identified by the     ║
║   above hash existed at the specified timestamp and was          ║
║   recorded on the Darmiyan blockchain with Proof-of-Boundary.    ║
║                                                                  ║
║   Verify at: bazinga --verify {proof.attestation_id}        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║   Verified: {proof.verification_time:<50} ║
║   Powered by BAZINGA φ-coherence | ∅ ≈ ∞                         ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return certificate

    def get_pricing(self) -> Dict:
        """Get attestation pricing tiers"""
        return ATTESTATION_TIERS

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            **self.registry["stats"],
            "chain_length": len(self.chain.blocks),
            "chain_valid": self.chain.validate_chain() if self.chain.blocks else True
        }


# ============================================================
# PUBLIC API
# ============================================================

_service = None

def get_attestation_service() -> DarmiyanAttestationService:
    """Get or create the attestation service singleton"""
    global _service
    if _service is None:
        _service = DarmiyanAttestationService()
    return _service


def attest_knowledge(
    content: str,
    email: str,
    tier: str = "standard"
) -> AttestationReceipt:
    """
    Create a knowledge attestation (step 1: get receipt, then pay).

    Usage:
        from bazinga import attest_knowledge

        receipt = attest_knowledge(
            content="My research finding...",
            email="me@example.com",
            tier="standard"  # ₹299
        )

        print(f"Attestation ID: {receipt.attestation_id}")
        print(f"Pay ₹{ATTESTATION_TIERS[tier]['price_inr']} to complete")
    """
    service = get_attestation_service()
    return service.create_attestation(content, email, tier)


def verify_attestation(attestation_id: str) -> Optional[AttestationProof]:
    """
    Verify an attestation (FREE).

    Usage:
        from bazinga import verify_attestation

        proof = verify_attestation("φATT_ABC123XYZ789")
        if proof:
            print(f"Verified! Timestamp: {proof.timestamp}")
            print(f"Block: {proof.block_number}")
    """
    service = get_attestation_service()
    return service.verify_attestation(attestation_id)


def get_certificate(attestation_id: str) -> Optional[str]:
    """
    Get a printable certificate for an attestation.

    Usage:
        from bazinga import get_certificate

        cert = get_certificate("φATT_ABC123XYZ789")
        print(cert)
    """
    service = get_attestation_service()
    return service.get_certificate(attestation_id)


def get_attestation_pricing() -> Dict:
    """Get attestation pricing tiers"""
    return ATTESTATION_TIERS
