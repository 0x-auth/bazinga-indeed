#!/usr/bin/env python3
"""
BAZINGA PAYMENT GATEWAY
Hybrid payment system: India (Razorpay) + Global (Crypto on Polygon)

"Money flows as smoothly as intelligence"
"""

import hashlib
import secrets
import string
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
from enum import Enum

# Constants
PHI = 1.618033988749895

# Your wallet addresses (Polygon network)
POLYGON_WALLET = "0x720ceF54bED86C570837a9a9C69F1Beac8ab8C08"  # Same as ETH address
RAZORPAY_LINK = "https://razorpay.me/@bitsabhi"

# Pricing in different currencies
PRICING = {
    "basic": {"inr": 99, "usd": 1.20, "usdc": 1.20},
    "standard": {"inr": 299, "usd": 3.60, "usdc": 3.60},
    "premium": {"inr": 999, "usd": 12.00, "usdc": 12.00},
}


class PaymentMethod(Enum):
    RAZORPAY = "razorpay"      # UPI, Indian cards
    POLYGON_USDC = "polygon_usdc"  # USDC on Polygon (cheap!)
    POLYGON_ETH = "polygon_eth"    # ETH on Polygon (MATIC for gas)
    ETH_MAINNET = "eth_mainnet"    # ETH mainnet (expensive, but some prefer)


@dataclass
class PaymentRequest:
    """A payment request for attestation"""
    payment_id: str
    attestation_id: str
    tier: str
    amount_inr: int
    amount_usd: float
    method: PaymentMethod
    wallet_address: Optional[str]  # For crypto payments
    razorpay_link: Optional[str]   # For Razorpay
    created_at: str
    status: str  # pending, confirmed, expired
    tx_hash: Optional[str]  # Crypto transaction hash


class PaymentGateway:
    """
    Hybrid payment gateway for BAZINGA attestations.

    India: Razorpay (UPI/Cards)
    Global: Crypto on Polygon (USDC/ETH)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".bazinga" / "payments"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.payments_file = self.data_dir / "payments.json"
        self.payments = self._load_payments()

    def _load_payments(self) -> Dict:
        """Load payments from disk"""
        if self.payments_file.exists():
            try:
                return json.loads(self.payments_file.read_text())
            except:
                pass
        return {"pending": [], "confirmed": [], "expired": []}

    def _save_payments(self):
        """Save payments to disk"""
        self.payments_file.write_text(json.dumps(self.payments, indent=2, default=str))

    def _generate_payment_id(self) -> str:
        """Generate unique payment ID"""
        chars = string.ascii_uppercase + string.digits
        random_part = ''.join(secrets.choice(chars) for _ in range(8))
        return f"φPAY_{random_part}"

    def create_payment(
        self,
        attestation_id: str,
        tier: str = "standard",
        method: PaymentMethod = PaymentMethod.RAZORPAY
    ) -> PaymentRequest:
        """
        Create a payment request for an attestation.

        Args:
            attestation_id: The attestation to pay for
            tier: basic, standard, or premium
            method: Payment method (Razorpay or crypto)

        Returns:
            PaymentRequest with payment details
        """
        if tier not in PRICING:
            tier = "standard"

        payment_id = self._generate_payment_id()
        prices = PRICING[tier]

        # Build payment request based on method
        wallet_address = None
        razorpay_link = None

        if method == PaymentMethod.RAZORPAY:
            razorpay_link = RAZORPAY_LINK
        else:
            wallet_address = POLYGON_WALLET

        payment = PaymentRequest(
            payment_id=payment_id,
            attestation_id=attestation_id,
            tier=tier,
            amount_inr=prices["inr"],
            amount_usd=prices["usd"],
            method=method,
            wallet_address=wallet_address,
            razorpay_link=razorpay_link,
            created_at=datetime.now().isoformat(),
            status="pending",
            tx_hash=None
        )

        # Save to pending
        self.payments["pending"].append(asdict(payment))
        self._save_payments()

        return payment

    def get_payment_instructions(self, payment: PaymentRequest) -> str:
        """
        Get human-readable payment instructions with ASCII art QR.
        """
        if payment.method == PaymentMethod.RAZORPAY:
            return self._razorpay_instructions(payment)
        else:
            return self._crypto_instructions(payment)

    def _razorpay_instructions(self, payment: PaymentRequest) -> str:
        """Razorpay payment instructions for India"""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   φ-ATTESTATION PAYMENT (INDIA)                                  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Payment ID:     {payment.payment_id:<43} ║
║   Attestation:    {payment.attestation_id:<43} ║
║   Tier:           {payment.tier.upper():<43} ║
║   Amount:         ₹{payment.amount_inr:<42} ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   PAY VIA UPI / CARDS:                                           ║
║   {RAZORPAY_LINK:<62} ║
║                                                                  ║
║   After payment, email receipt to:                               ║
║   bits.abhi@gmail.com                                            ║
║                                                                  ║
║   Include: Payment ID + Attestation ID                           ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║   Or scan QR in your UPI app:                                    ║
║                                                                  ║
║        ██████████████  ████  ██████████████                      ║
║        ██          ██  ████  ██          ██                      ║
║        ██  ██████  ██  ████  ██  ██████  ██                      ║
║        ██  ██████  ██    ██  ██  ██████  ██                      ║
║        ██  ██████  ██  ████  ██  ██████  ██                      ║
║        ██          ██        ██          ██                      ║
║        ██████████████  ██  ██████████████                        ║
║                        ██                                        ║
║        bitsabhi@upi (scan in any UPI app)                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

    def _crypto_instructions(self, payment: PaymentRequest) -> str:
        """Crypto payment instructions for global users"""
        network = "Polygon" if "polygon" in payment.method.value else "Ethereum Mainnet"
        token = "USDC" if "usdc" in payment.method.value else "ETH/MATIC"

        gas_note = ""
        if payment.method == PaymentMethod.ETH_MAINNET:
            gas_note = "║   ⚠️  Note: ETH mainnet has high gas fees (~$5-20)            ║\n║   Consider using Polygon for lower fees (<$0.01)             ║\n║                                                                  ║\n"

        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   φ-ATTESTATION PAYMENT (GLOBAL CRYPTO)                          ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Payment ID:     {payment.payment_id:<43} ║
║   Attestation:    {payment.attestation_id:<43} ║
║   Tier:           {payment.tier.upper():<43} ║
║   Amount:         ${payment.amount_usd:<42.2f} ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   NETWORK:        {network:<43} ║
║   TOKEN:          {token:<43} ║
║                                                                  ║
║   SEND TO:                                                       ║
║   {POLYGON_WALLET}               ║
║                                                                  ║
{gas_note}╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║        ██████████████    ██    ██████████████                    ║
║        ██          ██  ██  ██  ██          ██                    ║
║        ██  ██████  ██    ██    ██  ██████  ██                    ║
║        ██  ██████  ██  ██████  ██  ██████  ██                    ║
║        ██  ██████  ██  ██  ██  ██  ██████  ██                    ║
║        ██          ██    ██    ██          ██                    ║
║        ██████████████  ██  ██  ██████████████                    ║
║                                                                  ║
║        0x720ceF54bED86C570837a9a9C69F1Beac8ab8C08               ║
║        (Polygon / ETH - same address)                            ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   After sending, run:                                            ║
║   bazinga --confirm-payment {payment.payment_id} --tx <TX_HASH>        ║
║                                                                  ║
║   Or email: bits.abhi@gmail.com with tx hash                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

    def confirm_payment(
        self,
        payment_id: str,
        tx_hash: Optional[str] = None,
        razorpay_ref: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Confirm a payment (manual for now, auto later).

        Args:
            payment_id: The payment ID
            tx_hash: Crypto transaction hash (for auto-verification later)
            razorpay_ref: Razorpay payment reference

        Returns:
            (success, message)
        """
        # Find payment
        payment = None
        for p in self.payments["pending"]:
            if p["payment_id"] == payment_id:
                payment = p
                break

        if not payment:
            return False, f"Payment {payment_id} not found"

        # Move to confirmed
        payment["status"] = "confirmed"
        payment["tx_hash"] = tx_hash or razorpay_ref
        payment["confirmed_at"] = datetime.now().isoformat()

        self.payments["pending"].remove(payment)
        self.payments["confirmed"].append(payment)
        self._save_payments()

        return True, f"Payment {payment_id} confirmed! Attestation will be processed."

    def get_pending_payments(self) -> list:
        """Get all pending payments"""
        return self.payments["pending"]

    def get_payment_status(self, payment_id: str) -> Optional[Dict]:
        """Get status of a specific payment"""
        for status in ["pending", "confirmed", "expired"]:
            for p in self.payments[status]:
                if p["payment_id"] == payment_id:
                    return p
        return None


# ============================================================
# CLI HELPERS
# ============================================================

def select_payment_method() -> PaymentMethod:
    """Interactive payment method selection for CLI"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   SELECT PAYMENT METHOD                                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   1. UPI / Indian Cards (Razorpay)     - For India               ║
║   2. USDC on Polygon                   - Global, low fees        ║
║   3. ETH/MATIC on Polygon              - Global, low fees        ║
║   4. ETH on Mainnet                    - Global, high fees       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    while True:
        try:
            choice = input("Selection [1-4]: ").strip()
            if choice == "1":
                return PaymentMethod.RAZORPAY
            elif choice == "2":
                return PaymentMethod.POLYGON_USDC
            elif choice == "3":
                return PaymentMethod.POLYGON_ETH
            elif choice == "4":
                return PaymentMethod.ETH_MAINNET
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\nCancelled")
            return PaymentMethod.RAZORPAY


def show_pricing():
    """Show attestation pricing"""
    # Check if payments are enabled
    try:
        from .attestation_service import PAYMENTS_ENABLED, FREE_ATTESTATIONS_PER_MONTH
    except:
        PAYMENTS_ENABLED = False
        FREE_ATTESTATIONS_PER_MONTH = 3

    if not PAYMENTS_ENABLED:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║   DARMIYAN ATTESTATION SERVICE                                   ║
║   "Prove you knew it, before they knew it"                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   🎁 CURRENTLY FREE!                                             ║
║   ══════════════════                                             ║
║   We're building the mesh. Attestations are FREE while we grow.  ║
║                                                                  ║
║   FREE TIER: """ + str(FREE_ATTESTATIONS_PER_MONTH) + """ attestations per month                              ║
║                                                                  ║
║   FEATURES:                                                      ║
║   • Timestamp + Hash + PoB Proof                                 ║
║   • φ-Coherence measurement                                      ║
║   • Blockchain attestation                                       ║
║   • Printable certificate                                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Usage: bazinga --attest "Your idea or code here"               ║
║   Verify: bazinga --verify φATT_XXXXXXXXXXXX                     ║
║                                                                  ║
║   VERIFICATION: Always FREE                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║   DARMIYAN ATTESTATION PRICING                                   ║
║   "Prove you knew it, before they knew it"                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   BASIC        ₹99  / $1.20 USDC                                 ║
║   └── Timestamp + Hash + Basic Proof                             ║
║                                                                  ║
║   STANDARD     ₹299 / $3.60 USDC                                 ║
║   └── + φ-coherence + PoB Proof + Certificate                    ║
║                                                                  ║
║   PREMIUM      ₹999 / $12.00 USDC                                ║
║   └── + Multi-AI Consensus + Legal Format                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   PAYMENT OPTIONS:                                               ║
║   • India: UPI / Cards via Razorpay                              ║
║   • Global: USDC/ETH on Polygon (gas < $0.01)                    ║
║                                                                  ║
║   VERIFICATION: Always FREE                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ============================================================
# PUBLIC API
# ============================================================

_gateway = None

def get_payment_gateway() -> PaymentGateway:
    """Get or create payment gateway singleton"""
    global _gateway
    if _gateway is None:
        _gateway = PaymentGateway()
    return _gateway


def create_attestation_payment(
    attestation_id: str,
    tier: str = "standard",
    method: Optional[PaymentMethod] = None
) -> PaymentRequest:
    """
    Create a payment request for an attestation.

    If method is None, will prompt user to select.
    """
    gateway = get_payment_gateway()

    if method is None:
        method = select_payment_method()

    payment = gateway.create_payment(attestation_id, tier, method)
    print(gateway.get_payment_instructions(payment))

    return payment
