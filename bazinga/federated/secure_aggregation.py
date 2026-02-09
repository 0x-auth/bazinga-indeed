#!/usr/bin/env python3
"""
BAZINGA Secure Aggregation - Homomorphic Encryption for Federated Learning

Implements secure aggregation using Paillier cryptosystem:
- Additive homomorphism: E(a) * E(b) = E(a + b)
- Server aggregates encrypted gradients without seeing values
- Only the aggregated result is decrypted

Security Model:
- Honest-but-curious server
- Participants can't see each other's gradients
- Only sum is revealed

"Compute on secrets without revealing them."
"""

import math
import secrets
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_prime(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def generate_prime(bits: int) -> int:
    """Generate a random prime number with specified bits."""
    while True:
        p = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(p):
            return p


def mod_inverse(a: int, m: int) -> int:
    """Compute modular multiplicative inverse using extended Euclidean algorithm."""
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m


def L(x: int, n: int) -> int:
    """L function for Paillier: L(x) = (x-1)/n."""
    return (x - 1) // n


@dataclass
class PaillierKeyPair:
    """
    Paillier cryptosystem key pair.

    Public key: (n, g)
    Private key: (lambda, mu)
    """
    # Public key
    n: int          # n = p * q
    g: int          # Generator (typically n + 1)
    n_squared: int  # n²

    # Private key
    lambda_: int    # lcm(p-1, q-1)
    mu: int         # L(g^λ mod n²)^(-1) mod n

    @classmethod
    def generate(cls, key_bits: int = 1024) -> 'PaillierKeyPair':
        """
        Generate new Paillier key pair.

        Args:
            key_bits: Security parameter (bits of n)

        Returns:
            PaillierKeyPair
        """
        bits = key_bits // 2

        # Generate two primes
        p = generate_prime(bits)
        q = generate_prime(bits)

        # Ensure p != q
        while p == q:
            q = generate_prime(bits)

        # Compute n and n²
        n = p * q
        n_squared = n * n

        # g = n + 1 (simplified)
        g = n + 1

        # Lambda = lcm(p-1, q-1)
        lambda_ = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)

        # mu = L(g^λ mod n²)^(-1) mod n
        g_lambda = pow(g, lambda_, n_squared)
        mu = mod_inverse(L(g_lambda, n), n)

        return cls(
            n=n,
            g=g,
            n_squared=n_squared,
            lambda_=lambda_,
            mu=mu
        )

    def save(self, path: str, include_private: bool = True):
        """Save key pair to file."""
        data = {
            'n': self.n,
            'g': self.g,
            'n_squared': self.n_squared,
        }
        if include_private:
            data['lambda'] = self.lambda_
            data['mu'] = self.mu

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'PaillierKeyPair':
        """Load key pair from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            n=data['n'],
            g=data['g'],
            n_squared=data['n_squared'],
            lambda_=data.get('lambda', 0),
            mu=data.get('mu', 0)
        )


class PaillierEncryption:
    """
    Paillier encryption scheme with additive homomorphism.

    Properties:
        E(a) * E(b) mod n² = E(a + b)
        E(a)^k mod n² = E(k * a)
    """

    def __init__(self, key: PaillierKeyPair):
        """
        Initialize with key pair.

        Args:
            key: Paillier key pair
        """
        self.key = key

    def encrypt(self, plaintext: int) -> int:
        """
        Encrypt an integer.

        E(m) = g^m * r^n mod n²
        where r is random in Z*_n

        Args:
            plaintext: Integer to encrypt (0 <= m < n)

        Returns:
            Ciphertext
        """
        n = self.key.n
        g = self.key.g
        n_sq = self.key.n_squared

        # Random r in Z*_n
        r = secrets.randbelow(n - 1) + 1
        while math.gcd(r, n) != 1:
            r = secrets.randbelow(n - 1) + 1

        # c = g^m * r^n mod n²
        gm = pow(g, plaintext % n, n_sq)
        rn = pow(r, n, n_sq)
        ciphertext = (gm * rn) % n_sq

        return ciphertext

    def decrypt(self, ciphertext: int) -> int:
        """
        Decrypt a ciphertext.

        m = L(c^λ mod n²) * μ mod n

        Args:
            ciphertext: Encrypted value

        Returns:
            Plaintext integer
        """
        n = self.key.n
        n_sq = self.key.n_squared
        lambda_ = self.key.lambda_
        mu = self.key.mu

        # m = L(c^λ mod n²) * μ mod n
        c_lambda = pow(ciphertext, lambda_, n_sq)
        plaintext = (L(c_lambda, n) * mu) % n

        return plaintext

    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values (homomorphic addition).

        E(a) * E(b) = E(a + b)

        Args:
            c1, c2: Encrypted values

        Returns:
            E(a + b)
        """
        return (c1 * c2) % self.key.n_squared

    def multiply_constant(self, ciphertext: int, constant: int) -> int:
        """
        Multiply encrypted value by constant (scalar multiplication).

        E(a)^k = E(k * a)

        Args:
            ciphertext: Encrypted value
            constant: Scalar multiplier

        Returns:
            E(k * a)
        """
        return pow(ciphertext, constant, self.key.n_squared)


class SecureAggregator:
    """
    Secure gradient aggregator for federated learning.

    Workflow:
    1. Participants encrypt gradients with shared public key
    2. Aggregator sums encrypted gradients (homomorphic)
    3. Result decrypted reveals only the sum

    "The whole is revealed, but not the parts."
    """

    def __init__(
        self,
        key_bits: int = 1024,
        quantization_bits: int = 16,
    ):
        """
        Initialize secure aggregator.

        Args:
            key_bits: Security parameter for Paillier
            quantization_bits: Bits for gradient quantization
        """
        print(f"Generating {key_bits}-bit Paillier keys...")
        self.key = PaillierKeyPair.generate(key_bits)
        self.crypto = PaillierEncryption(self.key)
        self.quantization_bits = quantization_bits
        self.quantization_scale = 2 ** quantization_bits

        # Aggregation state
        self.encrypted_sums: Dict[str, int] = {}
        self.num_participants = 0
        self.round_id = 0

        print(f"SecureAggregator ready:")
        print(f"  Key bits: {key_bits}")
        print(f"  Quantization: {quantization_bits} bits")

    def quantize(self, value: float) -> int:
        """
        Quantize float to integer for encryption.

        Args:
            value: Float value (expected in [-1, 1] after clipping)

        Returns:
            Quantized integer
        """
        # Scale and shift to positive integer
        scaled = int((value + 1) * self.quantization_scale / 2)
        return max(0, min(self.quantization_scale, scaled))

    def dequantize(self, value: int, num_participants: int) -> float:
        """
        Dequantize integer back to float.

        Args:
            value: Quantized sum
            num_participants: Number of values summed

        Returns:
            Average float value
        """
        # Reverse the scaling
        avg_quantized = value / num_participants
        return (avg_quantized * 2 / self.quantization_scale) - 1

    def get_public_key(self) -> Dict[str, int]:
        """Get public key for participants."""
        return {
            'n': self.key.n,
            'g': self.key.g,
            'n_squared': self.key.n_squared,
        }

    def encrypt_gradients(
        self,
        gradients: Dict[str, List[float]],
    ) -> Dict[str, List[int]]:
        """
        Encrypt gradients for secure aggregation.

        Args:
            gradients: Dict of gradient name → list of float values

        Returns:
            Encrypted gradients
        """
        encrypted = {}

        for name, values in gradients.items():
            encrypted[name] = [
                self.crypto.encrypt(self.quantize(v))
                for v in values
            ]

        return encrypted

    def aggregate_encrypted(
        self,
        all_encrypted: List[Dict[str, List[int]]],
    ):
        """
        Aggregate encrypted gradients from all participants.

        Args:
            all_encrypted: List of encrypted gradient dicts from each participant
        """
        self.round_id += 1
        self.num_participants = len(all_encrypted)

        if self.num_participants == 0:
            return

        # Initialize sums with first participant
        first = all_encrypted[0]
        self.encrypted_sums = {
            name: values.copy()
            for name, values in first.items()
        }

        # Add remaining participants
        for encrypted in all_encrypted[1:]:
            for name, values in encrypted.items():
                if name in self.encrypted_sums:
                    for i, v in enumerate(values):
                        self.encrypted_sums[name][i] = self.crypto.add_encrypted(
                            self.encrypted_sums[name][i], v
                        )

        print(f"Aggregated {self.num_participants} participants (round {self.round_id})")

    def decrypt_aggregated(self) -> Dict[str, List[float]]:
        """
        Decrypt aggregated gradients to get average.

        Returns:
            Averaged gradients (sum / num_participants)
        """
        if not self.encrypted_sums:
            return {}

        decrypted = {}

        for name, encrypted_values in self.encrypted_sums.items():
            decrypted[name] = [
                self.dequantize(
                    self.crypto.decrypt(v),
                    self.num_participants
                )
                for v in encrypted_values
            ]

        return decrypted

    def clear_round(self):
        """Clear state for next round."""
        self.encrypted_sums = {}
        self.num_participants = 0


if TORCH_AVAILABLE:

    class TorchSecureAggregator(SecureAggregator):
        """
        Secure aggregator with PyTorch tensor support.

        Handles conversion between tensors and encrypted integers.
        """

        def encrypt_tensor(self, tensor: torch.Tensor) -> List[int]:
            """
            Encrypt a tensor.

            Args:
                tensor: PyTorch tensor (will be flattened)

            Returns:
                List of encrypted integers
            """
            # Flatten and convert to list
            flat = tensor.detach().cpu().flatten().tolist()

            # Encrypt each value
            return [self.crypto.encrypt(self.quantize(v)) for v in flat]

        def decrypt_to_tensor(
            self,
            encrypted: List[int],
            shape: Tuple[int, ...],
            num_participants: int,
        ) -> torch.Tensor:
            """
            Decrypt to tensor.

            Args:
                encrypted: Encrypted values
                shape: Original tensor shape
                num_participants: For averaging

            Returns:
                PyTorch tensor
            """
            # Decrypt
            decrypted = [
                self.dequantize(self.crypto.decrypt(v), num_participants)
                for v in encrypted
            ]

            # Convert to tensor and reshape
            return torch.tensor(decrypted).reshape(shape)

        def encrypt_gradients_torch(
            self,
            gradients: Dict[str, torch.Tensor],
        ) -> Tuple[Dict[str, List[int]], Dict[str, Tuple]]:
            """
            Encrypt gradient tensors.

            Args:
                gradients: Dict of gradient tensors

            Returns:
                (encrypted_grads, shapes)
            """
            encrypted = {}
            shapes = {}

            for name, grad in gradients.items():
                shapes[name] = tuple(grad.shape)
                encrypted[name] = self.encrypt_tensor(grad)

            return encrypted, shapes

        def aggregate_and_decrypt_torch(
            self,
            all_encrypted: List[Dict[str, List[int]]],
            shapes: Dict[str, Tuple],
        ) -> Dict[str, torch.Tensor]:
            """
            Aggregate encrypted gradients and decrypt to tensors.

            Args:
                all_encrypted: Encrypted gradients from all participants
                shapes: Original tensor shapes

            Returns:
                Averaged gradient tensors
            """
            # Aggregate
            self.aggregate_encrypted(all_encrypted)

            # Decrypt to tensors
            result = {}
            for name, encrypted_values in self.encrypted_sums.items():
                if name in shapes:
                    result[name] = self.decrypt_to_tensor(
                        encrypted_values,
                        shapes[name],
                        self.num_participants
                    )

            return result


# Simplified secure aggregation for testing (no heavy crypto)
class SimplifiedSecureAggregator:
    """
    Simplified secure aggregation using secret sharing.

    For development/testing when full Paillier is too slow.
    Still provides some privacy guarantee via masking.
    """

    def __init__(self, num_participants: int = 3):
        self.num_participants = num_participants
        self.masks: Dict[str, List[float]] = {}
        self.round_id = 0

    def generate_masks(self, shape: Tuple[int, ...]) -> List[List[float]]:
        """Generate random masks that sum to zero."""
        import random

        masks = []
        for _ in range(self.num_participants - 1):
            mask = [random.gauss(0, 1) for _ in range(math.prod(shape))]
            masks.append(mask)

        # Last mask ensures sum is zero
        last_mask = [-sum(m[i] for m in masks) for i in range(len(masks[0]))]
        masks.append(last_mask)

        return masks

    def mask_gradients(
        self,
        gradients: Dict[str, List[float]],
        participant_id: int,
    ) -> Dict[str, List[float]]:
        """Apply mask to gradients."""
        masked = {}
        for name, values in gradients.items():
            if name not in self.masks:
                self.masks[name] = self.generate_masks((len(values),))

            mask = self.masks[name][participant_id]
            masked[name] = [v + m for v, m in zip(values, mask)]

        return masked

    def aggregate(
        self,
        all_masked: List[Dict[str, List[float]]],
    ) -> Dict[str, List[float]]:
        """Aggregate masked gradients (masks cancel out)."""
        if not all_masked:
            return {}

        result = {}
        for name in all_masked[0]:
            num_values = len(all_masked[0][name])
            sums = [0.0] * num_values

            for masked in all_masked:
                if name in masked:
                    for i, v in enumerate(masked[name]):
                        sums[i] += v

            # Average
            result[name] = [s / len(all_masked) for s in sums]

        self.round_id += 1
        return result


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Secure Aggregation Test")
    print("=" * 60)

    # Test Paillier basics
    print("\n1. Testing Paillier Encryption:")
    key = PaillierKeyPair.generate(512)  # Small for testing
    crypto = PaillierEncryption(key)

    a, b = 42, 37
    ea = crypto.encrypt(a)
    eb = crypto.encrypt(b)
    esum = crypto.add_encrypted(ea, eb)

    print(f"  a = {a}, b = {b}")
    print(f"  E(a) = {ea % 10**10}... (truncated)")
    print(f"  Decrypted sum = {crypto.decrypt(esum)}")
    print(f"  Expected = {a + b}")
    assert crypto.decrypt(esum) == a + b, "Homomorphic addition failed!"
    print("  ✓ Homomorphic addition works!")

    # Test secure aggregation
    print("\n2. Testing Secure Aggregation:")
    aggregator = SecureAggregator(key_bits=512, quantization_bits=12)

    # Simulate 3 participants with gradients
    participants = [
        {'layer1': [0.1, 0.2, 0.3], 'layer2': [-0.1, 0.0, 0.1]},
        {'layer1': [0.2, 0.1, 0.4], 'layer2': [0.0, 0.1, -0.1]},
        {'layer1': [0.0, 0.3, 0.2], 'layer2': [0.1, -0.1, 0.0]},
    ]

    # Encrypt all
    all_encrypted = [aggregator.encrypt_gradients(p) for p in participants]

    # Aggregate
    aggregator.aggregate_encrypted(all_encrypted)

    # Decrypt
    result = aggregator.decrypt_aggregated()

    print(f"  Participants: {len(participants)}")
    print(f"  Aggregated result:")
    for name, values in result.items():
        print(f"    {name}: {[f'{v:.3f}' for v in values]}")

    # Verify (manually compute expected average)
    expected_l1 = [(0.1+0.2+0.0)/3, (0.2+0.1+0.3)/3, (0.3+0.4+0.2)/3]
    print(f"  Expected layer1: {[f'{v:.3f}' for v in expected_l1]}")

    print("\n✓ Secure Aggregation module ready!")
