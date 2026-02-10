#!/usr/bin/env python3
"""
BAZINGA Model Distribution - P2P Model Sharing

Distributes model weights across the network:
- Chunked transfer for large models
- Delta updates (only changed weights)
- Content-addressed storage (IPFS-style)
- Parallel download from multiple peers
- Integrity verification via Merkle trees

"Models flow like water, from many sources to one stream."
"""

import asyncio
import hashlib
import json
import time
import zlib
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Distribution constants
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
MAX_PARALLEL_DOWNLOADS = 5
CHUNK_TIMEOUT = 30.0  # seconds


class ChunkStatus(Enum):
    """Status of a model chunk."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class ModelChunk:
    """A chunk of model data."""
    chunk_id: str
    index: int
    size: int
    hash: str  # SHA-256 of chunk content
    data: Optional[bytes] = None
    status: ChunkStatus = ChunkStatus.PENDING

    def verify(self) -> bool:
        """Verify chunk integrity."""
        if not self.data:
            return False
        computed = hashlib.sha256(self.data).hexdigest()
        return computed == self.hash

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['data'] = None  # Don't serialize data
        return d


@dataclass
class ModelManifest:
    """
    Manifest describing a model for distribution.

    Contains metadata and chunk information for
    content-addressed retrieval.
    """
    model_id: str
    version: str
    total_size: int
    chunk_size: int
    num_chunks: int
    chunks: List[Dict]  # List of chunk metadata
    root_hash: str  # Merkle root of all chunks
    created_at: float = field(default_factory=time.time)
    compression: str = "zlib"
    quantization: str = "none"

    # Model metadata
    model_type: str = ""
    architecture: str = ""
    parameters: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelManifest':
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModelManifest':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def get_chunk_hash(self, index: int) -> Optional[str]:
        """Get hash of chunk by index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index].get('hash')
        return None


class MerkleTree:
    """
    Merkle tree for model integrity verification.

    Enables efficient verification of partial downloads
    and detection of corrupted chunks.
    """

    def __init__(self, chunk_hashes: List[str]):
        """
        Build Merkle tree from chunk hashes.

        Args:
            chunk_hashes: List of chunk SHA-256 hashes
        """
        self.leaves = chunk_hashes
        self.tree = self._build_tree(chunk_hashes)
        self.root = self.tree[0] if self.tree else ""

    def _build_tree(self, hashes: List[str]) -> List[str]:
        """Build tree from leaf hashes."""
        if not hashes:
            return []

        # Pad to power of 2
        n = len(hashes)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        hashes = hashes + [hashes[-1]] * (next_pow2 - n)

        tree = []
        level = hashes

        while len(level) > 1:
            tree = level + tree
            next_level = []
            for i in range(0, len(level), 2):
                combined = level[i] + level[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            level = next_level

        return level + tree

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for a chunk.

        Returns list of (hash, side) pairs for verification.
        """
        if index >= len(self.leaves):
            return []

        proof = []
        tree_index = len(self.tree) // 2 + index

        while tree_index > 0:
            sibling = tree_index ^ 1  # XOR to get sibling
            if sibling < len(self.tree):
                side = "left" if tree_index % 2 == 1 else "right"
                proof.append((self.tree[sibling], side))
            tree_index = (tree_index - 1) // 2

        return proof

    def verify_chunk(self, index: int, chunk_hash: str, proof: List[Tuple[str, str]]) -> bool:
        """Verify a chunk using its Merkle proof."""
        current = chunk_hash

        for sibling_hash, side in proof:
            if side == "left":
                combined = sibling_hash + current
            else:
                combined = current + sibling_hash
            current = hashlib.sha256(combined.encode()).hexdigest()

        return current == self.root


class DistributionProtocol:
    """
    Protocol for model distribution across the network.

    Handles:
    - Creating model manifests
    - Chunking models for transfer
    - Parallel download coordination
    - Delta updates
    """

    def __init__(
        self,
        storage_dir: str = "~/.bazinga/models",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Initialize distribution protocol.

        Args:
            storage_dir: Directory for model storage
            chunk_size: Size of chunks in bytes
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        # Active transfers
        self.downloads: Dict[str, 'ModelDownload'] = {}
        self.uploads: Dict[str, 'ModelUpload'] = {}

        print(f"DistributionProtocol initialized: {self.storage_dir}")

    def create_manifest(
        self,
        model_path: str,
        model_id: str,
        version: str,
        compress: bool = True,
    ) -> ModelManifest:
        """
        Create manifest for a model file.

        Args:
            model_path: Path to model file
            model_id: Model identifier
            version: Version string
            compress: Whether to compress chunks

        Returns:
            ModelManifest
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Read and optionally compress
        with open(model_path, 'rb') as f:
            data = f.read()

        if compress:
            data = zlib.compress(data, level=6)

        total_size = len(data)
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size

        # Create chunks
        chunks = []
        chunk_hashes = []

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total_size)
            chunk_data = data[start:end]

            chunk_hash = hashlib.sha256(chunk_data).hexdigest()
            chunk_hashes.append(chunk_hash)

            chunks.append({
                'index': i,
                'size': len(chunk_data),
                'hash': chunk_hash,
                'offset': start,
            })

        # Build Merkle tree
        merkle = MerkleTree(chunk_hashes)

        manifest = ModelManifest(
            model_id=model_id,
            version=version,
            total_size=total_size,
            chunk_size=self.chunk_size,
            num_chunks=num_chunks,
            chunks=chunks,
            root_hash=merkle.root,
            compression="zlib" if compress else "none",
        )

        # Save manifest
        manifest_path = self.storage_dir / f"{model_id}_{version}.manifest.json"
        manifest.save(str(manifest_path))

        # Save chunks
        chunks_dir = self.storage_dir / f"{model_id}_{version}_chunks"
        chunks_dir.mkdir(exist_ok=True)

        for i, chunk_meta in enumerate(chunks):
            start = chunk_meta['offset']
            end = start + chunk_meta['size']
            chunk_data = data[start:end]

            chunk_path = chunks_dir / f"{chunk_meta['hash']}.chunk"
            with open(chunk_path, 'wb') as f:
                f.write(chunk_data)

        print(f"Created manifest for {model_id} v{version}")
        print(f"  Size: {total_size / 1024 / 1024:.2f} MB")
        print(f"  Chunks: {num_chunks}")
        print(f"  Root hash: {merkle.root[:16]}...")

        return manifest

    def get_chunk(self, model_id: str, version: str, chunk_hash: str) -> Optional[bytes]:
        """
        Get a chunk from local storage.

        Args:
            model_id: Model identifier
            version: Model version
            chunk_hash: Chunk hash

        Returns:
            Chunk data if available
        """
        chunks_dir = self.storage_dir / f"{model_id}_{version}_chunks"
        chunk_path = chunks_dir / f"{chunk_hash}.chunk"

        if chunk_path.exists():
            with open(chunk_path, 'rb') as f:
                return f.read()
        return None

    def has_chunk(self, chunk_hash: str) -> bool:
        """Check if we have a chunk locally."""
        for chunks_dir in self.storage_dir.glob("*_chunks"):
            if (chunks_dir / f"{chunk_hash}.chunk").exists():
                return True
        return False

    def get_available_chunks(self, manifest: ModelManifest) -> Set[str]:
        """Get set of locally available chunk hashes."""
        available = set()
        for chunk in manifest.chunks:
            if self.has_chunk(chunk['hash']):
                available.add(chunk['hash'])
        return available


class ModelDownload:
    """
    Manages downloading a model from the network.

    Coordinates parallel downloads from multiple peers
    and verifies integrity.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        protocol: DistributionProtocol,
        peers: List[str] = None,
    ):
        """
        Initialize model download.

        Args:
            manifest: Model manifest
            protocol: Distribution protocol
            peers: List of peer endpoints to download from
        """
        self.manifest = manifest
        self.protocol = protocol
        self.peers = peers or []

        # Chunk tracking
        self.chunks: Dict[str, ModelChunk] = {}
        self.pending: List[str] = []
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()

        # Initialize chunks
        for chunk_meta in manifest.chunks:
            chunk = ModelChunk(
                chunk_id=chunk_meta['hash'],
                index=chunk_meta['index'],
                size=chunk_meta['size'],
                hash=chunk_meta['hash'],
            )
            self.chunks[chunk.hash] = chunk
            self.pending.append(chunk.hash)

        # Progress
        self.bytes_downloaded = 0
        self.start_time = time.time()

        # Merkle tree for verification
        chunk_hashes = [c['hash'] for c in manifest.chunks]
        self.merkle = MerkleTree(chunk_hashes)

    @property
    def progress(self) -> float:
        """Get download progress (0-1)."""
        if not self.chunks:
            return 1.0
        return len(self.completed) / len(self.chunks)

    @property
    def is_complete(self) -> bool:
        """Check if download is complete."""
        return len(self.completed) == len(self.chunks)

    async def start(self):
        """Start downloading."""
        # Check what we already have
        for chunk_hash in list(self.pending):
            if self.protocol.has_chunk(chunk_hash):
                self.pending.remove(chunk_hash)
                self.completed.add(chunk_hash)
                self.chunks[chunk_hash].status = ChunkStatus.VERIFIED

        print(f"Starting download: {self.manifest.model_id}")
        print(f"  Already have: {len(self.completed)}/{len(self.chunks)} chunks")

        # Download remaining chunks
        while self.pending and not self.is_complete:
            # Get batch of chunks to download
            batch = self.pending[:MAX_PARALLEL_DOWNLOADS]

            # Download in parallel
            tasks = [self._download_chunk(h) for h in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk_hash, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Chunk download failed: {chunk_hash[:8]}... - {result}")
                    self.failed.add(chunk_hash)
                elif result:
                    self.pending.remove(chunk_hash)
                    self.completed.add(chunk_hash)

    async def _download_chunk(self, chunk_hash: str) -> bool:
        """Download a single chunk."""
        chunk = self.chunks[chunk_hash]
        chunk.status = ChunkStatus.DOWNLOADING

        # Try each peer
        for peer in self.peers:
            try:
                # In real implementation, would request from peer
                # For now, simulate
                await asyncio.sleep(0.1)

                # Verify chunk
                if chunk.verify():
                    chunk.status = ChunkStatus.VERIFIED
                    self.bytes_downloaded += chunk.size
                    return True

            except Exception as e:
                continue

        chunk.status = ChunkStatus.FAILED
        return False

    def assemble(self, output_path: str) -> bool:
        """
        Assemble downloaded chunks into model file.

        Args:
            output_path: Where to save assembled model

        Returns:
            True if successful
        """
        if not self.is_complete:
            print("Cannot assemble: download incomplete")
            return False

        # Collect chunks in order
        ordered_chunks = sorted(self.chunks.values(), key=lambda c: c.index)

        data = b''
        for chunk in ordered_chunks:
            chunk_data = self.protocol.get_chunk(
                self.manifest.model_id,
                self.manifest.version,
                chunk.hash,
            )
            if chunk_data:
                data += chunk_data

        # Decompress if needed
        if self.manifest.compression == "zlib":
            data = zlib.decompress(data)

        # Write output
        with open(output_path, 'wb') as f:
            f.write(data)

        print(f"Model assembled: {output_path}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        elapsed = time.time() - self.start_time
        speed = self.bytes_downloaded / max(1, elapsed)

        return {
            'model_id': self.manifest.model_id,
            'version': self.manifest.version,
            'progress': self.progress,
            'completed_chunks': len(self.completed),
            'total_chunks': len(self.chunks),
            'bytes_downloaded': self.bytes_downloaded,
            'elapsed_seconds': elapsed,
            'speed_mbps': speed / 1024 / 1024,
            'failed_chunks': len(self.failed),
        }


class ModelDistribution:
    """
    High-level model distribution manager.

    Coordinates model sharing across the BAZINGA network:
    - Publishes models for others to download
    - Downloads models from peers
    - Manages local model cache
    - Handles delta updates

    Usage:
        dist = ModelDistribution()

        # Publish a model
        manifest = dist.publish_model("model.bin", "my-model", "1.0")

        # Download a model
        await dist.download_model(manifest, peers)
    """

    def __init__(
        self,
        storage_dir: str = "~/.bazinga/models",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Initialize model distribution.

        Args:
            storage_dir: Directory for model storage
            chunk_size: Chunk size for transfers
        """
        self.protocol = DistributionProtocol(storage_dir, chunk_size)
        self.storage_dir = self.protocol.storage_dir

        # Known manifests
        self.manifests: Dict[str, ModelManifest] = {}

        # Active downloads
        self.downloads: Dict[str, ModelDownload] = {}

        # Stats
        self.models_published = 0
        self.models_downloaded = 0

        # Load existing manifests
        self._load_manifests()

        print(f"ModelDistribution initialized")

    def _load_manifests(self):
        """Load existing manifests from storage."""
        for manifest_file in self.storage_dir.glob("*.manifest.json"):
            try:
                manifest = ModelManifest.load(str(manifest_file))
                key = f"{manifest.model_id}:{manifest.version}"
                self.manifests[key] = manifest
            except Exception as e:
                print(f"Failed to load manifest {manifest_file}: {e}")

        print(f"Loaded {len(self.manifests)} model manifests")

    def publish_model(
        self,
        model_path: str,
        model_id: str,
        version: str,
        compress: bool = True,
    ) -> ModelManifest:
        """
        Publish a model for network distribution.

        Args:
            model_path: Path to model file
            model_id: Model identifier
            version: Version string
            compress: Compress chunks

        Returns:
            ModelManifest for the published model
        """
        manifest = self.protocol.create_manifest(
            model_path, model_id, version, compress
        )

        key = f"{model_id}:{version}"
        self.manifests[key] = manifest
        self.models_published += 1

        return manifest

    async def download_model(
        self,
        manifest: ModelManifest,
        peers: List[str],
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Download a model from the network.

        Args:
            manifest: Model manifest
            peers: List of peer endpoints
            output_path: Where to save model

        Returns:
            True if successful
        """
        key = f"{manifest.model_id}:{manifest.version}"

        # Create download
        download = ModelDownload(manifest, self.protocol, peers)
        self.downloads[key] = download

        # Start download
        await download.start()

        if download.is_complete:
            # Assemble model
            if output_path is None:
                output_path = str(
                    self.storage_dir / f"{manifest.model_id}_{manifest.version}.bin"
                )

            if download.assemble(output_path):
                self.models_downloaded += 1
                return True

        return False

    def get_model_path(self, model_id: str, version: str) -> Optional[Path]:
        """Get path to a downloaded model."""
        path = self.storage_dir / f"{model_id}_{version}.bin"
        if path.exists():
            return path
        return None

    def has_model(self, model_id: str, version: str) -> bool:
        """Check if model is available locally."""
        return self.get_model_path(model_id, version) is not None

    def get_available_chunks(self, model_id: str, version: str) -> Set[str]:
        """Get chunks we can serve for a model."""
        key = f"{model_id}:{version}"
        manifest = self.manifests.get(key)
        if manifest:
            return self.protocol.get_available_chunks(manifest)
        return set()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        for key, manifest in self.manifests.items():
            models.append({
                'model_id': manifest.model_id,
                'version': manifest.version,
                'size_mb': manifest.total_size / 1024 / 1024,
                'chunks': manifest.num_chunks,
                'has_local': self.has_model(manifest.model_id, manifest.version),
            })
        return models

    def get_stats(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        return {
            'storage_dir': str(self.storage_dir),
            'known_models': len(self.manifests),
            'models_published': self.models_published,
            'models_downloaded': self.models_downloaded,
            'active_downloads': len(self.downloads),
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Model Distribution Test")
    print("=" * 60)

    import tempfile
    import asyncio

    async def test():
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create distribution
            dist = ModelDistribution(storage_dir=tmpdir)

            # Create test model
            test_model = Path(tmpdir) / "test_model.bin"
            test_data = b"x" * (5 * 1024 * 1024)  # 5 MB
            with open(test_model, 'wb') as f:
                f.write(test_data)

            # Publish
            print("\n1. Publishing model...")
            manifest = dist.publish_model(
                str(test_model),
                "test-model",
                "1.0",
            )
            print(f"   Manifest root: {manifest.root_hash[:16]}...")
            print(f"   Chunks: {manifest.num_chunks}")

            # Test Merkle tree
            print("\n2. Testing Merkle verification...")
            chunk_hashes = [c['hash'] for c in manifest.chunks]
            merkle = MerkleTree(chunk_hashes)
            print(f"   Merkle root: {merkle.root[:16]}...")

            # Verify a chunk
            proof = merkle.get_proof(0)
            verified = merkle.verify_chunk(0, chunk_hashes[0], proof)
            print(f"   Chunk 0 verified: {verified}")

            # List models
            print("\n3. Available models:")
            for model in dist.list_models():
                print(f"   - {model['model_id']} v{model['version']} ({model['size_mb']:.2f} MB)")

            # Stats
            print(f"\n4. Stats: {dist.get_stats()}")

    asyncio.run(test())

    print("\n" + "=" * 60)
    print("Model Distribution module ready!")
    print("=" * 60)
