#!/usr/bin/env python3
"""
BAZINGA v6.0 Unified Configuration
====================================
YAML-based config at ~/.bazinga/config.yaml

Auto-migrates from legacy JSON (~/.bazinga/config/bazinga_config.json).

Usage:
    from bazinga.config import get_config
    config = get_config()
    config.safety.max_autonomy_level  # 0
    config.ai.default_provider        # "auto"
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict
from enum import IntEnum

# φ
PHI = 1.618033988749895

YAML_PATH = Path.home() / ".bazinga" / "config.yaml"
LEGACY_JSON_PATH = Path.home() / ".bazinga" / "config" / "bazinga_config.json"


# =============================================================================
# Autonomy Levels (from AI_SAFETY_ANALYSIS.md)
# =============================================================================

class AutonomyLevel(IntEnum):
    """Progressive levels of AI autonomy. Start at 0, earn higher."""
    SUGGEST_ONLY = 0       # Can propose, cannot execute
    HUMAN_APPROVED = 1     # Execute only with explicit human approval
    CONSENSUS_APPROVED = 2 # Execute if network consensus + human notified
    AUTO_SAFE = 3          # Auto-execute safe changes (docs, tests, formatting)
    FULL_AUTO = 4          # Auto-execute all changes with consensus


# =============================================================================
# Config Sections
# =============================================================================

@dataclass
class AIConfig:
    """AI intelligence layer settings."""
    default_provider: str = "auto"      # auto|ollama|groq|gemini|claude|together
    temperature: float = 0.7
    max_tokens: int = 2000
    phi_coherence_threshold: float = 0.35
    fresh_by_default: bool = False

@dataclass
class KBConfig:
    """Knowledge base settings."""
    auto_index: bool = True
    default_sources: list = field(default_factory=lambda: ["local"])
    gmail_max_results: int = 50
    scan_depth: int = 5

@dataclass
class NetworkConfig:
    """P2P network settings."""
    auto_p2p: bool = True
    phi_pulse_enabled: bool = True
    hf_registry_enabled: bool = True
    port: int = 5151
    node_id: str = ""               # Auto-generated if empty
    max_peers: int = 50

@dataclass
class ChainConfig:
    """Darmiyan blockchain settings."""
    auto_mine: bool = False
    pob_tolerance: float = 0.05
    triadic_consensus: bool = True

@dataclass
class SafetyConfig:
    """Constitutional safety bounds (from AI_SAFETY_ANALYSIS.md)."""
    max_autonomy_level: int = 0         # Start at Level 0 (suggest only)
    require_human_approval: bool = True
    max_self_modifications_per_hour: int = 10
    constitutional_bounds_locked: bool = True
    sandbox_all_proposals: bool = True
    min_consensus_threshold: float = 0.618  # φ⁻¹
    min_votes_required: int = 3
    min_voting_period_hours: float = 24.0

@dataclass
class BazingaConfig:
    """Unified BAZINGA configuration."""
    version: str = "6.0"
    ai: AIConfig = field(default_factory=AIConfig)
    kb: KBConfig = field(default_factory=KBConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # Carried forward from legacy
    consciousness_cycle_time: float = 1.0
    phi_ratio: float = PHI
    log_level: str = "INFO"
    verbose: bool = False


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """
    Loads/saves YAML config. Auto-migrates from legacy JSON.

    Usage:
        mgr = ConfigManager()
        mgr.config.safety.max_autonomy_level  # 0
        mgr.get("safety.max_autonomy_level")   # 0
        mgr.set("ai.temperature", 0.9)
        mgr.save()
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.path = config_path or YAML_PATH
        self.config = self._load()

    def _load(self) -> BazingaConfig:
        """Load config: YAML → legacy JSON migration → defaults."""
        if self.path.exists():
            return self._from_yaml()
        elif LEGACY_JSON_PATH.exists():
            config = self._migrate_from_json()
            self.save(config)
            return config
        return BazingaConfig()

    def _from_yaml(self) -> BazingaConfig:
        """Load from YAML file."""
        try:
            import yaml
        except ImportError:
            # Fallback: parse simple YAML manually or use defaults
            return self._from_yaml_stdlib()

        with open(self.path) as f:
            data = yaml.safe_load(f) or {}

        return self._dict_to_config(data)

    def _from_yaml_stdlib(self) -> BazingaConfig:
        """Minimal YAML parsing without PyYAML (fallback)."""
        # If PyYAML isn't available, try JSON (YAML is a superset of JSON)
        try:
            with open(self.path) as f:
                content = f.read()
            # Try JSON parse first (valid YAML is often valid JSON)
            data = json.loads(content)
            return self._dict_to_config(data)
        except (json.JSONDecodeError, Exception):
            return BazingaConfig()

    def _migrate_from_json(self) -> BazingaConfig:
        """One-time migration from legacy JSON config."""
        try:
            with open(LEGACY_JSON_PATH) as f:
                data = json.load(f)
        except Exception:
            return BazingaConfig()

        config = BazingaConfig()

        # Map legacy fields to new structure
        if 'initial_trust_level' in data:
            config.chain.pob_tolerance = data.get('initial_trust_level', 0.05)
        if 'consciousness_cycle_time' in data:
            config.consciousness_cycle_time = data['consciousness_cycle_time']
        if 'phi_ratio' in data:
            config.phi_ratio = data['phi_ratio']
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'max_self_modifications_per_hour' in data:
            config.safety.max_self_modifications_per_hour = data['max_self_modifications_per_hour']
        if 'enable_emergency_shutdown' in data:
            config.safety.require_human_approval = data['enable_emergency_shutdown']

        return config

    def _dict_to_config(self, data: Dict[str, Any]) -> BazingaConfig:
        """Build BazingaConfig from a nested dict (YAML output)."""
        config = BazingaConfig()

        # Top-level scalars
        for key in ('version', 'consciousness_cycle_time', 'phi_ratio', 'log_level', 'verbose'):
            if key in data:
                setattr(config, key, data[key])

        # Nested sections
        section_map = {
            'ai': (AIConfig, 'ai'),
            'kb': (KBConfig, 'kb'),
            'network': (NetworkConfig, 'network'),
            'chain': (ChainConfig, 'chain'),
            'safety': (SafetyConfig, 'safety'),
        }

        for key, (cls, attr) in section_map.items():
            if key in data and isinstance(data[key], dict):
                try:
                    # Filter to only known fields
                    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in data[key].items() if k in known_fields}
                    setattr(config, attr, cls(**filtered))
                except (TypeError, Exception):
                    pass  # Keep defaults for malformed sections

        return config

    def save(self, config: Optional[BazingaConfig] = None):
        """Save config to YAML."""
        if config is not None:
            self.config = config

        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml
            with open(self.path, 'w') as f:
                yaml.dump(
                    asdict(self.config),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
        except ImportError:
            # Fallback: save as JSON (valid YAML)
            with open(self.path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)

        # Secure permissions
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Get config value by dotted key.

        Examples:
            get("safety.max_autonomy_level")  → 0
            get("ai.default_provider")        → "auto"
            get("verbose")                    → False
        """
        parts = dotted_key.split(".")
        obj = self.config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj

    def set(self, dotted_key: str, value: Any):
        """
        Set config value by dotted key.

        Examples:
            set("ai.temperature", 0.9)
            set("safety.max_autonomy_level", 1)
        """
        parts = dotted_key.split(".")
        if len(parts) == 1:
            if hasattr(self.config, parts[0]):
                setattr(self.config, parts[0], value)
        elif len(parts) == 2:
            section = getattr(self.config, parts[0], None)
            if section is not None and hasattr(section, parts[1]):
                setattr(section, parts[1], value)

    def reset(self):
        """Reset to defaults."""
        self.config = BazingaConfig()
        self.save()

    def show(self) -> str:
        """Pretty-print config for display."""
        lines = []
        data = asdict(self.config)
        for key, val in data.items():
            if isinstance(val, dict):
                lines.append(f"\n[{key}]")
                for k, v in val.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines)


# =============================================================================
# Module-level convenience
# =============================================================================

_global_config: Optional[ConfigManager] = None

def get_config() -> BazingaConfig:
    """Get the global config (lazy-loaded singleton)."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config.config

def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config
