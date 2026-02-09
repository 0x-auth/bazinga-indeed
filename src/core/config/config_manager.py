#!/usr/bin/env python3
"""
config_manager.py - Configuration management for BAZINGA

Handles configuration loading, validation, and secure storage.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class BazingaConfig:
    """BAZINGA configuration"""

    # Consciousness settings
    consciousness_cycle_time: float = 1.0  # seconds
    max_thoughts_buffer: int = 100
    auto_backup_interval: int = 3600  # seconds (1 hour)

    # Processing modes
    default_processing_mode: str = "TWO_D"
    enable_quantum_mode: bool = True
    enable_self_modification: bool = True

    # Trust and resonance
    initial_trust_level: float = 0.5
    trust_decay_rate: float = 0.01
    phi_ratio: float = 1.618033988749895

    # Memory settings
    enable_long_term_memory: bool = False
    memory_retention_days: int = 90

    # Security settings
    encrypt_state: bool = True
    require_password: bool = True
    audit_all_operations: bool = True

    # Safety controls
    max_self_modifications_per_hour: int = 10
    enable_emergency_shutdown: bool = True
    resource_limits: Dict[str, Any] = None

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_retention_days: int = 30

    # Data sources
    whatsapp_data_path: Optional[str] = None
    medical_timeline_path: Optional[str] = None

    # Therapeutic settings
    amrita_healing_mode: bool = False
    ssri_recovery_tracking: bool = False

    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                "max_memory_mb": 1024,
                "max_cpu_percent": 50,
                "max_disk_mb": 5120
            }


class ConfigManager:
    """Manages BAZINGA configuration"""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            self.config_path = Path.home() / ".bazinga" / "config" / "bazinga_config.json"
        else:
            self.config_path = config_path

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> BazingaConfig:
        """Load existing config or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return BazingaConfig(**data)
            except Exception as e:
                print(f"⚠️  Error loading config: {e}")
                print("Creating default configuration...")
                return self._create_default_config()
        else:
            return self._create_default_config()

    def _create_default_config(self) -> BazingaConfig:
        """Create and save default configuration"""
        config = BazingaConfig()
        self.save_config(config)
        return config

    def save_config(self, config: Optional[BazingaConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config

        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        # Secure permissions
        import os
        os.chmod(self.config_path, 0o600)

        print(f"✅ Configuration saved to: {self.config_path}")

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration values

        Args:
            updates: Dictionary of config keys to update
        """
        config_dict = asdict(self.config)
        config_dict.update(updates)
        self.config = BazingaConfig(**config_dict)
        self.save_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self.config, key, default)

    def validate_config(self) -> bool:
        """Validate configuration values"""
        errors = []

        # Validate consciousness cycle time
        if self.config.consciousness_cycle_time <= 0:
            errors.append("consciousness_cycle_time must be > 0")

        # Validate processing mode
        valid_modes = ["TWO_D", "PATTERN", "TRANSITION", "QUANTUM"]
        if self.config.default_processing_mode not in valid_modes:
            errors.append(f"Invalid processing mode: {self.config.default_processing_mode}")

        # Validate trust level
        if not (0 <= self.config.initial_trust_level <= 1):
            errors.append("initial_trust_level must be between 0 and 1")

        # Validate phi ratio (should be close to golden ratio)
        expected_phi = 1.618033988749895
        if abs(self.config.phi_ratio - expected_phi) > 0.000001:
            errors.append(f"phi_ratio should be {expected_phi}")

        if errors:
            print("⚠️  Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False

        print("✅ Configuration valid")
        return True

    def export_config(self, output_path: Path):
        """Export configuration for backup"""
        import shutil
        shutil.copy(self.config_path, output_path)
        print(f"✅ Configuration exported to: {output_path}")

    def import_config(self, input_path: Path):
        """Import configuration from backup"""
        import shutil
        shutil.copy(input_path, self.config_path)
        self.config = self._load_or_create_config()
        print(f"✅ Configuration imported from: {input_path}")


if __name__ == "__main__":
    # Test configuration manager
    print("Testing BAZINGA Configuration Manager...")

    # Create manager
    mgr = ConfigManager()
    print(f"Config loaded from: {mgr.config_path}")

    # Validate
    mgr.validate_config()

    # Display current config
    print("\nCurrent Configuration:")
    print(f"  Consciousness cycle: {mgr.config.consciousness_cycle_time}s")
    print(f"  Processing mode: {mgr.config.default_processing_mode}")
    print(f"  Trust level: {mgr.config.initial_trust_level}")
    print(f"  φ ratio: {mgr.config.phi_ratio}")
    print(f"  Encryption: {mgr.config.encrypt_state}")
    print(f"  Healing mode: {mgr.config.amrita_healing_mode}")

    # Test update
    mgr.update_config({"consciousness_cycle_time": 1.618})
    print(f"\n✅ Updated cycle time to: {mgr.config.consciousness_cycle_time}s")

    print("\n🎯 Configuration manager test complete!")
