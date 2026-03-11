from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    # Audio constraints
    clip_sec: float = 3.0
    duration_tolerance_sec: float = 0.10  # allow up to +50ms
    target_sr: int = 22050  # common for audio ML

    # Features
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 0.0
    fmax: Optional[float] = None  # None => sr/2

    # Normalization
    feature_norm: str = "per_clip"  # {"none","per_clip"}

    # Model
    model_path: str = "models/dummy.json"
    labels: List[str] = field(default_factory=lambda: ["ambulance", "firetruck", "traffic"])

    # Runtime
    max_latency_ms: int = 500

    @staticmethod
    def load(path: str | Path) -> "Config":
        """Load config from a YAML file."""
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        return Config(**data)

    def to_dict(self) -> Dict:
        return {
            "clip_sec": self.clip_sec,
            "target_sr": self.target_sr,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "feature_norm": self.feature_norm,
            "model_path": self.model_path,
            "labels": list(self.labels),
            "max_latency_ms": self.max_latency_ms,
        }