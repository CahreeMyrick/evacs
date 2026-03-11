from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Dict
import numpy as np
import random
import os

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass
class Timer:
    name: str
    t0: float = 0.0
    elapsed_ms: float = 0.0

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (perf_counter() - self.t0) * 1000.0

@dataclass
class StageTimes:
    times_ms: Dict[str, float]

    @property
    def total_ms(self) -> float:
        return float(sum(self.times_ms.values()))

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / (s + 1e-12)