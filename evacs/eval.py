from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

from .config import Config
from .pipeline import classify_file

@dataclass
class EvalResult:
    accuracy: float
    confusion: np.ndarray
    labels: List[str]

def evaluate_folder(root: str, cfg: Config) -> EvalResult:
    """
    Expects folder structure:
      root/
        ambulance/*.wav
        firetruck/*.wav
        traffic/*.wav
    """
    labels = cfg.labels
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    conf = np.zeros((len(labels), len(labels)), dtype=np.int32)

    total = 0
    correct = 0

    rootp = Path(root)
    for true_lab in labels:
        folder = rootp / true_lab
        if not folder.exists():
            continue
        for wav in folder.glob("*.wav"):
            pred = classify_file(str(wav), cfg).prediction.label
            i = label_to_idx[true_lab]
            j = label_to_idx[pred]
            conf[i, j] += 1
            total += 1
            correct += int(pred == true_lab)

    acc = (correct / total) if total > 0 else 0.0
    return EvalResult(accuracy=float(acc), confusion=conf, labels=labels)