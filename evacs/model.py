from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path
import json
import numpy as np
import os

from .errors import ModelLoadError, InferenceError
from .features import FeatureTensor, vectorize_logmel
from .utils import softmax

@dataclass(frozen=True)
class Prediction:
    label: str
    probs: Dict[str, float]

@dataclass
class ModelHandle:
    kind: str
    labels: List[str]
    payload: Dict[str, Any]
def load_model(model_path: str, labels: List[str]) -> ModelHandle:
    p = Path(model_path)

    if not p.exists():
        # default dummy so pipeline still works
        dummy = {"kind": "dummy_linear", "seed": 0}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dummy, indent=2))

    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise ModelLoadError(f"Failed to read model file: {model_path}") from e

    kind = data.get("kind", "dummy_linear")

    # Use labels from file if present; otherwise use cfg labels.
    file_labels = data.get("labels", None)
    eff_labels = list(file_labels) if file_labels else list(labels)

    if kind == "torchscript_cnn":
        # expects:
        # { "kind": "torchscript_cnn", "pt_path": "models/cnn_logmel.pt", "labels": [...] }

        pt_path = data.get("pt_path", None)
        if not pt_path:
            raise ModelLoadError("Invalid torchscript_cnn model: missing pt_path")

        # Resolve relative paths against the JSON file location
        pt_path_resolved = Path(pt_path)
        if not pt_path_resolved.is_absolute():
            pt_path_resolved = (p.parent / pt_path_resolved).resolve()

        if not pt_path_resolved.is_file():
            raise ModelLoadError(
                f"Invalid torchscript_cnn model: pt_path not found: {pt_path_resolved}"
            )

        # Write back resolved path so inference is stable no matter cwd
        data["pt_path"] = str(pt_path_resolved)

    return ModelHandle(kind=kind, labels=eff_labels, payload=data)
def _predict_dummy(model: ModelHandle, feats: FeatureTensor) -> Prediction:
    X = np.asarray(feats.X, dtype=np.float32)
    stats = np.array([X.mean(), X.std(), X.min(), X.max()], dtype=np.float32)
    k = len(model.labels)
    logits = np.zeros((k,), dtype=np.float32)
    for i in range(k):
        logits[i] = (stats @ np.array([0.7, -0.2, 0.1, 0.4], dtype=np.float32)) + 0.05 * (i - 1)
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    label = model.labels[idx]
    return Prediction(label=label, probs={model.labels[i]: float(probs[i]) for i in range(k)})

def _predict_torchscript_cnn(model: ModelHandle, feats: FeatureTensor) -> Prediction:
    import torch  # local import so numpy baseline still works without torch

    labels = model.labels
    pt_path = model.payload["pt_path"]

    m = torch.jit.load(pt_path, map_location="cpu")
    m.eval()

    X = np.asarray(feats.X, dtype=np.float32)[None, None, :, :]  # (1,1,M,T)
    xt = torch.from_numpy(X)

    with torch.no_grad():
        logits = m(xt).numpy().reshape(-1)  # (K,)
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return Prediction(label=labels[idx], probs={labels[i]: float(probs[i]) for i in range(len(labels))})
def _predict_softmax_linear(model: ModelHandle, feats: FeatureTensor) -> Prediction:
    """
    payload:
      - W: list[list[float]] shape (K, D)
      - b: list[float] shape (K,)
      - x_mean: list[float] shape (D,)
      - x_std: list[float] shape (D,)
      - labels: list[str]
    """
    labels = model.labels
    k = len(labels)

    W = np.asarray(model.payload["W"], dtype=np.float32)   # (K, D)
    b = np.asarray(model.payload["b"], dtype=np.float32)   # (K,)
    x_mean = np.asarray(model.payload.get("x_mean", None), dtype=np.float32) if "x_mean" in model.payload else None
    x_std = np.asarray(model.payload.get("x_std", None), dtype=np.float32) if "x_std" in model.payload else None

    x = vectorize_logmel(feats)  # (D,)
    if x_mean is not None and x_std is not None:
        x = (x - x_mean) / (x_std + 1e-6)

    logits = W @ x + b
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return Prediction(label=labels[idx], probs={labels[i]: float(probs[i]) for i in range(k)})

def predict(model: ModelHandle, feats: FeatureTensor) -> Prediction:
    try:
        if model.kind == "softmax_linear":
            return _predict_softmax_linear(model, feats)
        if model.kind == "torchscript_cnn":
          return _predict_torchscript_cnn(model, feats)
        return _predict_dummy(model, feats)
    except Exception as e:
        raise InferenceError("Inference failed") from e
    
