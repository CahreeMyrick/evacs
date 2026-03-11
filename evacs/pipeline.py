from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from .config import Config
from .io import load_wav, AudioBuffer
from .preprocess import to_mono, normalize_peak, pad_or_trim, resample_linear
from .features import log_mel, FeatureTensor
from .model import load_model, predict, Prediction
from .utils import Timer, StageTimes

@dataclass
class PipelineResult:
    prediction: Prediction
    stage_times: StageTimes
    features: Optional[FeatureTensor] = None

def classify_file(path: str, cfg: Config, return_features: bool = False) -> PipelineResult:
    times = {}

    with Timer("load") as t:
        buf = load_wav(path, max_duration_sec=cfg.clip_sec, tolerance_sec=cfg.duration_tolerance_sec)
    times["load_ms"] = t.elapsed_ms

    with Timer("preprocess") as t:
        buf2 = to_mono(buf)
        buf2 = resample_linear(buf2, target_sr=cfg.target_sr)
        buf2 = normalize_peak(buf2, peak=0.99)
        buf2 = pad_or_trim(buf2, clip_sec=cfg.clip_sec)
    times["preprocess_ms"] = t.elapsed_ms

    with Timer("features") as t:
        feats = log_mel(buf2, cfg)
    times["features_ms"] = t.elapsed_ms

    with Timer("inference") as t:
        model = load_model(cfg.model_path, labels=cfg.labels)
        pred = predict(model, feats)
    times["inference_ms"] = t.elapsed_ms

    result = PipelineResult(
        prediction=pred,
        stage_times=StageTimes(times_ms=times),
        features=feats if return_features else None,
    )
    return result